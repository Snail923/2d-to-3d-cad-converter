"""Approach 2 step 2: SymPointV2 labels -> building model with approach 1's build stages.

The labels come from an *oracle* that reads the benchmark's layer names (what a perfect model
would answer), so these tests measure the extraction + build half of approach 2 exactly; the
real model's quality is measured with ``compare`` on a GPU machine."""
import json
import os
import socket
import subprocess
import sys
import textwrap

import pytest

from architectural_converter.benchmark.dxf_writer import LAYER_SCHEMES
from architectural_converter.benchmark.generate import DEFAULT_DIR
from architectural_converter.benchmark.metrics import evaluate
from architectural_converter.models.base_entity import Status
from sympoint_converter import CLASSES, load_config
from sympoint_converter.__main__ import main
from sympoint_converter.compare import compare_one, to_html
from sympoint_converter.pipeline import SymPointPipeline
from sympoint_converter.sympoint_io import export_tiles, load_drawing

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROLE = {"wall": "wall", "door": "single door", "window": "window", "column": "wall", "furniture": "bed", "hatch": "wall"}
ORACLE = {}
for _sch in LAYER_SCHEMES.values():
    for _role, _ly in _sch.items():
        if _role in ROLE and _ly != "0":
            ORACLE.setdefault(_ly, ROLE[_role])


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SYMPOINT_STATE_DIR", str(tmp_path / "state"))


def oracle_answer(req):
    tiles = []
    for t in req["tiles"]:
        sem = []
        for lid in t["sympoint"]["layerIds"]:
            c = ORACLE.get(t["layer_names"][lid - 1])
            sem.append([CLASSES.index(c), 0.95, 1.0] if c else [CLASSES.index("wall"), 0.1, 0.05])
        tiles.append({"id": t["id"], "semantic": sem, "instances": []})
    return {"model": "oracle", "tiles": tiles}


def _bench(name):
    p = os.path.join(DEFAULT_DIR, f"{name}.dxf")
    with open(p.replace(".dxf", ".reference.json"), encoding="utf-8") as fh:
        return p, json.load(fh)


def _run(path, cfg=None):
    cfg = cfg or load_config()
    req = export_tiles(load_drawing(path), cfg)
    return SymPointPipeline(cfg, response=oracle_answer(req), labels_progress=lambda m: None)


@pytest.mark.parametrize("name", ["01_residential_apartment", "02_multi_room_residential", "03_commercial",
                                  "07_diagonal_curved", "11_single_line", "13_wide_polyline"])
def test_perfect_labels_rebuild_the_benchmark(name):
    """Double lines, closed polylines and loose symbols, columns, curved walls, single-line walls
    and wide polylines: with perfect labels the build is (almost) exact."""
    path, ref = _bench(name)
    st = _run(path).process(path)
    m = evaluate(st.model, ref)
    assert m["wall_precision"] >= 0.95 and m["wall_recall"] >= 0.95, m
    assert m["room_precision"] >= 0.95 and m["room_recall"] >= 0.9, m
    for k in ("door_precision", "door_recall", "window_precision", "window_recall", "column_precision", "column_recall"):
        assert m[k] >= 0.9, (k, m)
    hosted = [o for o in st.model["openings"] if o["type"] in ("door", "window") and o["status"] != "REJECTED"]
    assert hosted and all(o["host_wall"] for o in hosted)
    assert "sympoint_wall_faces" in {d for w in st.model["walls"] for d in w["detection_methods"]} or name in (
        "11_single_line", "13_wide_polyline")
    assert st.logs["detect"]["approach"] == "sympoint"
    assert not any(k.startswith("detect:") and k.split(":")[1].endswith("_detector") for k in st.timings)


def test_columns_and_no_end_cap_walls():
    path, ref = _bench("03_commercial")
    st = _run(path).process(path)
    assert len([c for c in st.model["columns"] if c["status"] != "REJECTED"]) == len(ref["columns"])
    ts = [w["thickness"] for w in st.model["walls"] if w["status"] != "REJECTED" and w["thickness"]]
    assert max(ts) <= 0.31                       # no "wall" across a door / window gap


def test_corrections_and_rerun():
    path, _ = _bench("01_residential_apartment")
    pipe = _run(path)
    st = pipe.process(path)
    wid = next(w["id"] for w in st.model["walls"] if w["status"] == "ACCEPTED")
    st = pipe.rerun(st, {"status_overrides": {wid: "REJECTED"}})
    w = next(w for w in st.model["walls"] if w["id"] == wid)
    assert w["status"] == "REJECTED" and w["status_reason"] == "manual correction"


def test_compare_one_and_report():
    path, _ = _bench("01_residential_apartment")
    cfg = load_config()
    req = export_tiles(load_drawing(path), cfg)
    r = compare_one(path, cfg, response=oracle_answer(req), progress=lambda m: None)
    assert r["approach1"]["scores"]["wall_recall"] >= 0.95 and r["approach2"]["scores"]["wall_recall"] >= 0.95
    assert r["agreement"]["approach1_wall_length_found"] >= 0.9
    assert r["approach2"]["counts"]["doors"] == r["approach1"]["counts"]["doors"]
    page = to_html([r])
    assert page.count("<svg") >= 2 and "approach 2" in page


def test_build_cli_outputs(tmp_path):
    path, _ = _bench("01_residential_apartment")
    cfg = load_config()
    resp = tmp_path / "resp.json"
    resp.write_text(json.dumps(oracle_answer(export_tiles(load_drawing(path), cfg))))
    assert main(["build", path, "--response", str(resp), "-o", str(tmp_path)]) == 0
    for ext in (".model.json", ".model.sqlite", ".overlay.svg", ".report.html", ".html", ".glb"):
        assert (tmp_path / f"01_residential_apartment.sympoint{ext}").exists(), ext
    m = json.loads((tmp_path / "01_residential_apartment.sympoint.model.json").read_text())
    assert m["statistics"]["counts"]["rooms"].get("ACCEPTED", 0) >= 7


# ------------------------------------------------------------------ automatic service
FAKE_SERVER = textwrap.dedent('''
    import argparse, json, os, sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    sys.path.insert(0, {root!r})
    from sympoint_converter import CLASSES
    ORACLE = {oracle!r}
    ap = argparse.ArgumentParser()
    for k in ("--config", "--checkpoint", "--host", "--port"):
        ap.add_argument(k)
    a = ap.parse_args()
    crash_flag = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_once")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass
        def _send(self, code, obj):
            out = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        def do_GET(self):
            self._send(200, {{"ok": True, "model": "SymPointV2", "device": "fake GPU"}})
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if os.path.exists(crash_flag):
                os.remove(crash_flag)
                self._send(500, {{"error": "AcceleratorError: CUDA error ... the service stops now. Restart it"}})
                os._exit(3)
            tiles = []
            for t in req["tiles"]:
                sem = []
                for lid in t["sympoint"]["layerIds"]:
                    c = ORACLE.get(t["layer_names"][lid - 1])
                    sem.append([CLASSES.index(c), 0.95, 1.0] if c else [32, 0.1, 0.05])
                tiles.append({{"id": t["id"], "semantic": sem, "instances": []}})
            self._send(200, {{"model": "SymPointV2", "tiles": tiles, "seconds": 0.0}})
    ThreadingHTTPServer((a.host, int(a.port)), H).serve_forever()
''')


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "sympointv2"
    home.mkdir()
    (home / "sympoint_server.py").write_text(FAKE_SERVER.format(root=ROOT, oracle=ORACLE))
    (home / "weights.pth").write_bytes(b"")
    (home / "svg_pointT_release.yaml").write_text("")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("SYMPOINT_HOME", str(home))
    monkeypatch.setenv("SYMPOINT_PYTHON", sys.executable)
    monkeypatch.setenv("SYMPOINT_URL", f"http://127.0.0.1:{port}")
    yield home
    main(["server", "stop"])


def test_service_starts_automatically_and_labels_are_cached(fake_home, tmp_path, capsys):
    path, _ = _bench("01_residential_apartment")
    assert main(["build", path, "-o", str(tmp_path / "a"), "--no-3d"]) == 0
    out = capsys.readouterr().out
    assert "starting the SymPointV2 service" in out and "labels: service" in out
    assert main(["server", "status"]) == 0 and '"running": true' in capsys.readouterr().out
    assert main(["server", "stop"]) == 0
    assert main(["build", path, "-o", str(tmp_path / "b"), "--no-3d"]) == 0      # no service needed now
    assert "labels: cache" in capsys.readouterr().out


def test_service_restarts_once_after_a_cuda_crash(fake_home, tmp_path, capsys):
    path, _ = _bench("01_residential_apartment")
    assert main(["server", "start"]) == 0
    (fake_home / "crash_once").write_text("1")
    assert main(["build", path, "-o", str(tmp_path), "--no-3d", "--no-cache"]) == 0
    out = capsys.readouterr().out
    assert "restarting the SymPointV2 service once" in out and "labels: service" in out


def test_missing_setup_is_explained(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SYMPOINT_HOME", str(tmp_path / "nowhere"))
    monkeypatch.setenv("SYMPOINT_URL", "http://127.0.0.1:9")
    path, _ = _bench("01_residential_apartment")
    assert main(["build", path, "-o", str(tmp_path), "--no-cache"]) == 1
    assert "SymPointV2 checkout not found" in capsys.readouterr().err


def test_approach_1_is_unchanged_by_approach_2_config():
    from architectural_converter.pipeline.pipeline_config import PipelineConfig
    base = PipelineConfig()
    assert "sympoint_label" not in base["detection"]["weights"]["wall"]
    sp = SymPointPipeline(load_config())
    assert sp.cfg["detection"]["weights"]["wall"]["sympoint_label"] > 0
    assert sp.cfg["detection"]["weights"]["wall"]["parallelism"] == base["detection"]["weights"]["wall"]["parallelism"]
    assert Status.ACCEPTED.value == "ACCEPTED"
    subprocess.run([sys.executable, "-c", "import architectural_converter.pipeline.extraction_pipeline"], check=True, cwd=ROOT)
