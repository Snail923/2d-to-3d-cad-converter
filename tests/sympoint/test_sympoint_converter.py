"""Approach 2 (SymPointV2): request format, tiling / ownership, label mapping, viewer and CLI.

The real model needs a CUDA GPU (sympoint_converter/service); here a fake service answers per DXF
layer, so everything on the converter side is tested exactly."""
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from architectural_converter.benchmark.dxf_writer import Style, render_plan
from architectural_converter.benchmark.plan_spec import PlanSpec
from sympoint_converter import BG, CLASSES, export_tiles, labels_from_response, load_config, load_drawing, primitives
from sympoint_converter.__main__ import main
from sympoint_converter.viewer import render_html, summarize

LAYER_CLASS = {"A-WALL": "wall", "A-GLAZ": "window", "A-DOOR": "single door", "I-FURN": "bed"}


def _plan():
    s = PlanSpec("sp", "two rooms, door, windows, furniture")
    s.chain([(0, 0), (8, 0), (8, 5), (0, 5)], 0.3, closed=True)
    s.wall((4, 0), (4, 5), 0.12)
    s.opening_at((4, 2.5), 0.9, "door")
    s.opening_at((2, 0), 1.5, "window")
    s.opening_at((6, 5), 1.2, "window")
    s.furniture.append((6, 2, 1.6, 2.0, 0))
    doc, _ = render_plan(s, Style(layers="aia", door_repr="loose", window_repr="loose", furniture_repr="polyline"))
    p = os.path.join(tempfile.mkdtemp(prefix="spc_"), "sp.dxf")
    doc.saveas(p)
    return p


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Labels cache, pid and log files of the tests never touch ~/.cache."""
    monkeypatch.setenv("SYMPOINT_STATE_DIR", str(tmp_path / "state"))


@pytest.fixture(scope="module")
def drawing_path():
    return _plan()


@pytest.fixture(scope="module")
def drawing(drawing_path):
    return load_drawing(drawing_path)


def fake_answer(request):
    """Like the service: per primitive [class index, score] (by DXF layer); one door instance."""
    tiles = []
    for t in request["tiles"]:
        names = t["layer_names"]
        sem, door = [], []
        for k, lid in enumerate(t["sympoint"]["layerIds"]):
            c = LAYER_CLASS.get(names[lid - 1])
            sem.append([CLASSES.index(c), 0.9] if c else [CLASSES.index("wall"), 0.1])
            if c == "single door":
                door.append(k)
        inst = [{"label": CLASSES.index("single door"), "score": 0.8, "members": door}] if door else []
        tiles.append({"id": t["id"], "semantic": sem, "instances": inst})
    return {"model": "SymPointV2", "tiles": tiles, "seconds": 0.01}


def test_request_matches_sympoint_loader_format(drawing):
    cfg = load_config()
    req = export_tiles(drawing, cfg)
    assert req["tiles"] and req["primitives"] == sum(sum(t["own"]) for t in req["tiles"])
    for t in req["tiles"]:
        sp = t["sympoint"]
        n = len(t["ids"])
        assert all(len(a) == 8 for a in sp["args"])
        for k in ("commands", "lengths", "semanticIds", "instanceIds", "rgb", "layerIds", "widths"):
            assert len(sp[k]) == n
        assert set(sp["commands"]) <= {0, 1, 2, 3} and set(sp["semanticIds"]) == {BG}
        assert min(sp["layerIds"]) >= 1                     # SymPointV2 pads with max(layerIds) + 1
        a = np.array(sp["args"]).reshape(-1, 4, 2)
        # the loader divides x by width and y by height: everything must land in [0, 1]
        assert a[..., 0].min() >= -1e-6 and a[..., 0].max() <= sp["width"] + 1e-6
        assert a[..., 1].min() >= -1e-6 and a[..., 1].max() <= sp["height"] + 1e-6
        assert np.all(np.array(sp["lengths"]) > 0)


def test_every_primitive_owned_once_also_when_tiles_are_split(drawing):
    prims = primitives(drawing, load_config())
    for cfg in (load_config({"tile_m": 3.0, "overlap_m": 1.0}),
                load_config({"tile_m": 30.0, "max_primitives_per_tile": 15})):
        req = export_tiles(drawing, cfg, prims)
        owned = [p for t in req["tiles"] for p, o in zip(t["pidx"], t["own"]) if o]
        assert sorted(owned) == list(range(len(prims)))
    split = export_tiles(drawing, load_config({"tile_m": 30.0, "max_primitives_per_tile": 15}), prims)
    assert any("q" in t["id"] for t in split["tiles"])


def test_labels_map_back_to_dxf_entities(drawing):
    cfg = load_config({"tile_m": 4.0, "overlap_m": 1.0})
    prims = primitives(drawing, cfg)
    req = export_tiles(drawing, cfg, prims)
    lab = labels_from_response(req, fake_answer(req), min_score=cfg["min_label_score"])
    raw = drawing.raw_entities
    assert lab["entities"]
    for rid, e in lab["entities"].items():
        assert e["label"] == LAYER_CLASS[raw[rid].layer]
    labelled_layers = {raw[p[0]].layer for k, p in enumerate(prims) if k in lab["primitives"]}
    assert labelled_layers <= set(LAYER_CLASS)            # low-score answers (score 0.1) were dropped
    assert {i["label"] for i in lab["instances"]} == {"single door"}
    assert all(raw[m].layer == "A-DOOR" for i in lab["instances"] for m in i["members"])


def test_summary_and_viewer(drawing):
    cfg = load_config()
    prims = primitives(drawing, cfg)
    req = export_tiles(drawing, cfg, prims)
    lab = labels_from_response(req, fake_answer(req), cfg["min_label_score"])
    s = summarize(prims, lab, drawing)
    assert set(s["groups"]) <= {"wall", "door", "window", "furniture"} and s["groups"]["wall"] > 0.3
    wall_layer = next(x for x in s["layers"] if x["layer"] == "A-WALL")
    assert wall_layer["dominant_class"] == "wall" and wall_layer["groups"]["wall"] == pytest.approx(1.0)
    page = render_html(prims, lab, drawing, "t", {"file": "sp.dxf"})
    assert page.startswith("<!doctype html>") and "</script></body></html>" in page and "__PAYLOAD__" not in page
    payload = json.loads(page.split("const P=", 1)[1].split(";const D=P.data", 1)[0])
    assert len(payload["data"]["cls"]) == len(prims)
    assert payload["data"]["offs"][-1] * 2 == len(payload["data"]["coords"])


@pytest.fixture()
def fake_service():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            out = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            self._send({"ok": True, "model": "SymPointV2", "device": "fake"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send(fake_answer(req))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_cli_end_to_end(drawing_path, fake_service, tmp_path):
    assert main(["health", "--url", fake_service]) == 0
    assert main(["label", drawing_path, "--url", fake_service, "-o", str(tmp_path), "--tile-m", "3", "30"]) == 0
    for sfx in ("_t3", "_t30"):
        for ext in (".sympoint.html", ".sympoint.labels.json", ".sympoint.response.json"):
            assert (tmp_path / f"sp{sfx}{ext}").exists()
    lab = json.loads((tmp_path / "sp_t30.sympoint.labels.json").read_text())
    assert all(e["handle"] and e["label"] == LAYER_CLASS[e["layer"]] for e in lab["entities"].values())
    # offline round trip: request file, then the saved answer gives the same labels
    off = tmp_path / "off"
    assert main(["label", drawing_path, "-o", str(off), "--tile-m", "30", "--offline"]) == 0
    req = json.loads((off / "sp.sympoint.request.json").read_text())
    (off / "resp.json").write_text(json.dumps(fake_answer(req)))
    assert main(["label", drawing_path, "-o", str(off), "--tile-m", "30", "--response", str(off / "resp.json")]) == 0
    lab2 = json.loads((off / "sp.sympoint.labels.json").read_text())
    assert lab2["entities"] == lab["entities"]


def test_unreachable_service_is_reported(drawing_path, tmp_path, capsys):
    assert main(["label", drawing_path, "--url", "http://127.0.0.1:9", "-o", str(tmp_path)]) == 1
    assert "Cannot reach the SymPointV2 service" in capsys.readouterr().err


def test_approach_1_does_not_depend_on_approach_2():
    root = os.path.join(os.path.dirname(__file__), "..", "..", "architectural_converter")
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".py"):
                assert "sympoint_converter" not in open(os.path.join(dp, f), encoding="utf-8").read(), f


def test_share_and_mass_answers(drawing):
    """Current service: [class, share 0-1, mass]. Lines the model barely covers (low mass) are
    unlabelled; old raw summed scores (> 1) are clamped."""
    cfg = load_config()
    req = export_tiles(drawing, cfg)
    ans = fake_answer(req)
    for t in ans["tiles"]:
        t["semantic"] = [[c, s, 1.2 if s > 0.5 else 0.05] for c, s in t["semantic"]]
        t["semantic"][0][1] = 0.95
    lab = labels_from_response(req, ans, cfg["min_label_score"], cfg["min_label_mass"])
    assert lab["entities"] and all(e["score"] <= 1 for e in lab["entities"].values())
    raw = drawing.raw_entities
    assert all(raw[rid].layer in LAYER_CLASS for rid in lab["entities"])
    old = fake_answer(req)
    for t in old["tiles"]:
        t["semantic"] = [[c, 1.4] for c, _ in t["semantic"]]
    lab_old = labels_from_response(req, old, cfg["min_label_score"])
    assert all(e["score"] == 1.0 for e in lab_old["entities"].values())
