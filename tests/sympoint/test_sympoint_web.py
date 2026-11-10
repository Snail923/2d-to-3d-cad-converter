"""Web app integration of approach 2: the browser bridge (runs in Pyodide; tested here in CPython)
and the files shipped to the browser."""
import importlib.util
import json
import os

import pytest

from architectural_converter import web_bridge as wb1
from sympoint_converter import web_bridge as wb2

from .test_sympoint_build import _bench, oracle_answer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def both():
    """Approach 1 then approach 2 on the same drawing, like the page's 'Compare both'."""
    path, ref = _bench("01_residential_apartment")
    wb2.reset()
    p1 = json.loads(wb1.process(path))
    wb2.remember("rules")
    req = json.loads(wb2.request(path, json.dumps({"tile_m": 20})))
    p2 = json.loads(wb2.process(path, json.dumps(oracle_answer(req)), None, json.dumps({"tile_m": 20})))
    return path, ref, p1, p2


def test_payloads_of_both_methods(both):
    _, _, p1, p2 = both
    assert "sympoint" not in p1 and p1["model"]["walls"]
    sp = p2["sympoint"]
    assert sp["classes"][32] == "wall" and len(sp["colors"]) == len(sp["classes"])
    assert sp["entity_class"] and all(len(v) == 2 for v in sp["entity_class"].values())
    assert sp["summary"]["groups"]["wall"] > 0.3 and sp["extract"]["walls"] > 0
    assert {"model", "view", "mesh", "context"} <= set(p2)            # same payload as approach 1
    assert all("sympoint_wall_faces" in w["detection_methods"] for w in p2["model"]["walls"] if not w["is_hypothesis"]
               and w["detection_methods"] and w["detection_methods"] != ["manual"])


def test_switching_compare_and_scores(both):
    _, ref, p1, p2 = both
    c = json.loads(wb2.compare())
    assert c["rules"]["doors"] == c["sympoint"]["doors"] == 6
    assert c["agreement"]["wall length of approach 1 found by approach 2"] >= 0.95
    sc = json.loads(wb2.evaluate_both(json.dumps(ref)))
    assert sc["rules"]["wall_recall"] >= 0.95 and sc["sympoint"]["wall_recall"] >= 0.95
    # exports / evaluate act on the active result
    assert wb2.activate("rules")
    assert json.loads(wb1.export_model_json())["walls"][0]["detection_methods"] != ["sympoint_wall_faces"]
    assert wb2.activate("sympoint")
    assert "sympoint_wall_faces" in json.loads(wb1.export_model_json())["walls"][0]["detection_methods"]


def test_corrections_rerun_per_method(both):
    _, _, p1, p2 = both
    wb2.activate("sympoint")
    wid = next(w["id"] for w in p2["model"]["walls"] if w["status"] == "ACCEPTED")
    out = json.loads(wb2.rerun(json.dumps({"status_overrides": {wid: "REJECTED"}})))
    assert "sympoint" in out and next(w for w in out["model"]["walls"] if w["id"] == wid)["status"] == "REJECTED"
    wb2.activate("rules")
    out1 = json.loads(wb2.rerun(json.dumps({})))
    assert "sympoint" not in out1
    wb2.activate("sympoint")                     # the corrected approach-2 state was kept
    w = next(w for w in json.loads(wb1.export_model_json())["walls"] if w["id"] == wid)
    assert w["status"] == "REJECTED"


def test_site_ships_both_packages_but_not_the_gpu_service():
    spec = importlib.util.spec_from_file_location("build_site", os.path.join(ROOT, "scripts", "build_site.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    files = mod.engine_files()
    for f in ("architectural_converter/web_bridge.py", "sympoint_converter/web_bridge.py",
              "sympoint_converter/service_manager.py", "sympoint_converter/config.json"):
        assert f in files, f
    assert not any(f.startswith("sympoint_converter/service/") for f in files)
    assert not any("/drawings/" in f or "__pycache__" in f for f in files)


def test_safe_mode_and_floor_layout_switch(both):
    path, _, _, p2 = both
    assert p2["mesh"]["metadata"].get("stacked_in_3d") is False          # approach 2: side by side
    req = json.loads(wb2.request(path, json.dumps({"tile_m": 20})))
    p = json.loads(wb2.process(path, json.dumps(oracle_answer(req)), None, json.dumps({"tile_m": 20}), True))
    logs = p["model"]["statistics"]["logs"]
    assert logs["topology"]["door_rules"].get("skipped") == "disabled" and logs["topology"]["recovery"].get("skipped")
    m = json.loads(wb2.mesh(True))
    assert "meshes" in m and "stacked_in_3d" in m["metadata"]
