import json
import os

from architectural_converter import web_bridge

DRAW = os.path.join(os.path.dirname(__file__), "..", "..", "architectural_converter", "benchmark", "drawings")


def test_bridge_roundtrip():
    out = json.loads(web_bridge.process(os.path.join(DRAW, "12_mline.dxf")))
    assert {"model", "view", "mesh", "context"} <= set(out)
    assert out["view"]["lines"] and out["mesh"]["meshes"]
    wid = next(w["id"] for w in out["model"]["walls"] if w["status"] == "ACCEPTED")
    out2 = json.loads(web_bridge.rerun(json.dumps({"status_overrides": {wid: "REJECTED"}})))
    assert next(w for w in out2["model"]["walls"] if w["id"] == wid)["status"] == "REJECTED"
    assert web_bridge.export_sqlite_bytes()[:15] == b"SQLite format 3"
    assert web_bridge.export_glb_bytes()[:4] == b"glTF"
    assert "<html>" in web_bridge.export_report_html()
    with open(os.path.join(DRAW, "12_mline.reference.json"), encoding="utf-8") as fh:
        m = json.loads(web_bridge.evaluate(fh.read()))
    assert "wall_precision" in m
