import json
import os
import sqlite3
import tempfile

import pytest

from architectural_converter.ingestion.dwg_converter import DWGConversionError, convert_dwg_to_dxf
from architectural_converter.models.base_entity import Status
from architectural_converter.pipeline.database_exporter import export_json, export_sqlite, phase2_view
from architectural_converter.pipeline.extraction_pipeline import STAGES, ExtractionPipeline
from architectural_converter.reconstruction3d import build_3d

HERE = os.path.dirname(__file__)
DRAW = os.path.join(HERE, "..", "..", "architectural_converter", "benchmark", "drawings")


@pytest.fixture(scope="module")
def apt():
    p = ExtractionPipeline()
    return p, p.process(os.path.join(DRAW, "01_residential_apartment.dxf"))


def test_model_contract(apt):
    _, st = apt
    m = st.model
    assert m["schema"] == "archmodel" and m["units"] == "m"
    for key in ("walls", "openings", "rooms", "columns", "relationships", "validation", "hypotheses", "source"):
        assert key in m
    w = next(w for w in m["walls"] if w["status"] == "ACCEPTED")
    for key in ("id", "geometry", "thickness", "classification", "confidence", "status", "detection_methods",
                "source_entities", "evidence", "height"):
        assert key in w
    assert w["height"] is None  # unknown stays unknown
    assert w["source_entities"] and {"handle", "layer", "type"} <= set(w["source_entities"][0])
    o = next(o for o in m["openings"] if o["type"] == "door")
    assert o["host_wall"] and o["height"] is None
    json.dumps(m)  # serialisable


def test_exports(apt):
    _, st = apt
    d = tempfile.mkdtemp()
    export_json(st.model, os.path.join(d, "m.json"))
    export_sqlite(st.model, os.path.join(d, "m.sqlite"))
    con = sqlite3.connect(os.path.join(d, "m.sqlite"))
    n = con.execute("select count(*) from walls where status='ACCEPTED'").fetchone()[0]
    assert n == sum(1 for w in st.model["walls"] if w["status"] == "ACCEPTED")
    assert con.execute("select count(*) from source_entities").fetchone()[0] > 0
    assert con.execute("select count(*) from relationships where type='hosts'").fetchone()[0] > 0


def test_corrections_rerun(apt):
    p, st = apt
    wid = next(w.id for w in st.building.walls.values() if w.status == Status.ACCEPTED)
    corr = {"status_overrides": {wid: "REJECTED"},
            "added_walls": [{"start": [1.0, 1.0], "end": [1.0, 2.5], "thickness": 0.1, "note": "test"}]}
    st2 = p.rerun(st, corr)
    assert st2.building.walls[wid].status == Status.REJECTED
    assert "manual_wall_0001" in st2.building.walls
    assert st2.building.walls["manual_wall_0001"].status == Status.ACCEPTED
    assert st2.model["corrections"]["status_overrides"][wid] == "REJECTED"
    st3 = p.rerun(st2, {})  # undo
    assert st3.building.walls[wid].status == Status.ACCEPTED


def test_stages_individually():
    p = ExtractionPipeline()
    st = p.process(os.path.join(DRAW, "07_diagonal_curved.dxf"), until="context")
    assert st.completed == STAGES[:3]
    for stage in STAGES[3:]:
        p.run_stage(st, stage)
    assert st.model is not None


def test_phase2_3d(apt):
    p, st = apt
    m3 = build_3d(st.model, p.cfg)
    assert m3.metadata["counts"]["wall"] > 0 and m3.metadata["counts"]["floor"] == 7
    assert "wall_height_m" in m3.metadata["defaults_used"]
    glb = m3.export_glb()
    assert glb[:4] == b"glTF"
    v = phase2_view(st.model)
    assert all(w["status"] != "REJECTED" for w in v["walls"])


def test_dwg_converter_reports_missing_oda(monkeypatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("ODA_FILE_CONVERTER", raising=False)
    f = tempfile.NamedTemporaryFile(suffix=".dwg", delete=False)
    f.write(b"AC1032")
    f.close()
    from architectural_converter.ingestion import dwg_converter
    monkeypatch.setattr(dwg_converter, "DEFAULT_PATHS", [])
    with pytest.raises(DWGConversionError):
        convert_dwg_to_dxf(f.name)


def test_dwg_converter_invokes_oda(monkeypatch, tmp_path):
    exe = tmp_path / "ODAFileConverter"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    dwg = tmp_path / "a.dwg"
    dwg.write_bytes(b"x")
    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"] = cmd
        out = cmd[2]
        with open(os.path.join(out, "a.dxf"), "w") as fh:
            fh.write("0\nEOF\n")

        class R:
            returncode = 0
            stderr = ""
        return R()
    path = convert_dwg_to_dxf(str(dwg), converter=str(exe), runner=fake_run)
    assert path.endswith("a.dxf") and calls["cmd"][4] == "DXF"
