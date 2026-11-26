"""Phase 2 meshes carry the numbers they are built from (``params``), so the web 3D editor can rebuild and
edit elements without the pipeline. The params must describe the same geometry as the vertices."""
import json
import os

import numpy as np
import pytest

from architectural_converter.benchmark.generate import DEFAULT_DIR
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.reconstruction3d import build_3d

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mesh(path):
    pipe = ExtractionPipeline()
    st = pipe.process(path)
    return build_3d(st.model, pipe.cfg).to_json()


def _check(j):
    walls = prisms = 0
    for m in j["meshes"]:
        p = m.get("params")
        if not p:
            continue
        P = np.array(m["positions"]).reshape(-1, 3)
        if p["kind"] == "wall":
            a, b, t = np.array(p["start"]), np.array(p["end"]), p["thickness"]
            u = (b - a) / np.linalg.norm(b - a)
            n = np.array([-u[1], u[0]])
            s, c = (P[:, :2] - a) @ u, (P[:, :2] - a) @ n
            assert s.min() > -1e-3 and s.max() < np.linalg.norm(b - a) + 1e-3, m["id"]
            assert np.abs(c).max() <= t / 2 + 1e-3, m["id"]
            assert abs(P[:, 2].min() - p["z0"]) < 1e-3 and abs(P[:, 2].max() - (p["z0"] + p["height"])) < 1e-3, m["id"]
            for o in p["openings"]:
                assert 0 <= o["s0"] < o["s1"] <= np.linalg.norm(b - a) + 1e-3 and 0 <= o["z0"] < o["z1"] <= p["height"] + 1e-3
            walls += 1
        elif p["kind"] == "prism":
            R = np.array(p["ring"])
            assert np.allclose(P[:, :2].min(0), R.min(0), atol=1e-3) and np.allclose(P[:, :2].max(0), R.max(0), atol=1e-3), m["id"]
            assert abs(P[:, 2].min() - p["z0"]) < 1e-3 and abs(P[:, 2].max() - p["z1"]) < 1e-3, m["id"]
            prisms += 1
    return walls, prisms


@pytest.mark.parametrize("name", ["01_residential_apartment", "03_commercial", "07_diagonal_curved"])
def test_params_describe_the_mesh(name):
    j = _mesh(os.path.join(DEFAULT_DIR, f"{name}.dxf"))
    walls, prisms = _check(j)
    assert walls >= 5 and prisms >= 3
    assert any(m.get("params", {}).get("kind") == "glass_of" for m in j["meshes"])


def test_params_follow_stacked_floors(tmp_path):
    """Multi-storey: the params are moved with the vertices onto each storey's level."""
    from architectural_converter.benchmark.permit_set import build_permit_set
    doc = build_permit_set()
    doc = doc[0] if isinstance(doc, tuple) else doc
    path = str(tmp_path / "permit.dxf")
    doc.saveas(path)
    j = _mesh(path)
    assert j["metadata"].get("stacked_in_3d")
    _check(j)
    zs = {round(m["params"]["z0"], 2) for m in j["meshes"] if (m.get("params") or {}).get("kind") == "wall"}
    assert len(zs) >= 3, zs                     # walls on several storeys, each at its own level
