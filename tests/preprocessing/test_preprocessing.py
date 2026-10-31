import numpy as np

from architectural_converter.geometry.geometry_adapter import build_segments, build_strokes
from architectural_converter.ingestion import ingest
from architectural_converter.pipeline.pipeline_config import PipelineConfig
from architectural_converter.preprocessing.coordinate_normalizer import normalize
from architectural_converter.preprocessing.duplicate_removal import mark_duplicates
from architectural_converter.preprocessing.layer_analyzer import category_scores, tokenize
from architectural_converter.preprocessing.tolerance import estimate_tolerances
from architectural_converter.preprocessing.unit_detection import detect_units
from tests.helpers import box_room, new_doc, save


def _prep(path):
    d = ingest(path)
    d.units = detect_units(d)
    normalize(d)
    return d


def test_unit_inference_without_header():
    for k, unit in ((1000, "mm"), (100, "cm"), (1, "m")):
        doc, msp = new_doc(insunits=0)
        box_room(msp, 8, 6, 0.2, k=k)
        doc.blocks.new("DR").add_arc((0, 0), 1, 0, 90)
        msp.add_blockref("DR", (2 * k, 0.2 * k), dxfattribs={"xscale": 0.9 * k, "yscale": 0.9 * k})
        d = _prep(save(doc))
        assert d.units.unit == unit, (k, d.units)


def test_wrong_header_is_overridden_by_geometry():
    doc, msp = new_doc(insunits=6)  # says metres, geometry is mm
    box_room(msp, 8, 6, 0.2, k=1000)
    d = _prep(save(doc))
    assert d.units.unit == "mm"
    assert "warning" in d.units.evidence


def test_offset_normalisation_preserves_raw_coordinates():
    doc, msp = new_doc(insunits=6)
    msp.add_line((500000, 4000000), (500006, 4000000))
    d = _prep(save(doc))
    g = [g for g in d.entities if g.kind == "line"][0]
    assert np.all(np.abs(g.points) < 10)
    raw = d.raw_entities[g.raw_id].raw_coords
    assert raw[0] == [500000.0, 4000000.0]
    back = d.transform.inverse(g.points)
    assert np.allclose(back[0], [500000, 4000000])


def test_duplicates_and_fragment_strokes():
    doc, msp = new_doc()
    msp.add_line((0, 0), (5, 0))
    msp.add_line((0, 0), (5, 0))          # exact
    msp.add_line((1, 0), (2, 0))          # contained
    msp.add_line((5.002, 0), (9, 0))      # fragment with 2 mm gap
    d = _prep(save(doc))
    segs = build_segments(d)
    cfg = PipelineConfig()
    tol = estimate_tolerances(d, segs, cfg)
    stats = mark_duplicates(segs, tol.duplicate, tol.angle)
    assert stats["exact"] == 1 and stats["contained"] == 1
    strokes = build_strokes(segs, tol.snap, tol.angle)
    assert len(strokes) == 1 and abs(strokes[0].length - 9) < 1e-6
    assert len(strokes[0].members) == 2


def test_layer_tokens_and_hints():
    cats = PipelineConfig()["layer_semantics"]["categories"]
    assert tokenize("A-WALL-FULL") == ["a", "wall", "full"]
    assert tokenize("MurPorteur") == ["mur", "porteur"]
    assert category_scores("A-WALL", cats)["wall"] == 1.0
    assert category_scores("MURS", cats)["wall"] == 1.0
    assert "door" in category_scores("A-DOOR", cats)
    assert "furniture" in category_scores("I-FURN", cats)
    assert category_scores("L01", cats) == {}
