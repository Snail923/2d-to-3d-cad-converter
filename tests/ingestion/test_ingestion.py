import math

from architectural_converter.ingestion import ingest
from tests.helpers import new_doc, save


def test_entities_handles_layers_and_nested_blocks():
    doc, msp = new_doc()
    inner = doc.blocks.new("INNER")
    inner.add_line((0, 0), (1, 0), dxfattribs={"layer": "0"})
    outer = doc.blocks.new("OUTER")
    outer.add_blockref("INNER", (2, 0))
    outer.add_circle((0, 0), 0.5, dxfattribs={"layer": "KEEP"})
    msp.add_blockref("OUTER", (10, 10), dxfattribs={"layer": "A-WALL", "rotation": 90})
    msp.add_line((0, 0), (3, 0), dxfattribs={"layer": "L1"})
    msp.add_text("ROOM", dxfattribs={"insert": (1, 1), "height": 0.2})
    d = ingest(save(doc))
    types = {r.dxftype for r in d.raw_entities.values()}
    assert {"LINE", "INSERT", "CIRCLE", "TEXT"} <= types
    # nested line: layer 0 inherits the outer INSERT's layer, coordinates transformed (rotation 90)
    nested = [r for r in d.raw_entities.values() if r.dxftype == "LINE" and len(r.block_path) == 2]
    assert len(nested) == 1
    r = nested[0]
    assert r.layer == "A-WALL"
    assert r.source_block_handle is not None
    (x0, y0), (x1, y1) = r.raw_coords
    assert math.isclose(x0, 10, abs_tol=1e-9) and math.isclose(y0, 12, abs_tol=1e-9)
    assert math.isclose(x1, 10, abs_tol=1e-9) and math.isclose(y1, 13, abs_tol=1e-9)
    circ = [r for r in d.raw_entities.values() if r.dxftype == "CIRCLE"][0]
    assert circ.layer == "KEEP"
    assert len(d.inserts) == 2
    top = [g for g in d.entities if g.kind == "line" and not g.block_path][0]
    assert d.raw_entities[top.raw_id].handle is not None


def test_polyline_bulge_hatch_mline_dimension_text():
    doc, msp = new_doc()
    msp.add_lwpolyline([(0, 0, 0, 0, 1), (2, 0), (2, 2)], format="xyseb")
    h = msp.add_hatch()
    h.paths.add_polyline_path([(0, 0), (4, 0), (4, 1), (0, 1)], is_closed=True)
    h.paths.add_polyline_path([(1, 0.25), (2, 0.25), (2, 0.75), (1, 0.75)], is_closed=True, flags=0)
    msp.add_mline([(0, 5), (5, 5)])
    msp.add_mtext("BEDROOM\\P12.5 m²", dxfattribs={"insert": (3, 3), "char_height": 0.25})
    dim = msp.add_linear_dim(base=(0, -1), p1=(0, 0), p2=(3, 0))
    dim.render()
    msp.add_solid([(0, 0), (1, 0), (0, 1), (1, 1)])
    msp.add_spline([(0, 0), (1, 2), (3, 1), (4, 4)])
    msp.add_ellipse((0, 0), major_axis=(2, 0), ratio=0.5)
    d = ingest(save(doc))
    kinds = {g.kind for g in d.entities}
    assert {"polyline", "line", "arc", "hatch", "mline", "text", "dimension", "polygon", "curve"} <= kinds
    hatch = [g for g in d.entities if g.kind == "hatch"][0]
    assert len(hatch.data["loops"]) == 2
    arc = [g for g in d.entities if g.kind == "arc"][0]
    assert arc.parent is not None  # bulge part keeps link to its polyline
    dimg = [g for g in d.entities if g.kind == "dimension"][0]
    assert abs(dimg.data["measurement"] - 3.0) < 1e-9
    assert any("BEDROOM" in g.data.get("text", "") for g in d.entities if g.kind == "text")


def test_minsert_and_mirrored_arc_is_ccw():
    doc, msp = new_doc()
    b = doc.blocks.new("D")
    b.add_arc((0, 0), 1, 0, 90)
    msp.add_blockref("D", (0, 0), dxfattribs={"yscale": -1})
    ins = msp.add_blockref("D", (10, 0))
    ins.dxf.column_count = 3
    ins.dxf.column_spacing = 2
    d = ingest(save(doc))
    arcs = [g for g in d.entities if g.kind == "arc"]
    assert len(arcs) == 4
    for a in arcs:
        assert 0 < a.data["sweep"] < math.pi  # 90 degree swing stays 90 degrees
