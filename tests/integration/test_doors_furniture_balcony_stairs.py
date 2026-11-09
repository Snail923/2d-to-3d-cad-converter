"""Door orientation, furniture symbols, balconies and multi-flight stairs.

Every drawing is synthetic: the rules are geometric / semantic, so they must work on any plan,
not on one sample drawing."""
import math
import os
import tempfile

from architectural_converter.benchmark.dxf_writer import Style, render_plan
from architectural_converter.benchmark.plan_spec import PlanSpec
from architectural_converter.models.base_entity import Status
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.pipeline.pipeline_config import PipelineConfig
from architectural_converter.reconstruction3d import build_3d


def _run(doc, name):
    p = os.path.join(tempfile.mkdtemp(prefix="acv_"), name)
    doc.saveas(p)
    return ExtractionPipeline(PipelineConfig()).process(p)


def _w(st, p):
    """Model coordinates (normalised to the drawing's lower-left corner) -> drawing coordinates."""
    if hasattr(p, "centroid"):
        p = (p.centroid.x, p.centroid.y)
    x, y = st.drawing.transform.inverse([p])[0]
    return float(x), float(y)


def _live(coll):
    return [x for x in coll.values() if x.status != Status.REJECTED]


def _two_rooms():
    s = PlanSpec("rooms", "bedroom + living, the bedroom door gap is drawn without a symbol")
    s.chain([(0, 0), (8, 0), (8, 5), (0, 5)], 0.3, closed=True)
    s.wall((4, 0), (4, 5), 0.12)
    s.opening_at((8, 2.5), 1.0, "door")
    s.opening_at((4, 2.5), 0.9, "opening")          # gap y 2.05 .. 2.95
    s.opening_at((2, 0), 1.5, "window")
    s.opening_at((6, 0), 1.5, "window")
    s.label("BEDROOM", (2, 2.5))
    s.label("LIVING", (6, 2.5))
    doc, _ = render_plan(s, Style(units="m", layers="aia"))
    return doc


def _door_with_frame_line(doc, layer="A-DOOR"):
    """Door block whose open leaf is a 2-point polyline and which also holds a frame line of
    radius length starting near the hinge, ending close to (not on) the closed-leaf point. The
    frame line comes last in the block, so 'last line found' would take it for the leaf."""
    blk = doc.blocks.new("DOOR_FRAMED")
    r = 0.9
    blk.add_arc((0, 0), r, 0, 90)
    blk.add_lwpolyline([(0, 0), (0, r)])                               # open leaf, tip on the arc
    blk.add_line((0.015, 0.03), (r - 0.005, 0.03))                     # frame / threshold line
    # hinge at the bottom jamb of the gap; local x (closed leaf) -> +y, local y (open) -> -x
    doc.modelspace().add_blockref("DOOR_FRAMED", (4.0, 2.05), dxfattribs={"layer": layer, "rotation": 90})


def test_door_chord_follows_closed_leaf_not_frame_line():
    doc = _two_rooms()
    _door_with_frame_line(doc)
    st = _run(doc, "door.dxf")
    b = st.building
    in_gap = [o for o in _live(b.openings) if abs(_w(st, o.center)[0] - 4.0) < 0.2 and 1.9 < _w(st, o.center)[1] < 3.1]
    assert len(in_gap) == 1, [(o.id, o.type, o.center) for o in in_gap]   # no duplicate gap opening
    d = in_gap[0]
    assert d.type == "door" and d.host_wall is not None
    assert abs(d.p1[0] - d.p0[0]) < 0.05 and abs(abs(d.p1[1] - d.p0[1]) - 0.9) < 0.1   # along the wall
    assert d.swing.get("leaf_known") is True
    assert not [i for i in st.report.issues if i.code == "no_access"]


def _tv_unit(doc, name, layer, at):
    """Furniture drawn with window-like equal parallel lines (a TV unit / low cabinet)."""
    if name not in doc.blocks:
        blk = doc.blocks.new(name)
        for y in (0.0, 0.2):
            blk.add_line((0, y), (1.2, y))
        blk.add_line((0, 0), (0, 0.2))
        blk.add_line((1.2, 0), (1.2, 0.2))
    doc.modelspace().add_blockref(name, at, dxfattribs={"layer": layer, "rotation": 90})


def test_furniture_against_a_wall_is_not_a_window():
    doc = _two_rooms()
    _door_with_frame_line(doc)
    _tv_unit(doc, "TV_UNIT", "I-FURN", (4.28, 3.2))          # furniture layer, beside the partition
    _tv_unit(doc, "B12", "0", (0.37, 1.0))                   # no semantics at all, beside the facade
    st = _run(doc, "furniture.dxf")
    tv = {r.id for r in st.drawing.inserts.values() if r.name in ("TV_UNIT", "B12")}
    assert tv
    wrong = [o for o in _live(st.building.openings) if o.symbol_insert in tv]
    assert not wrong, [(o.id, o.type, o.status_reason) for o in wrong]
    assert len([o for o in _live(st.building.openings) if o.type == "window"]) == 2


def _balcony_plan(rail_layer=None):
    s = PlanSpec("balcony", "flat with a balcony (thin railing) and a bay (full walls)")
    s.chain([(0, 0), (9, 0), (9, 5), (0, 5)], 0.3, closed=True)
    s.wall((4.5, 0), (4.5, 5), 0.12)
    s.chain([(0.8, 5.0), (0.8, 6.5), (3.8, 6.5), (3.8, 5.0)], 0.1)          # balcony railing
    s.chain([(5.5, 5.0), (5.5, 6.4), (8.0, 6.4), (8.0, 5.0)], 0.3)          # bay: a real room
    s.opening_at((9, 2.5), 1.0, "door")
    s.opening_at((4.5, 2.5), 0.9, "door")
    s.opening_at((2.8, 5.0), 0.9, "door", swing=1)
    s.opening_at((1.6, 5.0), 1.0, "window")
    s.opening_at((6.75, 5.0), 1.6, "opening")
    s.opening_at((6.75, 6.4), 1.5, "window")
    s.opening_at((2, 0), 1.5, "window")
    doc, _ = render_plan(s, Style(units="m", layers="aia"))
    return doc


def test_balcony_with_railing_is_open_and_its_facade_is_exterior():
    st = _run(_balcony_plan(), "balcony.dxf")
    b = st.building
    balconies = [r for r in _live(b.rooms) if r.classification == "balcony"]
    assert len(balconies) == 1, [(r.id, r.classification, round(r.area, 1)) for r in _live(b.rooms)]
    bal = balconies[0]
    assert 3.0 < bal.area < 4.5 and _w(st, bal.polygon)[1] > 5.0
    railings = [w for w in _live(b.walls) if w.classification == "railing"]
    assert len(railings) >= 3 and all(w.thickness < 0.15 for w in railings)
    # the bay with full-height walls is an ordinary room
    bay = [r for r in _live(b.rooms) if _w(st, r.polygon)[0] > 5 and _w(st, r.polygon)[1] > 5]
    assert bay and all(r.classification != "balcony" for r in bay)
    # the facade between flat and balcony: exterior wall, its window stays a window, door is exterior
    ops = [o for o in _live(b.openings) if abs(_w(st, o.center)[1] - 5.0) < 0.3 and _w(st, o.center)[0] < 4]
    assert {o.type for o in ops} == {"door", "window"}
    assert all(o.classification.startswith("exterior") for o in ops)
    assert not [i for i in st.report.issues if i.code == "no_access"]
    m3 = build_3d(st.model, PipelineConfig())
    rail_ids = {w.id for w in railings}
    # balcony wall = a low solid parapet with glass above it up to the railing height
    tops = [m["vertices"][:, 2].max() for m in m3.meshes if m["id"] in rail_ids]
    assert tops and all(abs(z - 0.5) < 1e-6 for z in tops)
    glass = [m for m in m3.meshes if m["id"].removesuffix("_glass") in rail_ids and m["type"] == "window"]
    assert len(glass) == len(tops)
    assert all(abs(m["vertices"][:, 2].min() - 0.5) < 1e-6 and abs(m["vertices"][:, 2].max() - 1.1) < 1e-6 for m in glass)
    assert "railing_height_m" in m3.metadata["defaults_used"]


def test_balcony_from_layer_semantics():
    """The railing is a full 0.3 m wall here, but its layer says balcony."""
    s = PlanSpec("terrace", "terrace whose parapet is as thick as the facade")
    s.chain([(0, 0), (9, 0), (9, 5), (0, 5)], 0.3, closed=True)
    s.wall((4.5, 0), (4.5, 5), 0.12)
    s.opening_at((9, 2.5), 1.0, "door")
    s.opening_at((4.5, 2.5), 0.9, "door")
    s.opening_at((2.8, 5.0), 0.9, "door")
    doc, _ = render_plan(s, Style(units="m", layers="aia"))
    msp = doc.modelspace()
    for (a, b) in (((0.8, 5.15), (0.8, 6.5)), ((0.8, 6.5), (3.8, 6.5)), ((3.8, 6.5), (3.8, 5.15)),
                   ((1.1, 5.15), (1.1, 6.2)), ((1.1, 6.2), (3.5, 6.2)), ((3.5, 6.2), (3.5, 5.15))):
        msp.add_line(a, b, dxfattribs={"layer": "A-BALCONY"})
    st = _run(doc, "terrace.dxf")
    assert [r for r in _live(st.building.rooms) if r.classification == "balcony"]


def _switchback_doc(arrow=True, with_outline=True):
    s = PlanSpec("stair", "stair hall with a switchback stair")
    s.chain([(0, -0.6), (4.6, -0.6), (4.6, 7.0), (0, 7.0)], 0.2, closed=True)
    s.opening_at((2.3, 7.0), 1.2, "door")
    doc, _ = render_plan(s, Style(units="m", layers="aia"))
    msp = doc.modelspace()
    lay = {"layer": "A-STAIR"}
    for k in range(10):                                   # flight A: treads y 1.0 .. 3.7
        msp.add_line((1.0, 1.0 + 0.3 * k), (2.1, 1.0 + 0.3 * k), dxfattribs=lay)
    for k in range(11):                                   # flight B: one more tread at the top
        msp.add_line((2.4, 1.0 + 0.3 * k), (3.5, 1.0 + 0.3 * k), dxfattribs=lay)
    for x in (1.0, 2.1, 2.4, 3.5):                        # stringers
        msp.add_line((x, 0.7), (x, 4.3 if x > 2.2 else 4.0), dxfattribs=lay)
    if with_outline:                                      # landing edge drawn as a double line
        msp.add_line((1.0, 0.7), (3.5, 0.7), dxfattribs=lay)
        msp.add_line((1.0, 0.55), (3.5, 0.55), dxfattribs=lay)
    if arrow:                                             # chevron at the top of flight B: B climbs +y
        msp.add_line((2.8, 4.15), (2.95, 4.3), dxfattribs=lay)
        msp.add_line((2.95, 4.3), (3.1, 4.15), dxfattribs=lay)
    return doc


def test_switchback_stair_flights_climb_in_opposite_directions():
    st = _run(_switchback_doc(), "stair.dxf")
    b = st.building
    flights = sorted(_live(b.stairs), key=lambda f: f.flight_index)
    assert len(flights) == 2
    A, B = flights
    assert A.assembly and A.assembly == B.assembly and A.assembly_kind == "switchback"
    assert B.direction_source == "arrow" and B.run[1] > 0.99      # read from the chevron
    assert A.direction_source == "assembly" and A.run[1] < -0.99   # opposite, propagated
    assert A.polygon.centroid.x < B.polygon.centroid.x
    # A climbs from the hall down-plan to the landing; B starts at the landing height
    assert A.landing is not None and _w(st, A.landing)[1] < 1.0
    assert abs(B.base_height - (A.treads + 1) * A.rise) < 1e-6 and B.base_height > 0
    assert A.rise == B.rise
    # the landing edge drawn with stair lines is not a wall
    stair_raw = set().union(*(f.source_entities for f in flights))
    walls_from_stair_layer = [w for w in _live(b.walls)
                              if all(st.drawing.raw_entities[r].layer == "A-STAIR" for r in w.source_entities)]
    assert not walls_from_stair_layer, [w.id for w in walls_from_stair_layer]
    assert stair_raw
    m3 = build_3d(st.model, PipelineConfig())
    z = {m["id"]: m["vertices"][:, 2].max() for m in m3.meshes if m["type"] == "stair"}
    assert z[B.id] > z[A.id] + 1.0


def test_switchback_without_marker_still_alternates():
    st = _run(_switchback_doc(arrow=False), "stair2.dxf")
    flights = sorted(_live(st.building.stairs), key=lambda f: f.flight_index)
    assert len(flights) == 2
    A, B = flights
    assert A.assembly == B.assembly
    assert math.isclose(A.run[1], -B.run[1], abs_tol=1e-6)        # opposite directions
    assert A.landing is not None and _w(st, A.landing)[1] < 1.0     # the closed end is the landing
