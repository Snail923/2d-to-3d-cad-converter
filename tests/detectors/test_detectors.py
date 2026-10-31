from architectural_converter.models.base_entity import Status
from tests.helpers import box_room, new_doc, run, save


def _walls(st, status=(Status.ACCEPTED,)):
    return [w for w in st.walls if w.status in status]


def test_parallel_walls_and_thickness_cluster():
    doc, msp = new_doc()
    box_room(msp, 6, 4, 0.2)
    st = run(save(doc), until="resolve")
    ws = _walls(st)
    assert len(ws) == 4
    assert all(abs(w.thickness - 0.2) < 1e-6 for w in ws)
    assert abs(st.ctx.thickness_clusters[0]["thickness"] - 0.2) < 1e-3
    for w in ws:
        assert w.source_entities and "parallel_wall_detector" in w.detection_methods


def test_gap_between_two_walls_is_not_a_wall():
    # two 0.2 m walls separated by a 0.3 m cavity: the cavity must not become a wall
    doc, msp = new_doc()
    for y in (0.0, 0.2, 0.5, 0.7):
        msp.add_line((0, y), (8, y), dxfattribs={"layer": "A-WALL"})
    st = run(save(doc), until="resolve")
    ts = sorted(round(w.thickness, 3) for w in _walls(st, (Status.ACCEPTED, Status.NEEDS_REVIEW)))
    assert ts == [0.2, 0.2]


def test_door_block_and_loose_door_and_window_block():
    doc, msp = new_doc()
    box_room(msp, 8, 5, 0.2)
    d = doc.blocks.new("DOOR")
    d.add_line((0, 0), (0, 1))
    d.add_arc((0, 0), 1, 0, 90)
    msp.add_blockref("DOOR", (1, 0.2), dxfattribs={"xscale": 0.9, "yscale": 0.9})
    msp.add_line((5, 0.2), (5, 1.1), dxfattribs={"layer": "A-DOOR"})
    msp.add_arc((5, 0.2), 0.9, 0, 90, dxfattribs={"layer": "A-DOOR"})
    win = doc.blocks.new("WIN")
    for y in (-0.5, 0, 0.5):
        win.add_line((0, y), (1, y))
    msp.add_blockref("WIN", (3, 4.9), dxfattribs={"xscale": 1.2, "yscale": 0.2})
    st = run(save(doc), until="detect")
    doors = st.ctx.candidates["door_detector"]
    wins = st.ctx.candidates["window_detector"]
    assert len(doors) == 2 and all(abs(o.width - 0.9) < 1e-6 for o in doors)
    assert {"symbol"} <= set(next(o for o in doors if o.symbol_insert).evidence)
    assert len(wins) == 1 and abs(wins[0].width - 1.2) < 1e-6


def test_columns_and_furniture_layer():
    doc, msp = new_doc()
    box_room(msp, 20, 12, 0.3)
    for x in (5, 10, 15):
        msp.add_lwpolyline([(x - .25, 5.75), (x + .25, 5.75), (x + .25, 6.25), (x - .25, 6.25)], close=True,
                           dxfattribs={"layer": "S-COLS"})
    msp.add_lwpolyline([(3, 2), (3.8, 2), (3.8, 2.8), (3, 2.8)], close=True, dxfattribs={"layer": "I-FURN"})
    st = run(save(doc), until="resolve")
    acc = [c for c in st.columns if c.status == Status.ACCEPTED]
    assert len(acc) == 3
    furn = [c for c in st.columns if c.polygon.centroid.x < 4]
    assert all(c.status != Status.ACCEPTED for c in furn)


def test_single_line_walls_strict_in_double_line_drawing():
    doc, msp = new_doc()
    box_room(msp, 6, 4, 0.2)
    msp.add_line((1, 1), (4, 1))  # stray line inside a double-line drawing, unknown layer
    st = run(save(doc), until="detect")
    assert st.ctx.candidates.get("single_line_wall_detector", []) == []


def test_single_line_drawing_produces_unknown_thickness_walls():
    doc, msp = new_doc()
    pts = [(0, 0), (6, 0), (6, 4), (0, 4)]
    for a, b in zip(pts, pts[1:] + pts[:1]):
        msp.add_line(a, b, dxfattribs={"layer": "WALLS"})
    msp.add_line((3, 0), (3, 4), dxfattribs={"layer": "WALLS"})
    st = run(save(doc))
    ws = [w for w in st.building.walls.values() if w.status != Status.REJECTED]
    assert len(ws) >= 5 and all(w.thickness is None for w in ws)
    assert sum(1 for r in st.building.rooms.values() if r.status == Status.ACCEPTED) == 2
