from architectural_converter.models.base_entity import Status
from tests.helpers import new_doc, run, save


def _live(st):
    return [w for w in st.building.walls.values() if w.status != Status.REJECTED]


def test_l_and_t_junctions_close_room():
    # walls drawn with interrupted faces at corners / T junction (typical union outline)
    doc, msp = new_doc()
    t = 0.2
    outer = [(0, 0), (8, 0), (8, 5), (0, 5)]
    for a, b in zip(outer, outer[1:] + outer[:1]):
        msp.add_line(a, b)
    # inner faces of two rooms separated by a wall at x = 4
    for x0, x1 in ((t, 4 - t / 2), (4 + t / 2, 8 - t)):
        pts = [(x0, t), (x1, t), (x1, 5 - t), (x0, 5 - t)]
        for a, b in zip(pts, pts[1:] + pts[:1]):
            msp.add_line(a, b)
    st = run(save(doc))
    rooms = [r for r in st.building.rooms.values() if r.status == Status.ACCEPTED]
    assert len(rooms) == 2
    assert any(w.reconstruction.get("extensions") for w in _live(st))


def test_door_gap_bridged_and_hosted_not_closed_silently():
    doc, msp = new_doc()
    t = 0.2
    # bottom wall with a 0.9 m gap at x 2..2.9, door symbol in the gap
    for (a, b) in (((0, 0), (2, 0)), ((2.9, 0), (6, 0)), ((t, t), (2, t)), ((2.9, t), (6 - t, t)),
                   ((2, 0), (2, t)), ((2.9, 0), (2.9, t))):
        msp.add_line(a, b)
    for (a, b) in (((6, 0), (6, 4)), ((6, 4), (0, 4)), ((0, 4), (0, 0)), ((6 - t, t), (6 - t, 4 - t)),
                   ((6 - t, 4 - t), (t, 4 - t)), ((t, 4 - t), (t, t))):
        msp.add_line(a, b)
    msp.add_line((2, t), (2, t + 0.9), dxfattribs={"layer": "A-DOOR"})
    msp.add_arc((2, t), 0.9, 0, 90, dxfattribs={"layer": "A-DOOR"})
    st = run(save(doc))
    doors = [o for o in st.building.openings.values() if o.type == "door" and o.status != Status.REJECTED]
    assert len(doors) == 1 and doors[0].host_wall is not None
    assert abs(doors[0].width - 0.9) < 0.02
    host = st.building.walls[doors[0].host_wall]
    assert doors[0].id in host.openings
    assert sum(1 for r in st.building.rooms.values() if r.status == Status.ACCEPTED) == 1


def test_unexplained_gap_becomes_flagged_opening():
    doc, msp = new_doc()
    t = 0.2
    for (a, b) in (((0, 0), (2, 0)), ((3, 0), (6, 0)), ((t, t), (2, t)), ((3, t), (6 - t, t)),
                   ((2, 0), (2, t)), ((3, 0), (3, t)), ((6, 0), (6, 4)), ((6, 4), (0, 4)), ((0, 4), (0, 0)),
                   ((6 - t, t), (6 - t, 4 - t)), ((6 - t, 4 - t), (t, 4 - t)), ((t, 4 - t), (t, t))):
        msp.add_line(a, b)
    st = run(save(doc))
    ops = [o for o in st.building.openings.values() if o.type == "opening"]
    assert len(ops) == 1 and ops[0].status == Status.NEEDS_REVIEW and ops[0].is_hypothesis
    codes = {i.code for i in st.report.issues}
    assert "unidentified_opening" in codes
