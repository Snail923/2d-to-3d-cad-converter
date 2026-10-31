from architectural_converter.models.base_entity import Status
from tests.helpers import box_room, new_doc, run, save


def test_room_label_area_and_narrow_cavity():
    doc, msp = new_doc()
    box_room(msp, 6, 4, 0.2)
    # inner double wall creating a 0.15 m cavity -> narrow region must be rejected
    for x in (3.0, 3.2, 3.35, 3.55):
        msp.add_line((x, 0.2), (x, 3.8))
    msp.add_mtext("KITCHEN\\P8.04 m²", dxfattribs={"insert": (1.5, 2), "char_height": 0.2, "attachment_point": 5})
    st = run(save(doc))
    rooms = [r for r in st.building.rooms.values() if r.status == Status.ACCEPTED]
    assert len(rooms) == 2
    k = [r for r in rooms if r.name == "KITCHEN"][0]
    assert k.classification == "kitchen"
    assert abs(k.area - 2.8 * 3.6) < 0.05
    assert any(r.status == Status.REJECTED and "narrow" in (r.status_reason or "") for r in st.building.rooms.values())


def test_missing_wall_hypothesis_is_proposed_not_added():
    doc, msp = new_doc()
    box_room(msp, 10, 8, 0.2)
    # partition from the bottom wall that stops 4.5 m short of the top wall: too wide to be an
    # opening, so the engine must not bridge it, only propose a missing-wall hypothesis
    msp.add_line((5.0, 0.2), (5.0, 3.3))
    msp.add_line((5.15, 0.2), (5.15, 3.3))
    msp.add_line((5.0, 3.3), (5.15, 3.3))
    msp.add_text("OFFICE", dxfattribs={"insert": (2, 2), "height": 0.2})
    msp.add_text("STORE", dxfattribs={"insert": (8, 2), "height": 0.2})
    st = run(save(doc))
    b = st.building
    assert len([r for r in b.rooms.values() if r.status == Status.ACCEPTED]) == 1
    assert len(b.hypotheses) >= 1
    h = list(b.hypotheses.values())[0]
    assert h.status == Status.NEEDS_REVIEW and h.confidence <= 0.5
    assert h.id not in b.walls
    assert len(b.room_hypotheses) >= 2
