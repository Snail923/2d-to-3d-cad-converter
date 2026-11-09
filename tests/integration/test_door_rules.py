"""Doors need walls (topology/door_rules.py): missing walls are re-checked around doors."""
import os
import tempfile

from architectural_converter.benchmark.dxf_writer import Style, render_plan
from architectural_converter.benchmark.plan_spec import PlanSpec
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.pipeline.pipeline_config import PipelineConfig


def _run(spec, overrides=None):
    doc, _ = render_plan(spec, Style(layers="aia", door_repr="block", window_repr="block", dims=False))
    p = os.path.join(tempfile.mkdtemp(prefix="acv_"), spec.name + ".dxf")
    doc.saveas(p)
    return ExtractionPipeline(PipelineConfig(overrides)).process(p)


def _live_rooms(st):
    return [r for r in st.model["rooms"] if r["status"] != "REJECTED"]


def _door_walls(st):
    return [w for w in st.model["walls"] if "door_implied" in w["detection_methods"] and w["status"] != "REJECTED"]


def test_wall_completed_along_a_door_line():
    """The partition holding the door stops 3.65 m short of the facade (wider than any opening,
    so it is not bridged as one): the wall on the door's line is completed up to the facade and
    the two rooms close."""
    s = PlanSpec("door_line", "partition with a door stops short of the facade")
    s.chain([(0, 0), (8, 0), (8, 7), (0, 7)], 0.3, closed=True)
    s.wall((4, 0), (4, 3.2), 0.12)
    s.opening_at((4, 2.5), 0.9, "door")
    s.opening_at((2, 0), 1.2, "window")
    st = _run(s)
    added = _door_walls(st)
    assert added, st.logs["topology"]["door_rules"]
    w = added[0]
    assert w["status"] == "NEEDS_REVIEW" and "needs a wall here" in w["status_reason"]
    door = next(o for o in st.model["openings"] if o["type"] == "door")
    xs = [p[0] for p in (w["geometry"]["start"], w["geometry"]["end"])]
    assert all(abs(x - door["position"]["center"][0]) < 0.05 for x in xs)      # on the door's line
    assert door["host_wall"]
    assert len(_live_rooms(st)) == 2
    off = _run(s, {"reconstruction": {"door_rules": {"enabled": False}}})
    assert len(_live_rooms(off)) == 1 and not _door_walls(off)


def test_room_with_a_door_gets_its_missing_wall():
    """The right room's facade is missing over 3.35 m (wider than any opening): the room is open
    to the outside, so it is no room and has no floor. It has a door, so the missing-room
    hypothesis closing it becomes a wall and the room exists."""
    s = PlanSpec("door_room", "right room open to the outside; it has a door")
    s.chain([(8, 1.5), (8, 0), (0, 0), (0, 5), (8, 5)], 0.3)
    s.wall((4, 0), (4, 5), 0.12)
    s.opening_at((4, 2.5), 0.9, "door")
    s.opening_at((6, 0), 1.0, "door")
    st = _run(s)
    promoted = st.model["statistics"]["logs"]["topology"].get("door_room_walls") or []
    walls = _door_walls(st)
    assert promoted and walls, st.model["statistics"]["logs"]["topology"]
    assert any("closes a room that has door" in (w["status_reason"] or "") for w in walls)
    assert len(_live_rooms(st)) == 2
    assert st.logs["ground"]["unassigned_floor_pieces"] == 0                 # the room has its floor now


def test_open_plan_room_is_not_split():
    """A short partition without a door inside one room: the two parts form one (open-plan)
    room and no wall is invented to split it, even though the room has a door."""
    s = PlanSpec("open_plan", "partition stops 2.95 m short, no door on it")
    s.chain([(0, 0), (8, 0), (8, 5), (0, 5)], 0.3, closed=True)
    s.wall((4, 0), (4, 1.9), 0.12)
    s.opening_at((6, 0), 1.0, "door")
    st = _run(s)
    assert not _door_walls(st) and len(_live_rooms(st)) == 1


def test_complete_rooms_are_not_touched():
    s = PlanSpec("closed", "closed rooms with doors")
    s.chain([(0, 0), (8, 0), (8, 5), (0, 5)], 0.3, closed=True)
    s.wall((4, 0), (4, 5), 0.12)
    s.opening_at((4, 2.5), 0.9, "door")
    s.opening_at((6, 0), 1.0, "door")
    st = _run(s)
    assert not _door_walls(st) and len(_live_rooms(st)) == 2


def test_area_without_room_gets_ground_and_a_warning():
    """The facade of the left room has a 3.5 m hole (wider than any opening, no door): the room
    is not closed, so it has no room floor. Its area still gets a ground slab in 3D and a
    validation warning points at it."""
    from architectural_converter.reconstruction3d import build_3d
    s = PlanSpec("ground", "left room open to the outside")
    s.chain([(2, 0), (0, 0), (0, 5), (8, 5), (8, 0), (5.5, 0)], 0.3)
    s.wall((6, 0), (6, 5), 0.12)
    s.opening_at((7, 5), 1.0, "window")
    st = _run(s)
    warn = [i for i in st.model["validation"]["issues"] if i["code"] == "floor_without_room"]
    assert warn and st.logs["ground"]["area_m2"] > 15
    m3 = build_3d(st.model, PipelineConfig())
    ground = [m for m in m3.meshes if m["type"] == "floor" and m["classification"] == "ground"]
    assert ground and m3.metadata["ground_slabs"]["area_m2"] > 15
    assert all(abs(m["vertices"][:, 2].max()) < 1e-9 for m in ground)        # at floor level
    off = _run(s, {"phase2": {"ground": {"enabled": False}}})
    assert not [i for i in off.model["validation"]["issues"] if i["code"] == "floor_without_room"]


def test_closed_building_needs_no_ground():
    s = PlanSpec("closed2", "closed rooms")
    s.chain([(0, 0), (8, 0), (8, 5), (0, 5)], 0.3, closed=True)
    s.wall((4, 0), (4, 5), 0.12)
    s.opening_at((4, 2.5), 0.9, "door")
    st = _run(s)
    assert st.logs["ground"]["unassigned_floor_pieces"] == 0
