"""Approach 2: gaps left in walls are typed from the model's labels of the lines inside them."""
import numpy as np

from architectural_converter.models.base_entity import Status
from architectural_converter.models.opening import Opening
from architectural_converter.models.wall import Wall
from sympoint_converter import CLASSES, load_config
from sympoint_converter.gap_typing import type_gaps


def _scene(door_elsewhere: bool):
    """A wall along x with a 0.9 m gap at s = 2.0-2.9 holding door-labelled lines (leaf drawn
    closed across the gap + frame lines)."""
    wall = Wall(id="wall_1", kind="straight", u=(1.0, 0.0), c=0.0, s0=0.0, s1=5.0, thickness=0.2)
    gap = Opening(id="opening_1", type="opening", p0=(2.0, 0.0), p1=(2.9, 0.0), width=0.9, host_wall="wall_1",
                  host_interval=(2.0, 2.9), status=Status.NEEDS_REVIEW)
    wall.openings.append(gap.id)
    prims, plab = [], {}
    for k, (p, q) in enumerate((((2.0, 0.05), (2.9, 0.05)), ((2.0, -0.05), (2.9, -0.05)), ((2.45, -0.1), (2.45, 0.1)))):
        P = np.array([p, q], float)
        prims.append((f"e_{k}", "Door", 0, P, float(np.linalg.norm(P[1] - P[0])), P))
        plab[k] = [CLASSES.index("single door"), 0.9]
    other = Wall(id="wall_2", kind="straight", u=(0.0, 1.0), c=-2.0, s0=0.1, s1=3.0, thickness=0.12)
    doors = []
    if door_elsewhere:
        # the same door placed across the gap, along the perpendicular wall its open leaf rests on
        d = Opening(id="door_1", type="door", p0=(2.0, 0.1), p1=(2.0, 1.0), width=0.9, host_wall="wall_2",
                    host_interval=(0.1, 1.0), status=Status.ACCEPTED)
        d.source_entities |= {"e_0", "e_1"}
        other.openings.append(d.id)
        doors.append(d)
    return wall, other, gap, doors, prims, {"primitives": plab}


def test_misplaced_door_is_moved_into_its_gap():
    wall, other, gap, doors, prims, labels = _scene(True)
    log = type_gaps([gap], doors, [], [wall, other], prims, labels, load_config())
    d = doors[0]
    assert log["doors_moved"] == 1 and gap.status == Status.REJECTED
    assert d.host_wall == "wall_1" and sorted(d.host_interval) == [2.0, 2.9] and d.p0 == (2.0, 0.0)
    assert d.id in wall.openings and d.id not in other.openings
    assert any("moved into the wall gap" in n for n in d.notes)


def test_gap_with_door_lines_and_no_door_object_becomes_a_door():
    wall, other, gap, doors, prims, labels = _scene(False)
    log = type_gaps([gap], doors, [], [wall, other], prims, labels, load_config())
    assert log["gaps_to_doors"] == 1 and gap.type == "door" and gap.status != Status.REJECTED


def test_gap_with_unlabelled_or_wall_lines_stays_a_gap():
    wall, other, gap, doors, prims, labels = _scene(False)
    labels = {"primitives": {k: [CLASSES.index("wall"), 0.9] for k in labels["primitives"]}}
    log = type_gaps([gap], doors, [], [wall, other], prims, labels, load_config())
    assert gap.type == "opening" and log["gaps_to_doors"] == 0
