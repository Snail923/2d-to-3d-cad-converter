"""Rooms inside a separate cluster of walls (an island inside the exterior wall ring) are found once.

The island's own holes are rooms; the exterior ring's hole minus the island must not return them again
(it did, on the SNOOBAR permit set: 386 of 1025 rooms were exact copies)."""
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from architectural_converter.topology.room_detector import holes_of


def test_rooms_of_an_island_are_not_returned_twice():
    outer = box(0, 0, 20, 12).difference(box(0.3, 0.3, 19.7, 11.7))          # exterior wall ring
    island = box(5, 3, 15, 9).difference(box(5.2, 3.2, 9.9, 8.8)).difference(box(10.1, 3.2, 14.8, 8.8))  # 2 rooms
    holes = holes_of(unary_union([outer, island]))
    areas = sorted(round(h.area, 2) for h in holes)
    assert len(holes) == 3, areas                     # the corridor ring around the island + its two rooms
    for room in (box(5.2, 3.2, 9.9, 8.8), box(10.1, 3.2, 14.8, 8.8)):
        assert sum(1 for h in holes if h.intersection(room).area > 0.5 * room.area) == 1
    ring = max(holes, key=lambda h: h.area)
    assert abs(ring.area - (19.4 * 11.4 - 10 * 6)) < 0.5   # outside the island only
    assert isinstance(ring, Polygon)
