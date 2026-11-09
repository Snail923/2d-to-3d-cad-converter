"""Ground (floor) coverage: every area inside the building needs a floor.

Floors come from rooms. A room that was not closed (a wall missing, a region rejected as a room)
leaves a hole in the floor. The building outline is closed over notches up to ``2 * close_m``
wide (an opening left by a missing outer wall), and every part of it that is neither a room, a
wall nor a column gets a *ground* slab in 3D and a validation warning, so it is visible and can
be fixed (add the wall / draw the room)."""
from __future__ import annotations

from shapely.geometry import Polygon

from ..geometry.geometric_utils import polygons_of, robust_difference, robust_union, safe_buffer


def unassigned_floor(outline_polys, covered_polys, close_m: float = 2.0, min_area_m2: float = 1.0,
                     min_width_m: float = 0.4):
    """Parts of the (closed) building outline not covered by rooms / walls / columns."""
    outline = [p for p in outline_polys if p is not None and not p.is_empty]
    if not outline:
        return []
    shell = robust_union([Polygon(p.exterior) for p in outline])      # inner courtyards included
    if close_m > 0:
        closed = safe_buffer(safe_buffer(shell, close_m, join_style=2), -close_m, join_style=2)
        if closed is not None and not closed.is_empty:
            # closing the mouth of an open room encloses the room: fill what became inside
            shell = robust_union([shell] + [Polygon(p.exterior) for p in polygons_of(closed)])
    cover = robust_union([p for p in covered_polys if p is not None and not p.is_empty]) if covered_polys else None
    rest = robust_difference(shell, safe_buffer(cover, 0.02)) if cover is not None else shell
    out = []
    for p in polygons_of(rest):
        if p.area < min_area_m2:
            continue
        core = safe_buffer(p, -min_width_m / 2)
        if core is None or core.is_empty:
            continue                                   # a sliver along a wall face
        out.append(p)
    return out
