"""Level outlines for stacking: every storey has a floor below it and a roof above it.

Room floors only exist where a room was closed. When floors are stacked, each storey needs one
continuous slab under it and one over it, so each floor-plan level gets an *outline*: the closed
shape of everything drawn on it (walls, rooms, columns, ground pieces), gaps closed, inner voids
filled. When the plan does not close into a shape (walls with wide gaps, rooms not found: the
closed shape covers less than ``min_hull_share`` of the hull of its walls), the outline is made by
connecting the outer vertices of the plan's walls (concave hull, else convex hull).

Phase 2 only extrudes these outlines (base slab, roof slab); no geometry is decided there."""
from __future__ import annotations

from shapely.geometry import MultiPoint, Polygon

from ..geometry.geometric_utils import polygons_of, robust_union, safe_buffer
from .ground import _close_enclosed


def _hull(points, ratio: float):
    mp = MultiPoint(points)
    try:
        import shapely
        h = shapely.concave_hull(mp, ratio=ratio, allow_holes=False)
        if h is not None and not h.is_empty and h.geom_type == "Polygon" and h.is_valid and h.area > 0:
            return h
    except Exception:  # noqa: BLE001  (older GEOS: no concave hull)
        pass
    return mp.convex_hull


def level_outline(geoms, wall_points, close_m: float = 2.0, wide_close_m=(), min_enclosure: float = 0.6,
                  min_hull_share: float = 0.5, hull_ratio: float = 0.25, min_area_m2: float = 1.0,
                  simplify_m: float = 0.02):
    """([Polygon without holes, ...], source) for one level; source = 'plan' | 'wall_vertices' | None."""
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    pts = [tuple(p) for p in wall_points]
    hull = MultiPoint(pts).convex_hull if len(pts) >= 3 else None
    shells = []
    if geoms:
        shell = robust_union([Polygon(p.exterior) for g in geoms for p in polygons_of(g)])
        if close_m > 0:
            closed = safe_buffer(safe_buffer(shell, close_m, join_style=2), -close_m, join_style=2)
            if closed is not None and not closed.is_empty:
                shell = robust_union([shell] + [Polygon(p.exterior) for p in polygons_of(closed)])
        for d in wide_close_m or ():
            if d > close_m:
                try:
                    shell = _close_enclosed(shell, float(d), min_enclosure, min_area_m2)
                except Exception:  # noqa: BLE001
                    break
        for p in polygons_of(shell):
            p = Polygon(p.exterior)
            core = safe_buffer(p, -0.25)
            if p.area >= min_area_m2 and core is not None and not core.is_empty:     # not a lone wall body
                shells.append(p)
    source = "plan"
    area = sum(p.area for p in shells)
    if hull is not None and hull.geom_type == "Polygon" and hull.area >= min_area_m2 \
            and area < min_hull_share * hull.area:
        # the plan does not close: connect the vertices of its walls
        h = _hull(pts, hull_ratio)
        shells = [Polygon(p.exterior) for p in polygons_of(robust_union([h] + shells))]
        source = "wall_vertices"
    out = []
    for p in shells:
        q = p.simplify(simplify_m) if simplify_m > 0 else p
        if q.is_valid and not q.is_empty and q.geom_type == "Polygon":
            p = Polygon(q.exterior)
        if p.area >= min_area_m2:
            out.append(p)
    return out, (source if out else None)
