"""Wall classification: exterior / interior / partition (+ openings and columns)."""
from __future__ import annotations

from ..geometry.geometric_utils import bbox_pad, polygons_of, sample_points

from ..models.base_entity import Status
from .architectural_rules import building_footprint, partition_threshold
from ..geometry.geometric_utils import safe_buffer as _sbuf


def classify_walls(walls, solid, tol, cfg):
    cc = cfg["classification"]
    import numpy as np
    import shapely
    from shapely.strtree import STRtree
    fp = building_footprint(solid)
    step = 0.1
    ring_pts = None
    if fp is not None:
        rings = [g.exterior for g in polygons_of(fp)]
        arr = np.vstack([sample_points(r, step) for r in rings]) if rings else np.zeros((0, 2))
        ring_pts = shapely.points(arr) if len(arr) else None
    ptree = STRtree(ring_pts) if ring_pts is not None else None
    d = tol.snap * 3 + max(tol.snap, 1e-4) * 2
    for w in walls:
        if w.status == Status.REJECTED:
            continue
        p = w.polygon(max(tol.snap * 2, 0.01))
        if p is None or ptree is None:
            w.classification = "interior_wall"
            continue
        L = max(w.length, 1e-9)
        cand = ptree.query(bbox_pad(p, d))
        ext = float((shapely.distance(ring_pts[cand], p) <= d).sum()) * step if len(cand) else 0.0
        w.classification = "exterior_wall" if ext >= cc["exterior_face_fraction"] * L else "interior_wall"
    inherit_facade(walls, tol)
    limit = partition_threshold([w for w in walls if w.status != Status.REJECTED], cfg)
    for w in walls:
        if w.classification == "interior_wall" and w.thickness and w.thickness <= limit:
            w.classification = "partition_wall"
        if w.classification == "interior_wall" and not w.thickness:
            w.notes.append("thickness unknown: interior/partition undecided")
    return fp, limit


FACADE_LINE_NOTE = "runs inside an exterior wall: part of the facade"


def inherit_facade(walls, tol, min_share: float = 0.5) -> int:
    """A wall whose centre line runs (mostly) inside the body of an exterior wall is a second
    line of that facade (duplicate / overlapping candidate, render or cladding line): it is
    exterior too. Otherwise its windows would read as interior windows. Railings keep their
    class (they may continue past the facade along a balcony)."""
    from shapely.strtree import STRtree
    from ..geometry.geometric_utils import robust_intersection
    live = [w for w in walls if w.status != Status.REJECTED]
    ext = [w for w in live if w.classification == "exterior_wall" and w.polygon() is not None]
    if not ext:
        return 0
    tree = STRtree([w.polygon() for w in ext])
    n = 0
    for w in live:
        if w.classification in ("exterior_wall", "railing"):
            continue
        ax = w.axis
        if ax is None or ax.length <= 0:
            continue
        cover = 0.0
        for k in tree.query(ax):
            v = ext[int(k)]
            if v is w or (v.thickness and w.thickness and v.thickness < w.thickness):
                continue
            cover = max(cover, robust_intersection(ax, _sbuf(v.polygon(), tol.snap)).length)
        if cover >= min_share * ax.length:
            w.classification = "exterior_wall"
            w.notes.append(FACADE_LINE_NOTE)
            n += 1
    return n


def classify_openings(openings, walls):
    wmap = {w.id: w for w in walls}
    for o in openings:
        host = wmap.get(o.host_wall)
        ext = host is not None and host.classification == "exterior_wall"
        if o.type == "door":
            o.classification = "exterior_door" if ext else "interior_door"
        elif o.type == "window":
            o.classification = "exterior_window" if ext else "interior_window"
        else:
            o.classification = "exterior_opening" if ext else "passage"


def classify_columns(columns, walls, tol):
    polys = [w.polygon(tol.snap * 2) for w in walls if w.status != Status.REJECTED]
    polys = [p for p in polys if p is not None]
    for c in columns:
        touching = any(p.distance(c.polygon) <= tol.snap * 2 for p in polys)
        c.classification = "engaged_column" if touching else "free_standing_column"
