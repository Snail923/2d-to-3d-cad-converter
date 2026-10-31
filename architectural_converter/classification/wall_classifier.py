"""Wall classification: exterior / interior / partition (+ openings and columns)."""
from __future__ import annotations

from ..geometry.geometric_utils import robust_intersection

from ..models.base_entity import Status
from .architectural_rules import building_footprint, partition_threshold


def classify_walls(walls, solid, tol, cfg):
    cc = cfg["classification"]
    fp = building_footprint(solid)
    ring = fp.boundary if fp is not None else None
    for w in walls:
        if w.status == Status.REJECTED:
            continue
        p = w.polygon(max(tol.snap * 2, 0.01))
        if p is None or ring is None:
            w.classification = "interior_wall"
            continue
        L = max(w.length, 1e-9)
        ext = robust_intersection(p.buffer(tol.snap * 3), ring).length
        w.classification = "exterior_wall" if ext >= cc["exterior_face_fraction"] * L else "interior_wall"
    limit = partition_threshold([w for w in walls if w.status != Status.REJECTED], cfg)
    for w in walls:
        if w.classification == "interior_wall" and w.thickness and w.thickness <= limit:
            w.classification = "partition_wall"
        if w.classification == "interior_wall" and not w.thickness:
            w.notes.append("thickness unknown: interior/partition undecided")
    return fp, limit


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
