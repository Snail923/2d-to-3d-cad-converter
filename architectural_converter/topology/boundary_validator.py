"""Boundary support of a room polygon (used for manual rooms and hypotheses)."""
from __future__ import annotations

from shapely.ops import unary_union

from ..models.base_entity import Status


def boundary_support(polygon, walls, tol) -> dict:
    acc = [w.polygon(max(tol.snap * 2, 0.01)) for w in walls if w.status == Status.ACCEPTED]
    allw = [w.polygon(max(tol.snap * 2, 0.01)) for w in walls if w.status != Status.REJECTED]
    L = polygon.exterior.length or 1.0
    out = {}
    for key, polys in (("accepted_walls", acc), ("any_walls", allw)):
        polys = [p for p in polys if p is not None]
        if polys:
            u = unary_union(polys).buffer(tol.snap * 4)
            out[key] = polygon.exterior.intersection(u).length / L
        else:
            out[key] = 0.0
    out["unsupported"] = 1.0 - out["any_walls"]
    return out
