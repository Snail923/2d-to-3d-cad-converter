"""Stage 5 orchestration: fragments -> junctions -> openings -> boundary association."""
from __future__ import annotations

from ..models.base_entity import Status
from .endpoint_snapping import snap_centerline_endpoints
from .intersection_resolver import resolve_arc_junctions, resolve_junctions
from .line_merging import merge_collinear
from .opening_reconstruction import OpeningReconstructor
from .wall_pairing import associate_faces


def _rehost(live, openings):
    """Point openings at the wall that absorbed their host during the final merges."""
    by_id = {o.id: o for o in openings}
    for w in live:
        for oid in w.openings:
            o = by_id.get(oid)
            if o is None or o.host_wall == w.id:
                continue
            o.host_wall = w.id
            if o.host_interval:
                half = abs(o.host_interval[1] - o.host_interval[0]) / 2
                mid = (o.center[0] * w.u[0] + o.center[1] * w.u[1])
                o.host_interval = (mid - half, mid + half)


def reconstruct(walls, doors, windows, ctx, cfg):
    tol = ctx.tol
    live = [w for w in walls if w.status != Status.REJECTED]
    rejected = [w for w in walls if w.status == Status.REJECTED]
    n_before = len(live)
    live = merge_collinear(live, tol)
    snapped = snap_centerline_endpoints(live, tol)
    ext = resolve_junctions(live, tol, cfg)
    ext += resolve_arc_junctions(live, tol, cfg)
    live = merge_collinear(live, tol)   # T-junctions are filled now: walls run through them
    orc = OpeningReconstructor(cfg, ctx.ids, tol)
    generic = orc.run(live, doors, windows)
    live = merge_collinear(live, tol)
    ext += resolve_junctions(live, tol, cfg)
    _rehost(live, doors + windows + generic)
    associate_faces(live, ctx)
    log = {"walls_before": n_before, "walls_after": len(live), "snapped_endpoints": snapped,
           "junction_extensions": ext, **orc.log}
    return live + rejected, generic, log
