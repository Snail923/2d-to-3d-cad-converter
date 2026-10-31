"""Helpers shared by wall detectors (evidence features and candidate construction)."""
from __future__ import annotations

import math

from ..geometry.geometric_utils import dot
from ..geometry.segment_pairing import cluster_consistency
from ..models.wall import Wall


def thickness_plausibility(t: float | None, cfg) -> float | None:
    if t is None:
        return None
    th = cfg["thickness"]
    lo, hi = th["plausible_min_m"], th["plausible_max_m"]
    mn, mx = th["min_m"], th["max_m"]
    if lo <= t <= hi:
        return 1.0
    if t < lo:
        return max(0.0, (t - mn) / max(lo - mn, 1e-9))
    return max(0.0, (mx - t) / max(mx - hi, 1e-9))


def stroke_members(stroke, u, s0, s1):
    """Members of ``stroke`` overlapping [s0, s1] given in frame ``u``."""
    if dot(stroke.u, u) >= 0:
        return stroke.members_in(s0, s1)
    return stroke.members_in(-s1, -s0)


def wall_from_piece(pc, ctx, method: str, strokes=None) -> Wall:
    strokes = strokes if strokes is not None else ctx.strokes
    cfg = ctx.cfg
    tol = ctx.tol
    si, sj = strokes[pc.i], strokes[pc.j]
    w = Wall(id=ctx.ids.new("wall"), kind="straight", u=pc.u, c=pc.c, s0=pc.s0, s1=pc.s1,
             thickness=pc.thickness, thickness_source="measured", detected_interval=(pc.s0, pc.s1))
    w.face_raw = []
    for st in (si, sj):
        face = set()
        for sid, rid in stroke_members(st, pc.u, pc.s0, pc.s1):
            w.face_segments.add(sid)
            w.source_entities.add(rid)
            face.add(rid)
        w.face_raw.append(face)
    at = tol.pair_angle_tol(pc.len_i, pc.len_j)
    w.add_evidence("parallelism", 1.0 - min(1.0, pc.angle_diff / max(at, 1e-9)) * 0.5)
    w.add_evidence("endpoint_alignment", min(1.0, max(pc.length / max(pc.len_i, 1e-9), pc.length / max(pc.len_j, 1e-9))))
    w.add_evidence("thickness_plausibility", thickness_plausibility(pc.thickness, cfg))
    th = cfg["thickness"]
    w.add_evidence("thickness_consistency", cluster_consistency(pc.thickness, ctx.thickness_clusters,
                                                                th["cluster_rel_tol"], th["cluster_abs_tol_m"]))
    w.add_evidence("emptiness", 1.0)
    lay = (sum(ctx.wall_evidence(l) for l in si.layers) / len(si.layers) +
           sum(ctx.wall_evidence(l) for l in sj.layers) / len(sj.layers)) / 2
    w.add_evidence("layer_match", lay)
    w.detection_methods.add(method)
    return w


def aspect_score(length: float, t: float | None) -> float | None:
    if not t:
        return None
    r = length / t
    return float(max(0.0, min(1.0, (r - 0.5) / 2.5)))
