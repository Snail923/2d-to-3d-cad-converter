"""Associate boundary linework with final walls (traceability + boundary support).

For every straight wall, strokes that run along either face (parallel, offset = c +- t/2)
are linked as face segments. The covered fraction of each face is stored as evidence of
how well the wall is supported by drawn boundaries; when both faces are present the
thickness is re-measured from them."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point

from ..geometry.geometric_utils import dot, merge_intervals, normal


def associate_faces(walls, ctx) -> None:
    tol = ctx.tol
    if ctx.stroke_tree is None:
        return
    for w in walls:
        if w.kind != "straight" or not w.thickness or w.length <= 0:
            continue
        poly = w.polygon()
        n = normal(w.u)
        cov = {-1: [], 1: []}
        offs = {-1: [], 1: []}
        for k in ctx.stroke_tree.query(poly.buffer(tol.snap * 3)):
            st = ctx.strokes[int(k)]
            if abs(abs(dot(st.u, w.u)) - 1) > 1e-3:
                continue
            o = dot(st.p0, n) - w.c
            for side in (-1, 1):
                if abs(o - side * w.thickness / 2) <= max(tol.snap * 2, 0.05 * w.thickness):
                    sg = 1.0 if dot(st.u, w.u) >= 0 else -1.0
                    a, b = sorted((sg * st.s0, sg * st.s1))
                    a, b = max(a, w.s0), min(b, w.s1)
                    if b - a > tol.snap:
                        cov[side].append((a, b))
                        offs[side].append(o)
                        for sid, rid in st.members_in(*sorted((sg * a, sg * b))):
                            w.face_segments.add(sid)
                            w.source_entities.add(rid)
        frac = {}
        for side in (-1, 1):
            L = sum(b - a for a, b in merge_intervals(cov[side]))
            frac[side] = min(1.0, L / w.length)
        w.reconstruction["face_support"] = {"left": round(frac[1], 3), "right": round(frac[-1], 3)}
        if offs[-1] and offs[1] and w.thickness_source == "measured":
            t_new = float(np.median(offs[1]) - np.median(offs[-1]))
            if abs(t_new - w.thickness) <= 0.15 * w.thickness:
                w.thickness = t_new
