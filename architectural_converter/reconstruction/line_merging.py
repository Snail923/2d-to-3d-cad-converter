"""Collinear merging of fragmented wall candidates.

Two candidates on the same axis band (same direction, offset and thickness) are merged when
* the gap between them is within the snapping tolerance (drafting fragmentation), or
* the gap is filled by a crossing wall (the wall runs straight through a junction).
Gaps of opening size are *not* merged here (see ``opening_reconstruction``)."""
from __future__ import annotations


from shapely.strtree import STRtree

from ..classification.candidate_resolver import aligned, same_band
from ..geometry.geometric_utils import rect_polygon, robust_union, robust_intersection
from ..models.wall import Wall


def merge_into(m: Wall, w: Wall, note: str, gap: tuple[float, float] | None = None) -> None:
    c, s0, s1 = aligned(w, m.u)
    Lm, Lw = m.length, s1 - s0
    if m.thickness and w.thickness and m.thickness_source == w.thickness_source:
        m.thickness = (m.thickness * Lm + w.thickness * Lw) / max(Lm + Lw, 1e-12)
    elif w.thickness and not m.thickness:
        m.thickness, m.thickness_source = w.thickness, w.thickness_source
        m.kind = "straight"
    if (m.thickness is None) == (w.thickness is None):
        m.c = (m.c * Lm + c * Lw) / max(Lm + Lw, 1e-12)
    m.s0, m.s1 = min(m.s0, s0), max(m.s1, s1)
    di = m.detected_interval or (m.s0, m.s1)
    wi = (s0, s1) if w.detected_interval is None else tuple(sorted((aligned(w, m.u)[1], aligned(w, m.u)[2])))
    m.detected_interval = (min(di[0], wi[0]), max(di[1], wi[1]))
    m.absorb(w)
    m.face_segments |= w.face_segments
    for oid in w.openings:
        if oid not in m.openings:
            m.openings.append(oid)
    m.reconstruction.setdefault("merged", []).append({"wall": w.id, "reason": note,
                                                      "gap": [round(x, 4) for x in gap] if gap else None})
    for k, v in w.reconstruction.items():
        if k == "merged":
            m.reconstruction["merged"].extend(v)
        elif k not in m.reconstruction:
            m.reconstruction[k] = v


def _gap(a: Wall, b: Wall):
    c, s0, s1 = aligned(b, a.u)
    if s0 >= a.s1:
        return a.s1, s0
    if a.s0 >= s1:
        return s1, a.s0
    return 0.0, 0.0


def merge_collinear(walls: list[Wall], tol, covered_factor: float = 1.3) -> list[Wall]:
    walls = list(walls)
    straight = [w for w in walls if w.kind in ("straight", "centerline")]
    if not straight:
        return walls
    maxt = max([x.thickness or 0 for x in walls] + [tol.snap * 2])
    reach = maxt * covered_factor + tol.snap
    alive = set(range(len(straight)))
    tree = STRtree([w.axis.buffer(reach) for w in straight])
    bodies = [(o, o.polygon(tol.snap * 2)) for o in walls]
    bodies = [(o, p) for o, p in bodies if p is not None]
    btree = STRtree([p for _, p in bodies]) if bodies else None
    for i in range(len(straight)):
        if i not in alive:
            continue
        a = straight[i]
        changed = True
        while changed:
            changed = False
            for j in sorted(int(k) for k in tree.query(a.axis.buffer(reach))):
                if j == i or j not in alive:
                    continue
                b = straight[j]
                if not same_band(a, b, tol):
                    continue
                g0, g1 = _gap(a, b)
                gap = g1 - g0
                reason = None
                if gap <= tol.snap:
                    reason = "fragment" if gap > 0 else "overlap"
                elif gap <= maxt * covered_factor:
                    t = a.thickness or tol.snap * 2
                    gp = rect_polygon(a.u, a.c, g0, g1, t)
                    polys = [p for o, p in (bodies[int(k)] for k in btree.query(gp, predicate="intersects"))
                             if o is not a and o is not b] if btree is not None else []
                    cover = robust_union(polys)
                    if not cover.is_empty and robust_intersection(cover.buffer(tol.snap), gp).area >= 0.9 * gp.area:
                        reason = "continuous through junction"
                if reason:
                    merge_into(a, b, reason, (g0, g1) if gap > 0 else None)
                    walls.remove(b)
                    alive.discard(j)
                    changed = True
                    break
    return walls
