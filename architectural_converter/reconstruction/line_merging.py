"""Collinear merging of fragmented wall candidates.

Two candidates on the same axis band (same direction, offset and thickness) are merged when
* the gap between them is within the snapping tolerance (drafting fragmentation), or
* the gap is filled by a crossing wall (the wall runs straight through a junction).
Gaps of opening size are *not* merged here (see ``opening_reconstruction``)."""
from __future__ import annotations

from shapely.ops import unary_union

from ..classification.candidate_resolver import aligned, same_band
from ..geometry.geometric_utils import rect_polygon
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
    maxt = max([x.thickness or 0 for x in walls] + [tol.snap * 2])
    i = 0
    while i < len(straight):
        a = straight[i]
        merged = False
        j = i + 1
        while j < len(straight):
            b = straight[j]
            if not same_band(a, b, tol):
                j += 1
                continue
            g0, g1 = _gap(a, b)
            gap = g1 - g0
            reason = None
            if gap <= tol.snap:
                reason = "fragment" if gap > 0 else "overlap"
            elif gap <= maxt * covered_factor:
                t = a.thickness or tol.snap * 2
                gp = rect_polygon(a.u, a.c, g0, g1, t)
                polys = [o.polygon(tol.snap * 2) for o in walls if o is not a and o is not b]
                cover = unary_union([p for p in polys if p is not None and p.intersects(gp)])
                if not cover.is_empty and cover.buffer(tol.snap).intersection(gp).area >= 0.9 * gp.area:
                    reason = "continuous through junction"
            if reason:
                merge_into(a, b, reason, (g0, g1) if gap > 0 else None)
                walls.remove(b)
                straight.pop(j)
                merged = True
                continue
            j += 1
        if not merged:
            i += 1
    return walls
