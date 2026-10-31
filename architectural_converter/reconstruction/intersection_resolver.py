"""Wall intersection / junction resolution.

Parallel-pair detection stops a wall where its boundary lines stop, which is typically the
*inner* corner of an L-junction or the face of the wall it meets at a T-junction. Each wall
end is extended to the axis intersection with the wall it meets (T), or to the outer corner
(L: intersection + half the other wall's thickness / sin(angle)), but only when:
* the walls are not parallel,
* the required extension is small (<= factor x max(thickness)), and
* the intersection lies on (or right next to) the other wall.
Every extension is recorded in ``wall.reconstruction['extensions']``; the detected interval
is kept unchanged. Centre-line walls snap with the endpoint tolerance only."""
from __future__ import annotations

import math

from shapely.geometry import Point

from ..geometry.geometric_utils import dot, frame_point, line_intersection
from ..models.wall import Wall


def _end(w: Wall, e: int):
    p = w.point_at(w.s1 if e else w.s0)
    out = w.u if e else (-w.u[0], -w.u[1])
    return p, out


def resolve_junctions(walls: list[Wall], tol, cfg) -> int:
    rc = cfg["reconstruction"]
    factor = rc["junction_extension_factor"]
    min_ang = math.radians(rc["min_junction_angle_deg"])
    straight = [w for w in walls if w.kind in ("straight", "centerline")]
    count = 0
    plan = []
    for w in straight:
        for e in (0, 1):
            p, out = _end(w, e)
            best = None
            for v in straight:
                if v is w:
                    continue
                sin = abs(w.u[0] * v.u[1] - w.u[1] * v.u[0])
                if sin < math.sin(min_ang):
                    continue
                hit = line_intersection(p, out, v.point_at(v.s0), v.u)
                if hit is None:
                    continue
                P, a, b = hit       # a: distance along out from the end, b: position along v from v.s0
                tw = w.thickness or 0.0
                tv = v.thickness or 0.0
                if w.thickness and v.thickness:
                    max_ext = factor * max(tw, tv) + tol.snap
                else:
                    max_ext = max(4 * tol.snap, (tw + tv) / 2 * factor + 2 * tol.snap)
                if a < -(tv / 2 / sin + tol.snap) or a > max_ext:
                    continue
                vlen = v.s1 - v.s0
                reach = tw / 2 / sin + max_ext
                if b < -reach or b > vlen + reach:
                    continue
                at_v_end = b < tv / 2 / sin + tol.snap or b > vlen - tv / 2 / sin - tol.snap
                if b < 0 or b > vlen:
                    at_v_end = True
                # L: to the outer corner; T: through to the far face of the wall met (fully embedded)
                target = a + tv / 2 / sin
                if target <= tol.precision:
                    continue
                cand = (abs(a), target, v, at_v_end, P)
                if best is None or cand[0] < best[0]:
                    best = cand
            if best is not None:
                plan.append((w, e, best))
    for w, e, (_, target, v, at_end, P) in plan:
        if target > 0:
            if e:
                w.s1 += target
            else:
                w.s0 -= target
            w.reconstruction.setdefault("extensions", []).append(
                {"end": "end" if e else "start", "length": round(target, 4), "to_wall": v.id,
                 "junction": "L" if at_end else "T", "point": [round(P[0], 4), round(P[1], 4)]})
            count += 1
    return count


# ---------------------------------------------------------------------------- curved walls
def _arc_sweep(a):
    return (a["end"] - a["start"]) % (2 * math.pi) or 2 * math.pi


def _ang_dist_outside(theta, a):
    """Angular distance of theta outside the arc's range (0 if inside)."""
    sw = _arc_sweep(a)
    d = (theta - a["start"]) % (2 * math.pi)
    if d <= sw:
        return 0.0
    return min(d - sw, 2 * math.pi - d)


def resolve_arc_junctions(walls: list[Wall], tol, cfg) -> int:
    """Straight <-> curved wall junctions (arc pairs stop where their boundary arcs stop)."""
    factor = cfg["reconstruction"]["junction_extension_factor"]
    arcs = [w for w in walls if w.kind == "arc" and w.arc]
    straight = [w for w in walls if w.kind in ("straight", "centerline")]
    n = 0
    plan_s, plan_a = [], []
    for w in straight:
        for e in (0, 1):
            p, out = _end(w, e)
            for A in arcs:
                a = A.arc
                cx, cy, r = a["center_x"], a["center_y"], a["radius"]
                max_ext = factor * max(w.thickness or 0, A.thickness or 0) + tol.snap
                dx, dy = p[0] - cx, p[1] - cy
                bq = out[0] * dx + out[1] * dy
                cq = dx * dx + dy * dy - r * r
                disc = bq * bq - cq
                if disc < 0:
                    continue
                for s in sorted((-bq - math.sqrt(disc), -bq + math.sqrt(disc)), key=abs):
                    if not (-(A.thickness or 0) / 2 - tol.snap <= s <= max_ext):
                        continue
                    P = (p[0] + out[0] * s, p[1] + out[1] * s)
                    th = math.atan2(P[1] - cy, P[0] - cx)
                    outside = _ang_dist_outside(th, a)
                    if outside * r > max_ext:
                        continue
                    tan = (-math.sin(th), math.cos(th))
                    sinp = max(0.2, abs(out[0] * tan[1] - out[1] * tan[0]))
                    at_end = outside > 0 or min((th - a["start"]) % (2 * math.pi),
                                                 (a["end"] - th) % (2 * math.pi)) * r < (w.thickness or 0) / 2 + tol.snap
                    target = s + ((A.thickness or 0) / 2 / sinp if at_end else 0.0)
                    if target > tol.precision:
                        plan_s.append((w, e, target, A.id))
                    break
    for A in arcs:
        a = A.arc
        cx, cy, r = a["center_x"], a["center_y"], a["radius"]
        for which in ("start", "end"):
            th = a[which]
            E = (cx + r * math.cos(th), cy + r * math.sin(th))
            tan = (-math.sin(th), math.cos(th))
            out = tan if which == "end" else (-tan[0], -tan[1])
            for v in straight:
                max_ext = factor * max(v.thickness or 0, A.thickness or 0) + tol.snap
                hit = line_intersection(E, out, v.point_at(v.s0), v.u)
                if hit is None:
                    continue
                P, s, b = hit
                if not (-tol.snap <= s <= max_ext):
                    continue
                vlen = v.s1 - v.s0
                if b < -max_ext or b > vlen + max_ext:
                    continue
                sinp = max(0.2, abs(out[0] * v.u[1] - out[1] * v.u[0]))
                at_end = b < (v.thickness or 0) / 2 + tol.snap or b > vlen - (v.thickness or 0) / 2 - tol.snap
                ext = s + ((v.thickness or 0) / 2 / sinp if at_end else 0.0)
                if ext > tol.precision:
                    plan_a.append((A, which, ext / r, v.id))
                break
    for w, e, target, aid in plan_s:
        if e:
            w.s1 += target
        else:
            w.s0 -= target
        w.reconstruction.setdefault("extensions", []).append(
            {"end": "end" if e else "start", "length": round(target, 4), "to_wall": aid, "junction": "curved"})
        n += 1
    for A, which, dth, vid in plan_a:
        if which == "end":
            A.arc["end"] = (A.arc["end"] + dth) % (2 * math.pi)
        else:
            A.arc["start"] = (A.arc["start"] - dth) % (2 * math.pi)
        A.reconstruction.setdefault("extensions", []).append(
            {"end": which, "angle_rad": round(dth, 5), "to_wall": vid, "junction": "curved"})
        n += 1
    return n
