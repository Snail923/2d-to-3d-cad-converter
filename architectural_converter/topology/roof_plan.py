"""Roof outline from the roof plan of the drawing.

A roof plan shows the pitched roof as areas hatched with a roof-tile pattern (AutoCAD's AR-RSHKE,
AR-RROOF, or any pattern / layer named roof, shake, shingle ...), one hatch per roof face, with
what is not tiled left out: a flat deck or an open core in the middle is a hole in the hatched
area. The union of those hatches is the roof's outline, as drawn.

Roof plans are drawn on their own sheet. The sheet is brought onto the reference floor plan with
the same linework vote that aligns the floor plans (the plot lines, the building outline and the
grid repeat on every sheet)."""
from __future__ import annotations

import math

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon, box

from ..geometry.geometric_utils import polygons_of, robust_union, safe_buffer


def is_roof_hatch(region, keys) -> bool:
    name = f"{getattr(region, 'pattern', '') or ''} {getattr(region, 'layer', '') or ''}".upper()
    return any(k.upper() in name for k in keys)


def roof_areas(hatch_regions, keys, min_area_m2: float = 10.0, min_width_m: float = 2.0) -> list:
    """Tiled areas drawn in plan: unions of roof hatches wide enough to be a roof seen from above
    (a roof seen in an elevation is a thin band)."""
    polys = [h.polygon for h in hatch_regions if is_roof_hatch(h, keys) and h.polygon is not None and not h.polygon.is_empty]
    if not polys:
        return []
    merged = safe_buffer(safe_buffer(robust_union(polys), 0.08, join_style=2), -0.08, join_style=2)
    out = []
    for p in polygons_of(merged):
        solid = Polygon(p.exterior)
        core = safe_buffer(solid, -min_width_m / 2)
        if p.area >= min_area_m2 and core is not None and not core.is_empty:
            out.append(p)
    return out


def _points(strokes, bbox, pad):
    reg = box(bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    P = []
    for st in strokes:
        if reg.contains(st.line()):
            P.append(st.p0)
            P.append(st.p1)
    return np.array(P, dtype=float) if P else np.zeros((0, 2))


def align_roof(area, strokes, ref_bbox, tol, pad: float = 12.0):
    """(translation onto the reference floor plan, votes, points) for a roof area on its sheet."""
    from .level_analysis import _align  # noqa: PLC0415
    grid = max(tol.snap * 4, 0.02)
    d, votes, n = _align(_points(strokes, ref_bbox, pad), _points(strokes, area.bounds, pad), grid)
    return d, votes, n


def roof_for_level(areas, strokes, ref_bbox, tol, target, below=None, min_overlap: float = 0.6, min_area_m2: float = 10.0,
                   ref_translation=(0.0, 0.0)):
    """The drawn roof that belongs over ``target`` (the attic storey's outline in the stacking
    frame): the aligned roof area that lies on it, cut to ``below`` (the storey under it: a roof is
    not wider than the building). ``ref_bbox`` is the sheet of a placed floor plan the roof sheet is
    aligned onto, ``ref_translation`` that floor plan's own translation into the stacking frame.
    Returns (polygons with their holes, info) or (None, info)."""
    best = None
    for a in areas:
        d, votes, n = align_roof(a, strokes, ref_bbox, tol)
        if d is None or votes < max(8, 0.05 * n):
            continue
        d = (d[0] + ref_translation[0], d[1] + ref_translation[1])
        moved = translate(a, d[0], d[1])
        solid = Polygon(moved.exterior)
        ov = solid.intersection(target).area / max(solid.area, 1e-9)
        if ov >= min_overlap and (best is None or votes > best[1]):
            best = (moved, votes, n, d, ov)
    if best is None:
        return None, {"roof_areas": len(areas), "matched": False}
    moved, votes, n, d, ov = best
    if below is not None and not below.is_empty:
        moved = moved.intersection(below)
    parts = [p for p in polygons_of(moved) if p.area >= min_area_m2]
    return parts or None, {"roof_areas": len(areas), "matched": bool(parts), "votes": votes, "points": n,
                           "translation": [round(d[0], 3), round(d[1], 3)], "overlap": round(ov, 3)}


# ----------------------------------------------------------------------------- regular roof outline
# A tile hatch is not a roof outline yet: it leaves out every dormer (a narrow notch pointing into
# the roof), the box of every label written on it, and the core that rises through the roof, and
# its sides wander by the width of a hatch line. Built as drawn, each of these becomes a valley or
# a hip of its own and the roof folds into a herringbone. The rules below turn the hatch into the
# outline a hip roof is set out on, with nothing specific to one drawing:
#
# 1. Dormers and slivers: notches narrower than ``dormer_max_width_m`` are closed (morphological
#    closing). Those entered from the eave line and deep enough are kept as dormers.
# 2. Small holes (label boxes, gaps of the hatch) are filled; larger holes stay open (a court).
# 3. Abutments: an untiled pocket of the roof (inside its closure) that holds enclosed rooms of the
#    storey and is walled in by the roof on most of its perimeter is a core rising through the
#    roof: it becomes part of the roof body (flat top, not tiled), so the planes run into its
#    walls instead of down into the pocket. An untiled pocket without rooms (an entrance recess,
#    open air) stays out of the roof: its sides are eaves.
# 4. Rectified: sides within ``angle_tol_deg`` of the building's two main directions are made
#    exactly parallel / perpendicular to them, jogs shorter than ``min_edge_m`` are removed.

REGULARIZE = {"dormer_max_width_m": 1.6, "dormer_min_depth_m": 0.5, "dormer_min_area_m2": 0.3,
              "fill_holes_below_m2": 4.0, "angle_tol_deg": 8.0, "min_edge_m": 0.25,
              "abutment_closing_m": 4.0, "abutment_min_area_m2": 2.0, "abutment_wall_margin_m": 0.35,
              "abutment_min_enclosure": 0.5, "max_area_change": 0.08}


def _close(g, r):
    return safe_buffer(safe_buffer(g, r, join_style=2, mitre_limit=10.0), -r, join_style=2, mitre_limit=10.0)


def _open(g, r):
    return safe_buffer(safe_buffer(g, -r, join_style=2, mitre_limit=10.0), r, join_style=2, mitre_limit=10.0)


def dominant_axis(geom) -> float:
    """Main direction of an outline (radians, in [0, pi/2)): length-weighted mean of its sides'
    directions modulo a right angle."""
    sx = sy = 0.0
    for p in polygons_of(geom):
        for ring in [p.exterior, *p.interiors]:
            P = np.asarray(ring.coords, float)
            E = P[1:] - P[:-1]
            L = np.hypot(E[:, 0], E[:, 1])
            a = np.arctan2(E[:, 1], E[:, 0]) * 4.0
            sx += float((L * np.cos(a)).sum())
            sy += float((L * np.sin(a)).sum())
    return (math.atan2(sy, sx) / 4.0) % (math.pi / 2)


def _rectify_ring(P, tol, min_edge):
    """Ring (open, in the building's axis frame) with its near-axis sides made exactly horizontal /
    vertical and short jogs between two parallel sides removed."""
    n = len(P)
    E = np.roll(P, -1, 0) - P
    L = np.hypot(E[:, 0], E[:, 1])
    ang = np.arctan2(E[:, 1], E[:, 0]) % math.pi
    cls = ["H" if min(a, math.pi - a) < tol else "V" if abs(a - math.pi / 2) < tol else "O" for a in ang]
    if all(c == cls[0] for c in cls):
        return P
    start = next(i for i in range(n) if cls[i] != cls[i - 1])
    chains = []                                   # [class, length, weighted value, first vertex index]
    for k in range(n):
        i = (start + k) % n
        val = (P[i, 1] + P[(i + 1) % n, 1]) / 2 if cls[i] == "H" else (P[i, 0] + P[(i + 1) % n, 0]) / 2
        if chains and chains[-1][0] == cls[i] and cls[i] != "O":
            c = chains[-1]
            c[2] = (c[2] * c[1] + val * L[i]) / max(c[1] + L[i], 1e-12)
            c[1] += L[i]
        else:
            chains.append([cls[i], L[i], val, i])
    changed = True
    while changed and len(chains) > 4:            # short sides between two parallel ones are a jog
        changed = False
        m = len(chains)
        for k in range(m):
            if chains[k][1] >= min_edge or chains[k - 1][1] < min_edge:
                continue
            j = k                                 # the run of short sides starting at k
            while chains[j % m][1] < min_edge and (j - k) < m - 2:
                j += 1
            a, b = chains[k - 1], chains[j % m]
            run = sum(chains[i % m][1] for i in range(k, j))
            if a is b or a[0] != b[0] or a[0] == "O" or run >= 2 * min_edge:
                continue
            a[2] = (a[2] * a[1] + b[2] * b[1]) / max(a[1] + b[1], 1e-12)
            a[1] += b[1] + run
            drop = {i % m for i in range(k, j + 1)}
            chains = [x for i, x in enumerate(chains) if i not in drop]
            changed = True
            break
    out = []
    for k, c in enumerate(chains):
        prev = chains[k - 1]
        x, y = P[c[3]]
        for ch in (prev, c):
            if ch[0] == "H":
                y = ch[2]
            elif ch[0] == "V":
                x = ch[2]
        out.append((x, y))
    return np.asarray(out, float)


def rectify(geom, angle_tol_deg: float = 8.0, min_edge_m: float = 0.25, max_area_change: float = 0.08):
    """``geom`` with its sides made exactly parallel / perpendicular to its main directions (see
    above). Parts that would change by more than ``max_area_change`` are kept as they are."""
    from shapely.affinity import rotate
    theta = dominant_axis(geom)
    tol = math.radians(angle_tol_deg)
    out = []
    for p in polygons_of(geom):
        c = p.centroid
        r = rotate(p, -theta, origin=c, use_radians=True)
        try:
            rings = []
            for ring in [r.exterior, *r.interiors]:
                ring = ring.simplify(min(0.05, min_edge_m / 4))      # hatch wobble, not a side
                P = np.asarray(ring.coords, float)[:-1]
                Q = _rectify_ring(P, tol, min_edge_m) if len(P) >= 4 else P
                if len(Q) >= 3:
                    rings.append(Q)
            q = Polygon(rings[0], [h for h in rings[1:] if Polygon(h).area > 1e-6]) if rings else None
            if q is None or not q.is_valid:
                q = None if q is None else max(polygons_of(q.buffer(0)), key=lambda g: g.area, default=None)
            if q is not None:
                q = rotate(q, theta, origin=c, use_radians=True)
            if q is None or q.is_empty or abs(q.area - p.area) > max_area_change * p.area \
                    or q.symmetric_difference(p).area > 2 * max_area_change * p.area:
                q = p
        except Exception:  # noqa: BLE001
            q = p
        out.append(q)
    return out


def _mouth(notch, solid):
    """Where a notch opens onto the eave line: (a, b) end points on that line, or None for an
    inner hole."""
    edge = notch.boundary.intersection(safe_buffer(solid.boundary, 0.02))
    segs = [g for g in getattr(edge, "geoms", [edge]) if g.geom_type == "LineString" and g.length > 0.05]
    if not segs:
        return None
    pts = np.vstack([np.asarray(s.coords, float) for s in segs])
    mid = pts.mean(axis=0)
    best = None                                   # the eave side the notch opens onto
    for p in polygons_of(solid):
        R = np.asarray(p.exterior.coords, float)
        for a, b in zip(R[:-1], R[1:]):
            ab = b - a
            L2 = float(ab @ ab)
            if L2 < 1e-12:
                continue
            t = min(1.0, max(0.0, float((mid - a) @ ab) / L2))
            dist = float(np.linalg.norm(a + t * ab - mid))
            if best is None or dist < best[0]:
                best = (dist, a, ab / math.sqrt(L2))
    _, a0, d = best
    t = (pts - a0) @ d
    a, b = a0 + float(t.min()) * d, a0 + float(t.max()) * d
    return (a, b) if float(np.linalg.norm(b - a)) > 0.2 else None


def _snap_to(geom, ref, tol: float, theta: float):
    """``geom`` with every vertex coordinate (in the building's axis frame) that lies within
    ``tol`` of a vertex coordinate of ``ref`` moved onto it: shared sides line up exactly."""
    from shapely.affinity import rotate
    o = ref.centroid
    R = rotate(ref, -theta, origin=o, use_radians=True)
    xs = np.unique(np.concatenate([np.asarray(r.coords)[:, 0] for p in polygons_of(R) for r in [p.exterior, *p.interiors]]))
    ys = np.unique(np.concatenate([np.asarray(r.coords)[:, 1] for p in polygons_of(R) for r in [p.exterior, *p.interiors]]))

    def snap(v, vals):
        k = int(np.argmin(np.abs(vals - v)))
        return float(vals[k]) if abs(vals[k] - v) <= tol else float(v)
    out = []
    for p in polygons_of(rotate(geom, -theta, origin=o, use_radians=True)):
        ring = [(snap(x, xs), snap(y, ys)) for x, y in p.exterior.coords]
        q = Polygon(ring)
        q = q if q.is_valid else q.buffer(0)
        out.extend(polygons_of(rotate(q, theta, origin=o, use_radians=True)))
    return out


def _fill_holes(geom, below_m2):
    """``geom`` with its holes smaller than ``below_m2`` filled; (geometry, number filled)."""
    parts, n = [], 0
    for p in polygons_of(geom):
        holes = [h for h in p.interiors if Polygon(h).area >= below_m2]
        n += len(p.interiors) - len(holes)
        parts.append(Polygon(p.exterior, holes))
    return robust_union(parts), n


def regularize_roof(tiled, rooms=(), with_dormers: bool = True, opts: dict | None = None, walls=()):
    """Regular roof outline from the drawn (tiled) area, see the rules above. ``rooms``: polygons of
    the enclosed rooms of the storey, ``walls``: its wall bodies (same frame); a core is its rooms
    and the walls around them (without walls: the rooms grown by ``abutment_wall_margin_m``). Returns {"parts": [Polygon], "abutments":
    [Polygon], "dormers": [{"outline", "mouth", "tip", "width", "depth"}], "info": {...}}."""
    from shapely.geometry import Point as _Point
    from ..geometry.geometric_utils import robust_difference, robust_intersection
    o = dict(REGULARIZE, **(opts or {}))
    T = robust_union(polygons_of(tiled))
    if T.is_empty:
        return None
    info = {"dormers": 0, "holes_filled": 0, "abutments": 0, "area_drawn_m2": round(T.area, 2)}
    r = o["dormer_max_width_m"] / 2
    Tc, info["holes_filled"] = _fill_holes(robust_union([_close(T, r), T]), o["fill_holes_below_m2"])
    dormers = []
    if with_dormers:
        for q in polygons_of(robust_difference(Tc, T)):
            if q.area < o["dormer_min_area_m2"]:
                continue
            mouth = _mouth(q, robust_union([Polygon(p.exterior) for p in polygons_of(Tc)]))
            if mouth is None:
                continue
            a, b = mouth
            w = float(np.linalg.norm(b - a))
            if w > o["dormer_max_width_m"] + 0.2:
                continue
            m = (a + b) / 2
            nrm = np.array([-(b - a)[1], (b - a)[0]]) / max(w, 1e-9)
            if not Tc.contains(_Point(*(m + 0.05 * nrm))):
                nrm = -nrm
            depth = max(float((np.asarray(v) - m) @ nrm) for v in q.exterior.coords)
            if depth < o["dormer_min_depth_m"]:
                continue
            dormers.append({"outline": q, "mouth": (a, b), "tip": m + depth * nrm, "width": w, "depth": depth})
    info["dormers"] = len(dormers)
    abuts = []
    room_geom = robust_union([g for g in rooms if g is not None and not g.is_empty]) if rooms else None
    wall_geom = robust_union([g for g in walls if g is not None and not g.is_empty]) if walls else None
    wall_geom = None if wall_geom is not None and wall_geom.is_empty else wall_geom
    if room_geom is not None and not room_geom.is_empty:
        closure = robust_union([Polygon(p.exterior) for p in polygons_of(_close(Tc, o["abutment_closing_m"]))])
        edge = safe_buffer(Tc.boundary, 0.05)
        for q in polygons_of(robust_difference(closure, Tc)):
            if q.area < o["abutment_min_area_m2"]:
                continue
            held = robust_intersection(q, room_geom)
            if held.is_empty or held.area < o["abutment_min_area_m2"] / 2:
                continue
            grown = safe_buffer(held, o["abutment_wall_margin_m"], join_style=2)
            if wall_geom is not None:             # the core's own walls: those along its rooms
                own = [w for w in polygons_of(robust_intersection(wall_geom, grown))
                       if w.intersects(safe_buffer(held, 0.05, join_style=2))]
                grown = _close(robust_union([held, *own]), 0.1)
            a = robust_intersection(q, grown)
            for p in polygons_of(_open(a, 0.2)):
                if p.area < o["abutment_min_area_m2"]:
                    continue
                if p.boundary.intersection(edge).length / max(p.length, 1e-9) >= o["abutment_min_enclosure"]:
                    abuts.append(Polygon(p.exterior))
    info["abutments"] = len(abuts)
    body, _ = _fill_holes(robust_union([Tc, *abuts, _close(robust_union([Tc, *abuts]), o["min_edge_m"])]),
                          o["fill_holes_below_m2"])         # no hairline left between a core and the tiles
    out = [p for p in rectify(body, o["angle_tol_deg"], o["min_edge_m"], o["max_area_change"]) if not p.is_empty]
    whole = robust_union(out)
    theta = dominant_axis(whole)
    abuts = [q for a in abuts for p in rectify(a, o["angle_tol_deg"], o["min_edge_m"], o["max_area_change"])
             for q in _snap_to(p, whole, o["min_edge_m"], theta)]
    abuts = [p for a in abuts for p in polygons_of(robust_intersection(a, whole)) if p.area >= o["abutment_min_area_m2"]]
    info["area_regular_m2"] = round(whole.area, 2)
    info["axis_deg"] = round(math.degrees(theta), 2)

    return {"parts": out, "abutments": abuts, "dormers": dormers, "info": info}
