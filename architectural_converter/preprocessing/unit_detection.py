"""Drawing unit inference.

``$INSUNITS`` is used as a prior, never trusted blindly: many drawings carry a wrong or
missing unit code. Each candidate unit is scored by how plausible the drawing's
measurements become in metres:

* robust overall extent (building size)
* dominant spacing between parallel overlapping lines (wall thickness)
* radius of quarter-circle arcs (door swings)
* typical annotation text height
"""
from __future__ import annotations

import math

import numpy as np

from ..models.drawing import Drawing, UnitInfo

INSUNITS_TO_M = {1: ("in", 0.0254), 2: ("ft", 0.3048), 4: ("mm", 0.001), 5: ("cm", 0.01),
                 6: ("m", 1.0), 14: ("dm", 0.1), 10: ("yd", 0.9144)}
CANDIDATES = {"mm": 0.001, "cm": 0.01, "dm": 0.1, "m": 1.0, "in": 0.0254, "ft": 0.3048}
NAME_TO_M = dict(CANDIDATES)

# plausible ranges in metres: (low, high, weight)
RANGES = {
    "extent": (6.0, 400.0, 1.0),
    "thickness": (0.07, 0.5, 1.6),
    "door_radius": (0.6, 1.25, 1.4),
    "text_height": (0.08, 0.8, 0.4),
}


def _plaus(v: float | None, lo: float, hi: float) -> float | None:
    if v is None or v <= 0:
        return None
    if lo <= v <= hi:
        return 1.0
    b = lo if v < lo else hi
    return math.exp(-2.0 * abs(math.log(v / b)))


def _all_points(d: Drawing) -> np.ndarray:
    pts = [g.points for g in d.entities if g.points is not None and g.kind not in ("text", "dimension")]
    pts += [loop for g in d.entities if g.kind == "hatch" for loop in g.data.get("loops", [])]
    if not pts:
        return np.zeros((0, 2))
    return np.vstack(pts)


def _dominant_parallel_spacing(d: Drawing, extent: float) -> float | None:
    segs = []
    for g in d.entities:
        if g.kind == "line" and g.points is not None and not g.block_path:
            p0, p1 = g.points[0], g.points[1]
            L = float(np.hypot(*(p1 - p0)))
            if L > 0:
                segs.append((L, p0, p1))
    if len(segs) < 2:
        return None
    segs.sort(key=lambda s: -s[0])
    segs = segs[:1500]
    P0 = np.array([s[1] for s in segs])
    P1 = np.array([s[2] for s in segs])
    D = P1 - P0
    L = np.hypot(D[:, 0], D[:, 1])
    U = D / L[:, None]
    flip = (U[:, 1] < 0) | ((np.abs(U[:, 1]) < 1e-12) & (U[:, 0] < 0))
    U[flip] *= -1
    ang = np.arctan2(U[:, 1], U[:, 0]) % np.pi
    N = np.stack([-U[:, 1], U[:, 0]], axis=1)
    maxd = extent * 0.06
    dists, weights = [], []
    for i in range(len(segs)):
        da = np.abs(ang - ang[i]) % np.pi
        da = np.minimum(da, np.pi - da)
        mid = (P0 + P1) / 2
        off = (mid - mid[i]) @ N[i]
        s_i0, s_i1 = sorted((P0[i] @ U[i], P1[i] @ U[i]))
        sj0 = np.minimum(P0 @ U[i], P1 @ U[i])
        sj1 = np.maximum(P0 @ U[i], P1 @ U[i])
        ov = np.minimum(sj1, s_i1) - np.maximum(sj0, s_i0)
        m = (da < math.radians(1.5)) & (np.abs(off) > extent * 1e-5) & (np.abs(off) < maxd) & (ov > 0)
        m[: i + 1] = False
        if np.any(m):
            dists.extend(np.abs(off[m]).tolist())
            weights.extend(ov[m].tolist())
    if not dists:
        return None
    dists = np.array(dists)
    weights = np.array(weights)
    # weight short spacings higher (nearest neighbour); histogram in log space
    logd = np.log(dists)
    if float(np.ptp(logd)) < 1e-6:  # all spacings identical (e.g. one simple room)
        return float(dists[0])
    hist, edges = np.histogram(logd, bins=60, weights=weights / np.sqrt(dists))
    k = int(np.argmax(hist))
    return float(math.exp((edges[k] + edges[k + 1]) / 2))


def detect_units(d: Drawing, forced: str | None = None) -> UnitInfo:
    ins = int(d.metadata.get("$INSUNITS", 0) or 0)
    header_unit, header_scale = INSUNITS_TO_M.get(ins, ("unitless", None))
    if forced:
        return UnitInfo(ins, header_unit, forced, NAME_TO_M[forced], 1.0, {"forced": True})
    pts = _all_points(d)
    if len(pts) == 0:
        unit = header_unit if header_scale else "m"
        return UnitInfo(ins, header_unit, unit, header_scale or 1.0, 0.1, {"reason": "no geometry"})
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    extent = float(np.hypot(*(hi - lo))) or 1.0
    spacing = _dominant_parallel_spacing(d, extent)
    radii = [g.data["radius"] for g in d.entities if g.kind == "arc"
             and 70 <= math.degrees(g.data.get("sweep", 0)) <= 110]
    door_r = float(np.median(radii)) if len(radii) >= 1 else None
    th = [g.data["height"] for g in d.entities if g.kind == "text" and g.data.get("height", 0) > 0]
    text_h = float(np.median(th)) if th else None
    raw = {"extent": extent, "thickness": spacing, "door_radius": door_r, "text_height": text_h}

    scores = {}
    for name, f in CANDIDATES.items():
        s = 0.0
        wsum = 0.0
        for key, (a, b, w) in RANGES.items():
            v = raw[key]
            p = _plaus(v * f if v else None, a, b)
            if p is not None:
                s += w * p
                wsum += w
        s = s / wsum if wsum else 0.0
        if header_scale is not None and abs(f - header_scale) < 1e-12:
            s += 0.25  # header prior
        scores[name] = round(s, 4)
    best = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0)
    conf = float(max(0.05, min(1.0, 0.5 + margin)))
    ev = {"measurements_drawing_units": raw, "scores": scores}
    if header_scale is not None and best != header_unit:
        ev["warning"] = f"$INSUNITS says {header_unit} but geometry is more plausible in {best}"
    return UnitInfo(ins, header_unit, best, CANDIDATES[best], round(conf, 3), ev)
