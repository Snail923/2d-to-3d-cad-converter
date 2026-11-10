"""Pitched (hip) roof over a storey outline.

Every side of the outline carries a roof plane rising inwards at the same pitch, so the planes
meet in hips, valleys and ridges exactly as a hip roof is drawn in plan (triangles over the short
sides, trapezia over the long ones). The surface is built from its contour lines: the outline
moved inwards by d (mitred corners) is the roof's contour at height d x tan(pitch). A height limit
gives a flat top (a roof deck) where the planes would rise above it.

    z(d) = eave + tan(pitch) * d,   d = distance moved inwards from the eave line

``roof_height_at`` gives the underside height at a point, used to trim the walls of the attic
room under the slope."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import triangulate


def _polys(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    return [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon" and p.area > 0]


def _inset(poly, d):
    return poly if d <= 0 else poly.buffer(-d, join_style=2, mitre_limit=10.0)


def _dense(ring, step):
    P = np.asarray(ring.coords, float)
    out = []
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
        out.extend(a + (b - a) * (k / n) for k in range(n))
    return out


def hip_roof(ring, pitch_deg: float, max_rise: float | None = None, eave_z: float = 0.0, contours: int = 12):
    """(V, F, rise) of the roof surface over the simple polygon ``ring``; rise = its height above
    the eave. Empty arrays when the outline is degenerate."""
    empty = (np.zeros((0, 3)), np.zeros((0, 3), int), 0.0)
    try:
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = max(_polys(poly.buffer(0)), key=lambda p: p.area)
        poly = Polygon(poly.exterior)
    except Exception:  # noqa: BLE001
        return empty
    if poly.is_empty or poly.area <= 0:
        return empty
    slope = math.tan(math.radians(pitch_deg))
    # how far in the planes go before they meet (ridge), by bisection on the inset
    lo, hi = 0.0, math.sqrt(poly.area)
    for _ in range(24):
        mid = (lo + hi) / 2
        if _polys(_inset(poly, mid)):
            lo = mid
        else:
            hi = mid
    d_top = lo
    if max_rise is not None and slope > 0:
        d_top = min(d_top, max_rise / slope)
    if d_top <= 1e-6:
        return empty
    step = max(0.4, d_top / max(contours, 1))
    levels = [d_top * k / contours for k in range(contours)] + [d_top]
    z_of: dict[tuple, float] = {}
    for d in levels:
        for p in _polys(_inset(poly, d)):
            for r in [p.exterior, *p.interiors]:
                for x, y in _dense(r, step * 1.5):
                    k = (round(float(x), 5), round(float(y), 5))
                    z_of[k] = max(z_of.get(k, 0.0), d * slope)
    pts = list(z_of)
    if len(pts) < 3:
        return empty
    grown = poly.buffer(1e-6)
    V, F, index = [], [], {}
    for t in triangulate(MultiPoint(pts)):
        if not grown.contains(t.representative_point()):
            continue
        tri = []
        for x, y in list(t.exterior.coords)[:3]:
            k = (round(float(x), 5), round(float(y), 5))
            if k not in z_of:                              # (numerical noise of the triangulation)
                k = min(pts, key=lambda q: (q[0] - x) ** 2 + (q[1] - y) ** 2)
            if k not in index:
                index[k] = len(V)
                V.append([k[0], k[1], eave_z + z_of[k]])
            tri.append(index[k])
        if len(set(tri)) == 3:
            F.append(tri)
    if not F:
        return empty
    return np.asarray(V, float), np.asarray(F, int), d_top * slope


def roof_height_at(x: float, y: float, rings, pitch_deg: float, max_rise: float | None, eave_z: float):
    """Height of the roof surface over (x, y), or None outside every roof outline."""
    slope = math.tan(math.radians(pitch_deg))
    p = Point(x, y)
    best = None
    for ring in rings:
        poly = ring if isinstance(ring, Polygon) else Polygon(ring)
        if not poly.buffer(1e-6).contains(p):
            continue
        rise = slope * poly.exterior.distance(p)
        if max_rise is not None:
            rise = min(rise, max_rise)
        z = eave_z + rise
        best = z if best is None else max(best, z)
    return best
