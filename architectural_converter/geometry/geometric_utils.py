"""Small, dependency-light geometric helpers used across the engine."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon, GeometryCollection
from shapely.geometry.base import BaseGeometry

EPS = 1e-12


def canonical_dir(dx: float, dy: float) -> tuple[float, float]:
    """Unit direction with angle in [0, pi)."""
    L = math.hypot(dx, dy)
    if L < EPS:
        return (1.0, 0.0)
    ux, uy = dx / L, dy / L
    if uy < -1e-12 or (abs(uy) <= 1e-12 and ux < 0):
        ux, uy = -ux, -uy
    return (ux, uy)


def dir_angle(u: Sequence[float]) -> float:
    return math.atan2(u[1], u[0]) % math.pi


def angle_diff(a: float, b: float) -> float:
    """Smallest difference between two undirected angles (radians)."""
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def normal(u: Sequence[float]) -> tuple[float, float]:
    return (-u[1], u[0])


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def frame_point(u: Sequence[float], s: float, c: float) -> tuple[float, float]:
    n = normal(u)
    return (s * u[0] + c * n[0], s * u[1] + c * n[1])


def rect_polygon(u: Sequence[float], c: float, s0: float, s1: float, t: float) -> Polygon:
    h = t / 2.0
    pts = [frame_point(u, s0, c - h), frame_point(u, s1, c - h),
           frame_point(u, s1, c + h), frame_point(u, s0, c + h)]
    return Polygon(pts)


def line_intersection(p, u, q, v):
    """Intersection of p + a*u and q + b*v. Returns (point, a, b) or None if parallel."""
    den = cross(u, v)
    if abs(den) < 1e-12:
        return None
    w = (q[0] - p[0], q[1] - p[1])
    a = cross(w, v) / den
    b = cross(w, u) / den
    return ((p[0] + a * u[0], p[1] + a * u[1]), a, b)


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def merge_intervals(intervals: Iterable[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    iv = sorted((min(a, b), max(a, b)) for a, b in intervals)
    out: list[list[float]] = []
    for a, b in iv:
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def subtract_intervals(base: tuple[float, float], cuts: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces = [base]
    for c0, c1 in merge_intervals(cuts):
        nxt = []
        for a, b in pieces:
            if c1 <= a or c0 >= b:
                nxt.append((a, b))
                continue
            if c0 > a:
                nxt.append((a, c0))
            if c1 < b:
                nxt.append((c1, b))
        pieces = nxt
    return pieces


def polygons_of(geom: BaseGeometry | None) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        out = []
        for g in geom.geoms:
            out.extend(polygons_of(g))
        return out
    return []


def inscribed_width(poly: Polygon, max_iter: int = 14) -> float:
    """Approximate width of the largest inscribed disc (2*r) by binary search on erosion."""
    if poly.is_empty or poly.area <= 0:
        return 0.0
    minx, miny, maxx, maxy = poly.bounds
    hi = min(maxx - minx, maxy - miny) / 2.0
    lo = 0.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if mid <= 0:
            break
        if poly.buffer(-mid, join_style=2).is_empty:
            hi = mid
        else:
            lo = mid
    return 2.0 * lo


def thin_fraction(poly: BaseGeometry, max_half_thickness: float) -> float:
    """Fraction of the area that disappears under a morphological opening of radius
    ``max_half_thickness``. 1.0 means the region is thin everywhere (wall-like)."""
    if poly.is_empty or poly.area <= 0:
        return 0.0
    opened = poly.buffer(-max_half_thickness, join_style=2).buffer(max_half_thickness, join_style=2)
    return float(max(0.0, 1.0 - opened.area / poly.area))


def mrr_dims(geom: BaseGeometry) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    """Minimum rotated rectangle: (width, length, long-axis unit direction, centre)."""
    r = geom.minimum_rotated_rectangle
    if not isinstance(r, Polygon):
        c = geom.centroid
        if isinstance(r, LineString) and r.length > 0:
            (x0, y0), (x1, y1) = r.coords[0], r.coords[-1]
            return 0.0, r.length, canonical_dir(x1 - x0, y1 - y0), (c.x, c.y)
        return 0.0, 0.0, (1.0, 0.0), (c.x, c.y)
    xs = list(r.exterior.coords)[:4]
    e0 = (xs[1][0] - xs[0][0], xs[1][1] - xs[0][1])
    e1 = (xs[2][0] - xs[1][0], xs[2][1] - xs[1][1])
    l0, l1 = math.hypot(*e0), math.hypot(*e1)
    c = r.centroid
    if l0 >= l1:
        return l1, l0, canonical_dir(*e0), (c.x, c.y)
    return l0, l1, canonical_dir(*e1), (c.x, c.y)


def ring_area_signed(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def sample_line(line: LineString, step: float) -> np.ndarray:
    if line.length <= 0:
        c = line.coords[0]
        return np.array([c])
    n = max(2, int(math.ceil(line.length / step)) + 1)
    return np.array([line.interpolate(d).coords[0] for d in np.linspace(0, line.length, n)])


def round_pt(p, nd: int = 4):
    return [round(float(p[0]), nd), round(float(p[1]), nd)]


def coords_list(geom: BaseGeometry, nd: int = 4):
    """JSON-friendly coordinates for polygons / lines."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return {"exterior": [round_pt(p, nd) for p in geom.exterior.coords],
                "holes": [[round_pt(p, nd) for p in r.coords] for r in geom.interiors]}
    if isinstance(geom, LineString):
        return [round_pt(p, nd) for p in geom.coords]
    if isinstance(geom, MultiPolygon):
        return [coords_list(g, nd) for g in geom.geoms]
    return None
