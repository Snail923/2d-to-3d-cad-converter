"""Pitched (hip) roof over a storey outline, laid in courses of tiles.

Every side of the outline carries a roof plane rising inwards at the same pitch, so the planes
meet in hips, valleys and ridges exactly as a hip roof is drawn in plan (triangles over the short
sides, trapezia over the long ones). The surface is built from its contour lines: the outline
moved inwards by d (mitred corners) is the roof's contour at height d x tan(pitch).

    z(d) = eave + tan(pitch) * d,   d = distance moved inwards from the eave line

* **Hollow roofs.** Holes of the outline (the untiled middle of a roof plan: a deck, a core) are
  left open: the planes stop at the hole's edge.
* **Height limit.** Where the planes would rise above ``max_rise`` the roof is flat (a deck).
* **Tiles.** The surface is laid in courses: the band between two contour lines ``course_m`` apart
  is one row of tiles, standing ``thickness_m`` proud at its lower edge and lying flush at its
  upper edge, where the next row laps over it. Courses are returned in two alternating sets so
  they can be coloured as rows.

``roof_height_at`` gives the surface height at a point, used to trim the walls of the attic room
under the slope."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import triangulate

EMPTY = (np.zeros((0, 3)), np.zeros((0, 3), int))


def _polys(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g] if g.area > 0 else []
    return [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon" and p.area > 0]


def as_polygon(fp):
    """Footprint entry (ring, or {"exterior", "holes"}) -> (outer polygon without holes, polygon with holes)."""
    if isinstance(fp, dict):
        ext, holes = fp["exterior"], fp.get("holes") or []
    else:
        ext, holes = fp, []
    outer = Polygon(ext)
    if not outer.is_valid:
        outer = Polygon(max(_polys(outer.buffer(0)), key=lambda p: p.area).exterior)
    mask = outer
    for h in holes:
        try:
            mask = mask.difference(Polygon(h).buffer(0))
        except Exception:  # noqa: BLE001
            pass
    return outer, mask


def _inset(poly, d):
    return poly if d <= 0 else poly.buffer(-d, join_style=2, mitre_limit=10.0)


def _dense(ring, step):
    P = np.asarray(ring.coords, float)
    out = []
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
        out.extend(a + (b - a) * (k / n) for k in range(n))
    return out


def _mesh(region, zfun, step):
    """Triangles over a polygonal region (holes allowed), z from ``zfun(x, y)``."""
    V, F = [], []
    for p in _polys(region):
        pts = {}
        for r in [p.exterior, *p.interiors]:
            for x, y in _dense(r, step):
                pts[(round(float(x), 5), round(float(y), 5))] = None
        keys = list(pts)
        if len(keys) < 3:
            continue
        grown = p.buffer(1e-6)
        index = {}
        for t in triangulate(MultiPoint(keys)):
            if not grown.contains(t.representative_point()):
                continue
            tri = []
            for x, y in list(t.exterior.coords)[:3]:
                k = (round(float(x), 5), round(float(y), 5))
                if k not in pts:
                    k = min(keys, key=lambda q: (q[0] - x) ** 2 + (q[1] - y) ** 2)
                if k not in index:
                    index[k] = len(V)
                    V.append([k[0], k[1], zfun(k[0], k[1])])
                tri.append(index[k])
            if len(set(tri)) == 3:
                F.append(tri)
    if not F:
        return EMPTY
    return np.asarray(V, float), np.asarray(F, int)


def _merge(parts):
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        if len(F):
            Vs.append(V)
            Fs.append(F + off)
            off += len(V)
    return (np.vstack(Vs), np.vstack(Fs)) if Vs else EMPTY


def hip_roof(footprint, pitch_deg: float, max_rise: float | None = None, eave_z: float = 0.0,
             course_m: float = 0.35, thickness_m: float = 0.0, max_courses: int = 60) -> dict:
    """{"courses": [(V, F), (V, F)] (alternating rows), "deck": (V, F) (flat top, if limited),
    "rise": height above the eave}. Empty meshes when the outline is degenerate."""
    out = {"courses": [EMPTY, EMPTY], "deck": EMPTY, "rise": 0.0}
    try:
        outer, mask = as_polygon(footprint)
    except Exception:  # noqa: BLE001
        return out
    if outer.is_empty or outer.area <= 0 or mask.is_empty:
        return out
    slope = math.tan(math.radians(pitch_deg))
    lo, hi = 0.0, math.sqrt(outer.area)            # how far in the planes go before they meet (ridge)
    for _ in range(24):
        mid = (lo + hi) / 2
        if _polys(_inset(outer, mid)):
            lo = mid
        else:
            hi = mid
    d_top, limited = lo, False
    if max_rise is not None and slope > 0 and max_rise / slope < d_top:
        d_top, limited = max_rise / slope, True
    if d_top <= 1e-6:
        return out
    n = int(min(max_courses, max(1, math.ceil(d_top / max(course_m, 0.05)))))
    c = d_top / n
    step = max(0.5, c * 1.5)
    rows = ([], [])
    prev = outer
    for k in range(n):
        nxt = _inset(outer, (k + 1) * c)
        band = prev.difference(nxt) if _polys(nxt) else prev
        band = band.intersection(mask)
        if _polys(band):
            ob, ib = prev.exterior, (nxt.boundary if _polys(nxt) else None)
            d0 = k * c

            def z(x, y, ob=ob, ib=ib, d0=d0):
                p = Point(x, y)
                do = ob.distance(p)
                di = ib.distance(p) if ib is not None else max(c - do, 0.0)
                f = do / (do + di) if do + di > 1e-9 else 0.0
                return eave_z + slope * (d0 + f * c) + thickness_m * (1.0 - f)
            rows[k % 2].append(_mesh(band, z, step))
        if not _polys(nxt):
            break
        prev = nxt if nxt.geom_type == "Polygon" else _MultiRing(nxt)   # an L / U plan splits into parts
    out["courses"] = [_merge(rows[0]), _merge(rows[1])]
    out["rise"] = d_top * slope
    if limited:
        top = _inset(outer, d_top)
        top = top.intersection(mask) if _polys(top) else None
        if top is not None and _polys(top):
            zt = eave_z + d_top * slope
            out["deck"] = _mesh(top, lambda x, y: zt, 1.5)
    return out


class _MultiRing:
    """A multi-part contour used like a polygon by the course loop (exterior = all its rings)."""

    def __init__(self, geom):
        self.geom = geom
        self.geom_type = "Polygon"
        self.exterior = geom.boundary

    def difference(self, other):
        return self.geom.difference(other)

    def intersection(self, other):
        return self.geom.intersection(other)


def roof_height_at(x: float, y: float, footprints, pitch_deg: float, max_rise: float | None, eave_z: float):
    """Height of the roof surface over (x, y); None outside the roof and in its open middle."""
    slope = math.tan(math.radians(pitch_deg))
    p = Point(x, y)
    best = None
    for fp in footprints:
        outer, mask = fp if isinstance(fp, tuple) else as_polygon(fp)
        if not mask.buffer(1e-6).contains(p):
            continue
        rise = slope * outer.exterior.distance(p)
        if max_rise is not None:
            rise = min(rise, max_rise)
        z = eave_z + rise
        best = z if best is None else max(best, z)
    return best
