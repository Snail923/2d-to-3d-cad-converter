"""Pitched (hip) roof over a storey outline, laid in courses of tiles.

Every side of the outline carries a roof plane rising inwards at the same pitch, so the planes
meet in hips, valleys and ridges exactly as a hip roof is drawn in plan (triangles over the short
sides, trapezia over the long ones): the straight skeleton of the outline.

    z(d) = eave + tan(pitch) * d,   d = how far the eave line has moved inwards

**Exact faces.** The surface is built by moving the eave line inwards in steps (every vertex moves
along its corner bisector, every side stays parallel to itself). Between two steps each side sweeps
a planar trapezium, so the mesh is made of the roof planes themselves: no triangle straddles a hip
or a valley. A step stops at every event (a side shrinking to nothing, an outline splitting in
two), where the outline is rebuilt from a fresh mitred inset and stepping goes on.

* **Holes** of the outline (an open court) are left open: the planes stop at the hole's edge.
* **Abutments** (``footprint["abutments"]``: a stair / lift core rising through the roof, found in
  Phase 1) are part of the roof body for the planes but are not tiled: they get a flat top at the
  roof's top height and vertical sides from the roof surface up to it.
* **Height limit.** Where the planes would rise above ``max_rise`` the roof is flat (a deck).
* **Tiles.** The surface is laid in courses: the band between two contour lines ``course_m`` apart
  is one row of tiles, standing ``thickness_m`` proud at its lower edge and lying flush at its
  upper edge, where the next row laps over it. Courses are returned in two alternating sets so
  they can be coloured as rows.
* **Dormers** (``footprint["dormers"]``: narrow notches of the drawn roof, Phase 1) are small
  gabled boxes standing on the eave line, their ridge running into the main roof.

``RoofSurface`` gives the surface height at a point (the same triangles that are drawn), used to
trim the walls of the attic room under the slope."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient
from shapely.prepared import prep
from shapely.strtree import STRtree

from ..geometry.geometric_utils import hole_free
from .mesh_utils import earclip

EMPTY = (np.zeros((0, 3)), np.zeros((0, 3), int))
MITRE = 100.0                     # mitre limit of the insets: corners stay sharp (straight skeleton)


def _polys(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g] if g.area > 0 else []
    return [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon" and p.area > 0]


def _union(polys):
    from shapely.ops import unary_union
    polys = [p for p in polys if p is not None and not p.is_empty]
    return unary_union(polys) if polys else Polygon()


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


def abutments_of(fp, outer=None):
    """The abutments of a footprint entry (cores rising through the roof) as one geometry."""
    rings = fp.get("abutments") if isinstance(fp, dict) else None
    if not rings:
        return Polygon()
    a = _union([Polygon(r).buffer(0) for r in rings if len(r) >= 3])
    return a.intersection(outer) if outer is not None else a


def _inset(poly, d):
    return poly if d <= 0 else poly.buffer(-d, join_style=2, mitre_limit=MITRE)


def ridge_depth(outer) -> float:
    """How far the eave line moves in before the planes meet (the ridge)."""
    lo, hi = 0.0, math.sqrt(max(outer.area, 1e-9))
    for _ in range(26):
        mid = (lo + hi) / 2
        if _polys(_inset(outer, mid)):
            lo = mid
        else:
            hi = mid
    return lo


# ----------------------------------------------------------------------------- straight skeleton
def _ring_frame(coords):
    """Vertices of a ring (interior on the left) and the mitre vector of every vertex: the vertex
    at inset d is P + d * M."""
    P = np.asarray(coords, float)
    if len(P) > 1 and np.allclose(P[0], P[-1]):
        P = P[:-1]
    keep = np.linalg.norm(P - np.roll(P, 1, 0), axis=1) > 1e-9
    P = P[keep]
    if len(P) < 3:
        return None, None
    E = np.roll(P, -1, 0) - P
    L = np.linalg.norm(E, axis=1)
    D = E / np.maximum(L, 1e-12)[:, None]
    N = np.column_stack([-D[:, 1], D[:, 0]])          # left normal = inwards
    Np = np.roll(N, 1, 0)
    den = 1.0 + np.einsum("ij,ij->i", Np, N)
    M = (Np + N) / np.maximum(den, 1e-6)[:, None]
    return P, M


def _shoelace(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def skeleton_quads(outer, d_end: float, breaks=()):
    """Planar pieces of the straight-skeleton roof between inset 0 and ``d_end``: a list of
    (4 x 2 corners, 4 insets). Every piece lies on the plane of one side of the outline. Steps
    also stop at ``breaks`` (the course lines of the tiles)."""
    quads = []
    outer = orient(outer, 1.0)
    breaks = sorted(b for b in breaks if 0 < b < d_end) + [d_end]
    depth, cur, guard = 0.0, outer, 0
    while depth < d_end - 1e-9 and guard < 4000:
        guard += 1
        frames = []
        for p in _polys(cur):
            P, M = _ring_frame(orient(p, 1.0).exterior.coords)   # an inset of a hole-free outline has no holes
            if P is not None:
                frames.append((P, M))
        if not frames:
            break
        s = next(b for b in breaks if b > depth + 1e-9) - depth
        for P, M in frames:                                       # the next side to shrink to nothing
            E = np.roll(P, -1, 0) - P
            L = np.linalg.norm(E, axis=1)
            rate = np.einsum("ij,ij->i", E / np.maximum(L, 1e-12)[:, None], np.roll(M, -1, 0) - M)
            t = np.where(rate < -1e-12, -L / np.where(rate < -1e-12, rate, -1.0), np.inf)
            s = min(s, float(t.min()))
        s = max(s, 0.0)
        ok = False
        for _ in range(40):                                       # an outline that splits: step up to the split
            if s <= 1e-6:
                break
            true_area = sum(p.area for p in _polys(_inset(outer, depth + s)))
            moved = sum(_shoelace(P + s * M) for P, M in frames)
            if abs(moved - true_area) <= 1e-5 + 1e-7 * outer.area:
                ok = True
                break
            s *= 0.5
        if ok:
            for P, M in frames:
                Q = P + s * M
                n = len(P)
                for i in range(n):
                    j = (i + 1) % n
                    quads.append((np.array([P[i], P[j], Q[j], Q[i]]), (depth, depth, depth + s, depth + s)))
            depth += s
        else:
            depth += 1e-4                                         # pass the split event itself
        cur = _inset(outer, depth) if depth < d_end - 1e-9 else Polygon()
    return quads


# ----------------------------------------------------------------------------- meshing helpers
def _plane(xy, z):
    """z = a x + b y + c through the corners of a planar piece."""
    A = np.column_stack([xy, np.ones(len(xy))])
    coef, *_ = np.linalg.lstsq(A, np.asarray(z, float), rcond=None)
    return coef


def _flat_mesh(region, zfun):
    """Triangles over a polygonal region (holes allowed), z from ``zfun(x, y)``."""
    V, F = [], []
    for piece in hole_free(region):
        piece = orient(piece.simplify(0), 1.0) if piece.simplify(0).geom_type == "Polygon" else orient(piece, 1.0)
        P = np.asarray(piece.exterior.coords, float)[:-1]
        if len(P) < 3:
            continue
        off = len(V)
        V.extend([x, y, zfun(x, y)] for x, y in P)
        F.extend([a + off, b + off, c + off] for a, b, c in earclip(P))
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


def _quads_mesh(items, mask, mask_p):
    """Mesh of planar pieces [(xy, z)], cut to ``mask``."""
    V, F = [], []
    for xy, z in items:
        poly = Polygon(xy)
        if poly.area < 1e-10:
            continue
        if mask_p.contains(poly):
            off = len(V)
            V.extend([float(x), float(y), float(zz)] for (x, y), zz in zip(xy, z))
            for tri in ((0, 1, 2), (0, 2, 3)):
                a, b, c = (xy[k] for k in tri)
                if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 1e-12:
                    F.append([off + k for k in tri])
            continue
        if not mask_p.intersects(poly):
            continue
        a, b, c = _plane(xy, z)
        sub = _flat_mesh(poly.intersection(mask), lambda x, y, a=a, b=b, c=c: a * x + b * y + c)
        if len(sub[1]):
            F.extend((sub[1] + len(V)).tolist())
            V.extend(sub[0].tolist())
    if not F:
        return EMPTY
    return np.asarray(V, float), np.asarray(F, int)


# ----------------------------------------------------------------------------- the roof surface
class RoofSurface:
    """Height of a hip roof at any point: the planar triangles of its straight skeleton (no tile
    thickness), flat above the height limit and over the abutments."""

    def __init__(self, footprint, pitch_deg: float, max_rise: float | None, eave_z: float, quads=None, d_top=None):
        self.outer, self.mask = as_polygon(footprint)
        self.abut = abutments_of(footprint, self.outer)
        self.slope = math.tan(math.radians(pitch_deg))
        self.eave = eave_z
        if d_top is None:
            d_top = ridge_depth(self.outer)
            if max_rise is not None and self.slope > 0 and max_rise / self.slope < d_top:
                d_top = max_rise / self.slope
        self.d_top = d_top
        self.z_top = eave_z + d_top * self.slope
        self.z_abut = eave_z + (max_rise if max_rise is not None else d_top * self.slope)
        if quads is None:
            quads = skeleton_quads(self.outer, d_top)
        self.tris = []
        for xy, d in quads:
            z = [eave_z + self.slope * x for x in d]
            for tri in ((0, 1, 2), (0, 2, 3)):
                t = xy[list(tri)]
                if abs((t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1]) - (t[1, 1] - t[0, 1]) * (t[2, 0] - t[0, 0])) > 1e-12:
                    self.tris.append((t, np.array([z[k] for k in tri])))
        self.tree = STRtree([Polygon(t) for t, _ in self.tris]) if self.tris else None
        top = _inset(self.outer, d_top - 1e-6)
        self.top = prep(top if _polys(top) else Polygon())
        self.mask_p = prep(self.mask.buffer(1e-6))
        self.abut_p = prep(self.abut.buffer(1e-6)) if not self.abut.is_empty else None

    def field(self, x: float, y: float) -> float | None:
        """Height of the roof planes over (x, y) inside the outline (abutments and holes included)."""
        p = Point(x, y)
        if not self.outer.buffer(1e-6).covers(p):
            return None
        if self.tree is not None:
            for i in self.tree.query(p):
                t, z = self.tris[int(i)]
                (x0, y0), (x1, y1), (x2, y2) = t
                det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                l0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / det
                l1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / det
                l2 = 1 - l0 - l1
                if min(l0, l1, l2) >= -1e-7:
                    return float(min(l0 * z[0] + l1 * z[1] + l2 * z[2], self.z_top))
        if self.top.covers(p) or self.tree is None:
            return float(self.z_top)          # the flat deck in the middle
        t, z = self.tris[int(self.tree.nearest(p))]   # a hairline crack between two pieces
        a, b, c = _plane(t, z)
        return float(min(max(a * x + b * y + c, self.eave), self.z_top))

    def height(self, x: float, y: float, near: float = 0.0) -> float | None:
        """Height of the roof over (x, y); the abutments' top over a core; None outside the roof and
        in its open holes. A point less than ``near`` outside the roof (the outer face of a wall
        standing on the eave line) gets the height at the nearest point of the roof."""
        p = Point(x, y)
        if self.abut_p is not None and self.abut_p.covers(p):
            return float(self.z_abut)
        if not self.mask_p.covers(p):
            if near <= 0 or self.mask.distance(p) > near:
                return None
            from shapely.ops import nearest_points
            q = nearest_points(self.mask, p)[0]
            c = self.mask.representative_point()
            x, y = q.x + (c.x - q.x) * 1e-6, q.y + (c.y - q.y) * 1e-6
            if self.abut_p is not None and self.abut_p.covers(Point(x, y)):
                return float(self.z_abut)
        z = self.field(x, y)
        return float(self.eave) if z is None else z


def hip_roof(footprint, pitch_deg: float, max_rise: float | None = None, eave_z: float = 0.0,
             course_m: float = 0.35, thickness_m: float = 0.0, max_courses: int = 60) -> dict:
    """{"courses": [(V, F), (V, F)] (alternating rows), "deck": (V, F) (flat top, if limited, and the
    tops of the abutments), "abutment_sides": (V, F), "rise": height above the eave, "surface":
    RoofSurface}. Empty meshes when the outline is degenerate."""
    out = {"courses": [EMPTY, EMPTY], "deck": EMPTY, "abutment_sides": EMPTY, "rise": 0.0, "surface": None}
    try:
        outer, mask = as_polygon(footprint)
    except Exception:  # noqa: BLE001
        return out
    if outer.is_empty or outer.area <= 0 or mask.is_empty:
        return out
    abut = abutments_of(footprint, outer)
    tiled = mask.difference(abut) if not abut.is_empty else mask
    slope = math.tan(math.radians(pitch_deg))
    d_top, limited = ridge_depth(outer), False
    if max_rise is not None and slope > 0 and max_rise / slope < d_top:
        d_top, limited = max_rise / slope, True
    if d_top <= 1e-6:
        return out
    n = int(min(max_courses, max(1, math.ceil(d_top / max(course_m, 0.05)))))
    c = d_top / n
    quads = skeleton_quads(outer, d_top, [k * c for k in range(1, n)])
    rows = ([], [])
    for xy, d in quads:
        k = min(n - 1, int((sum(d) / 4) // c))
        z = [eave_z + slope * x + thickness_m * (1.0 - (x - k * c) / c) for x in d]
        rows[k % 2].append((xy, z))
    tp = prep(tiled)
    out["courses"] = [_quads_mesh(rows[0], tiled, tp), _quads_mesh(rows[1], tiled, tp)]
    out["rise"] = d_top * slope
    surface = RoofSurface(footprint, pitch_deg, max_rise, eave_z, quads=quads, d_top=d_top)
    out["surface"] = surface
    decks = []
    if limited:
        top = _inset(outer, d_top)
        top = top.intersection(tiled) if _polys(top) else None
        if top is not None and _polys(top):
            zt = eave_z + d_top * slope
            decks.append(_flat_mesh(top, lambda x, y: zt))
    if not abut.is_empty:
        za = surface.z_abut
        decks.append(_flat_mesh(abut.intersection(mask), lambda x, y: za))
        out["abutment_sides"] = _abutment_sides(abut.intersection(mask), surface, za)
    out["deck"] = _merge(decks)
    return out


def _abutment_sides(abut, surface, z_top, step: float = 0.5):
    """Vertical faces around an abutment from the roof surface up to its flat top."""
    V, F = [], []
    for p in _polys(abut):
        for ring in [p.exterior, *p.interiors]:
            P = np.asarray(ring.coords, float)
            for a, b in zip(P[:-1], P[1:]):
                k = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
                for i in range(k):
                    q0, q1 = a + (b - a) * (i / k), a + (b - a) * ((i + 1) / k)
                    z0, z1 = surface.field(*q0), surface.field(*q1)
                    z0 = surface.eave if z0 is None else z0
                    z1 = surface.eave if z1 is None else z1
                    if z0 >= z_top - 1e-3 and z1 >= z_top - 1e-3:
                        continue
                    off = len(V)
                    V.extend([[q0[0], q0[1], z0], [q1[0], q1[1], z1], [q1[0], q1[1], z_top], [q0[0], q0[1], z_top]])
                    F.extend([[off, off + 1, off + 2], [off, off + 2, off + 3]])
    if not F:
        return EMPTY
    return np.asarray(V, float), np.asarray(F, int)


def dormer_meshes(dormer, surface, eave_z: float, pitch_deg: float = 45.0, min_rise: float = 0.4):
    """A gabled dormer standing in a notch of the drawn roof: a front wall on the eave line, two
    cheeks and a gable roof whose ridge runs level from the front into the main roof (the pointed
    shape a dormer has in plan). Returns ((V, F) walls, (V, F) roof) or None when it would be too
    low to stand out of the roof."""
    a, b = (np.asarray(p, float) for p in dormer["mouth"])
    tip = np.asarray(dormer["tip"], float)
    m = (a + b) / 2
    w = float(np.linalg.norm(b - a)) / 2
    u = tip - m
    L = float(np.linalg.norm(u))
    if w < 0.1 or L < 0.2:
        return None
    u /= L
    v = (b - a) / (2 * w)
    zr = surface.field(*tip)
    if zr is None or zr - eave_z < min_rise:
        return None
    ze = max(eave_z + 0.1, zr - w * math.tan(math.radians(pitch_deg)))
    # cross-section (across, up) of the dormer, extruded along u from the eave line to the tip
    sec = [(-w, 0.0), (w, 0.0), (w, ze), (0.0, zr), (-w, ze)]
    P = lambda s, t, z: [float(m[0] + v[0] * s + u[0] * t), float(m[1] + v[1] * s + u[1] * t), float(z)]  # noqa: E731
    front = [P(s, 0.0, z) for s, z in sec]
    back = [P(s, L, z) for s, z in sec]
    Vw = front + back
    Fw = [[0, 1, 2], [0, 2, 4], [2, 3, 4],                  # front: wall and gable
          [5, 7, 6], [5, 9, 7], [7, 9, 8],                  # back (inside the main roof)
          [1, 6, 7], [1, 7, 2], [0, 4, 9], [0, 9, 5]]       # cheeks
    Vr = [front[2], front[3], back[3], back[2], front[4], front[3], back[3], back[4]]
    Fr = [[0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7]]
    return (np.asarray(Vw, float), np.asarray(Fw, int)), (np.asarray(Vr, float), np.asarray(Fr, int))


_SURFACES: dict = {}


def roof_height_at(x: float, y: float, footprints, pitch_deg: float, max_rise: float | None, eave_z: float):
    """Height of the roof surface over (x, y); None outside the roof and in its open middle."""
    best = None
    for fp in footprints:
        key = (repr(fp), pitch_deg, max_rise, eave_z)
        s = _SURFACES.get(key)
        if s is None:
            if len(_SURFACES) > 64:
                _SURFACES.clear()
            s = _SURFACES[key] = RoofSurface(fp, pitch_deg, max_rise, eave_z)
        z = s.height(x, y)
        if z is not None:
            best = z if best is None else max(best, z)
    return best
