"""Plan specifications with exact ground truth (metres, world frame)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from ..geometry.geometric_utils import polygons_of
from ..models.wall import arc_points


@dataclass
class WallSpec:
    p0: tuple[float, float]
    p1: tuple[float, float]
    t: float
    arc: tuple[float, float, float, float, float] | None = None  # cx, cy, r, a0_deg, a1_deg (ccw)

    @property
    def length(self):
        if self.arc:
            cx, cy, r, a0, a1 = self.arc
            return r * math.radians((a1 - a0) % 360)
        return math.dist(self.p0, self.p1)

    @property
    def u(self):
        d = np.subtract(self.p1, self.p0)
        return d / np.hypot(*d)

    @property
    def n(self):
        u = self.u
        return np.array([-u[1], u[0]])

    def axis(self) -> LineString:
        if self.arc:
            cx, cy, r, a0, a1 = self.arc
            return LineString(arc_points(cx, cy, r, math.radians(a0), math.radians(a1), 1.0))
        return LineString([self.p0, self.p1])


@dataclass
class OpeningSpec:
    wall: int
    at: float                  # distance from wall.p0 to opening centre along the wall
    width: float
    kind: str = "door"         # door | double_door | sliding | window | opening
    swing: int = 1             # +1 swing to +n side, -1 to -n side
    hinge: str = "start"       # start | end

    def interval(self):
        return (self.at - self.width / 2, self.at + self.width / 2)


@dataclass
class PlanSpec:
    name: str
    description: str
    walls: list[WallSpec] = field(default_factory=list)
    openings: list[OpeningSpec] = field(default_factory=list)
    labels: list[tuple[str, tuple[float, float]]] = field(default_factory=list)
    columns: list[tuple[float, float, float, float, str]] = field(default_factory=list)  # x, y, w, h, shape
    furniture: list[tuple[float, float, float, float, float]] = field(default_factory=list)  # x, y, w, h, angle
    grid: bool = False

    # ---------------------------------------------------------------- builders
    def chain(self, pts, t, closed=False):
        idx = []
        seq = list(pts) + ([pts[0]] if closed else [])
        for a, b in zip(seq[:-1], seq[1:]):
            self.walls.append(WallSpec(tuple(map(float, a)), tuple(map(float, b)), t))
            idx.append(len(self.walls) - 1)
        return idx

    def wall(self, a, b, t):
        self.walls.append(WallSpec(tuple(map(float, a)), tuple(map(float, b)), t))
        return len(self.walls) - 1

    def arc_wall(self, cx, cy, r, a0, a1, t):
        p0 = (cx + r * math.cos(math.radians(a0)), cy + r * math.sin(math.radians(a0)))
        p1 = (cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1)))
        self.walls.append(WallSpec(p0, p1, t, arc=(cx, cy, r, a0, a1)))
        return len(self.walls) - 1

    def opening_at(self, point, width, kind="door", swing=1, hinge="start"):
        """Place an opening on the straight wall whose axis is nearest to ``point``."""
        best = None
        for k, w in enumerate(self.walls):
            if w.arc:
                continue
            d = w.axis().distance(Point(point))
            if best is None or d < best[0]:
                best = (d, k)
        k = best[1]
        w = self.walls[k]
        at = float(np.dot(np.subtract(point, w.p0), w.u))
        self.openings.append(OpeningSpec(k, at, width, kind, swing, hinge))
        return len(self.openings) - 1

    def label(self, text, pt):
        self.labels.append((text, (float(pt[0]), float(pt[1]))))

    # ---------------------------------------------------------------- geometry
    # junction handling: each wall end is extended into the walls it meets, and the
    # extension is clipped to those walls, so corners are filled without spikes.
    def _dir_at(self, w, pt):
        if w.arc:
            cx, cy, r, a0, a1 = w.arc
            a = math.atan2(pt[1] - cy, pt[0] - cx)
            return np.array([-math.sin(a), math.cos(a)])
        return w.u

    def _end_info(self, k, end):
        w = self.walls[k]
        if w.arc:
            cx, cy, r, a0, a1 = w.arc
            a = math.radians(a1 if end else a0)
            pt = np.array([cx + r * math.cos(a), cy + r * math.sin(a)])
            tan = np.array([-math.sin(a), math.cos(a)])
            out = tan if end else -tan
        else:
            pt = np.array(w.p1 if end else w.p0)
            out = w.u if end else -w.u
        partners, ext = [], 0.0
        for j, o in enumerate(self.walls):
            if j == k or o.axis().distance(Point(pt)) > 1e-6:
                continue
            partners.append(j)
            d = self._dir_at(o, pt)
            ang = abs(float(np.dot(d, out)))
            if ang > 0.999:
                continue
            ext = max(ext, o.t / 2 / math.sqrt(max(1e-9, 1 - ang * ang)))
        return pt, out, ext, partners

    def _base_polygon(self, k):
        w = self.walls[k]
        if w.arc:
            cx, cy, r, a0, a1 = w.arc
            outer = arc_points(cx, cy, r + w.t / 2, math.radians(a0), math.radians(a1), 1.0)
            inner = arc_points(cx, cy, r - w.t / 2, math.radians(a0), math.radians(a1), 1.0)
            return Polygon(outer + inner[::-1])
        u, n = w.u, w.n
        a, b = np.array(w.p0), np.array(w.p1)
        h = w.t / 2
        return Polygon([tuple(a - n * h), tuple(b - n * h), tuple(b + n * h), tuple(a + n * h)])

    def _ext_polygon(self, k, end):
        w = self.walls[k]
        pt, out, ext, partners = self._end_info(k, end)
        if ext <= 0:
            return None, partners
        n = np.array([-out[1], out[0]])
        h = w.t / 2
        q = pt + out * ext
        return Polygon([tuple(pt - n * h), tuple(q - n * h), tuple(q + n * h), tuple(pt + n * h)]), partners

    def wall_polygon(self, k) -> Polygon:
        if not hasattr(self, "_poly_cache") or self._poly_cache[0] != len(self.walls):
            base = [self._base_polygon(j) for j in range(len(self.walls))]
            exts = [[self._ext_polygon(j, e) for e in (0, 1)] for j in range(len(self.walls))]
            final = []
            for j in range(len(self.walls)):
                poly = base[j]
                for e in (0, 1):
                    ep, partners = exts[j][e]
                    if ep is None or not partners:
                        continue
                    clip = unary_union([base[o] for o in partners] +
                                       [x[0] for o in partners for x in exts[o] if x[0] is not None])
                    poly = poly.union(ep.intersection(clip))
                final.append(poly)
            self._poly_cache = (len(self.walls), final)
        return self._poly_cache[1][k]

    def opening_polygon(self, o: OpeningSpec, margin=0.002) -> Polygon:
        w = self.walls[o.wall]
        u, n = w.u, w.n
        s0, s1 = o.interval()
        h = w.t / 2 + margin
        base = np.array(w.p0)
        pts = [base + u * s0 - n * h, base + u * s1 - n * h, base + u * s1 + n * h, base + u * s0 + n * h]
        return Polygon([tuple(p) for p in pts])

    def solid(self, with_openings=True):
        walls = unary_union([self.wall_polygon(k) for k in range(len(self.walls))])
        if with_openings and self.openings:
            walls = walls.difference(unary_union([self.opening_polygon(o) for o in self.openings]))
        return walls

    def column_polygons(self):
        out = []
        for x, y, w, h, shape in self.columns:
            if shape == "circle":
                out.append(Point(x, y).buffer(w / 2, 32))
            else:
                out.append(Polygon([(x - w / 2, y - h / 2), (x + w / 2, y - h / 2), (x + w / 2, y + h / 2),
                                    (x - w / 2, y + h / 2)]))
        return out

    def reference(self) -> dict:
        solid_closed = self.solid(with_openings=False)
        cols = self.column_polygons()
        col_union = unary_union(cols) if cols else None
        rooms = []
        for poly in polygons_of(solid_closed):
            for ring in poly.interiors:
                rp = Polygon(ring)
                if rp.area < 1.0:
                    continue
                if col_union is not None:
                    rp = rp.difference(col_union)
                name = None
                for txt, pt in self.labels:
                    if rp.contains(Point(pt)):
                        name = txt.split("\n")[0]
                        break
                rooms.append({"polygon": [list(c) for c in rp.exterior.coords] if isinstance(rp, Polygon) else
                              [list(c) for c in max(polygons_of(rp), key=lambda g: g.area).exterior.coords],
                              "area": rp.area, "name": name})
        walls = []
        for w in self.walls:
            d = {"start": list(w.p0), "end": list(w.p1), "thickness": w.t, "kind": "arc" if w.arc else "straight"}
            if w.arc:
                d["arc"] = list(w.arc)
                d["axis"] = [list(c) for c in w.axis().coords]
            walls.append(d)
        doors, windows, openings = [], [], []
        for o in self.openings:
            w = self.walls[o.wall]
            s0, s1 = o.interval()
            p0 = np.array(w.p0) + w.u * s0
            p1 = np.array(w.p0) + w.u * s1
            rec = {"center": list((p0 + p1) / 2), "start": list(p0), "end": list(p1), "width": o.width,
                   "kind": o.kind, "wall": o.wall}
            if o.kind == "window":
                windows.append(rec)
            elif o.kind == "opening":
                openings.append(rec)
            else:
                doors.append(rec)
        columns = [{"center": [x, y], "size": [w, h], "shape": s} for x, y, w, h, s in self.columns]
        return {"name": self.name, "description": self.description, "walls": walls, "rooms": rooms,
                "doors": doors, "windows": windows, "openings": openings, "columns": columns}
