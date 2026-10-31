"""Adapters from normalised GeoEntities to detection primitives.

* ``Segment``  - one straight piece of source geometry (LINE, polyline part, MLINE line)
* ``Stroke``   - maximal collinear chain of segments (same layer & role) used by detectors;
                 keeps member segment ids for traceability (fragmentation is *identified*
                 here, not hidden: each stroke records how many fragments it joined)
* ``ArcPrim``  - circular arc (ARC entity or polyline bulge part)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree

from ..models.drawing import Drawing
from .geometric_utils import angle_diff, canonical_dir, dir_angle, normal


@dataclass
class Segment:
    id: str
    geo_id: str
    raw_id: str
    layer: str
    p0: np.ndarray
    p1: np.ndarray
    origin: str            # line | polyline | mline | hatch
    role: str = "linework"
    dup_of: str | None = None
    flags: set[str] = field(default_factory=set)

    @property
    def length(self) -> float:
        return float(np.hypot(*(self.p1 - self.p0)))

    @property
    def u(self) -> tuple[float, float]:
        d = self.p1 - self.p0
        return canonical_dir(float(d[0]), float(d[1]))

    def line(self) -> LineString:
        return LineString([tuple(self.p0), tuple(self.p1)])


@dataclass
class Stroke:
    id: str
    u: tuple[float, float]
    c: float
    s0: float
    s1: float
    layer: str
    role: str
    origin: str
    members: list[str]
    raw_ids: set[str]
    layers: set[str] = field(default_factory=set)
    member_ranges: list[tuple[str, str, float, float]] = field(default_factory=list)

    def members_in(self, a: float, b: float, tol: float = 1e-6) -> list[tuple[str, str]]:
        """(segment id, raw id) of members overlapping the interval [a, b] (stroke frame)."""
        return [(sid, rid) for sid, rid, x0, x1 in self.member_ranges if min(b, x1) - max(a, x0) > tol]

    @property
    def length(self) -> float:
        return self.s1 - self.s0

    @property
    def angle(self) -> float:
        return dir_angle(self.u)

    @property
    def n(self) -> tuple[float, float]:
        return normal(self.u)

    def point(self, s: float) -> tuple[float, float]:
        n = self.n
        return (s * self.u[0] + self.c * n[0], s * self.u[1] + self.c * n[1])

    @property
    def p0(self):
        return self.point(self.s0)

    @property
    def p1(self):
        return self.point(self.s1)

    def line(self) -> LineString:
        return LineString([self.p0, self.p1])


@dataclass
class ArcPrim:
    id: str
    geo_id: str
    raw_id: str
    layer: str
    center: tuple[float, float]
    radius: float
    start: float
    end: float
    sweep: float
    role: str = "linework"
    origin: str = "arc"
    block_path: tuple = ()

    def point_at(self, a: float):
        return (self.center[0] + self.radius * math.cos(a), self.center[1] + self.radius * math.sin(a))

    @property
    def p_start(self):
        return self.point_at(self.start)

    @property
    def p_end(self):
        return self.point_at(self.end)


def build_segments(d: Drawing, include_hatch: bool = False) -> list[Segment]:
    geo = d.geo_by_id()
    segs: list[Segment] = []
    n = 0
    for g in d.entities:
        if not g.valid or g.kind != "line":
            continue
        parent = geo.get(g.parent) if g.parent else None
        origin = "line"
        if parent is not None:
            origin = "mline" if parent.kind == "mline" else "polyline"
        n += 1
        segs.append(Segment(id=f"s_{n:06d}", geo_id=g.id, raw_id=g.raw_id, layer=g.layer,
                            p0=np.array(g.points[0], dtype=float), p1=np.array(g.points[1], dtype=float),
                            origin=origin, role=g.role))
    return segs


def hatch_segments(d: Drawing) -> list[Segment]:
    segs: list[Segment] = []
    n = 0
    for g in d.entities:
        if not g.valid or g.kind not in ("hatch", "polygon"):
            continue
        loops = g.data.get("loops") if g.kind == "hatch" else [g.points]
        for lp in loops:
            pts = np.asarray(lp)
            if len(pts) < 3:
                continue
            closed = pts if np.allclose(pts[0], pts[-1]) else np.vstack([pts, pts[:1]])
            for a, b in zip(closed[:-1], closed[1:]):
                if np.hypot(*(b - a)) <= 0:
                    continue
                n += 1
                segs.append(Segment(id=f"h_{n:06d}", geo_id=g.id, raw_id=g.raw_id, layer=g.layer,
                                    p0=np.array(a), p1=np.array(b), origin="hatch", role=g.role))
    return segs


def build_arcs(d: Drawing) -> list[ArcPrim]:
    out = []
    n = 0
    for g in d.entities:
        if not g.valid or g.kind != "arc":
            continue
        n += 1
        out.append(ArcPrim(id=f"a_{n:06d}", geo_id=g.id, raw_id=g.raw_id, layer=g.layer,
                           center=tuple(g.data["center"]), radius=float(g.data["radius"]),
                           start=float(g.data["start"]), end=float(g.data["end"]),
                           sweep=float(g.data["sweep"]), role=g.role, block_path=g.block_path))
    return out


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build_strokes(segments: list[Segment], snap: float, angle_tol: float, prefix: str = "k",
                  same_layer: bool = True) -> list[Stroke]:
    """Merge collinear, touching/overlapping segments (same layer & role) into strokes."""
    segs = [s for s in segments if s.dup_of is None and s.length > 0]
    if not segs:
        return []
    lines = [s.line() for s in segs]
    tree = STRtree(lines)
    uf = _UF(len(segs))
    us = [s.u for s in segs]
    angs = [dir_angle(u) for u in us]
    off_tol = max(snap * 0.6, 1e-6)
    for i, s in enumerate(segs):
        minx, miny, maxx, maxy = lines[i].bounds
        cand = tree.query(LineString([(minx - snap, miny - snap), (maxx + snap, maxy + snap)]).envelope)
        u = us[i]
        n = normal(u)
        for j in cand:
            j = int(j)
            if j <= i:
                continue
            t = segs[j]
            if same_layer and (t.layer != s.layer):
                continue
            if t.role != s.role or t.origin == "hatch" and s.origin != "hatch":
                continue
            if angle_diff(angs[i], angs[j]) > angle_tol:
                continue
            o0 = float(np.dot(t.p0 - s.p0, n))
            o1 = float(np.dot(t.p1 - s.p0, n))
            if abs(o0) > off_tol or abs(o1) > off_tol:
                continue
            a0, a1 = sorted((float(np.dot(s.p0, u)), float(np.dot(s.p1, u))))
            b0, b1 = sorted((float(np.dot(t.p0, u)), float(np.dot(t.p1, u))))
            gap = max(b0 - a1, a0 - b1)
            if gap <= snap:
                uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(len(segs)):
        groups.setdefault(uf.find(i), []).append(i)
    strokes: list[Stroke] = []
    for k, idx in enumerate(sorted(groups.values(), key=lambda g: g[0])):
        ref_u = us[idx[0]]
        vx = vy = 0.0
        for i in idx:
            u = us[i]
            sgn = 1.0 if (u[0] * ref_u[0] + u[1] * ref_u[1]) >= 0 else -1.0
            L = segs[i].length
            vx += sgn * u[0] * L
            vy += sgn * u[1] * L
        u = canonical_dir(vx, vy)
        n = normal(u)
        wsum = sum(segs[i].length for i in idx)
        c = sum(segs[i].length * float(np.dot((segs[i].p0 + segs[i].p1) / 2, n)) for i in idx) / wsum
        ss = [float(np.dot(p, u)) for i in idx for p in (segs[i].p0, segs[i].p1)]
        first = segs[idx[0]]
        ranges = []
        for i in idx:
            a, b = sorted((float(np.dot(segs[i].p0, u)), float(np.dot(segs[i].p1, u))))
            ranges.append((segs[i].id, segs[i].raw_id, a, b))
        strokes.append(Stroke(id=f"{prefix}_{k + 1:06d}", u=u, c=c, s0=min(ss), s1=max(ss), layer=first.layer,
                              role=first.role, origin=first.origin, members=[segs[i].id for i in idx],
                              raw_ids={segs[i].raw_id for i in idx}, layers={segs[i].layer for i in idx},
                              member_ranges=ranges))
    return strokes
