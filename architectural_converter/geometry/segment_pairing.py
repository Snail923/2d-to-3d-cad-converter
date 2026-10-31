"""Parallel boundary pairing (shared by the parallel, polyline and hatch wall detectors).

For strokes i, j:
1. directions agree within a length-aware angular tolerance,
2. perpendicular separation lies in the thickness search range,
3. both endpoints of j have nearly the same offset from i (true parallelism),
4. longitudinal overlap is long enough relative to the separation,
5. the strip between them is not crossed lengthwise by another parallel stroke
   ("emptiness"): sub-intervals that contain an in-between line are removed and
   reported as *band obstructions* (typical for windows drawn inside the wall band).
Then 6. side-consistency: a boundary line can carry wall material on one side only, so a
greedy pass keeps the strongest non-conflicting pairs (resolves e.g. the gap between two
adjacent walls looking like a wall)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from .geometric_utils import angle_diff, dot, merge_intervals, normal, subtract_intervals


@dataclass
class PairPiece:
    i: int
    j: int
    u: tuple[float, float]
    c_i: float
    c_j: float
    s0: float
    s1: float
    angle_diff: float
    offset_spread: float
    len_i: float
    len_j: float
    score: float = 0.0
    side_conflict: bool = False
    evidence: dict = field(default_factory=dict)

    @property
    def thickness(self) -> float:
        return abs(self.c_i - self.c_j)

    @property
    def c(self) -> float:
        return (self.c_i + self.c_j) / 2

    @property
    def length(self) -> float:
        return self.s1 - self.s0


@dataclass
class Obstruction:
    i: int
    j: int
    u: tuple[float, float]
    c_lo: float
    c_hi: float
    s0: float
    s1: float
    blockers: list[int]


def _aligned(st, u):
    """Return (s0, s1, offsets at ends) of stroke ``st`` expressed in frame u."""
    p0, p1 = st.p0, st.p1
    n = normal(u)
    a, b = dot(p0, u), dot(p1, u)
    oa, ob = dot(p0, n), dot(p1, n)
    if a <= b:
        return a, b, oa, ob
    return b, a, ob, oa


def find_pairs(strokes, tol, cfg_parallel, tmin=None, tmax=None, check_emptiness=True):
    """Return (pieces, obstructions) for the given strokes."""
    tmin = tol.thickness_min if tmin is None else tmin
    tmax = tol.thickness_max if tmax is None else tmax
    if len(strokes) < 2:
        return [], []
    lines = [s.line() for s in strokes]
    tree = STRtree(lines)
    angs = [s.angle for s in strokes]
    lens = [s.length for s in strokes]
    min_ov_abs = cfg_parallel["min_overlap_abs_m"]
    min_ov_fac = cfg_parallel["min_overlap_factor_of_thickness"]
    raw_pieces: list[PairPiece] = []
    for i, si in enumerate(strokes):
        minx, miny, maxx, maxy = lines[i].bounds
        env = Polygon([(minx - tmax, miny - tmax), (maxx + tmax, miny - tmax), (maxx + tmax, maxy + tmax),
                       (minx - tmax, maxy + tmax)])
        u = si.u
        n = normal(u)
        a0, a1, _, _ = _aligned(si, u)
        ci = si.c
        for j in tree.query(env):
            j = int(j)
            if j <= i:
                continue
            sj = strokes[j]
            ad = angle_diff(angs[i], angs[j])
            if ad > tol.pair_angle_tol(lens[i], lens[j]):
                continue
            b0, b1, ob0, ob1 = _aligned(sj, u)
            d = (ob0 + ob1) / 2 - ci
            ad_ = abs(d)
            if ad_ < tmin or ad_ > tmax:
                continue
            spread = abs(ob0 - ob1)
            if spread > max(2 * tol.snap, tol.parallel_offset_frac * ad_):
                continue
            o0, o1 = max(a0, b0), min(a1, b1)
            ov = o1 - o0
            if ov < max(min_ov_abs, min_ov_fac * ad_):
                continue
            raw_pieces.append(PairPiece(i=i, j=j, u=u, c_i=ci, c_j=ci + d, s0=o0, s1=o1, angle_diff=ad,
                                        offset_spread=spread, len_i=lens[i], len_j=lens[j]))
    if not check_emptiness:
        return raw_pieces, []
    pieces: list[PairPiece] = []
    obstructions: list[Obstruction] = []
    for pc in raw_pieces:
        lo, hi = sorted((pc.c_i, pc.c_j))
        margin = max(tol.snap, 0.02 * (hi - lo))
        if hi - lo <= 2 * margin:
            pieces.append(pc)
            continue
        strip = Polygon([_fp(pc.u, pc.s0, lo + margin), _fp(pc.u, pc.s1, lo + margin),
                         _fp(pc.u, pc.s1, hi - margin), _fp(pc.u, pc.s0, hi - margin)])
        cuts = []
        blockers = []
        for k in tree.query(strip):
            k = int(k)
            if k in (pc.i, pc.j):
                continue
            sk = strokes[k]
            if angle_diff(angs[pc.i], angs[k]) > tol.pair_angle_tol(lens[pc.i], lens[k]):
                continue
            k0, k1, ok0, ok1 = _aligned(sk, pc.u)
            if not (lo + margin < ok0 < hi - margin and lo + margin < ok1 < hi - margin):
                continue
            # a line running straight through the whole wall (grid / axis line) is not an obstruction
            if k0 < pc.s0 - tol.snap and k1 > pc.s1 + tol.snap:
                continue
            c0, c1 = max(k0, pc.s0), min(k1, pc.s1)
            if c1 - c0 > tol.snap:
                cuts.append((c0, c1))
                blockers.append(k)
        if not cuts:
            pieces.append(pc)
            continue
        cuts = merge_intervals(cuts, gap=tol.snap)
        for c0, c1 in cuts:
            obstructions.append(Obstruction(i=pc.i, j=pc.j, u=pc.u, c_lo=lo, c_hi=hi, s0=c0, s1=c1,
                                            blockers=sorted(set(blockers))))
        thick = hi - lo
        for s0, s1 in subtract_intervals((pc.s0, pc.s1), cuts):
            if s1 - s0 >= max(min_ov_abs, min_ov_fac * thick):
                pieces.append(PairPiece(i=pc.i, j=pc.j, u=pc.u, c_i=pc.c_i, c_j=pc.c_j, s0=s0, s1=s1,
                                        angle_diff=pc.angle_diff, offset_spread=pc.offset_spread,
                                        len_i=pc.len_i, len_j=pc.len_j))
    return pieces, obstructions


def _fp(u, s, c):
    n = normal(u)
    return (s * u[0] + c * n[0], s * u[1] + c * n[1])


def thickness_clusters(values, weights, rel_tol, abs_tol, min_share):
    """1-D clustering of thickness values weighted by overlap length."""
    if len(values) == 0:
        return []
    order = np.argsort(values)
    v = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    groups = []
    cur = [0]
    for k in range(1, len(v)):
        ref = np.average(v[cur], weights=w[cur])
        if v[k] - ref <= max(abs_tol, rel_tol * ref):
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)
    total = float(w.sum())
    out = []
    for g in groups:
        ww = float(w[g].sum())
        out.append({"thickness": float(np.average(v[g], weights=w[g])), "weight": ww,
                    "share": ww / total if total else 0.0, "count": len(g),
                    "min": float(v[g].min()), "max": float(v[g].max())})
    out.sort(key=lambda c: -c["weight"])
    sig = [c for c in out if c["share"] >= min_share]
    top = sig[0]["weight"] if sig else 1.0
    for c in out:
        c["significant"] = c["share"] >= min_share
        c["relative_weight"] = c["weight"] / top if top else 0.0
    return out


def cluster_consistency(t, clusters, rel_tol, abs_tol) -> float:
    best = 0.0
    for c in clusters:
        if not c.get("significant"):
            continue
        if abs(t - c["thickness"]) <= max(abs_tol, rel_tol * c["thickness"]) * 1.5:
            best = max(best, math.sqrt(c["relative_weight"]))
    return best


def resolve_side_conflicts(pieces, strokes, snap):
    """Greedy: highest score first; a stroke cannot carry material on both sides over the
    same interval. Returns the accepted pieces; conflicting ones are flagged."""
    claims: dict[int, list[tuple[int, float, float]]] = {}
    accepted = []

    def side_interval(k, pc):
        st = strokes[k]
        sgn = 1.0 if dot(st.u, pc.u) >= 0 else -1.0
        c_self = pc.c_i if k == pc.i else pc.c_j
        c_other = pc.c_j if k == pc.i else pc.c_i
        side = int(np.sign((c_other - c_self) * sgn)) or 1
        a, b = sorted((sgn * pc.s0, sgn * pc.s1))
        return side, a, b

    for pc in sorted(pieces, key=lambda p: -p.score):
        conflict = False
        sides = []
        for k in (pc.i, pc.j):
            side, a, b = side_interval(k, pc)
            sides.append((k, side, a, b))
            for (sd, x0, x1) in claims.get(k, []):
                if sd != side and min(b, x1) - max(a, x0) > max(snap, 0.25 * (b - a)):
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            pc.side_conflict = True
            continue
        for k, side, a, b in sides:
            claims.setdefault(k, []).append((side, a, b))
        accepted.append(pc)
    return accepted
