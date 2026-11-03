"""Stair flight detection (deterministic).

A straight flight is drawn as a run of parallel tread (nosing) lines of similar length whose
spacing - the *going* - is regular and in the range of real stairs. Geometry alone is not
proof (shelving, louvres or exploded hatches can look alike), so the candidate is scored on:

* ``regularity``  - how constant the spacing and tread lengths are,
* ``tread_count`` - more treads, less chance of coincidence,
* ``layer_match`` - tread lines on a layer whose name says stair,
* ``stair_text``  - an "UP" / "DN" marker or a "stair" room label on or next to the flight,
* ``stringers``   - lines along both ends of the treads (the stair sides / walls).

A geometry-only candidate is capped by the stair gate so it lands in NEEDS_REVIEW.

Accepted stairs also *veto wall candidates* built from their tread lines: two treads 0.3 m
apart look exactly like the two faces of a 0.3 m wall."""
from __future__ import annotations

import math
import re

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree

from ..geometry.geometric_utils import bbox_pad
from ..models.base_entity import Status
from ..models.stair import Stair


def _canon(u):
    ux, uy = float(u[0]), float(u[1])
    if ux < -1e-12 or (abs(ux) <= 1e-12 and uy < 0):
        ux, uy = -ux, -uy
    return ux, uy


def _chains(c, s0, s1, sc):
    """Successor links between treads of one orientation group, then maximal regular runs."""
    n = len(c)
    order = np.argsort(c)
    c, s0, s1 = c[order], s0[order], s1[order]
    L = s1 - s0
    g_lo, g_hi = sc["going_m"]
    succ = -np.ones(n, int)
    for i in range(n):
        a = np.searchsorted(c, c[i] + g_lo * 0.999, "left")
        b = np.searchsorted(c, c[i] + g_hi * 1.001, "right")
        if a >= b:
            continue
        js = np.arange(a, b)
        ov = np.minimum(s1[js], s1[i]) - np.maximum(s0[js], s0[i])
        ok = (ov >= sc["min_overlap"] * np.maximum(L[js], L[i])) & \
             (np.abs(L[js] - L[i]) <= sc["max_length_deviation"] * np.maximum(L[js], L[i]))
        js = js[ok]
        if len(js):
            succ[i] = js[np.argmin(c[js])]
    pred_count = np.bincount(succ[succ >= 0], minlength=n)
    has_pred = np.zeros(n, bool)
    has_pred[succ[succ >= 0]] = True
    runs = []
    for i in range(n):
        if has_pred[i] and pred_count[i] == 1:
            continue                                 # not a chain start
        chain = [i]
        k = i
        while succ[k] >= 0 and len(chain) < 400:
            k = succ[k]
            chain.append(k)
            if pred_count[k] > 1:
                break
        # split where the spacing jumps (landing, two flights in line)
        gaps = np.diff(c[chain])
        cur = [chain[0]]
        for j in range(1, len(chain)):
            if len(cur) >= 2:
                ref = float(np.median(np.diff(c[cur])))
                if abs(gaps[j - 1] - ref) > sc["max_gap_deviation"] * ref:
                    runs.append(cur)
                    cur = []
            cur.append(chain[j])
        runs.append(cur)
    return [[int(order[k]) for k in r] for r in runs if len(r) >= sc["min_treads"]]


def detect_stairs(ctx, cfg, ids, labels=()):
    sc = cfg["detection"]["stair"]
    strokes = ctx.strokes
    lo, hi = sc["tread_length_m"]
    cand = [k for k, st in enumerate(strokes) if lo <= st.length <= hi]
    if len(cand) < sc["min_treads"]:
        return [], {"candidates": len(cand), "flights": 0}
    ang = np.array([math.atan2(*_canon(strokes[k].u)[::-1]) % math.pi for k in cand])
    order = np.argsort(ang)
    tol = math.radians(sc["angle_tol_deg"])
    groups, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if ang[b] - ang[a] > tol:
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)
    if len(groups) > 1 and ang[groups[0][0]] + math.pi - ang[groups[-1][-1]] <= tol:   # wrap around 0/180 deg
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    txt_re = [re.compile(p, re.I) for p in sc["text_patterns"]]
    marks = [lb for lb in labels if any(r.match(lb.text.strip()) for r in txt_re) or lb.room_type == "stair"]
    mark_pts = [Point(lb.position) for lb in marks]
    mark_tree = STRtree(mark_pts) if mark_pts else None
    arrows = _ArrowIndex(ctx, sc)
    stairs = []
    for grp in groups:
        if len(grp) < sc["min_treads"]:
            continue
        ks = [cand[g] for g in grp]
        u = np.mean([_canon(strokes[k].u) for k in ks], axis=0)
        u /= np.linalg.norm(u)
        nrm = np.array([-u[1], u[0]])
        P0 = np.array([strokes[k].point(strokes[k].s0) for k in ks])
        P1 = np.array([strokes[k].point(strokes[k].s1) for k in ks])
        c = (P0 @ nrm + P1 @ nrm) / 2
        a0, a1 = P0 @ u, P1 @ u
        s0, s1 = np.minimum(a0, a1), np.maximum(a0, a1)
        for run in _chains(c, s0, s1, sc):
            cs = c[run]
            o = np.argsort(cs)
            run = [run[i] for i in o]
            cs = cs[o]
            gaps = np.diff(cs)
            going = float(np.median(gaps))
            lens = s1[run] - s0[run]
            width = float(np.median(lens))
            sa, sb = float(np.median(s0[run])), float(np.median(s1[run]))
            st = Stair(id=ids.new("stair"))
            st.u = (float(u[0]), float(u[1]))
            st.run = (float(nrm[0]), float(nrm[1]))
            base = float(cs[0])
            st.origin = (float(u[0] * sa + nrm[0] * base), float(u[1] * sa + nrm[1] * base))
            st.tread_offsets = [float(x - base) for x in cs]
            st.tread_span = (0.0, sb - sa)
            st.width, st.going = width, going
            corners = [(sa, base - going), (sb, base - going), (sb, cs[-1]), (sa, cs[-1])]
            st.polygon = Polygon([tuple(u * x + nrm * y) for x, y in corners])
            tk = [ks[i] for i in run]
            st.stroke_ids = [strokes[k].id for k in tk]
            for k in tk:
                st.source_entities |= strokes[k].raw_ids
            st.detection_methods.add("stair_detector")
            dev = max(float(np.max(np.abs(gaps - going))) / going, float(np.max(np.abs(lens - width))) / width)
            st.add_evidence("regularity", 1.0 - min(1.0, dev / max(sc["max_gap_deviation"], 1e-6)) * 0.5)
            st.add_evidence("tread_count", min(1.0, (len(run) - 2) / 8))
            lay = np.mean([max(ctx.layer_score(ly, "stair") for ly in strokes[k].layers) if strokes[k].layers else 0.0
                           for k in tk])
            if lay > 0:
                st.add_evidence("layer_match", float(lay))
            # stair marker text on / next to the flight
            if mark_tree is not None:
                zone = st.polygon.buffer(max(going * 3, 0.5))
                hits = [int(i) for i in mark_tree.query(zone, predicate="intersects")]
                if hits:
                    st.add_evidence("stair_text", 1.0)
                    near = [marks[i] for i in hits if txt_re[0].match(marks[i].text.strip())]
                    if near:
                        p = np.array(near[0].position) @ nrm
                        st.direction_known = True
                        if abs(p - cs[-1]) < abs(p - cs[0]):     # marker at the far end: flight starts there
                            _flip(st, nrm, u, sa, cs)
                    elif hits:
                        st.notes.append("walking direction unknown (no UP/DN marker)")
            if not st.direction_known:
                vote = _arrow_vote(arrows, st, u, nrm, sa, sb, cs)
                if vote:
                    st.direction_known = True
                    st.direction_source = "arrow"
                    st.add_evidence("stair_text", max(st.evidence.get("stair_text", 0.0), 0.8))
                    if vote < 0:                   # the arrow points up the flight towards -nrm
                        _flip(st, nrm, u, sa, cs)
            elif st.direction_source is None:
                st.direction_source = "text"
            st.add_evidence("stringers", _stringer_support(ctx, st, tk, u, nrm, sa, sb, cs))
            stairs.append(st)
    w = cfg["detection"]["weights"]["stair"]
    th = cfg["detection"]["status_thresholds"]
    gate = cfg["detection"].get("gates", {}).get("stair")
    for st in stairs:
        st.score(w, th, gate)
    return stairs, {"candidates": len(cand), "flights": len(stairs),
                    "accepted": sum(1 for s in stairs if s.status == Status.ACCEPTED)}


def _flip(st, nrm, u, sa, cs):
    top = float(cs[-1])
    st.run = (float(-nrm[0]), float(-nrm[1]))
    st.origin = (float(u[0] * sa + nrm[0] * top), float(u[1] * sa + nrm[1] * top))
    st.tread_offsets = [float(top - x) for x in cs[::-1]]
    g = st.going
    corners = [(sa, top + g), (sa + st.tread_span[1], top + g), (sa + st.tread_span[1], cs[0]), (sa, cs[0])]
    st.polygon = Polygon([tuple(u * x + nrm * y) for x, y in corners])


def _stringer_support(ctx, st, tread_ks, u, nrm, sa, sb, cs):
    """Share of the flight length bounded by lines along both tread ends."""
    if ctx.stroke_tree is None:
        return 0.0
    tset = set(tread_ks)
    span = float(cs[-1] - cs[0])
    if span <= 0:
        return 0.0
    tol = max(ctx.tol.snap * 4, 0.05) + st.going * 0.5
    scores = []
    for s_end in (sa, sb):
        p0 = u * s_end + nrm * cs[0]
        p1 = u * s_end + nrm * cs[-1]
        probe = LineString([tuple(p0), tuple(p1)])
        cover = 0.0
        for k in ctx.stroke_tree.query(bbox_pad(probe, tol)):
            k = int(k)
            if k in tset:
                continue
            s = ctx.strokes[k]
            if abs(abs(float(np.dot(s.u, nrm))) - 1.0) > 0.01:
                continue
            pa, pb = np.array(s.point(s.s0)), np.array(s.point(s.s1))
            if abs(float(pa @ u) - s_end) > tol:
                continue
            x0, x1 = sorted((float(pa @ nrm), float(pb @ nrm)))
            cover = max(cover, max(0.0, min(x1, cs[-1]) - max(x0, cs[0])))
        scores.append(min(1.0, cover / span))
    return float(np.mean(scores))          # a flight may be open (balustrade) on one side


def veto_tread_walls(stairs, walls, cfg):
    """Reject wall candidates made of a stair's tread lines (pairs of treads look like walls)."""
    share = cfg["detection"]["stair"]["wall_veto_overlap"]
    live = [s for s in stairs if s.status != Status.REJECTED]
    n = 0
    for s in live:
        zone = s.polygon.buffer(s.going * 0.5)
        for w in walls:
            if w.status == Status.REJECTED or not w.source_entities:
                continue
            inside = len(w.source_entities & s.source_entities) / len(w.source_entities)
            if inside < share:
                continue
            ax = w.axis
            if ax is not None and ax.length > 0 and zone.intersection(ax).length < share * ax.length:
                continue
            w.veto(f"stair treads of {s.id} paired as a wall")
            n += 1
    return n


def assign_rise(stairs, levels, cfg, assumed_storey_m: float | None = None):
    """Riser height and starting height of every flight.

    A single flight that can climb its whole storey at a comfortable rise climbs it. The flights
    of one assembly share the storey: rise = storey height / all their risers, each flight
    starting where the previous one ends (its landing). Otherwise the configured default rise."""
    sc = cfg["detection"]["stair"]
    groups: dict = {}
    for s in stairs:
        groups.setdefault(s.assembly or s.id, []).append(s)
    for flights in groups.values():
        flights.sort(key=lambda f: f.flight_index)
        lv = levels.get(flights[0].level)
        H = (lv.height if lv is not None and lv.height else None)
        risers = sum(f.treads + 1 for f in flights)
        lo_rise = sc.get("min_rise_m", 0.12)
        if H and H / risers <= sc["full_flight_max_rise_m"]:
            rise, src = H / risers, ("storey height / risers" if len(flights) == 1 else
                                     "storey height / risers of all flights")
        elif not H and len(flights) > 1 and assumed_storey_m and \
                lo_rise <= assumed_storey_m / risers <= sc["full_flight_max_rise_m"]:
            # a multi-flight stair climbs one whole storey: with no level annotation, the storey
            # assumed by the 3D model is used (recorded as an assumption, not a measurement)
            rise, src = assumed_storey_m / risers, f"assumed storey height {assumed_storey_m:.2f} m / risers of all flights"
        else:
            rise, src = sc["default_rise_m"], ("default rise (part of a multi-flight stair or storey height unknown)"
                                               if len(flights) == 1 else "default rise (storey height unknown)")
        z = 0.0
        for f in flights:
            f.rise, f.rise_source = rise, src
            f.base_height = z
            z += (f.treads + 1) * rise


# ---------------------------------------------------------------------------------------------
# walking direction from arrows
class _ArrowIndex:
    """Short strokes that can form an arrow head (chevron) on a flight's walking line."""

    def __init__(self, ctx, sc):
        lo, hi = sc.get("arrow_stroke_m", [0.04, 0.6])
        self.segs = [g for g in ctx.segments if g.dup_of is None and lo <= g.length <= hi]
        self.tree = STRtree([g.line() for g in self.segs]) if self.segs else None
        self.join = max(ctx.tol.snap * 2, 0.01)
        self.ang = [math.radians(a) for a in sc.get("arrow_half_angle_deg", [15.0, 75.0])]

    def chevrons(self, zone):
        """(apex, unit vector the arrow points to) for every symmetric pair of short strokes
        meeting at a common end point inside ``zone``."""
        if self.tree is None:
            return []
        segs = [self.segs[int(k)] for k in self.tree.query(zone, predicate="intersects")]
        out = []
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a, b = segs[i], segs[j]
                for pa, qa in ((a.p0, a.p1), (a.p1, a.p0)):
                    for pb, qb in ((b.p0, b.p1), (b.p1, b.p0)):
                        if float(np.hypot(*(pa - pb))) > self.join:
                            continue
                        apex = (pa + pb) / 2
                        da, db = qa - apex, qb - apex
                        la, lb = float(np.hypot(*da)), float(np.hypot(*db))
                        if la <= 0 or lb <= 0 or abs(la - lb) > 0.35 * max(la, lb):
                            continue
                        da, db = da / la, db / lb
                        axis = -(da + db)
                        n = float(np.hypot(*axis))
                        if n < 1e-6:
                            continue
                        axis /= n                  # bisector, pointing away from the tails
                        half = math.acos(max(-1.0, min(1.0, float(-da @ axis))))
                        if self.ang[0] <= half <= self.ang[1]:
                            out.append((apex, axis))
        return out


def _arrow_vote(arrows, st, u, nrm, sa, sb, cs) -> int:
    """+1 / -1 when arrow heads on the flight point along / against ``nrm``, 0 when none."""
    zone = st.polygon.buffer(max(st.going, 0.2))
    vote = 0
    for apex, axis in arrows.chevrons(zone):
        along = float(apex @ u)
        if not (sa - 0.05 <= along <= sb + 0.05):
            continue                               # arrow of a neighbouring flight
        a = float(axis @ nrm)
        if abs(a) < 0.9:                           # not pointing along the walking line
            continue
        vote += 1 if a > 0 else -1
    return (vote > 0) - (vote < 0)


# ---------------------------------------------------------------------------------------------
# multi-flight stairs
def _frame_extent(poly, v):
    vals = [float(np.dot(c, v)) for c in poly.exterior.coords]
    return min(vals), max(vals)


def _run_sign(st, nrm) -> int:
    return 1 if float(np.dot(st.run, nrm)) > 0 else -1


def _set_run(st, nrm, sign):
    """Make the flight climb towards ``sign * nrm`` (keeps the tread geometry)."""
    if _run_sign(st, nrm) == sign:
        return
    u = np.array(st.u)
    cur = np.array(st.run)
    o = np.array(st.origin)
    sa = float(o @ u)
    base = float(o @ cur)                         # tread positions in the current run frame
    cs = np.array([base + t for t in st.tread_offsets])
    # express tread positions along +nrm, then flip as the detector does
    pos = cs * float(np.dot(cur, nrm))
    order = np.sort(pos)
    if float(np.dot(cur, nrm)) < 0:
        # current run is -nrm: rebuild the +nrm description first
        st.run = (float(nrm[0]), float(nrm[1]))
        st.origin = (float(u[0] * sa + nrm[0] * order[0]), float(u[1] * sa + nrm[1] * order[0]))
        st.tread_offsets = [float(x - order[0]) for x in order]
        g = st.going
        corners = [(sa, order[0] - g), (sa + st.tread_span[1], order[0] - g), (sa + st.tread_span[1], order[-1]),
                   (sa, order[-1])]
        st.polygon = Polygon([tuple(u * x + nrm * y) for x, y in corners])
    else:
        _flip(st, nrm, u, sa, order)


def assemble_flights(stairs, walls, cfg, tol) -> dict:
    """Group flights into stairs and give every flight its walking direction and order.

    * **switchback** (U / dog-leg): parallel flights side by side (well <= ``max_well_m``) whose
      runs overlap. Consecutive flights climb in *opposite* directions and share a landing at
      the end where both meet.
    * **straight with landing**: parallel flights in line (same tread band), separated by a
      landing <= ``max_landing_m``. They climb in the *same* direction.

    A direction read from an UP / DN text or an arrow propagates to the other flights. The
    landing side of a switchback is where the flight ends line up and the stair well is closed
    beyond (a wall within one landing depth); the other end opens to the floor."""
    sc = cfg["detection"]["stair"]
    live = [s_ for s_ in stairs if s_.status != Status.REJECTED]
    log = {"assemblies": 0, "switchback": 0, "straight_with_landing": 0, "directions_propagated": 0}
    if len(live) < 2:
        return log
    max_well = sc.get("max_well_m", 0.8)
    max_land = sc.get("max_landing_m", 3.0)
    edges = []
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            A, B = live[i], live[j]
            if abs(float(np.dot(A.u, B.u))) < math.cos(math.radians(sc["angle_tol_deg"] * 2)):
                continue
            u = np.array(_canon(A.u))
            nrm = np.array([-u[1], u[0]])
            au, bu = _frame_extent(A.polygon, u), _frame_extent(B.polygon, u)
            an, bn = _frame_extent(A.polygon, nrm), _frame_extent(B.polygon, nrm)
            u_gap = max(bu[0] - au[1], au[0] - bu[1])
            n_gap = max(bn[0] - an[1], an[0] - bn[1])
            u_ov = min(au[1], bu[1]) - max(au[0], bu[0])
            n_ov = min(an[1], bn[1]) - max(an[0], bn[0])
            if -tol.snap <= u_gap <= max_well and n_ov >= 0.5 * min(an[1] - an[0], bn[1] - bn[0]):
                edges.append((u_gap, "switchback", A, B))
            elif u_ov >= 0.7 * min(au[1] - au[0], bu[1] - bu[0]) and -tol.snap <= n_gap <= max_land:
                edges.append((n_gap, "straight_with_landing", A, B))
    deg: dict[int, int] = {}
    links: dict[int, list] = {}
    for gap, kind, A, B in sorted(edges, key=lambda e: e[0]):
        if deg.get(id(A), 0) >= 2 or deg.get(id(B), 0) >= 2:
            continue
        deg[id(A)] = deg.get(id(A), 0) + 1
        deg[id(B)] = deg.get(id(B), 0) + 1
        links.setdefault(id(A), []).append((kind, B))
        links.setdefault(id(B), []).append((kind, A))
    seen = set()
    n_asm = 0
    for s0 in live:
        if id(s0) in seen or id(s0) not in links:
            continue
        # walk the chain from one end
        start = s0
        prev = None
        while True:
            nxt = [B for k, B in links[id(start)] if B is not prev]
            if len(links[id(start)]) == 1 or not nxt:
                break
            prev, start = start, nxt[0]
            if start is s0:
                break
        chain, kinds, prev, cur = [start], [], None, start
        while True:
            nxt = [(k, B) for k, B in links[id(cur)] if B is not prev and B not in chain]
            if not nxt:
                break
            k, B = nxt[0]
            kinds.append(k)
            chain.append(B)
            prev, cur = cur, B
        for f in chain:
            seen.add(id(f))
        n_asm += 1
        _orient_chain(chain, kinds, walls, tol, log)
        asm_id = f"stair_assembly_{n_asm:04d}"
        for k, f in enumerate(chain):
            f.assembly = asm_id
            f.assembly_kind = kinds[0] if len(set(kinds)) == 1 else "mixed"
            f.flight_index = k
        for k in kinds:
            log[k] += 1
    log["assemblies"] = n_asm
    return log


def _orient_chain(chain, kinds, walls, tol, log):
    u = np.array(_canon(chain[0].u))
    nrm = np.array([-u[1], u[0]])
    # relative directions: +1 same, -1 opposite to the first flight
    rel = [1]
    for k in kinds:
        rel.append(rel[-1] * (-1 if k == "switchback" else 1))
    known = [(i, _run_sign(f, nrm)) for i, f in enumerate(chain) if f.direction_known]
    if known:
        votes = sum(sg * rel[i] for i, sg in known)
        first = 1 if votes >= 0 else -1
        if any(sg * rel[i] != first for i, sg in known):
            chain[0].notes.append("stair markers disagree between flights: majority used")
    else:
        first = _run_sign(chain[0], nrm)
    for i, f in enumerate(chain):
        want = first * rel[i]
        if not f.direction_known or _run_sign(f, nrm) != want:
            if not f.direction_known:
                log["directions_propagated"] += 1 if known else 0
                f.direction_source = "assembly" if known else f.direction_source
                if known:
                    f.direction_known = True
            _set_run(f, nrm, want)
    # walking order. Straight flights: along the run. Switchback: the landing side decides.
    if kinds and kinds[0] == "straight_with_landing":
        key = lambda f: float(np.dot(np.array(f.origin), nrm)) * first   # noqa: E731
        chain.sort(key=key)
    elif kinds:
        side = _landing_side(chain[:2], nrm, walls, tol)
        # the first flight climbs TOWARDS the landing
        if _run_sign(chain[0], nrm) != side:
            chain.reverse()
        if not known:
            chain[0].notes.append("walking direction unknown: chosen from the landing side of the stair")
            for f in chain:
                f.direction_source = f.direction_source or "landing_side"
    _build_landings(chain, kinds, nrm, walls, tol)


def _ends(f, nrm):
    lo, hi = _frame_extent(f.polygon, nrm)
    return lo, hi


def _landing_side(pair, nrm, walls, tol) -> int:
    """+1 when the landing of a switchback is on the +nrm end, -1 on the -nrm end."""
    A, B = pair
    a, b = _ends(A, nrm), _ends(B, nrm)
    u = np.array(_canon(A.u))
    us = [_frame_extent(f.polygon, u) for f in pair]
    u0, u1 = min(x[0] for x in us), max(x[1] for x in us)
    depth = max(A.width, B.width)
    scores = {}
    for side in (1, -1):
        end = max(a[1], b[1]) if side > 0 else min(a[0], b[0])
        align = abs(a[1] - b[1]) if side > 0 else abs(a[0] - b[0])
        y0, y1 = (end, end + 1.5 * depth) if side > 0 else (end - 1.5 * depth, end)
        probe = Polygon([tuple(u * x + nrm * y) for x, y in ((u0, y0), (u1, y0), (u1, y1), (u0, y1))])
        cover = []
        for w in walls:
            if w.status == Status.REJECTED or abs(float(np.dot(w.u, u))) < 0.9:
                continue
            ax = w.axis
            if ax is not None and ax.intersects(probe):
                x = sorted(float(np.dot(c, u)) for c in ax.intersection(probe).coords) if \
                    ax.intersection(probe).geom_type == "LineString" else []
                if x:
                    cover.append((max(u0, x[0]), min(u1, x[-1])))
        # share of the stair width closed by walls parallel to the treads (merged intervals:
        # wall candidates are still fragmented at this stage)
        closed, cur = 0.0, None
        for x0, x1 in sorted(cover):
            if cur is None or x0 > cur[1]:
                if cur is not None:
                    closed += cur[1] - cur[0]
                cur = [x0, x1]
            else:
                cur[1] = max(cur[1], x1)
        if cur is not None:
            closed += cur[1] - cur[0]
        closed /= max(u1 - u0, 1e-9)
        scores[side] = (min(1.0, closed) >= 0.5, -round(align, 3))
    return 1 if scores[1] >= scores[-1] else -1


def _build_landings(chain, kinds, nrm, walls, tol):
    """Landing reached at the top of every flight that has a successor."""
    u = np.array(_canon(chain[0].u))
    for k in range(len(chain) - 1):
        A, B = chain[k], chain[k + 1]
        sgn = _run_sign(A, nrm)
        a, b = _ends(A, nrm), _ends(B, nrm)
        us = [_frame_extent(f.polygon, u) for f in (A, B)]
        u0, u1 = min(x[0] for x in us), max(x[1] for x in us)
        if kinds[k] == "switchback":
            # A climbs towards the landing, B (climbing back) starts on the same side. The landing
            # starts at A's last nosing and reaches one flight width past the outermost end.
            d = max(A.width, B.width)
            y0, y1 = (a[1], max(a[1], b[1]) + d) if sgn > 0 else (min(a[0], b[0]) - d, a[0])
        else:
            top = a[1] if sgn > 0 else a[0]
            start = b[0] if sgn > 0 else b[1]
            y0, y1 = sorted((top, start))
            u0, u1 = _frame_extent(A.polygon, u)
        if y1 - y0 <= tol.snap:
            continue
        land = Polygon([tuple(u * x + nrm * y) for x, y in ((u0, y0), (u1, y0), (u1, y1), (u0, y1))])
        # stop at the walls around the stair well
        for w in walls:
            if w.status == Status.REJECTED:
                continue
            p = w.polygon(tol.snap * 2)
            if p is not None and p.intersects(land):
                land = land.difference(p)
        if land.geom_type != "Polygon":
            parts = [g for g in getattr(land, "geoms", []) if g.geom_type == "Polygon"]
            ref = A.polygon
            land = min(parts, key=lambda g: g.distance(ref)) if parts else None
        if land is not None and not land.is_empty and land.area > 0.1:
            A.landing = land


def veto_stair_outline_walls(stairs, walls, cfg, layer_of, layer_score) -> int:
    """Reject wall candidates drawn only with stair linework (stringers, landing edges, the
    well outline) inside a flight: two stair lines 0.2 m apart look like a wall, and as a wall
    they block the stair and its landing. Grid / annotation layers do not count either way."""
    share = cfg["detection"]["stair"]["wall_veto_overlap"]
    live = [s_ for s_ in stairs if s_.status != Status.REJECTED]
    n = 0
    for s_ in live:
        zone = s_.polygon.buffer(max(s_.going, 0.2))
        for w in walls:
            if w.status == Status.REJECTED or not w.source_entities:
                continue
            lays = [layer_of(r) for r in w.source_entities]
            lays = [ly for ly in lays if ly is not None and layer_score(ly, "grid") < 0.5
                    and layer_score(ly, "annotation") < 0.5 and layer_score(ly, "dimension") < 0.5]
            if not lays or min(layer_score(ly, "stair") for ly in lays) < 0.5:
                continue
            ax = w.axis
            if ax is None or ax.length <= 0 or zone.intersection(ax).length < share * ax.length:
                continue
            w.veto(f"stair outline of {s_.id} (stair-layer lines inside the flight), not a wall")
            n += 1
    return n
