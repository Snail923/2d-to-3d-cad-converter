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


def assign_rise(stairs, levels, cfg):
    """Riser height: a flight that can climb its whole storey at a comfortable rise climbs it;
    otherwise it is one flight of a multi-flight stair and gets the configured default rise."""
    sc = cfg["detection"]["stair"]
    for s in stairs:
        lv = levels.get(s.level)
        H = (lv.height if lv is not None and lv.height else None)
        risers = s.treads + 1
        if H and H / risers <= sc["full_flight_max_rise_m"]:
            s.rise, s.rise_source = H / risers, "storey height / risers"
        else:
            s.rise, s.rise_source = sc["default_rise_m"], "default rise (part of a multi-flight stair or storey height unknown)"
