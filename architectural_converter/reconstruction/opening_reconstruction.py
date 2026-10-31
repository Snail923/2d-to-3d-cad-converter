"""Opening-aware wall reconstruction.

Free wall ends are examined along the wall axis. The gap to the next wall is classified:

* ``<= snap``                -> already handled by collinear merging
* ``(snap, min_opening)``    -> small gap. Closed only when a collinear partner of the same
                                thickness continues the wall and the gap is <= its thickness;
                                recorded as a low-confidence reconstruction (never silent)
* ``[min_opening, max_opening]`` -> opening slot. Door / window candidates whose chord lies
                                in the slot are hosted there. Without symbol evidence the slot
                                becomes an *unidentified opening* (NEEDS_REVIEW)
* larger                     -> left open (reported by validation / room recovery)

When the slot is bridged, the host wall becomes continuous across it (collinear partner
merged, or the wall extended to the perpendicular wall it reaches) and the opening keeps the
exact interval along the host wall, so Phase 2 can cut a void. Symbols that sit on a
continuous wall (no gap drawn) are hosted by projection."""
from __future__ import annotations

import math

from shapely.geometry import LineString, Point

from ..classification.candidate_resolver import aligned, same_band
from ..geometry.geometric_utils import dot, frame_point, line_intersection, rect_polygon, robust_intersection
from ..models.base_entity import Status
from ..models.opening import Opening
from ..models.wall import Wall
from .line_merging import merge_into


class OpeningReconstructor:
    def __init__(self, cfg, ids, tol):
        self.cfg = cfg
        self.ids = ids
        self.tol = tol
        rc = cfg["reconstruction"]
        self.min_open, self.max_open = rc["opening_width_m"]
        # wider slots are accepted only when a door/window symbol explains them
        self.max_symbol_open = max(self.max_open, cfg["detection"]["window"]["width_m"][1],
                                   cfg["detection"]["door"]["sliding_width_m"][1])
        self.log = {"hosted": 0, "unidentified": 0, "small_gaps_closed": 0, "merged_across": 0,
                    "extended_across": 0, "hosted_on_continuous_wall": 0, "unhosted": 0}

    # ------------------------------------------------------------------ helpers
    def _connected(self, w: Wall, e: int, walls) -> bool:
        p = w.point_at(w.s1 if e else w.s0)
        pt = Point(p)
        r = max(self.tol.snap * 2, 1e-4)
        for v in walls:
            if v is w:
                continue
            poly = v.polygon(self.tol.snap * 2)
            if poly is not None and poly.distance(pt) <= r:
                return True
        return False

    def _slot_hit(self, w: Wall, e: int, walls):
        """Nearest wall reached by the ray from end e of w (returns kind, wall, gap)."""
        p = w.point_at(w.s1 if e else w.s0)
        out = w.u if e else (-w.u[0], -w.u[1])
        far = (p[0] + out[0] * (self.max_symbol_open + 1.0), p[1] + out[1] * (self.max_symbol_open + 1.0))
        ray = LineString([p, far])
        best = None
        for v in walls:
            if v is w or v.kind not in ("straight", "centerline", "arc", "polygon"):
                continue
            if v.kind in ("straight", "centerline") and same_band(w, v, self.tol):
                c, s0, s1 = aligned(v, w.u)
                gap = (s0 - w.s1) if e else (w.s0 - s1)
                if gap > self.tol.snap * 0.5:
                    if best is None or gap < best[2]:
                        best = ("collinear", v, gap)
                continue
            poly = v.polygon(self.tol.snap * 2)
            if poly is None or not poly.intersects(ray):
                continue
            inter = robust_intersection(poly, ray)
            d = Point(p).distance(inter)
            if d <= self.tol.snap * 0.5:
                continue
            if v.kind in ("straight", "centerline") and abs(dot(v.u, w.u)) > 0.95:
                continue  # parallel offset wall: not a stop
            if best is None or d < best[2]:
                best = ("wall", v, d)
        return best

    def _match_symbol(self, w: Wall, a: float, b: float, cands, relative_to_chord: bool = False):
        """Best door/window whose chord lies within slot [a, b] on wall w."""
        t = w.thickness or 0.0
        best = None
        for o in cands:
            if o.host_wall is not None or o.status == Status.REJECTED:
                continue
            for p0, p1 in [(o.p0, o.p1)] + list(o.alt_chords):
                d = (p1[0] - p0[0], p1[1] - p0[1])
                L = math.hypot(*d)
                if L <= 0 or abs(dot(d, w.u)) / L < 0.9:
                    continue
                n = (-w.u[1], w.u[0])
                mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
                if abs(dot(mid, n) - w.c) > t / 2 + max(0.3, t):
                    continue
                x0, x1 = sorted((dot(p0, w.u), dot(p1, w.u)))
                ov = min(b, x1) - max(a, x0)
                if ov <= 0:
                    continue
                score = ov / (x1 - x0) if relative_to_chord else ov / max(b - a, x1 - x0)
                if score >= 0.6 and (best is None or score > best[0]):
                    best = (score, o, (p0, p1))
        return best

    def _host(self, o: Opening, w: Wall, a: float, b: float, score: float, chord, gap_match: bool):
        o.host_wall = w.id
        o.host_interval = (a, b)
        o.p0 = w.point_at(a)
        o.p1 = w.point_at(b)
        o.width = b - a
        o.alt_chords = []
        o.add_evidence("host_wall", 1.0)
        o.add_evidence("gap_match", score if gap_match else 0.3)
        if o.id not in w.openings:
            w.openings.append(o.id)
        self.log["hosted"] += 1

    # ------------------------------------------------------------------ main
    def run(self, walls: list[Wall], doors, windows):
        tol = self.tol
        symbols = list(doors) + list(windows)
        generic: list[Opening] = []
        live = [w for w in walls if w.status != Status.REJECTED]
        self.max_t = max([w.thickness or 0.0 for w in live] + [tol.snap])
        queue = [(w, e) for w in live if w.kind in ("straight", "centerline") for e in (0, 1)]
        done = set()
        while queue:
            w, e = queue.pop(0)
            if w not in live or (id(w), e) in done:
                continue
            done.add((id(w), e))
            connected = self._connected(w, e, live)
            hit = self._slot_hit(w, e, live)
            if hit is None:
                continue
            kind, v, gap = hit
            s_end = w.s1 if e else w.s0
            a, b = (s_end, s_end + gap) if e else (s_end - gap, s_end)
            if connected:
                # a junction end: continue only along the same wall, across an explained opening
                if kind != "collinear" or gap > self.max_symbol_open + 2 * self.max_t:
                    continue
                ca, cb = self._clear_interval(w, a, b, live, v)
                match = self._match_symbol(w, a, b, symbols, relative_to_chord=True)
                if match is None and (self._crossed(w, a, b, live, v) or not (self.min_open <= cb - ca <= self.max_open)):
                    continue  # e.g. partitions facing each other across a corridor
                host = self._merge_or_extend(w, e, kind, v, gap, live, note="bridged opening beside junction")
                if match is not None:
                    score, o, chord = match
                    x0, x1 = sorted((dot(chord[0], w.u), dot(chord[1], w.u)))
                    self._host(o, host, max(a, x0), min(b, x1), score, chord, True)
                else:
                    self._generic(host, ca, cb, generic)
                queue.append((host, e))
                done.discard((id(host), e))
                continue
            if gap < self.min_open:
                t = w.thickness or 0.0
                if kind == "collinear" and gap <= max(t, tol.snap * 4) * self.cfg["reconstruction"]["small_gap_close_max_factor_of_thickness"]:
                    self._merge_or_extend(w, e, kind, v, gap, live, note="small gap closed (hypothesis)")
                    w.reconstruction.setdefault("hypothesis_gaps", []).append(
                        {"interval": [round(a, 4), round(b, 4)], "confidence": 0.6})
                    self.log["small_gaps_closed"] += 1
                    queue.append((w, e))
                    done.discard((id(w), e))
                continue
            if gap > self.max_symbol_open:
                continue
            match = self._match_symbol(w, a, b, symbols)
            if gap > self.max_open and match is None:
                continue
            ca, cb = self._clear_interval(w, a, b, live, v)
            host = self._merge_or_extend(w, e, kind, v, gap, live, note="bridged opening")
            if match is not None:
                score, o, chord = match
                self._host(o, host, a, b, score, chord, True)
            else:
                self._generic(host, ca, cb, generic)
            queue.append((host, e))
            done.discard((id(host), e))
        # symbols not in any gap: host on a continuous wall by projection
        for o in symbols:
            if o.host_wall is not None or o.status == Status.REJECTED:
                continue
            best = None
            for w in live:
                if w.kind not in ("straight", "centerline"):
                    continue
                m = self._match_symbol(w, w.s0, w.s1, [o], relative_to_chord=True)
                if m is None:
                    continue
                score, _, (p0, p1) = m
                x0, x1 = sorted((dot(p0, w.u), dot(p1, w.u)))
                n = (-w.u[1], w.u[0])
                dist = abs(dot(((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2), n) - w.c)
                if best is None or dist < best[0]:
                    best = (dist, w, max(x0, w.s0), min(x1, w.s1))
            if best is not None and best[3] - best[2] > tol.snap:
                _, w, a, b = best
                self._host(o, w, a, b, 0.5, None, False)
                o.notes.append("hosted on continuous wall (no gap drawn)")
                self.log["hosted_on_continuous_wall"] += 1
            else:
                self.log["unhosted"] += 1
        walls[:] = [w for w in walls if w in live or w.status == Status.REJECTED]
        return generic

    def _generic(self, host, a, b, generic):
        o = Opening(id=self.ids.new("opening"), type="opening", subtype="unidentified")
        self._host(o, host, a, b, 1.0, None, True)
        o.evidence.pop("gap_match", None)
        o.add_evidence("gap_geometry", 1.0)
        lo, hi = self.cfg["validation"]["door_width_m"]
        o.add_evidence("width_plausibility", 1.0 if lo <= b - a <= hi else 0.5)
        o.notes.append("gap in wall without door/window symbol")
        o.is_hypothesis = True
        generic.append(o)
        self.log["unidentified"] += 1

    def _clear_interval(self, w, a, b, walls, partner):
        """Largest part of slot [a, b] on w's band that is not covered by other walls."""
        t = w.thickness or self.tol.snap * 2
        slot = rect_polygon(w.u, w.c, a, b, t * 0.98)
        others = [v.polygon(self.tol.snap * 2) for v in walls if v is not w and v is not partner]
        cover = [p for p in others if p is not None and p.intersects(slot)]
        if not cover:
            return a, b
        cuts = []
        for p in cover:
            inter = robust_intersection(p, slot)
            if inter.is_empty:
                continue
            xs = [dot(c, w.u) for g in getattr(inter, "geoms", [inter]) for c in getattr(g, "exterior", g).coords]
            cuts.append((min(xs), max(xs)))
        from ..geometry.geometric_utils import subtract_intervals
        pieces = subtract_intervals((a, b), cuts)
        if not pieces:
            return a, a
        return max(pieces, key=lambda ab: ab[1] - ab[0])

    def _crossed(self, w, a, b, walls, partner) -> bool:
        """True when another wall runs straight across the slot (it continues on both sides of w's band)."""
        t = w.thickness or self.tol.snap * 2
        n = (-w.u[1], w.u[0])
        for v in walls:
            if v is w or v is partner or v.kind not in ("straight", "centerline"):
                continue
            if abs(dot(v.u, w.u)) > 0.95:
                continue
            p0, p1 = v.point_at(v.s0), v.point_at(v.s1)
            o0, o1 = dot(p0, n) - w.c, dot(p1, n) - w.c
            reach = t / 2 + (v.thickness or 0) + self.tol.snap * 2
            if min(o0, o1) < -reach and max(o0, o1) > reach:
                hit = line_intersection(p0, v.u, w.point_at(0.0), w.u)
                pad = (v.thickness or 0) + self.tol.snap * 2   # crossing walls at the slot ends count
                if hit is not None and a - pad <= hit[2] <= b + pad:
                    return True
        return False

    def _interval_on(self, host: Wall, w: Wall, a: float, b: float):
        if host is w:
            return a, b
        # same direction frames (merged into host which may be flipped)
        if dot(host.u, w.u) >= 0:
            return a, b
        return -b, -a

    def _merge_or_extend(self, w: Wall, e: int, kind: str, v: Wall, gap: float, live, note: str) -> Wall:
        if kind == "collinear":
            merge_into(w, v, note, None)
            live.remove(v)
            self.log["merged_across"] += 1
            return w
        # extend w across the gap to the axis of the wall it reaches (or to its face for
        # polygon / arc walls)
        ext = gap
        if v.kind in ("straight", "centerline"):
            p = w.point_at(w.s1 if e else w.s0)
            out = w.u if e else (-w.u[0], -w.u[1])
            hit = line_intersection(p, out, v.point_at(v.s0), v.u)
            if hit is not None and hit[1] > 0:
                ext = hit[1]
        if e:
            w.s1 += ext
        else:
            w.s0 -= ext
        w.reconstruction.setdefault("extensions", []).append(
            {"end": "end" if e else "start", "length": round(ext, 4), "to_wall": v.id, "junction": "opening",
             "reason": note})
        self.log["extended_across"] += 1
        return w
