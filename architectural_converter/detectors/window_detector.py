"""Window detection.

Sources of evidence:
1. window symbols (blocks with window role) - chord along the block's long axis;
2. glazing line groups: >= 2 equal, parallel, overlapping thin lines (+ jambs);
3. wall-band obstructions reported by the parallel detector: a wall pair whose band is
   filled, over a window-sized interval, by lines parallel to the wall (glass/sill lines).
A two-line group whose spacing equals a dominant wall thickness is treated as wall-like
(negative evidence), so short wall pieces are not mistaken for windows."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import MultiPoint, Point

from ..geometry.geometric_utils import angle_diff, dot, frame_point, mrr_dims, normal, rect_polygon, robust_intersection
from ..geometry.segment_pairing import cluster_consistency
from ..models.opening import Opening
from .base_detector import BaseDetector, register


@register
class WindowDetector(BaseDetector):
    name = "window_detector"
    produces = "window"
    requires = ("parallel_wall_detector",)

    def detect(self, drawing, ctx):
        out = []
        out += self._symbols(drawing, ctx)
        out += self._glazing_groups(ctx)
        out += self._obstructions(ctx)
        out = self._dedupe(out, ctx)
        return self.finalize(out)

    def _mk(self, ctx, u, c, s0, s1, depth, subtype="fixed"):
        p0 = frame_point(u, s0, c)
        p1 = frame_point(u, s1, c)
        o = Opening(id=ctx.ids.new("window"), type="window", subtype=subtype, p0=p0, p1=p1, width=s1 - s0,
                    depth=depth)
        o.footprint = rect_polygon(u, c, s0, s1, max(depth or 0, ctx.tol.snap * 2))
        lo, hi = self.cfg["detection"]["window"]["width_m"]
        o.add_evidence("width_plausibility", 1.0 if lo <= o.width <= hi else 0.2)
        return o

    def _symbols(self, drawing, ctx):
        geo = drawing.geo_by_id()
        out = []
        refs = [r for r in drawing.inserts.values() if r.role == "window"]
        ids = {r.id for r in refs}
        for ref in refs:
            if any(p in ids for p in ref.parent_path):
                continue
            pts = [tuple(p) for g in ref.child_geo_ids if g in geo and geo[g].valid and geo[g].points is not None
                   and geo[g].kind not in ("text", "dimension") for p in geo[g].points]
            if len(pts) < 2:
                continue
            w, L, u, cen = mrr_dims(MultiPoint(pts))
            n = normal(u)
            o = self._mk(ctx, u, dot(cen, n), dot(cen, u) - L / 2, dot(cen, u) + L / 2, w)
            o.symbol_insert = ref.id
            o.source_entities.add(ref.id)
            o.add_evidence("symbol", 1.0)
            o.add_evidence("name_hint", min(1.0, ref.role_scores.get("window", 0)))
            o.add_evidence("layer_match", ctx.layer_score(ref.layer, "window"))
            out.append(o)
        return out

    def _glazing_groups(self, ctx):
        cfg = self.cfg["detection"]["window"]
        tol = ctx.tol
        lo, hi = cfg["width_m"]
        th = self.cfg["thickness"]
        idx = [k for k, st in enumerate(ctx.strokes) if lo * 0.9 <= st.length <= hi]
        used = set()
        out = []
        for a in idx:
            if a in used:
                continue
            A = ctx.strokes[a]
            n = normal(A.u)
            group = [a]
            for b in idx:
                if b == a or b in used:
                    continue
                B = ctx.strokes[b]
                if angle_diff(A.angle, B.angle) > tol.pair_angle_tol(A.length, B.length):
                    continue
                sgn = 1.0 if dot(A.u, B.u) >= 0 else -1.0
                b0, b1 = sorted((sgn * B.s0, sgn * B.s1))
                off = dot(B.p0, n) - A.c
                if abs(off) > tol.thickness_max * 1.2 or abs(off) < tol.snap:
                    continue
                ov = min(A.s1, b1) - max(A.s0, b0)
                if ov < 0.9 * max(A.length, B.length) or abs(A.length - B.length) > max(2 * tol.snap, cfg["length_match_tol"] * A.length):
                    continue
                group.append(b)
            if len(group) < cfg["min_lines"]:
                continue
            offs = sorted(dot(ctx.strokes[k].p0, n) for k in group)
            depth = offs[-1] - offs[0]
            if depth > tol.thickness_max * 1.2:
                continue
            if len(group) == 2 and any({pc.i, pc.j} == set(group) for pc in ctx.pairs):
                continue  # the two lines are the faces of a wall piece, not glazing
            in_band = self._in_band(ctx, A.u, offs[0], offs[-1], A.s0, A.s1)
            lay = max(ctx.layer_score(l, "window") for k in group for l in ctx.strokes[k].layers)
            furn = max(ctx.layer_score(l, "furniture") for k in group for l in ctx.strokes[k].layers)
            if in_band <= 0 and lay < 0.5:
                continue  # glazing must sit in a wall (or be explicitly on a window layer)
            used |= set(group)
            o = self._mk(ctx, A.u, (offs[0] + offs[-1]) / 2, A.s0, A.s1, depth)
            o.add_evidence("not_furniture", 1.0 - furn)
            nl = len(group)
            o.add_evidence("line_count", {2: 0.5, 3: 0.85}.get(nl, 1.0))
            o.add_evidence("length_match", 1.0)
            o.add_evidence("layer_match", lay)
            if nl == 2:
                cons = cluster_consistency(depth, ctx.thickness_clusters, th["cluster_rel_tol"], th["cluster_abs_tol_m"])
                o.add_evidence("not_wall_thickness", 1.0 - cons)
            else:
                o.add_evidence("not_wall_thickness", 1.0)
            o.add_evidence("jambs", self._jambs(ctx, A.u, offs[0], offs[-1], A.s0, A.s1))
            o.add_evidence("in_wall_band", in_band)
            for k in group:
                o.source_entities |= ctx.strokes[k].raw_ids
            o.detection_methods.add("glazing_lines")
            out.append(o)
        return out

    def _jambs(self, ctx, u, c_lo, c_hi, s0, s1):
        tol = ctx.tol
        hits = 0
        if ctx.stroke_tree is None:
            return 0.0
        for s in (s0, s1):
            p = frame_point(u, s, (c_lo + c_hi) / 2)
            for k in ctx.stroke_tree.query(Point(p).buffer(max(tol.snap * 3, (c_hi - c_lo) / 2 + tol.snap))):
                st = ctx.strokes[int(k)]
                if abs(dot(st.u, u)) < 0.1 and st.length >= (c_hi - c_lo) * 0.8 and st.line().distance(Point(p)) <= tol.snap * 3:
                    hits += 1
                    break
        return hits / 2.0

    def _in_band(self, ctx, u, c_lo, c_hi, s0, s1):
        tol = ctx.tol
        best = 0.0
        for pc in ctx.pairs:
            if abs(abs(dot(pc.u, u)) - 1) > 1e-3:
                continue
            sg = 1.0 if dot(pc.u, u) >= 0 else -1.0
            lo, hi = sorted((pc.c_i * sg, pc.c_j * sg))
            if not (lo - 2 * tol.snap <= c_lo and c_hi <= hi + 2 * tol.snap):
                continue
            a0, a1 = sorted((pc.s0 * sg, pc.s1 * sg))
            if min(abs(a1 - s0), abs(a0 - s1)) <= max(4 * tol.snap, 0.02):
                best = 1.0
        return best

    def _obstructions(self, ctx):
        cfg = self.cfg["detection"]["window"]
        lo, hi = cfg["width_m"]
        out = []
        th = self.cfg["thickness"]
        for ob in ctx.obstructions:
            L = ob.s1 - ob.s0
            if not (lo <= L <= hi):
                continue
            # the band must be a real wall band (a thickness class of this drawing) ...
            cons = cluster_consistency(ob.c_hi - ob.c_lo, ctx.thickness_clusters, th["cluster_rel_tol"], th["cluster_abs_tol_m"])
            if cons < cfg["obstruction_min_consistency"]:
                continue
            # ... and the wall must continue beside the window (same pair, adjacent piece)
            flank = any(pc.i == ob.i and pc.j == ob.j and min(abs(pc.s1 - ob.s0), abs(pc.s0 - ob.s1)) <= 2 * ctx.tol.snap
                        for pc in ctx.pairs)
            if not flank:
                continue
            o = self._mk(ctx, ob.u, (ob.c_lo + ob.c_hi) / 2, ob.s0, ob.s1, ob.c_hi - ob.c_lo)
            o.add_evidence("in_wall_band", 1.0)
            o.add_evidence("line_count", {1: 0.85}.get(len(ob.blockers), 1.0))
            o.add_evidence("not_wall_thickness", 1.0)
            o.add_evidence("layer_match", max(ctx.layer_score(l, "window") for k in ob.blockers for l in ctx.strokes[k].layers))
            for k in ob.blockers:
                o.source_entities |= ctx.strokes[k].raw_ids
            o.detection_methods.add("wall_band_obstruction")
            out.append(o)
        return out

    def _dedupe(self, cands, ctx):
        out: list[Opening] = []
        for o in sorted(cands, key=lambda x: -len(x.evidence)):
            merged = False
            for p in out:
                if p.footprint is not None and o.footprint is not None:
                    inter = robust_intersection(p.footprint.buffer(ctx.tol.snap * 2), o.footprint).area
                    if inter >= 0.5 * min(p.footprint.area, o.footprint.area):
                        p.absorb(o)
                        merged = True
                        break
            if not merged:
                out.append(o)
        return out
