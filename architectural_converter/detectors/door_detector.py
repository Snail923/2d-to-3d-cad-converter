"""Door detection from symbols (blocks) and loose swing geometry.

Evidence: block role / name hints, quarter-circle swing arcs with plausible radius, a leaf
line from the arc centre, layer hints. The *opening chord* is hinge -> closed-leaf point
(the arc end that is not the leaf tip). When the leaf is missing both chords are kept and
the host wall decides (the chord parallel to the wall wins).

Leaves from all sources (blocks and loose linework) are collected first; two leaves whose
closed points meet and whose hinges are r1 + r2 apart form a double door - also when they
come from two separate block inserts. Blocks with door role and no arc are sliding doors."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from ..geometry.geometric_utils import mrr_dims
from ..models.opening import Opening
from .base_detector import BaseDetector, register
from .symbol_analysis import door_arcs, leaf_line_for


def _sector(c, r, a0, a1):
    sweep = (a1 - a0) % (2 * math.pi) or 2 * math.pi
    n = max(4, int(math.degrees(sweep) / 6))
    pts = [tuple(c)] + [(c[0] + r * math.cos(a0 + sweep * k / n), c[1] + r * math.sin(a0 + sweep * k / n))
                        for k in range(n + 1)]
    return Polygon(pts)


@register
class DoorDetector(BaseDetector):
    name = "door_detector"
    produces = "door"

    def detect(self, drawing, ctx):
        cfg = self.cfg["detection"]["door"]
        tol = ctx.tol
        geo = drawing.geo_by_id()
        leaves = []
        out: list[Opening] = []
        door_refs = [r for r in drawing.inserts.values() if r.role == "door"]
        door_ids = {r.id for r in door_refs}
        for ref in door_refs:
            if any(p in door_ids for p in ref.parent_path):
                continue
            ents = [geo[g] for g in ref.child_geo_ids if g in geo and geo[g].valid]
            arcs = door_arcs(ents, cfg, tol.snap)
            lines = [g for g in ents if g.kind == "line"]
            if arcs:
                for lf in self._leaves(arcs, lines, cfg, tol):
                    lf.update({"ref": ref, "layer": ref.layer})
                    leaves.append(lf)
            else:
                o = self._sliding(ref, ents, cfg, ctx)
                if o is not None:
                    out.append(o)
        loose = [geo[a.geo_id] for a in ctx.arcs if a.role in ("linework", "symbol")]
        arcs = door_arcs(loose, cfg, tol.snap)
        if arcs:
            lines = [g for g in drawing.entities if g.kind == "line" and g.valid and g.role in ("linework", "symbol")]
            for lf in self._leaves(arcs, lines, cfg, tol):
                lf.update({"ref": None, "layer": lf["arc"].layer})
                leaves.append(lf)
        out += self._assemble(leaves, cfg, ctx)
        return self.finalize(out)

    # ------------------------------------------------------------------
    def _leaves(self, arcs, lines, cfg, tol):
        res = []
        for a in arcs:
            c = np.array(a.data["center"])
            r = a.data["radius"]
            p_start = c + r * np.array([math.cos(a.data["start"]), math.sin(a.data["start"])])
            p_end = c + r * np.array([math.cos(a.data["end"]), math.sin(a.data["end"])])
            leaf = leaf_line_for(a, lines, tol.snap, cfg["leaf_length_tol"])
            if leaf is not None:
                tip = leaf[1]
                closed = p_end if np.hypot(*(tip - p_start)) < np.hypot(*(tip - p_end)) else p_start
                chords = [(tuple(c), tuple(closed))]
            else:
                chords = [(tuple(c), tuple(p_end)), (tuple(c), tuple(p_start))]
            res.append({"arc": a, "c": c, "r": r, "chords": chords, "leaf": leaf,
                        "sector": _sector(c, r, a.data["start"], a.data["end"])})
        return res

    def _sliding(self, ref, ents, cfg, ctx):
        pts = [tuple(p) for g in ents if g.points is not None and g.kind != "text" for p in g.points]
        if len(pts) < 2:
            return None
        w, L, u, cen = mrr_dims(MultiPoint(pts))
        lo, hi = cfg["sliding_width_m"]
        if not (lo <= L <= hi):
            return None
        o = Opening(id=ctx.ids.new("door"), type="door", subtype="sliding",
                    p0=(cen[0] - u[0] * L / 2, cen[1] - u[1] * L / 2), p1=(cen[0] + u[0] * L / 2, cen[1] + u[1] * L / 2),
                    width=L, depth=w, symbol_insert=ref.id)
        o.source_entities.add(ref.id)
        o.footprint = MultiPoint(pts).convex_hull
        o.add_evidence("symbol", 1.0)
        o.add_evidence("name_hint", min(1.0, ref.role_scores.get("door", 0)))
        o.add_evidence("layer_match", ctx.layer_score(ref.layer, "door"))
        o.add_evidence("width_plausibility", 1.0)
        return o

    def _evidence(self, o, lfs, ctx):
        o.add_evidence("swing_arc", 1.0)
        o.add_evidence("leaf_line", sum(1.0 for L in lfs if L["leaf"] is not None) / len(lfs))
        refs = [L["ref"] for L in lfs if L["ref"] is not None]
        for L in lfs:
            o.source_entities.add(L["arc"].raw_id)
            if L["leaf"] is not None:
                o.source_entities.add(L["leaf"][0].raw_id)
            if L["ref"] is not None:
                o.source_entities.add(L["ref"].id)
        if refs:
            o.symbol_insert = refs[0].id
            o.add_evidence("symbol", 1.0)
            o.add_evidence("name_hint", min(1.0, max(r.role_scores.get("door", 0) for r in refs)))
        o.add_evidence("layer_match", max(ctx.layer_score(L["layer"], "door") for L in lfs))

    def _assemble(self, leaves, cfg, ctx):
        tol = ctx.tol
        out = []
        used = set()
        for i, A in enumerate(leaves):
            if i in used:
                continue
            for j in range(i + 1, len(leaves)):
                if j in used:
                    continue
                B = leaves[j]
                d = float(np.hypot(*(A["c"] - B["c"])))
                if abs(d - (A["r"] + B["r"])) > max(4 * tol.snap, 0.05 * d) or d > cfg["double_max_width_m"]:
                    continue
                if not any(math.dist(ca[1], cb[1]) <= max(4 * tol.snap, 0.04 * d) for ca in A["chords"] for cb in B["chords"]):
                    continue
                used |= {i, j}
                o = Opening(id=ctx.ids.new("door"), type="door", subtype="double_swing", p0=tuple(A["c"]), p1=tuple(B["c"]),
                            width=d, swing={"leaves": 2, "radii": [A["r"], B["r"]]})
                o.footprint = unary_union([A["sector"], B["sector"]])
                self._evidence(o, [A, B], ctx)
                o.add_evidence("width_plausibility", 1.0)
                out.append(o)
                break
        for i, A in enumerate(leaves):
            if i in used:
                continue
            p0, p1 = A["chords"][0]
            o = Opening(id=ctx.ids.new("door"), type="door", subtype="single_swing", p0=p0, p1=p1, width=A["r"],
                        hinge=tuple(A["c"]), alt_chords=A["chords"][1:],
                        swing={"leaves": 1, "radius": A["r"], "arc_start": A["arc"].data["start"],
                               "arc_end": A["arc"].data["end"]})
            o.footprint = A["sector"]
            self._evidence(o, [A], ctx)
            o.add_evidence("width_plausibility", 1.0 if 0.6 <= A["r"] <= 1.2 else 0.6)
            out.append(o)
        return out
