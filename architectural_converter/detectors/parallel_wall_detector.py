"""Detector A: walls drawn as two (approximately) parallel boundaries; also concentric arc
pairs for curved walls. Thickness classes are clustered from the drawing itself."""
from __future__ import annotations

import math

import numpy as np

from ..geometry.curve_processor import arc_overlap
from ..geometry.segment_pairing import (cluster_consistency, find_pairs, resolve_side_conflicts,
                                        thickness_clusters)
from ..models.wall import Wall
from .base_detector import BaseDetector, register
from .wall_common import thickness_plausibility, wall_from_piece


@register
class ParallelWallDetector(BaseDetector):
    name = "parallel_wall_detector"
    produces = "wall"

    def detect(self, drawing, ctx):
        cfg = self.cfg
        th = cfg["thickness"]
        tol = ctx.tol
        pieces, obstructions = find_pairs(ctx.strokes, tol, cfg["detection"]["parallel"])
        clusters = thickness_clusters([p.thickness for p in pieces], [p.length for p in pieces],
                                      th["cluster_rel_tol"], th["cluster_abs_tol_m"], th["cluster_min_share"])
        ctx.thickness_clusters = clusters
        for p in pieces:
            cons = cluster_consistency(p.thickness, clusters, th["cluster_rel_tol"], th["cluster_abs_tol_m"])
            lay = ctx.strokes_wall_evidence([p.i, p.j])
            plaus = thickness_plausibility(p.thickness, cfg)
            p.score = p.length * (0.2 + cons) * (0.2 + lay) * (0.2 + plaus)
        accepted = resolve_side_conflicts(pieces, ctx.strokes, tol.snap)
        ctx.pairs = accepted
        ctx.obstructions = obstructions
        ctx.stats["pair_pieces"] = len(pieces)
        ctx.stats["pair_pieces_side_conflict"] = len(pieces) - len(accepted)
        cands = [wall_from_piece(p, ctx, self.name) for p in accepted]
        cands += self._arc_walls(ctx)
        return self.finalize(cands)

    # ----------------------------------------------------------------- curved walls
    def _arc_walls(self, ctx):
        cfg = self.cfg
        tol = ctx.tol
        ap = cfg["detection"]["arc_pairs"]
        th = cfg["thickness"]
        arcs = [a for a in ctx.arcs if a.role == "linework"]
        out = []
        used = set()
        ctol = max(tol.snap * ap["center_tol_factor_of_snap"], 1e-4)
        for i, a in enumerate(arcs):
            for j in range(i + 1, len(arcs)):
                b = arcs[j]
                if math.dist(a.center, b.center) > ctol:
                    continue
                dr = abs(a.radius - b.radius)
                if not (tol.thickness_min <= dr <= tol.thickness_max):
                    continue
                ov = arc_overlap(a.start, a.sweep, b.start, b.sweep)
                if ov is None or ov[1] < math.radians(ap["min_sweep_deg"]):
                    continue
                r_lo, r_hi = sorted((a.radius, b.radius))
                # emptiness: another concentric arc strictly between
                blocked = any(math.dist(c.center, a.center) <= ctol and r_lo + tol.snap < c.radius < r_hi - tol.snap
                              and arc_overlap(c.start, c.sweep, ov[0], ov[1]) for k, c in enumerate(arcs) if k not in (i, j))
                if blocked:
                    continue
                cx = (a.center[0] + b.center[0]) / 2
                cy = (a.center[1] + b.center[1]) / 2
                r = (a.radius + b.radius) / 2
                w = Wall(id=ctx.ids.new("wall"), kind="arc", thickness=dr, thickness_source="measured",
                         arc={"center_x": cx, "center_y": cy, "radius": r, "start": ov[0], "end": (ov[0] + ov[1]) % (2 * math.pi)})
                w.source_entities |= {a.raw_id, b.raw_id}
                w.add_evidence("parallelism", 1.0 - 0.5 * math.dist(a.center, b.center) / ctol)
                w.add_evidence("endpoint_alignment", ov[1] / max(a.sweep, b.sweep))
                w.add_evidence("thickness_plausibility", thickness_plausibility(dr, cfg))
                w.add_evidence("thickness_consistency", cluster_consistency(dr, ctx.thickness_clusters,
                                                                            th["cluster_rel_tol"], th["cluster_abs_tol_m"]))
                w.add_evidence("emptiness", 1.0)
                w.add_evidence("layer_match", (ctx.wall_evidence(a.layer) + ctx.wall_evidence(b.layer)) / 2)
                used |= {i, j}
                out.append(w)
        return out
