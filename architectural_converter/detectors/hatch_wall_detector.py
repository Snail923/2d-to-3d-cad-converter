"""Detector C: hatch / solid-fill based walls.

Only fill regions that are *thin everywhere* (morphological opening removes them) are
treated as wall material; floor patterns, column fills and decorative hatches are not.
Their boundaries are paired like linework, with the extra condition that the material
between the pair lies inside the fill. Thin fill parts that cannot be paired (curved or
irregular walls) become polygon candidates."""
from __future__ import annotations

from shapely.geometry import Point

from ..geometry.geometric_utils import frame_point, polygons_of, rect_polygon, thin_fraction, inscribed_width, robust_union
from ..geometry.geometry_adapter import Segment, build_strokes
from ..geometry.segment_pairing import find_pairs, resolve_side_conflicts
from ..models.wall import Wall
from .base_detector import BaseDetector, register
from .wall_common import thickness_plausibility, wall_from_piece


@register
class HatchWallDetector(BaseDetector):
    name = "hatch_wall_detector"
    produces = "wall"
    requires = ("parallel_wall_detector",)

    def detect(self, drawing, ctx):
        tol = ctx.tol
        regions = [r for r in ctx.hatch_regions if r.cls == "wall_fill" and r.role == "linework"]
        if not regions:
            return []
        segs = []
        n = 0
        import numpy as np
        for r in regions:
            for ring in [r.polygon.exterior, *r.polygon.interiors]:
                cs = list(ring.coords)
                for a, b in zip(cs[:-1], cs[1:]):
                    if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= 0:
                        continue
                    n += 1
                    segs.append(Segment(id=f"hs_{n:06d}", geo_id=r.geo_id, raw_id=r.raw_id, layer=r.layer,
                                        p0=np.array(a), p1=np.array(b), origin="hatch", role="linework"))
        strokes = build_strokes(segs, tol.snap, tol.angle, prefix="hk", same_layer=False)
        pieces, _ = find_pairs(strokes, tol, self.cfg["detection"]["parallel"])
        fill = robust_union([r.polygon for r in regions])
        good = []
        for p in pieces:
            mid = frame_point(p.u, (p.s0 + p.s1) / 2, p.c)
            if fill.buffer(tol.snap).contains(Point(mid)):
                p.score = p.length
                good.append(p)
        good = resolve_side_conflicts(good, strokes, tol.snap)
        out = []
        covered = []
        for p in good:
            w = wall_from_piece(p, ctx, self.name, strokes=strokes)
            w.face_segments = set()
            w.add_evidence("hatch_fill", 1.0)
            w.add_evidence("emptiness", 1.0)
            out.append(w)
            covered.append(rect_polygon(p.u, p.c, p.s0, p.s1, p.thickness).buffer(tol.snap * 2))
        # thin fill parts not explained by pairs
        rest = fill.difference(robust_union(covered)) if covered else fill
        for part in polygons_of(rest):
            if part.area < (tol.thickness_min * 4) ** 2:
                continue
            tf = thin_fraction(part, tol.thickness_max / 2 * 1.05)
            if tf < self.cfg["detection"]["hatch"]["thin_fraction_min"]:
                continue
            t_est = 2 * part.area / max(part.exterior.length, 1e-9)
            w = Wall(id=ctx.ids.new("wall"), kind="polygon", shape=part, thickness=t_est)
            for r in regions:
                if r.polygon.intersects(part):
                    w.source_entities.add(r.raw_id)
            w.add_evidence("hatch_fill", 1.0)
            w.add_evidence("thinness", tf)
            w.add_evidence("thickness_plausibility", thickness_plausibility(t_est, self.cfg))
            w.add_evidence("layer_match", sum(ctx.wall_evidence(r.layer) for r in regions) / len(regions))
            w.notes.append("unpaired thin hatch region")
            out.append(w)
        ctx.stats["wall_fill_regions"] = len(regions)
        return self.finalize(out)
