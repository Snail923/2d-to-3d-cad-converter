"""Column detection: compact closed shapes (polylines, circles, solids, fills, closed loops
of loose lines) and column symbols. Evidence distinguishes columns from furniture:
fill, layer/name hints, repetition of identical sizes, alignment on a grid."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import polygonize

from ..geometry.geometric_utils import mrr_dims, polygons_of, robust_union, robust_intersection
from ..models.column import Column
from .base_detector import BaseDetector, register


@register
class ColumnDetector(BaseDetector):
    name = "column_detector"
    produces = "column"

    def _ok(self, poly):
        cc = self.cfg["detection"]["column"]
        lo, hi = cc["size_m"]
        w, L, _, _ = mrr_dims(poly)
        if w <= 0:
            return None
        if not (lo * 0.9 <= w and L <= hi and L / w <= cc["max_aspect"]):
            return None
        return w, L

    def detect(self, drawing, ctx):
        geo = drawing.geo_by_id()
        tol = ctx.tol
        raw: list[tuple[Polygon, set, str, str, float]] = []   # poly, sources, layer, origin, name_hint
        for g in drawing.entities:
            if not g.valid or g.role not in ("linework", "column"):
                continue
            if g.role == "column":
                continue  # handled per insert below
            if g.kind in ("polyline", "polygon") and g.closed and g.points is not None and len(g.points) >= 3:
                p = Polygon(g.points).buffer(0)
            elif g.kind == "circle":
                p = Polygon(g.points)
            else:
                continue
            for q in polygons_of(p):
                if self._ok(q):
                    raw.append((q, {g.raw_id}, g.layer, g.kind, 0.0))
        for h in ctx.hatch_regions:
            if h.cls == "column_fill" and h.role == "linework" and self._ok(h.polygon):
                raw.append((h.polygon, {h.raw_id}, h.layer, "fill", 0.0))
        for ref in drawing.inserts.values():
            if ref.role != "column":
                continue
            pts = [tuple(p) for gid in ref.child_geo_ids if gid in geo and geo[gid].points is not None
                   and geo[gid].kind not in ("text", "dimension") for p in geo[gid].points]
            if len(pts) >= 3:
                hull = MultiPoint(pts).convex_hull
                if isinstance(hull, Polygon) and self._ok(hull):
                    raw.append((hull, {ref.id}, ref.layer, "symbol", min(1.0, ref.role_scores.get("column", 0))))
        # closed loops of short loose lines (4-line squares)
        hi = self.cfg["detection"]["column"]["size_m"][1]
        short = [st.line() for st in ctx.strokes if st.length <= hi * 1.05]
        if short:
            noded = robust_union(short)
            for face in polygonize(noded):
                if self._ok(face) and face.area / max(face.minimum_rotated_rectangle.area, 1e-12) > 0.9:
                    srcs = set()
                    covered = 0.0
                    ring = face.exterior.buffer(tol.snap * 2)
                    for st in ctx.strokes:
                        if st.length <= hi * 1.05 and st.line().within(ring):
                            srcs |= st.raw_ids
                            covered += st.length
                    # the loop must be drawn by its own short lines, not cut out of longer wall faces
                    if srcs and covered >= 0.9 * face.exterior.length:
                        raw.append((face, srcs, "", "loop", 0.0))
        # merge duplicates (same footprint)
        merged: list[dict] = []
        for poly, srcs, layer, origin, nh in raw:
            for m in merged:
                inter = robust_intersection(m["poly"], poly).area
                if inter >= 0.6 * min(m["poly"].area, poly.area):
                    m["srcs"] |= srcs
                    m["origins"].add(origin)
                    m["layers"].add(layer)
                    m["name_hint"] = max(m["name_hint"], nh)
                    break
            else:
                merged.append({"poly": poly, "srcs": set(srcs), "origins": {origin}, "layers": {layer}, "name_hint": nh})
        fills = [h.polygon for h in ctx.hatch_regions if h.cls in ("column_fill", "wall_fill")]
        fill_u = robust_union(fills) if fills else None
        sizes = []
        for m in merged:
            w, L = self._ok(m["poly"])
            m["size"] = (w, L)
            sizes.append((w, L))
        out = []
        for m in merged:
            p = m["poly"]
            w, L = m["size"]
            c = p.centroid
            col = Column(id=ctx.ids.new("column"), polygon=p, size=(w, L))
            circ = 4 * math.pi * p.area / max(p.length ** 2, 1e-12)
            col.shape = "circular" if circ > 0.95 else ("rectangular" if p.area / max(w * L, 1e-12) > 0.9 else "polygonal")
            col.source_entities |= m["srcs"]
            col.add_evidence("size_plausibility", 1.0)
            col.add_evidence("shape", 1.0 if col.shape != "polygonal" else 0.5)
            filled = "fill" in m["origins"] or (fill_u is not None and robust_intersection(fill_u, p).area >= 0.8 * p.area)
            col.add_evidence("filled", 1.0 if filled else 0.0)
            lay = max((ctx.layer_score(l, "column") for l in m["layers"] if l), default=0.0)
            furn = max((ctx.layer_score(l, "furniture") for l in m["layers"] if l), default=0.0)
            col.add_evidence("layer_match", lay if lay > 0 else (0.5 if not furn else 0.0))
            col.add_evidence("not_furniture", 1.0 - furn)
            if m["name_hint"]:
                col.add_evidence("name_hint", m["name_hint"])
            same = sum(1 for (w2, L2) in sizes if abs(w2 - w) <= 0.03 * w + tol.snap and abs(L2 - L) <= 0.03 * L + tol.snap) - 1
            col.add_evidence("repetition", min(1.0, same / 2.0))
            aligned = 0
            for o in merged:
                if o is m:
                    continue
                oc = o["poly"].centroid
                if abs(oc.x - c.x) <= tol.snap * 4 or abs(oc.y - c.y) <= tol.snap * 4:
                    aligned += 1
            col.add_evidence("grid_alignment", min(1.0, aligned / 2.0))
            out.append(col)
        return self.finalize(out)
