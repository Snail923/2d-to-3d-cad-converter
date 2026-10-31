"""Detector B: polyline-based wall representations.

* closed thin polylines (wall outlines) - distinguished from columns (compact), furniture
  and floor/room outlines (not thin) by area, aspect, erosion thinness and layer hints;
* constant-width polylines (the width *is* the wall thickness);
* MLINE entities (style width x scale is the thickness).
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon

from ..geometry.geometric_utils import (canonical_dir, dot, mrr_dims, normal, polygons_of, thin_fraction)
from ..models.wall import Wall
from .base_detector import BaseDetector, register
from ..geometry.segment_pairing import cluster_consistency
from .wall_common import aspect_score, thickness_plausibility


def _consistency(t, ctx, cfg):
    th = cfg["thickness"]
    return cluster_consistency(t, ctx.thickness_clusters, th["cluster_rel_tol"], th["cluster_abs_tol_m"]) if t else 0.0


@register
class PolylineWallDetector(BaseDetector):
    name = "polyline_wall_detector"
    produces = "wall"
    requires = ("parallel_wall_detector",)

    def detect(self, drawing, ctx):
        out = []
        out += self._closed_outlines(drawing, ctx)
        out += self._wide_polylines(drawing, ctx)
        out += self._mlines(drawing, ctx)
        return self.finalize(out)

    def _closed_outlines(self, drawing, ctx):
        cfg = self.cfg
        pc = cfg["detection"]["polyline"]
        tol = ctx.tol
        out = []
        for g in drawing.entities:
            if g.kind != "polyline" or not g.closed or not g.valid or g.role != "linework":
                continue
            if g.data.get("width", 0) > 0:
                continue
            poly = Polygon(g.points)
            if not poly.is_valid:
                poly = poly.buffer(0)
            for p in polygons_of(poly):
                if p.area <= tol.thickness_min ** 2:
                    continue
                per = p.exterior.length
                t_est = 2 * p.area / max(per, 1e-9)
                if not (tol.thickness_min * 0.8 <= t_est <= tol.thickness_max):
                    continue
                tf = thin_fraction(p, tol.thickness_max / 2 * 1.05)
                w_, L, u, cen = mrr_dims(p)
                if L / max(t_est, 1e-9) < pc["min_aspect"] or tf < 1 - pc["max_thin_residual"]:
                    continue
                rect_like = p.area / max(w_ * L, 1e-12) > 0.95
                if rect_like:
                    n = normal(u)
                    c = dot(cen, n)
                    s_c = dot(cen, u)
                    wall = Wall(id=ctx.ids.new("wall"), kind="straight", u=u, c=c, s0=s_c - L / 2, s1=s_c + L / 2,
                                thickness=w_, detected_interval=(s_c - L / 2, s_c + L / 2))
                else:
                    wall = Wall(id=ctx.ids.new("wall"), kind="polygon", shape=p, thickness=t_est)
                wall.source_entities.add(g.raw_id)
                wall.add_evidence("closed_outline", 1.0)
                wall.add_evidence("thinness", tf)
                wall.add_evidence("aspect", aspect_score(L, t_est))
                wall.add_evidence("thickness_plausibility", thickness_plausibility(wall.thickness, cfg))
                wall.add_evidence("thickness_consistency", _consistency(wall.thickness, ctx, cfg))
                wall.add_evidence("layer_match", ctx.wall_evidence(g.layer))
                out.append(wall)
        return out

    def _wide_polylines(self, drawing, ctx):
        cfg = self.cfg
        tol = ctx.tol
        out = []
        for g in drawing.entities:
            if g.kind != "polyline" or not g.valid or g.role != "linework":
                continue
            wdt = float(g.data.get("width", 0) or 0)
            if not (tol.thickness_min <= wdt <= tol.thickness_max):
                continue
            pts = g.points
            seq = list(pts) + ([pts[0]] if g.closed else [])
            for a, b in zip(seq[:-1], seq[1:]):
                d = np.asarray(b) - np.asarray(a)
                L = float(np.hypot(*d))
                if L <= tol.snap:
                    continue
                u = canonical_dir(float(d[0]), float(d[1]))
                n = normal(u)
                s0, s1 = sorted((dot(a, u), dot(b, u)))
                w = Wall(id=ctx.ids.new("wall"), kind="straight", u=u, c=dot(a, n), s0=s0, s1=s1, thickness=wdt,
                         thickness_source="polyline_width", detected_interval=(s0, s1))
                w.source_entities.add(g.raw_id)
                w.add_evidence("polyline_width", 1.0)
                w.add_evidence("thickness_plausibility", thickness_plausibility(wdt, cfg))
                w.add_evidence("aspect", aspect_score(L, wdt))
                w.add_evidence("layer_match", ctx.wall_evidence(g.layer))
                w.notes.append("thickness from polyline constant width")
                out.append(w)
        return out

    def _mlines(self, drawing, ctx):
        cfg = self.cfg
        tol = ctx.tol
        geo_children: dict[str, list] = {}
        for g in drawing.entities:
            if g.parent and g.kind == "line":
                geo_children.setdefault(g.parent, []).append(g)
        out = []
        for g in drawing.entities:
            if g.kind != "mline" or not g.valid or g.role != "linework":
                continue
            kids = geo_children.get(g.id, [])
            pts = g.points
            seq = list(pts) + ([pts[0]] if g.closed else [])
            for a, b in zip(seq[:-1], seq[1:]):
                d = np.asarray(b) - np.asarray(a)
                L = float(np.hypot(*d))
                if L <= tol.snap:
                    continue
                u = canonical_dir(float(d[0]), float(d[1]))
                n = normal(u)
                offs = []
                for k in kids:
                    kd = k.points[1] - k.points[0]
                    if abs(abs(dot(canonical_dir(float(kd[0]), float(kd[1])), u)) - 1) > 1e-4:
                        continue
                    mid = (k.points[0] + k.points[1]) / 2
                    s = dot(mid, u)
                    if min(dot(a, u), dot(b, u)) - tol.snap <= s <= max(dot(a, u), dot(b, u)) + tol.snap:
                        offs.append(dot(mid, n))
                if len(offs) >= 2:
                    c, t = (max(offs) + min(offs)) / 2, max(offs) - min(offs)
                else:
                    c, t = dot(a, n), float(g.data.get("width", 0) or 0)
                if not (tol.thickness_min <= t <= tol.thickness_max):
                    continue
                s0, s1 = sorted((dot(a, u), dot(b, u)))
                w = Wall(id=ctx.ids.new("wall"), kind="straight", u=u, c=c, s0=s0, s1=s1, thickness=t,
                         thickness_source="mline", detected_interval=(s0, s1))
                w.source_entities.add(g.raw_id)
                w.add_evidence("mline", 1.0)
                w.add_evidence("thickness_plausibility", thickness_plausibility(t, cfg))
                w.add_evidence("layer_match", ctx.wall_evidence(g.layer))
                out.append(w)
        return out
