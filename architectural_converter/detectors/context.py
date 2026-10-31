"""Drawing context (Stage 3 deliverable): semantic hints and statistics shared by detectors."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from ..geometry.geometric_utils import inscribed_width, mrr_dims, polygons_of, thin_fraction, robust_difference
from ..geometry.geometry_adapter import ArcPrim, Segment, Stroke, build_arcs, build_strokes
from ..models.base_entity import IdFactory
from ..preprocessing.layer_analyzer import LayerProfile, analyze_layers
from ..preprocessing.tolerance import Tolerances
from .symbol_analysis import analyze_symbols
from .text_detector import TextDetector

LINEWORK_ROLES = ("linework",)


@dataclass
class HatchRegion:
    geo_id: str
    raw_id: str
    layer: str
    polygon: Polygon
    cls: str              # wall_fill | column_fill | area_fill | other
    pattern: str
    solid: bool
    thin_fraction: float
    role: str


@dataclass
class DrawingContext:
    cfg: Any
    tol: Tolerances
    segments: list[Segment]
    strokes: list[Stroke] = field(default_factory=list)
    stroke_tree: STRtree | None = None
    arcs: list[ArcPrim] = field(default_factory=list)
    layers: dict[str, LayerProfile] = field(default_factory=dict)
    symbol_roles: dict[str, str] = field(default_factory=dict)
    labels: list = field(default_factory=list)
    dimensions: list[dict] = field(default_factory=list)
    hatch_regions: list[HatchRegion] = field(default_factory=list)
    area_outlines: list[dict] = field(default_factory=list)
    thickness_clusters: list[dict] = field(default_factory=list)
    pairs: list = field(default_factory=list)
    obstructions: list = field(default_factory=list)
    candidates: dict[str, list] = field(default_factory=dict)
    ids: IdFactory = field(default_factory=IdFactory)
    stats: dict[str, Any] = field(default_factory=dict)

    # convenience ----------------------------------------------------------------
    def wall_evidence(self, layer: str) -> float:
        p = self.layers.get(layer)
        return p.wall_evidence if p else self.cfg["layer_semantics"]["wall_evidence"]["_unknown"]

    def layer_score(self, layer: str, cat: str) -> float:
        p = self.layers.get(layer)
        return p.category_scores.get(cat, 0.0) if p else 0.0

    def layer_primary(self, layer: str) -> str | None:
        p = self.layers.get(layer)
        return p.primary if p else None

    def strokes_wall_evidence(self, idxs) -> float:
        vals = []
        for k in idxs:
            st = self.strokes[k]
            vals.append(np.mean([self.wall_evidence(ly) for ly in st.layers]))
        return float(np.mean(vals)) if vals else 0.5

    def stroke_raw_ids(self, idxs) -> set[str]:
        out: set[str] = set()
        for k in idxs:
            out |= self.strokes[k].raw_ids
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "tolerances": self.tol.to_dict(),
            "layers": {k: v.to_dict() for k, v in sorted(self.layers.items())},
            "symbol_roles": _count(self.symbol_roles.values()),
            "labels": len(self.labels),
            "dimensions": len(self.dimensions),
            "hatch_regions": _count(h.cls for h in self.hatch_regions),
            "thickness_clusters": self.thickness_clusters,
            "strokes": len(self.strokes),
            "segments": len(self.segments),
            "stats": self.stats,
        }


def _count(it):
    out: dict[str, int] = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def loops_to_polygons(loops) -> list[Polygon]:
    """Even-odd nesting of boundary loops into polygons with holes."""
    polys = []
    for lp in loops:
        try:
            p = Polygon(lp)
            if not p.is_valid:
                p = p.buffer(0)
            polys.extend(polygons_of(p))
        except Exception:
            continue
    polys = [p for p in polys if p.area > 0]
    polys.sort(key=lambda p: -p.area)
    depth = []
    for i, p in enumerate(polys):
        rp = p.representative_point()
        depth.append(sum(1 for j in range(i) if polys[j].contains(rp)))
    shells = []
    for i, p in enumerate(polys):
        if depth[i] % 2 == 0:
            shells.append(p)
        else:
            for k in range(len(shells) - 1, -1, -1):
                if shells[k].contains(p.representative_point()):
                    shells[k] = robust_difference(shells[k], p)
                    break
    out = []
    for s in shells:
        out.extend(polygons_of(s))
    return out


def analyze_fills(drawing, cfg, tol) -> list[HatchRegion]:
    regions = []
    tmax = tol.thickness_max
    clo, chi = cfg["detection"]["column"]["size_m"]
    for g in drawing.entities:
        if not g.valid:
            continue
        if g.kind == "hatch":
            polys = loops_to_polygons(g.data.get("loops", []))
            pattern, solid = g.data.get("pattern", ""), bool(g.data.get("solid"))
        elif g.kind == "polygon" and g.data.get("filled"):
            polys = polygons_of(Polygon(g.points).buffer(0))
            pattern, solid = "SOLID", True
        else:
            continue
        for poly in polys:
            if poly.area <= tol.duplicate ** 2:
                continue
            tf = thin_fraction(poly, tmax / 2 * 1.05)
            w, L, _, _ = mrr_dims(poly)
            if clo <= w and L <= chi and L / max(w, 1e-9) <= cfg["detection"]["column"]["max_aspect"] \
                    and poly.area >= 0.8 * w * L * 0.78:
                cls = "column_fill"
            elif tf >= cfg["detection"]["hatch"]["thin_fraction_min"] and inscribed_width(poly) <= tmax * 1.05:
                cls = "wall_fill"
            elif tf >= cfg["detection"]["hatch"]["thin_fraction_min"]:
                cls = "wall_fill"
            elif inscribed_width(poly) > 1.2:
                cls = "area_fill"
            else:
                cls = "other"
            regions.append(HatchRegion(geo_id=g.id, raw_id=g.raw_id, layer=g.layer, polygon=poly, cls=cls,
                                       pattern=pattern, solid=solid, thin_fraction=tf, role=g.role))
    return regions


def collect_dimensions(drawing) -> list[dict]:
    out = []
    for g in drawing.entities:
        if g.kind != "dimension" or not g.valid:
            continue
        m = g.data.get("measurement")
        if m is None:
            continue
        out.append({"raw_id": g.raw_id, "measurement": float(m), "dimtype": g.data.get("dimtype"),
                    "defpoints": g.data.get("defpoints", {}), "layer": g.layer})
    return out


def collect_area_outlines(drawing, tol) -> list[dict]:
    """Large closed non-thin polylines: possible room / floor outlines (context only)."""
    out = []
    for g in drawing.entities:
        if g.kind == "polyline" and g.closed and g.valid and g.role == "linework" and len(g.points) >= 4:
            p = Polygon(g.points)
            if not p.is_valid:
                p = p.buffer(0)
            if p.area > 2.0 and inscribed_width(p) > tol.thickness_max * 1.5:
                out.append({"geo_id": g.id, "raw_id": g.raw_id, "layer": g.layer, "polygon": p})
    return out


def build_context(drawing, segments: list[Segment], tol: Tolerances, cfg) -> DrawingContext:
    ctx = DrawingContext(cfg=cfg, tol=tol, segments=segments)
    ctx.layers = analyze_layers(drawing, segments, cfg, tol)
    ctx.symbol_roles = analyze_symbols(drawing, cfg, tol)
    geo = drawing.geo_by_id()
    for s in segments:
        s.role = geo[s.geo_id].role
    ctx.labels = TextDetector(cfg).detect(drawing, ctx)
    ctx.dimensions = collect_dimensions(drawing)
    ctx.hatch_regions = analyze_fills(drawing, cfg, tol)
    ctx.area_outlines = collect_area_outlines(drawing, tol)
    live = [s for s in segments if s.role in LINEWORK_ROLES and s.dup_of is None
            and ctx.layer_primary(s.layer) not in ("dimension", "annotation", "ignore")]
    ctx.strokes = build_strokes(live, tol.snap, tol.angle)
    ctx.stroke_tree = STRtree([st.line() for st in ctx.strokes]) if ctx.strokes else None
    ctx.arcs = build_arcs(drawing)
    for a in ctx.arcs:
        a.role = geo[a.geo_id].role
    frag = sum(1 for st in ctx.strokes if len(st.members) > 1)
    ctx.stats["fragmented_strokes"] = frag
    ctx.stats["segments_in_fragmented_strokes"] = sum(len(st.members) for st in ctx.strokes if len(st.members) > 1)
    return ctx
