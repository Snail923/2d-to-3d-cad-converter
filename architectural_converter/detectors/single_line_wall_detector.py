"""Detector E: single-line (centre-line) walls.

A single line is never converted into a wall by default. It is evaluated only when it is
not already explained as a boundary of a geometric wall, and it needs supporting evidence:
* layer semantics,
* network connectivity (both ends attached to other lines or to detected walls),
* participation in enclosed room-sized faces of the line network,
* membership in a significant connected network (gap-tolerant along the line direction,
  so door gaps do not split a wall network); small isolated networks (furniture, symbols)
  are excluded.
Thickness is left unknown (``None``) - it is never invented here."""
from __future__ import annotations

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union

from ..geometry.geometric_utils import dot, inscribed_width
from ..models.base_entity import Status
from ..models.wall import Wall
from .base_detector import BaseDetector, register

GEOMETRIC = ("parallel_wall_detector", "polyline_wall_detector", "hatch_wall_detector")


@register
class SingleLineWallDetector(BaseDetector):
    name = "single_line_wall_detector"
    produces = "wall"
    requires = GEOMETRIC + ("layer_wall_detector", "door_detector", "window_detector", "column_detector")

    def detect(self, drawing, ctx):
        cfg = self.cfg["detection"]["single_line"]
        tol = ctx.tol
        geom_walls = [w for d in GEOMETRIC for w in ctx.candidates.get(d, []) if w.status != Status.REJECTED]
        faces_used = set()
        for w in geom_walls:
            faces_used |= w.face_segments
        wall_polys = [p for p in (w.polygon() for w in geom_walls) if p is not None]
        wall_union = unary_union(wall_polys).buffer(tol.snap) if wall_polys else None
        symbol_fp = [o.footprint for d in ("door_detector", "window_detector") for o in ctx.candidates.get(d, [])
                     if o.footprint is not None and o.status != Status.REJECTED]
        symbol_union = unary_union(symbol_fp).buffer(tol.snap) if symbol_fp else None
        # drawing convention: if most wall-like linework is already explained by double-line
        # walls, an unpaired single line needs explicit wall-layer evidence
        tot = sum(st.length for st in ctx.strokes if np.mean([ctx.wall_evidence(l) for l in st.layers]) >= 0.3)
        paired = sum(st.length for st in ctx.strokes if any(m in faces_used for m in st.members))
        share = paired / tot if tot > 0 else 0.0
        strict = share >= cfg["double_line_share_for_strict_mode"]
        ctx.stats["double_line_share"] = round(share, 3)
        cand = []
        for k, st in enumerate(ctx.strokes):
            if st.length < cfg["min_length_m"]:
                continue
            if any(m in faces_used for m in st.members):
                continue
            lay = float(np.mean([ctx.wall_evidence(l) for l in st.layers]))
            if lay < 0.3 or (strict and lay < 0.9):
                continue
            ln = st.line()
            if wall_union is not None and ln.within(wall_union):
                continue
            if symbol_union is not None and ln.intersection(symbol_union).length > 0.5 * ln.length:
                continue
            cand.append((k, st, lay))
        if not cand:
            return []
        lines = [st.line() for _, st, _ in cand]
        # ---- connectivity graph with gap-tolerant bridging along the line direction
        max_gap = self.cfg["reconstruction"]["opening_width_m"][1]
        G = nx.Graph()
        for a in range(len(cand)):
            G.add_node(a)
        for a in range(len(cand)):
            la = lines[a].buffer(tol.snap * 2)
            for b in range(a + 1, len(cand)):
                if la.intersects(lines[b]):
                    G.add_edge(a, b, kind="touch")
        ends_conn = np.zeros((len(cand), 2), dtype=float)
        bridges = []
        for a, (k, st, lay) in enumerate(cand):
            for e, (p, out) in enumerate(((st.p0, -1.0), (st.p1, 1.0))):
                pt = Point(p)
                for b in range(len(cand)):
                    if b != a and lines[b].distance(pt) <= tol.snap * 2:
                        ends_conn[a, e] = 1.0
                        break
                if not ends_conn[a, e] and wall_union is not None and wall_union.buffer(tol.snap).contains(pt):
                    ends_conn[a, e] = 1.0
                if not ends_conn[a, e]:
                    ray = LineString([p, (p[0] + out * st.u[0] * max_gap, p[1] + out * st.u[1] * max_gap)])
                    best = None
                    for b in range(len(cand)):
                        if b == a or lines[b].distance(ray) > tol.snap * 2:
                            continue
                        inter = lines[b].buffer(tol.snap * 2).intersection(ray)
                        d = pt.distance(inter)
                        if best is None or d < best[0]:
                            best = (d, b)
                    if best is not None:
                        d, b = best
                        G.add_edge(a, b, kind="gap")
                        # a bridged opening-sized gap along the wall direction: possible opening
                        bridges.append(LineString([p, (p[0] + out * st.u[0] * d, p[1] + out * st.u[1] * d)]))
                        ends_conn[a, e] = 0.75
        comps = list(nx.connected_components(G))
        comp_len = {}
        for ci, comp in enumerate(comps):
            L = sum(cand[a][1].length for a in comp)
            for a in comp:
                comp_len[a] = L
        largest = max(comp_len.values())
        # ---- enclosed faces of the network (+ geometric walls as boundaries)
        net = [ln for ln in lines] + bridges
        if wall_union is not None:
            net.append(wall_union.boundary)
        faces = [f for f in polygonize(unary_union(net)) if f.area >= cfg["min_room_face_area_m2"]
                 and inscribed_width(f) >= 0.6]
        face_bnd = unary_union([f.exterior for f in faces]).buffer(tol.snap * 2) if faces else None
        out = []
        for a, (k, st, lay) in enumerate(cand):
            share = comp_len[a] / largest
            if share < cfg["min_component_share"] and lay < 0.9:
                continue
            ln = lines[a]
            cyc = ln.intersection(face_bnd).length / ln.length if face_bnd is not None else 0.0
            conn = float(min(ends_conn[a, 0], 1.0) + min(ends_conn[a, 1], 1.0)) / 2.0
            if cyc < 0.5 and lay < 0.9 and conn < 1.0:
                continue
            w = Wall(id=ctx.ids.new("wall"), kind="centerline", u=st.u, c=st.c, s0=st.s0, s1=st.s1,
                     thickness=None, thickness_source="unknown", detected_interval=(st.s0, st.s1))
            w.face_segments = set(st.members)
            w.source_entities = set(st.raw_ids)
            w.add_evidence("layer_match", lay)
            w.add_evidence("network_connectivity", conn)
            w.add_evidence("cycle_support", min(1.0, cyc))
            w.notes.append("single-line wall: thickness unknown")
            out.append(w)
        return self.finalize(out)
