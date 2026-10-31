"""Topology-consistency evidence for walls (second scoring pass).

A wall whose both ends attach to other walls (or columns), or which hosts openings, and which
belongs to the main wall network is topologically consistent; isolated pieces (furniture
outlines, stray pairs) score low."""
from __future__ import annotations

import networkx as nx
from shapely.geometry import Point
from shapely.strtree import STRtree

from ..classification.candidate_resolver import same_band
from ..geometry.geometric_utils import dot
from ..models.base_entity import Status
from .graph_builder import build_wall_graph


def topology_evidence(walls, columns, tol):
    G = build_wall_graph(walls, tol)
    size = {}
    for comp in nx.connected_components(G):
        for n in comp:
            size[n] = len(comp)
    largest = max(size.values()) if size else 1
    bodies = [(w, w.polygon(tol.snap * 2)) for w in walls]
    bodies = [(w, p) for w, p in bodies if p is not None]
    bodies += [(c, c.polygon.buffer(tol.snap * 2)) for c in columns
               if c.polygon is not None and c.status != Status.REJECTED]
    tree = STRtree([p for _, p in bodies]) if bodies else None
    for w in walls:
        w.connections = set(G.neighbors(w.id)) if w.id in G else set()
        if w.kind not in ("straight", "centerline"):
            w.set_evidence("topology_consistency", 1.0 if w.connections else 0.3)
            continue
        ends = 0
        for e in (0, 1):
            pt = Point(w.point_at(w.s1 if e else w.s0))
            hit = False
            if tree is not None:
                for k in tree.query(pt.buffer(tol.snap * 2)):
                    v, p = bodies[int(k)]
                    if v is w or p.distance(pt) > tol.snap * 2:
                        continue
                    # lying alongside a parallel wall is not a junction (only a collinear continuation is)
                    if getattr(v, "kind", "") in ("straight", "centerline") and abs(dot(v.u, w.u)) > 0.95 \
                            and not same_band(v, w, tol):
                        continue
                    hit = True
                    break
            ends += hit
        comp_share = size.get(w.id, 1) / largest
        attach = 1.0 if (w.openings or ends == 2) else 0.5 * (ends / 2)
        w.set_evidence("topology_consistency", 0.35 * (ends / 2) + 0.35 * attach + 0.3 * min(1.0, comp_share * 3))
    return G
