"""Architectural connectivity graphs (NetworkX + Shapely).

* ``build_axis_graph`` - planar graph of wall centre lines: axes are noded at their
  intersections, nearby nodes are merged (snap tolerance), edges carry the wall id.
  Degree-1 nodes are free wall ends.
* ``build_wall_graph`` - one node per wall, an edge where wall bodies touch (junction).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

from ..geometry.endpoint_processor import cluster_centers, cluster_points


def build_axis_graph(walls, tol) -> nx.Graph:
    G = nx.Graph()
    axes = [(w.id, w.axis) for w in walls if not w.axis.is_empty and w.axis.length > 0]
    if not axes:
        return G
    noded = unary_union([a for _, a in axes])
    pieces = list(getattr(noded, "geoms", [noded]))
    pieces = [p for p in pieces if isinstance(p, LineString) and p.length > 0]
    if not pieces:
        return G
    P = np.array([c for p in pieces for c in (p.coords[0], p.coords[-1])])
    labels = cluster_points(P, max(tol.snap, 1e-6))
    centers = cluster_centers(P, labels)
    tree = STRtree([a for _, a in axes])
    for k, p in enumerate(pieces):
        a, b = labels[2 * k], labels[2 * k + 1]
        if a == b:
            continue
        mid = p.interpolate(0.5, normalized=True)
        wid = None
        best = None
        for i in tree.query(mid.buffer(tol.snap)):
            d = axes[int(i)][1].distance(mid)
            if best is None or d < best:
                best, wid = d, axes[int(i)][0]
        G.add_node(int(a), pos=tuple(centers[a]))
        G.add_node(int(b), pos=tuple(centers[b]))
        G.add_edge(int(a), int(b), wall=wid, length=p.length)
    return G


def build_wall_graph(walls, tol) -> nx.Graph:
    G = nx.Graph()
    polys = []
    for w in walls:
        G.add_node(w.id, kind=w.kind, status=w.status.value)
        polys.append(w.polygon(tol.snap * 2))
    idx = [i for i, p in enumerate(polys) if p is not None]
    if not idx:
        return G
    tree = STRtree([polys[i] for i in idx])
    for a_pos, i in enumerate(idx):
        pa = polys[i].buffer(tol.snap * 1.5)
        for j_pos in tree.query(pa):
            j = idx[int(j_pos)]
            if j <= i:
                continue
            if pa.intersects(polys[j]):
                inter = pa.intersection(polys[j])
                c = inter.centroid
                G.add_edge(walls[i].id, walls[j].id, point=(c.x, c.y))
    return G


def free_ends(walls, tol):
    """Wall ends not touching any other wall body: list of (wall, end, point)."""
    out = []
    polys = {w.id: w.polygon(tol.snap * 2) for w in walls}
    items = [(w, p) for w in walls for p in [polys[w.id]] if p is not None]
    tree = STRtree([p for _, p in items]) if items else None
    for w in walls:
        if w.kind not in ("straight", "centerline"):
            continue
        for e in (0, 1):
            pt = Point(w.point_at(w.s1 if e else w.s0))
            ok = False
            if tree is not None:
                for k in tree.query(pt.buffer(tol.snap * 2)):
                    v, p = items[int(k)]
                    if v is not w and p.distance(pt) <= tol.snap * 2:
                        ok = True
                        break
            if not ok:
                out.append((w, e, (pt.x, pt.y)))
    return out
