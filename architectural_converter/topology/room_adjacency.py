"""Room <-> wall, room <-> opening and room <-> room relationships."""
from __future__ import annotations

import networkx as nx
from shapely.strtree import STRtree

from ..geometry.geometric_utils import rect_polygon


def relate_rooms(rooms, walls, openings, tol) -> nx.Graph:
    G = nx.Graph()
    for r in rooms:
        G.add_node(r.id, name=r.name, area=r.area)
    if not rooms:
        return G
    tree = STRtree([r.polygon for r in rooms])
    for w in walls:
        p = w.polygon(max(tol.snap * 2, 0.01))
        if p is None:
            continue
        touch = []
        for k in tree.query(p.buffer(tol.snap * 4)):
            r = rooms[int(k)]
            shared = r.polygon.exterior.intersection(p.buffer(tol.snap * 4)).length
            if shared > max(0.1, tol.snap * 10):
                touch.append(r)
                r.bounding_walls.add(w.id)
                w.rooms.add(r.id)
        for i in range(len(touch)):
            for j in range(i + 1, len(touch)):
                a, b = touch[i].id, touch[j].id
                touch[i].adjacent_rooms.add(b)
                touch[j].adjacent_rooms.add(a)
                if G.has_edge(a, b):
                    G[a][b]["walls"].append(w.id)
                else:
                    G.add_edge(a, b, walls=[w.id], openings=[])
    wmap = {w.id: w for w in walls}
    for o in openings:
        if o.host_wall not in wmap or not o.host_interval:
            continue
        w = wmap[o.host_wall]
        span = rect_polygon(w.u, w.c, o.host_interval[0], o.host_interval[1],
                            (w.thickness or tol.snap * 2) + tol.snap * 8)
        hit = [rooms[int(k)] for k in tree.query(span) if rooms[int(k)].polygon.intersects(span)]
        o.rooms = [r.id for r in hit]
        for r in hit:
            r.openings.add(o.id)
        for i in range(len(hit)):
            for j in range(i + 1, len(hit)):
                a, b = hit[i], hit[j]
                if o.type in ("door", "opening"):
                    a.connected_rooms.add(b.id)
                    b.connected_rooms.add(a.id)
                if G.has_edge(a.id, b.id):
                    G[a.id][b.id]["openings"].append(o.id)
                else:
                    G.add_edge(a.id, b.id, walls=[], openings=[o.id])
    return G
