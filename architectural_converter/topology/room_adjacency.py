"""Room <-> wall, room <-> opening and room <-> room relationships."""
from __future__ import annotations

import networkx as nx
from shapely.geometry import LineString
from shapely.strtree import STRtree

from ..geometry.geometric_utils import rect_polygon, robust_intersection, sample_points, bbox_pad


def relate_rooms(rooms, walls, openings, tol) -> nx.Graph:
    G = nx.Graph()
    for r in rooms:
        G.add_node(r.id, name=r.name, area=r.area)
    if not rooms:
        return G
    import shapely
    tree = STRtree([r.polygon for r in rooms])
    step = 0.1
    samples = [shapely.points(sample_points(r.polygon.exterior, step)) for r in rooms]
    d = tol.snap * 4
    for w in walls:
        p = w.polygon(max(tol.snap * 2, 0.01))
        if p is None:
            continue
        touch = []
        for k in tree.query(bbox_pad(p, d)):
            r = rooms[int(k)]
            shared = float((shapely.distance(samples[int(k)], p) <= d).sum()) * step
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
    # an opening may sit in one line of a wall drawn as several parallel candidates (facade +
    # render line, double partition): when the probe across the host wall finds no room on one
    # side, look past the whole wall stack on that side and take the nearest room there
    max_t = max([w.thickness or 0.0 for w in walls] + [tol.snap * 2])
    for o in openings:
        if o.host_wall not in wmap or not o.host_interval:
            continue
        w = wmap[o.host_wall]
        t = w.thickness or tol.snap * 2
        a, b = o.host_interval
        span = rect_polygon(w.u, w.c, a, b, t + tol.snap * 8)
        hit = [rooms[int(k)] for k in tree.query(span) if rooms[int(k)].polygon.intersects(span)]
        axis = LineString([w.point_at(a), w.point_at(b)])
        for sgn in (1.0, -1.0):
            near = rect_polygon(w.u, w.c + sgn * (t / 2 + tol.snap * 2), a, b, tol.snap * 4)
            if any(r.polygon.intersects(near) for r in hit):
                continue
            reach = t / 2 + 2 * max_t + tol.snap * 4
            side = rect_polygon(w.u, w.c + sgn * reach / 2, a, b, reach)
            best = None
            for k in tree.query(side):
                r = rooms[int(k)]
                part = robust_intersection(r.polygon, side)
                if part.is_empty or part.area <= 0:
                    continue
                d = part.distance(axis)
                if best is None or d < best[0]:
                    best = (d, r)
            if best is not None and best[1] not in hit:
                hit.append(best[1])
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
