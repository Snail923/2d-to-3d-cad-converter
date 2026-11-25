"""Wall structure: walls belong to a building.

The model calls a line "wall" from its local look; double lines of a site boundary, a kerb, a
furniture outline or a detail drawing look the same. In the built result the difference shows:
the walls of a building form groups that enclose rooms and carry doors and windows.

After rooms are known, every group of connected walls is checked. A group none of whose walls
bounds a room, hosts a door or window, or is a balcony railing closes nothing:

* next to rooms (the building was recognised there) the group is not part of it: rejected,
  whatever its size (plot limits, road and kerb lines, furniture);
* away from every room, small and medium groups are rejected (detail drawings, site lines);
* large groups away from every room are kept for review only when they look like a floor whose
  rooms could not be closed: several of their walls are junctions (three or more walls meet).
  A loop or chain of walls, each joined only to the next (a plot outline, a sheet border, a road
  edge), is rejected.

One more case is the empty enclosure: a wall group that closes around a single large space with
no door, no window and no room name, while the drawing has real rooms elsewhere, is a plot or
sheet outline; that "room" and its walls are rejected.

Other rooms are not affected (the removed walls bound none), so nothing has to be recomputed."""
from __future__ import annotations

import networkx as nx
from shapely.geometry import box as _box
from shapely.strtree import STRtree

from architectural_converter.models.base_entity import Status
from architectural_converter.topology.graph_builder import build_wall_graph

from . import options


def _live(e):
    return e is not None and e.status != Status.REJECTED


def prune_wall_groups(building, tol, cfg) -> dict:
    o = options(cfg, "wall_structure")
    if not o.get("enabled", True):
        return {"skipped": "disabled"}
    b = building
    walls = [w for w in b.walls.values() if _live(w) and not w.is_hypothesis]
    rooms = [r for r in b.rooms.values() if _live(r) and r.polygon is not None and not r.is_hypothesis]
    log = {"groups": 0, "rejected_walls": 0, "review_walls": 0, "rejected_length_m": 0.0}
    if not walls or not rooms:
        return {**log, "skipped": "no rooms"}           # nothing to compare against: keep everything
    bound = {wid for r in rooms for wid in r.bounding_walls}
    hosts = {o_.host_wall for o_ in b.openings.values() if _live(o_) and o_.type in ("door", "window") and o_.host_wall}
    G = build_wall_graph(walls, tol)
    by_id = {w.id: w for w in walls}
    tree = STRtree([r.polygon for r in rooms])
    max_walls, max_len = int(o.get("small_group_max_walls", 6)), float(o.get("small_group_max_length_m", 20.0))
    far = float(o.get("far_from_rooms_m", 3.0))
    far_walls, far_len = int(o.get("far_group_max_walls", 20)), float(o.get("far_group_max_length_m", 60.0))
    min_junctions, junction_share = int(o.get("floor_min_junction_walls", 3)), float(o.get("floor_min_junction_share", 0.15))
    comps = [[by_id[i] for i in comp if i in by_id] for comp in nx.connected_components(G)]
    # empty enclosures: one large unnamed space, no opening in any of its walls
    big = float(o.get("empty_enclosure_min_area_m2", 150.0))
    log["empty_enclosures"] = 0
    if len(rooms) > 1:
        for ws in comps:
            ids_ = {w.id for w in ws}
            if not ws or ids_ & hosts or any(w.manual_status is not None for w in ws):
                continue
            mine = [r for r in rooms if _live(r) and r.bounding_walls & ids_]
            if len(mine) == 1 and mine[0].manual_status is None and not mine[0].label_texts \
                    and mine[0].area >= big and set(mine[0].bounding_walls) <= ids_ \
                    and not any(_live(o_) and o_.host_wall in ids_ for o_ in b.openings.values()):
                mine[0].veto("an enclosure with no door, window or name next to a drawing with real rooms: "
                             "a plot / sheet outline, not a room")
                bound -= ids_
                log["empty_enclosures"] += 1
        rooms = [r for r in rooms if _live(r)]
        tree = STRtree([r.polygon for r in rooms])
    for ws in comps:
        if not ws or any(w.id in bound or w.id in hosts or w.manual_status is not None
                         or w.evidence.get("railing_label", 0.0) >= 0.5
                         or w.detection_methods & {"door_implied", "label_implied"} for w in ws):
            continue
        log["groups"] += 1
        length = sum(w.length for w in ws)
        near = False
        for w in ws:
            ax = w.axis
            x0, y0, x1, y1 = ax.bounds
            if any(rooms[int(k)].polygon.distance(ax) <= far
                   for k in tree.query(_box(x0 - far, y0 - far, x1 + far, y1 + far))):
                near = True
                break
        small = len(ws) <= max_walls and length <= max_len
        junctions = sum(1 for w in ws if G.degree(w.id) >= 3)
        floor_like = junctions >= max(min_junctions, junction_share * len(ws))
        for w in ws:
            if small or near or (len(ws) <= far_walls and length <= far_len) or not floor_like:
                w.veto("not part of a building: its wall group encloses no room and has no door or window"
                       + (" (the rooms next to it are closed without it)" if near
                          else f", and is more than {far:g} m from every room"
                          + ("" if floor_like or small else ": a loop / chain of lines, not the walls of a floor")))
                log["rejected_walls"] += 1
                log["rejected_length_m"] += w.length
            elif w.status == Status.ACCEPTED:
                w.status = Status.NEEDS_REVIEW
                w.status_reason = "its wall group encloses no room and has no door or window (free-standing?)"
                w.notes.append("wall structure: connected to no wall that bounds a room")
                log["review_walls"] += 1
    log["rejected_length_m"] = round(log["rejected_length_m"], 2)
    return log
