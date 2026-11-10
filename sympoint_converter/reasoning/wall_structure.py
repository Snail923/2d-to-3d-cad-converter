"""Wall structure: walls belong to a building.

The model calls a line "wall" from its local look; double lines of a site boundary, a kerb, a
furniture outline or a detail drawing look the same. In the built result the difference shows:
the walls of a building form groups that enclose rooms and carry doors and windows.

After rooms are known, every group of connected walls is checked. A group none of whose walls
bounds a room, hosts a door or window, or is a balcony railing closes nothing:

* small groups (a few walls, short total length) are rejected, and so are medium groups far from
  every room (detail drawings, site lines);
* larger groups are kept for review (free-standing walls exist, and a floor whose rooms could
  not be closed must not disappear), with the reason in their notes.

Rooms are not affected (the removed walls bound none), so nothing has to be recomputed."""
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
    for comp in nx.connected_components(G):
        ws = [by_id[i] for i in comp if i in by_id]
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
        for w in ws:
            if small or (not near and len(ws) <= far_walls and length <= far_len):
                w.veto("not part of a building: its wall group encloses no room and has no door or window"
                       + ("" if near else f", and is more than {far:g} m from every room"))
                log["rejected_walls"] += 1
                log["rejected_length_m"] += w.length
            elif w.status == Status.ACCEPTED:
                w.status = Status.NEEDS_REVIEW
                w.status_reason = "its wall group encloses no room and has no door or window (free-standing?)"
                w.notes.append("wall structure: connected to no wall that bounds a room")
                log["review_walls"] += 1
    log["rejected_length_m"] = round(log["rejected_length_m"], 2)
    return log
