"""Room access reasoning: every room must be reachable through doors or openings.

A room nobody can enter is a logical contradiction, and the most common reason is a door
that was read as something else (a wardrobe or sliding door read as a window, glazed doors,
a door symbol on an unexpected layer). The solver therefore:

1. builds the access graph: rooms + OUTSIDE, connected by doors and openings (windows do not
   give access),
2. picks the entry points of every floor: OUTSIDE, rooms holding a stair flight and
   circulation rooms (lobby, corridor, hall, entrance, stair...). A floor without any of
   them uses its largest connected group of rooms,
3. for every unreachable room, looks at the windows on its boundary that lead into a room
   (never to the outside) and re-types the most door-like one as a door, then repeats until
   nothing changes.

The choice is ranked deterministically: no window symbol / layer evidence first, then a
door-like width, then the weakest window confidence. Re-typed openings stay traceable (notes,
``status_reason``) and NEEDS_REVIEW for review. Rooms that
still have no access are reported by validation."""
from __future__ import annotations

import networkx as nx

from ..models.base_entity import Status

OUTSIDE = "__outside__"
STRONG_WINDOW_EVIDENCE = ("name_hint", "layer_match")   # names / layers, not geometry-only symbols


def _live(e):
    return e is not None and e.status != Status.REJECTED


def access_graph(b) -> nx.Graph:
    G = nx.Graph()
    G.add_node(OUTSIDE)
    for r in b.rooms.values():
        if _live(r):
            G.add_node(r.id)
    for o in b.openings.values():
        if not _live(o) or o.type not in ("door", "opening"):
            continue
        rooms = [r for r in o.rooms if r in G]
        host = b.walls.get(o.host_wall)
        if host is not None and host.classification == "exterior_wall" and len(rooms) == 1:
            G.add_edge(rooms[0], OUTSIDE, opening=o.id)
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                G.add_edge(rooms[i], rooms[j], opening=o.id)
    return G


def _sources(b, G, cfg, rooms_of_level):
    words = tuple(cfg["classification"]["access"]["entry_room_words"])
    stair_rooms = set()
    for s in b.stairs.values():
        if not _live(s):
            continue
        c = s.polygon.representative_point()
        for r in rooms_of_level:
            if r.polygon.intersects(s.polygon) or r.polygon.contains(c):
                stair_rooms.add(r.id)
    src = {r.id for r in rooms_of_level if r.id in stair_rooms or
           any(w in f"{r.name or ''} {r.classification or ''}".lower() for w in words)}
    if any(G.has_edge(OUTSIDE, r.id) for r in rooms_of_level):
        src.add(OUTSIDE)
    if not src and rooms_of_level:
        ids = {r.id for r in rooms_of_level}
        comps = [c & ids for c in nx.connected_components(G.subgraph(ids))]
        src = max(comps, key=len) if comps else set()
    return src


def _rank(o, cfg):
    ac = cfg["classification"]["access"]
    lo, hi = ac["door_width_m"]
    strong = any(o.evidence.get(k, 0.0) >= 0.5 for k in STRONG_WINDOW_EVIDENCE)
    width_ok = lo <= o.width <= hi
    return (strong, not width_ok, o.confidence)


def solve_access(b, cfg, locked=frozenset()) -> dict:
    ac = cfg["classification"]["access"]
    if not ac.get("enabled", True):
        return {"skipped": "disabled"}
    th = cfg["detection"]["status_thresholds"]
    min_area = cfg["validation"]["min_room_area_m2"]
    log = {"unreachable_before": 0, "windows_to_doors": [], "unreachable_after": 0}
    levels = {}
    for r in b.rooms.values():
        if _live(r):
            levels.setdefault(r.level, []).append(r)
    for it in range(ac["max_iterations"] + 1):
        G = access_graph(b)
        bad_all = []
        for lvl, rooms in levels.items():
            src = _sources(b, G, cfg, rooms)
            reach = set()
            for s_ in src:
                if s_ in G:
                    reach |= nx.node_connected_component(G, s_)
            bad_all += [r for r in rooms if r.id not in reach and r.area >= min_area
                        and r.classification != "shaft_or_void"]
        if it == 0:
            log["unreachable_before"] = len(bad_all)
        log["unreachable_after"] = len(bad_all)
        if it == ac["max_iterations"]:
            break
        fixed = False
        for r in sorted(bad_all, key=lambda x: x.id):
            cands = []
            for oid in sorted(r.openings):
                o = b.openings.get(oid)
                if not _live(o) or o.type != "window" or o.id in locked or o.manual_status is not None:
                    continue
                others = [x for x in o.rooms if x != r.id and x in G]
                if not others:
                    continue          # a window to the outside never explains a missing door
                # (a window between two rooms may sit in an exterior wall: the facade towards a
                # balcony or terrace, where a French window is the usual access)
                cands.append(o)
            if not cands:
                continue
            o = min(cands, key=lambda x: _rank(x, cfg))
            o.type, o.subtype, o.sill_height = "door", "access_inferred", None
            o.classification = "interior_door"
            o.notes.append(f"room {r.id}{' (' + r.name + ')' if r.name else ''} had no access: "
                           "the most door-like window on its boundary is read as a door")
            o.status_reason = "access reasoning: window re-typed as door"
            o.confidence = max(th["review"], min(o.confidence, round(th["accept"] - 0.01, 4)))
            o.status = Status.NEEDS_REVIEW
            log["windows_to_doors"].append({"opening": o.id, "room": r.id})
            fixed = True
            break                     # one change per pass, then recompute reachability
        if not fixed:
            break
    return log
