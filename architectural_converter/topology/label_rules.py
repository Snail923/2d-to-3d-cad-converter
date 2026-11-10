"""Room names need rooms: re-check missing walls around room labels.

A room has one name, and a named space is a room. Two rules use the labels of the drawing on the
missing-boundary hypotheses (walls proposed from free wall ends, see missing_room_recovery):

1. **One room, one name.** A detected room that holds several names of different kinds (a balcony
   and a bedroom, a bathroom and a corridor) misses a wall between them. A hypothesis that splits
   it so that the names fall on different sides becomes a wall. Names of spaces that are commonly
   open to each other (living / dining / kitchen ..., ``open_plan_types``) do not require one.
2. **A name needs a room.** A room name that lies in no room, where a hypothesis closes a room
   around it, turns that hypothesis into a wall.

Walls added here are NEEDS_REVIEW (``label_implied``) and record the names in their notes."""
from __future__ import annotations

import math

from ..detectors.text_detector import room_type_of
from ..geometry.geometric_utils import dot, robust_intersection
from ..models.base_entity import Status


def _live(e):
    return e is not None and e.status != Status.REJECTED


def label_types(room, keywords) -> list[str]:
    """Room types named by the labels inside a room (one per typed label)."""
    return [t for t in (room_type_of(x, keywords) for x in room.label_texts) if t]


def room_label_type(room, keywords) -> str | None:
    """The type a room's labels give it. Several different names in one room mean a wall is
    missing; the room is then typed by an indoor name (never 'balcony': an outdoor space that
    is not separated from an indoor one is not a balcony yet)."""
    types = label_types(room, keywords)
    if not types:
        return None
    kinds = list(dict.fromkeys(types))
    if len(kinds) > 1:
        specific = [t for t in kinds if t not in ("balcony", "guest_room")] or [t for t in kinds if t != "balcony"]
        if specific:
            return specific[0]
    return kinds[0]


def _aligned(h, by_id) -> bool:
    src = by_id.get((h.reconstruction.get("hypothesis") or {}).get("from_wall"))
    if src is None or src.kind not in ("straight", "centerline"):
        return True
    al = abs(dot(src.u, h.u))
    return al >= math.cos(math.radians(2.0)) or al <= math.sin(math.radians(2.0))


def promote_label_rooms(hyps, hrooms, tol, cfg, ids, rooms=(), walls=()) -> tuple[list, dict]:
    """(hypotheses turned into walls, log)."""
    rc = cfg["reconstruction"].get("label_rules", {})
    log = {"split_rooms": [], "named_rooms": []}
    if not rc.get("enabled", False):
        return [], {"skipped": "disabled"}
    kw = cfg["classification"]["room_keywords"]
    open_types = set(rc.get("open_plan_types", ()))
    conf = float(rc.get("added_wall_confidence", 0.6))
    min_part = float(rc.get("min_part_share", 0.05))
    by_hyp: dict[str, list] = {}
    for r in hrooms:
        for hid in r.hypothesis_walls:
            by_hyp.setdefault(hid, []).append(r)
    live_rooms = [r for r in rooms if _live(r) and r.polygon is not None]
    multi = [(r, label_types(r, kw)) for r in live_rooms]
    multi = [(r, t) for r, t in multi if len(t) > 1]
    by_id = {w.id: w for w in walls}
    out, taken = [], []
    for h in sorted(hyps, key=lambda x: x.length):
        if not h.is_hypothesis or not _aligned(h, by_id):
            continue
        created = [r for r in by_hyp.get(h.id, []) if r.polygon is not None and r.polygon.area > 0]
        if not created:
            continue
        if any(robust_intersection(r.polygon, t).area > 0.5 * min(r.polygon.area, t.area) for r in created for t in taken):
            continue
        why = None
        for hr in created:
            inside = label_types(hr, kw)
            if not inside:
                continue
            host = None
            for R, types in multi:
                a = robust_intersection(hr.polygon, R.polygon).area
                if a >= 0.8 * hr.polygon.area:
                    host = (R, types, a)
                    break
            if host is not None:
                R, types, a = host
                if not (min_part * R.polygon.area <= a <= (1 - min_part) * R.polygon.area):
                    continue                                   # not a split of the room
                rest = list(types)
                for t in inside:
                    if t in rest:
                        rest.remove(t)
                if not rest:
                    continue                                   # all names on one side
                if all(t in open_types for t in inside + rest):
                    continue                                   # spaces that may be open to each other
                why = ("split", R, inside, rest)
                break
            # rule 2: the name is in no room yet
            covered = sum(robust_intersection(hr.polygon, x.polygon).area for x in live_rooms)
            if covered <= 0.5 * hr.polygon.area:
                why = ("named", hr, inside, [])
                break
        if why is None:
            continue
        taken += [x.polygon for x in created]
        kind, R, inside, rest = why
        h.is_hypothesis = False
        h.reconstruction.setdefault("hypothesis", {})["was"] = h.id
        h.id = ids.new("wall")
        h.detection_methods.add("label_implied")
        h.add_evidence("labels_require_wall", 1.0)
        h.confidence = conf
        h.status = Status.NEEDS_REVIEW
        if kind == "split":
            h.status_reason = (f"room {R.id} holds the names {', '.join(inside)} and {', '.join(rest)}: "
                               "one room has one name, a wall between them was missing")
            log["split_rooms"].append({"room": R.id, "wall": h.id, "names": [inside, rest]})
        else:
            h.status_reason = f"closes the room named {', '.join(inside)}: a named space is a room"
            log["named_rooms"].append({"wall": h.id, "names": inside})
        h.notes.append("missing wall required by the room names (was a missing-room hypothesis)")
        out.append(h)
    return out, log

