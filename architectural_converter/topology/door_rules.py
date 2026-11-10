"""Doors need walls: re-check missing walls around doors.

A door is always set in a wall, and a space with a door is a room. Two rules use that:

1. **Door line** (before room detection). Along the line of every live door, on both sides of the
   door, the wall must continue until it meets a crossing wall. Parts of that line not covered by
   a wall of the same band, up to ``max_gap_m`` from the door, are added as walls
   (``door_implied``). The door is hosted in the wall when it was not.

2. **Room with a door** (after missing-room recovery). A missing-boundary hypothesis that closes a
   room which has a door on its boundary is not a guess any more: the hypothesis becomes a wall.

Walls added here are NEEDS_REVIEW (they close rooms, appear in 3D in the review colour and can be
rejected in the review UI); each records the door and the reason in its notes and evidence."""
from __future__ import annotations

import math

from shapely.geometry import LineString, Point

from ..geometry.geometric_utils import canonical_dir, dot, frame_point, merge_intervals, normal, robust_intersection
from ..models.base_entity import Status
from ..models.wall import Wall


def _live(e):
    return e is not None and e.status != Status.REJECTED


def _door_line(o, walls_by_id):
    """(u, c, s0, s1, thickness) of a door: its host wall's band, else its own chord."""
    host = walls_by_id.get(o.host_wall) if o.host_wall else None
    d = (o.p1[0] - o.p0[0], o.p1[1] - o.p0[1])
    if math.hypot(*d) <= 1e-9:
        return None
    if host is not None and _live(host) and host.kind in ("straight", "centerline"):
        u = host.u
        c = host.c
        t = host.thickness
    else:
        u = canonical_dir(*d)
        c = dot(o.p0, normal(u))
        t = o.depth
    s0, s1 = sorted((dot(o.p0, u), dot(o.p1, u)))
    return u, c, s0, s1, t


def _covered(u, c, t, walls, tol):
    """Intervals along the line (frame u) covered by straight walls of the same band."""
    out = []
    band = max((t or 0.0) / 2, tol.snap * 2, 0.02)
    for w in walls:
        if w.kind not in ("straight", "centerline") or abs(abs(dot(w.u, u)) - 1) > 1e-3:
            continue
        sg = 1.0 if dot(w.u, u) >= 0 else -1.0
        if abs(sg * w.c - c) > band:
            continue
        a, b = sorted((sg * w.s0, sg * w.s1))
        out.append((a, b))
    return merge_intervals(out, tol.snap)


def _first_stop(u, c, s_from, direction, walls, tol, reach):
    """Distance along the line to the first crossing wall body (None if none within reach)."""
    p = frame_point(u, s_from, c)
    q = frame_point(u, s_from + direction * reach, c)
    ray = LineString([p, q])
    best = None
    for w in walls:
        if w.kind in ("straight", "centerline") and abs(dot(w.u, u)) > 0.95:
            continue                                  # parallel walls do not stop the line
        poly = w.polygon(max(tol.snap * 2, 0.06))
        if poly is None or not poly.intersects(ray):
            continue
        d = Point(p).distance(robust_intersection(poly, ray))
        if best is None or d < best:
            best = d
    return best


def complete_door_walls(walls, doors, tol, cfg, ids) -> dict:
    """Rule 1: add the missing wall pieces on the line of each door. Returns a log."""
    rc = cfg["reconstruction"].get("door_rules", {})
    log = {"doors_checked": 0, "walls_added": 0, "doors_hosted": 0, "added": []}
    if not rc.get("enabled", True):
        return {"skipped": "disabled"}
    reach = float(rc.get("max_gap_m", 4.0))
    min_piece = max(float(rc.get("min_piece_m", 0.05)), tol.snap * 2)
    conf = float(rc.get("added_wall_confidence", 0.6))
    live = [w for w in walls if _live(w)]
    by_id = {w.id: w for w in walls}
    for o in doors:
        if not _live(o) or o.type != "door":
            continue
        line = _door_line(o, by_id)
        if line is None:
            continue
        u, c, s0, s1, t = line
        log["doors_checked"] += 1
        cov = _covered(u, c, t, live, tol)
        hosted = o.host_wall is not None and _live(by_id.get(o.host_wall))
        if not hosted and not any(y >= s0 - reach and x <= s1 + reach for x, y in cov):
            # an unhosted door with no wall of its own line nearby: its line is not known well
            # enough (open-leaf chord, furniture) to draw a wall from it
            log.setdefault("skipped_no_wall_on_line", 0)
            log["skipped_no_wall_on_line"] += 1
            continue
        new_here = []
        for side in (-1, 1):
            start = s0 if side < 0 else s1
            stop = _first_stop(u, c, start, side, live, tol, reach)
            if stop is None:
                continue                              # nothing to close against: an open side
            a, b = sorted((start, start + side * stop))
            # the missing parts of [a, b] (outside existing walls of this band)
            cur, gaps = a, []
            for x, y in cov:
                if y <= cur or x >= b:
                    continue
                if x > cur:
                    gaps.append((cur, x))
                cur = max(cur, y)
            if cur < b:
                gaps.append((cur, b))
            for g0, g1 in gaps:
                if g1 - g0 < min_piece:
                    continue
                thick = t or next((w.thickness for w in live if w.thickness), None)
                w = Wall(id=ids.new("wall"), kind="straight" if thick else "centerline", u=u, c=c, s0=g0, s1=g1,
                         thickness=thick, thickness_source="door_band" if thick else "unknown",
                         detected_interval=(g0, g1))
                w.detection_methods.add("door_implied")
                w.add_evidence("door_requires_wall", 1.0)
                w.confidence = conf
                w.status = Status.NEEDS_REVIEW
                w.status_reason = f"door {o.id} needs a wall here (missing between the door and the next wall)"
                w.notes.append(f"added: the wall of door {o.id} did not reach the next crossing wall")
                new_here.append(w)
        if new_here:
            walls.extend(new_here)
            live.extend(new_here)
            log["walls_added"] += len(new_here)
            log["added"] += [w.id for w in new_here]
            if o.host_wall is None or not _live(by_id.get(o.host_wall)):
                # host the door in the band it sits in: the piece next to it
                host = min(new_here, key=lambda w: min(abs(w.s0 - s1), abs(w.s1 - s0)))
                host.s0, host.s1 = min(host.s0, s0), max(host.s1, s1)
                host.detected_interval = (host.s0, host.s1)
                o.host_wall = host.id
                o.host_interval = (s0, s1)
                host.openings.append(o.id)
                o.add_evidence("host_wall", 1.0)
                log["doors_hosted"] += 1
            for w in new_here:
                by_id[w.id] = w
    return log


def drop_unused(walls, rooms) -> list:
    """Walls added by the door rules that ended up bounding no room close nothing: rejected."""
    bound = {wid for r in rooms if _live(r) for wid in r.bounding_walls}
    out = []
    for w in walls:
        if w.detection_methods & {"door_implied", "label_implied"} and _live(w) and w.manual_status is None \
                and w.id not in bound:
            w.veto("door / room-name rule: the added wall does not close a room")
            out.append(w.id)
    return out


def promote_door_rooms(hyps, hrooms, openings, tol, cfg, ids, rooms=(), walls=()) -> list:
    """Rule 2: hypotheses that close a room with a door on its boundary become walls.

    Only for space that is not a room yet (a hypothesis that splits an existing room, e.g. an
    open-plan living / kitchen, is not promoted), and only walls that follow the direction of the
    wall they continue (no diagonal links across an entrance)."""
    rc = cfg["reconstruction"].get("door_rules", {})
    if not rc.get("enabled", True) or not rc.get("promote_hypotheses", True):
        return []
    conf = float(rc.get("added_wall_confidence", 0.6))
    doors = [o for o in openings if _live(o) and o.type == "door"]
    by_hyp: dict[str, list] = {}
    for r in hrooms:
        for hid in r.hypothesis_walls:
            by_hyp.setdefault(hid, []).append(r)
    live_rooms = [r.polygon for r in rooms if _live(r) and r.polygon is not None]
    by_id = {w.id: w for w in walls}
    out, taken = [], []
    # shortest first; one wall per enclosed area (several hypotheses often close the same room)
    for h in sorted(hyps, key=lambda x: x.length):
        rooms = by_hyp.get(h.id, [])
        if any(robust_intersection(r.polygon, t).area > 0.5 * min(r.polygon.area, t.area) for r in rooms for t in taken):
            continue
        # new space only: most of the enclosed area must not be a room already
        if any(sum(robust_intersection(r.polygon, x).area for x in live_rooms) > 0.5 * r.polygon.area for r in rooms):
            continue
        src = by_id.get((h.reconstruction.get("hypothesis") or {}).get("from_wall"))
        if src is not None and src.kind in ("straight", "centerline"):
            al = abs(dot(src.u, h.u))
            if min(al, 1 - al) > 1 - math.cos(math.radians(2.0)) and not al < math.sin(math.radians(2.0)):
                continue                                     # neither along nor across its wall
        hit = None
        for r in rooms:
            ring = r.polygon.exterior
            for o in doors:
                reach = (o.depth or 0.3) / 2 + max(tol.snap * 4, 0.1)
                if ring.distance(Point(o.center)) <= reach:
                    hit = (r, o)
                    break
            if hit:
                break
        if not hit:
            continue
        r, o = hit
        taken += [x.polygon for x in rooms]
        h.is_hypothesis = False
        h.reconstruction.setdefault("hypothesis", {})["was"] = h.id
        h.id = ids.new("wall")
        h.detection_methods.add("door_implied")
        h.add_evidence("door_requires_wall", 1.0)
        h.confidence = conf
        h.status = Status.NEEDS_REVIEW
        h.status_reason = f"closes a room that has door {o.id}: a room with a door needs all its walls"
        h.notes.append(f"missing wall of a room with door {o.id} (was a missing-room hypothesis)")
        out.append(h)
    return out
