"""Manual corrections, stored separately from the DXF source data.

Format (JSON, model coordinates in metres)::

    {
      "version": 1,
      "status_overrides": {"wall_0012": "REJECTED", "door_0003": "ACCEPTED", "room_0004": "ACCEPTED"},
      "added_walls": [{"start": [x, y], "end": [x, y], "thickness": 0.15, "note": "..."}],
      "room_overrides": [{"polygon": [[x, y], ...], "name": "Kitchen"}],
      "opening_type_overrides": {"opening_0002": "door", "window_0004": "sliding_or_closet"},
      "wall_edits": {"wall_0007": {"start": [x, y], "end": [x, y], "thickness": 0.2}},
      "wall_cuts": [{"wall": "wall_0007", "start": [x, y], "end": [x, y]}],
      "added_openings": [{"type": "door", "start": [x, y], "end": [x, y], "wall": "wall_0007"}],
      "opening_edits": {"door_0003": {"start": [x, y], "end": [x, y]}},
      "added_floors": [{"polygon": [[x, y], ...]}],
      "floor_cuts": [{"polygon": [[x, y], ...]}]
    }

``wall_edits`` moves the ends of a wall / sets its thickness; ``wall_cuts`` removes the part of a
wall between two points (the wall is shortened or split, openings in the removed part go with it);
``added_openings`` puts a door / window / opening into a wall (``wall`` optional: the nearest wall);
``opening_edits`` moves the ends of an opening along its wall. Closing an opening (wall instead of
the door / window / gap) is ``status_overrides: {id: "REJECTED"}``. ``added_floors`` adds floor
where no room is, ``floor_cuts`` removes floor (room floors, ground pieces and the storey slabs:
a stair void, a courtyard).

``added_walls`` also covers "connect endpoints" and accepted missing-wall hypotheses.
Corrections never modify RawEntity / GeoEntity data."""
from __future__ import annotations

import copy
import json
from typing import Any

from shapely.geometry import Polygon

from ..geometry.geometric_utils import canonical_dir, dot, normal
from ..models.base_entity import Status
from ..models.room import Room
from ..models.wall import Wall

EMPTY = {"version": 1, "status_overrides": {}, "added_walls": [], "room_overrides": [], "opening_type_overrides": {},
         "wall_edits": {}, "wall_cuts": [], "added_openings": [], "opening_edits": {}, "added_floors": [],
         "floor_cuts": []}

OPENING_TYPES = {"door": ("door", None), "window": ("window", None), "opening": ("opening", "passage"),
                 "sliding_or_closet": ("door", "sliding_or_closet")}


def normalize_corrections(c: dict | str | None) -> dict[str, Any]:
    if c is None:
        return copy.deepcopy(EMPTY)
    if isinstance(c, str):
        c = json.loads(c)
    out = copy.deepcopy(EMPTY)
    for k in out:
        if k in c and c[k] is not None:
            out[k] = c[k]
    return out


def apply_status_overrides(entities, corrections) -> int:
    ov = corrections.get("status_overrides", {})
    n = 0
    for e in entities:
        st = ov.get(e.id)
        if st:
            e.manual_status = Status(st)
            e.status = Status(st)
            e.status_reason = "manual correction"
            e.add_evidence("manual", 1.0 if st == "ACCEPTED" else 0.0)
            n += 1
    return n


def manual_walls(corrections, ids) -> list[Wall]:
    out = []
    for k, aw in enumerate(corrections.get("added_walls", [])):
        a, b = aw["start"], aw["end"]
        u = canonical_dir(b[0] - a[0], b[1] - a[1])
        n = normal(u)
        s0, s1 = sorted((dot(a, u), dot(b, u)))
        t = aw.get("thickness")
        w = Wall(id=f"manual_wall_{k + 1:04d}", kind="straight" if t else "centerline", u=u, c=dot(a, n), s0=s0, s1=s1,
                 thickness=float(t) if t else None, thickness_source="manual" if t else "unknown",
                 detected_interval=(s0, s1))
        w.detection_methods.add("manual")
        w.evidence = {"manual": 1.0}
        w.manual_status = Status.ACCEPTED
        w.status = Status.ACCEPTED
        w.confidence = 1.0
        w.status_reason = "manual correction"
        if aw.get("note"):
            w.notes.append(aw["note"])
        out.append(w)
    return out


def manual_rooms(corrections) -> list[Room]:
    out = []
    for k, ro in enumerate(corrections.get("room_overrides", [])):
        poly = Polygon(ro["polygon"]).buffer(0)
        r = Room(id=f"manual_room_{k + 1:04d}", polygon=poly, name=ro.get("name"))
        r.detection_methods.add("manual")
        r.evidence = {"manual": 1.0}
        r.manual_status = Status.ACCEPTED
        r.status = Status.ACCEPTED
        r.confidence = 1.0
        r.status_reason = "manual correction"
        if ro.get("name"):
            r.label_texts.append(ro["name"])
        out.append(r)
    return out


def apply_opening_type_overrides(openings, corrections) -> set[str]:
    """Re-type detected openings (door / window / opening / sliding_or_closet). Returns the ids
    that are locked against automatic re-typing by the plausibility rules and the access solver."""
    ov = corrections.get("opening_type_overrides", {})
    locked = set()
    for o in openings:
        t = ov.get(o.id)
        if not t or t not in OPENING_TYPES:
            continue
        typ, sub = OPENING_TYPES[t]
        o.type = typ
        o.subtype = sub or ("verified" if o.subtype in ("sliding_or_closet", "access_inferred", "unidentified") else o.subtype)
        if typ != "window":
            o.sill_height = None
        o.notes.append(f"type set to {t} by manual correction")
        locked.add(o.id)
    return locked


# ---------------------------------------------------------------------------- geometry edits
def _manual(e, note):
    e.manual_status = Status.ACCEPTED
    e.status = Status.ACCEPTED
    e.status_reason = "manual correction"
    e.confidence = max(float(e.confidence or 0.0), 1.0)
    e.add_evidence("manual", 1.0)
    e.detection_methods.add("manual")
    if note not in e.notes:
        e.notes.append(note)


def _reframe(w, a, b):
    u = canonical_dir(b[0] - a[0], b[1] - a[1])
    w.u, w.c = u, dot(a, normal(u))
    w.s0, w.s1 = sorted((dot(a, u), dot(b, u)))
    w.detected_interval = (w.s0, w.s1)
    w.__dict__.pop("_poly_cache", None)


def _rehost(o, w):
    lo, hi = sorted((dot(o.p0, w.u), dot(o.p1, w.u)))
    lo, hi = max(lo, w.s0), min(hi, w.s1)
    o.host_wall, o.host_interval = w.id, (lo, hi)
    o.p0, o.p1 = (w.u[0] * lo - w.u[1] * w.c, w.u[1] * lo + w.u[0] * w.c), (w.u[0] * hi - w.u[1] * w.c, w.u[1] * hi + w.u[0] * w.c)
    o.width = float(hi - lo)
    if o.id not in w.openings:
        w.openings.append(o.id)


def apply_geometry_edits(walls, doors, windows, generic, corrections, min_piece: float = 0.02) -> dict:
    """Wall edits, wall cuts, added and edited openings, on the reconstructed walls (before rooms
    are detected). New wall pieces / openings are appended to the given lists. Returns a log."""
    log = {"walls_edited": 0, "wall_cuts": 0, "wall_pieces_added": 0, "openings_added": 0, "openings_edited": 0,
           "skipped": []}
    by_id = {w.id: w for w in walls}
    ops = {o.id: o for o in list(doors) + list(windows) + list(generic)}
    straight = lambda w: w.kind in ("straight", "centerline")  # noqa: E731
    for wid, ed in (corrections.get("wall_edits") or {}).items():
        w = by_id.get(wid)
        if w is None or not straight(w):
            log["skipped"].append(f"wall_edits {wid}: no such straight wall")
            continue
        if ed.get("start") and ed.get("end"):
            _reframe(w, ed["start"], ed["end"])
            for oid in list(w.openings):
                if oid in ops:
                    _rehost(ops[oid], w)
        if ed.get("thickness"):
            w.thickness, w.thickness_source, w.kind = float(ed["thickness"]), "manual", "straight"
            w.__dict__.pop("_poly_cache", None)
        _manual(w, "geometry set by manual correction")
        log["walls_edited"] += 1
    pieces: dict[str, list] = {}
    for k, cut in enumerate(corrections.get("wall_cuts") or []):
        cand = pieces.get(cut.get("wall")) or ([by_id[cut["wall"]]] if cut.get("wall") in by_id else [])
        a, b = cut["start"], cut["end"]
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        if not cand:                                    # no wall named: the straight wall under the cut
            cand = [w for w in walls if straight(w) and w.status != Status.REJECTED
                    and abs(dot(mid, normal(w.u)) - w.c) <= (w.thickness or 0.1) / 2 + 0.05
                    and w.s0 - 1e-6 <= dot(mid, w.u) <= w.s1 + 1e-6]
        cand = [w for w in cand if straight(w)]
        if not cand:
            log["skipped"].append(f"wall_cuts {k}: no wall there")
            continue
        w = min(cand, key=lambda x: abs(min(max(dot(mid, x.u), x.s0), x.s1) - dot(mid, x.u)))
        lo, hi = sorted((dot(a, w.u), dot(b, w.u)))
        lo, hi = max(lo, w.s0), min(hi, w.s1)
        if hi - lo < min_piece:
            log["skipped"].append(f"wall_cuts {k}: nothing of {w.id} between the two points")
            continue
        root = cut.get("wall") or w.id
        group = pieces.setdefault(root, [w])
        left, right = lo - w.s0 >= min_piece, w.s1 - hi >= min_piece
        hosted = [ops[i] for i in w.openings if i in ops]
        new = None
        if left and right:
            new = copy.copy(w)
            new.id = f"{w.id}_{len(group) + 1}"
            new.openings, new.source_entities, new.detection_methods = [], set(w.source_entities), set(w.detection_methods)
            new.face_segments, new.notes, new.evidence = set(w.face_segments), list(w.notes), dict(w.evidence)
            new.connections, new.rooms = set(), set()
            new.__dict__.pop("_poly_cache", None)
            new.s0, new.s1 = hi, w.s1
            new.detected_interval = (new.s0, new.s1)
            w.s1 = lo
        elif left:
            w.s1 = lo
        elif right:
            w.s0 = hi
        else:
            w.manual_status = Status.REJECTED
            w.status, w.status_reason = Status.REJECTED, "manual correction: wall removed"
        w.detected_interval = (w.s0, w.s1)
        w.__dict__.pop("_poly_cache", None)
        if "part removed by manual correction" not in w.notes:
            w.notes.append("part removed by manual correction")
        for o in hosted:
            x0, x1 = sorted(o.host_interval) if o.host_interval else sorted((dot(o.p0, w.u), dot(o.p1, w.u)))
            m = (x0 + x1) / 2
            if lo - 1e-6 <= m <= hi + 1e-6 or w.status == Status.REJECTED:
                o.status, o.status_reason = Status.REJECTED, "veto: its part of the wall was removed by manual correction"
                o.host_wall = None
                w.openings = [i for i in w.openings if i != o.id]
            elif new is not None and m > hi:
                w.openings = [i for i in w.openings if i != o.id]
                o.host_wall = new.id
                new.openings.append(o.id)
        if new is not None:
            walls.append(new)
            by_id[new.id] = new
            group.append(new)
            log["wall_pieces_added"] += 1
        log["wall_cuts"] += 1
    for oid, ed in (corrections.get("opening_edits") or {}).items():
        o = ops.get(oid)
        w = by_id.get(o.host_wall) if o is not None and o.host_wall else None
        if o is None or w is None or not straight(w) or not (ed.get("start") and ed.get("end")):
            log["skipped"].append(f"opening_edits {oid}: no such opening in a straight wall")
            continue
        o.p0, o.p1 = tuple(ed["start"]), tuple(ed["end"])
        _rehost(o, w)
        _manual(o, "position set by manual correction")
        log["openings_edited"] += 1
    from ..models.opening import Opening  # noqa: PLC0415
    for k, ao in enumerate(corrections.get("added_openings") or []):
        a, b = ao["start"], ao["end"]
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        named = pieces.get(ao.get("wall")) or ([by_id[ao["wall"]]] if ao.get("wall") in by_id else [])
        named = [v for v in named if straight(v) and v.status != Status.REJECTED
                 and v.s0 - 0.05 <= dot(mid, v.u) <= v.s1 + 0.05]          # (the piece of a cut wall it lies on)
        w = named[0] if named else None
        if w is None:
            best = None
            for v in walls:
                if not straight(v) or v.status == Status.REJECTED:
                    continue
                d = abs(dot(mid, normal(v.u)) - v.c)
                s = dot(mid, v.u)
                if d <= (v.thickness or 0.1) / 2 + 0.3 and v.s0 - 0.05 <= s <= v.s1 + 0.05 and (best is None or d < best[0]):
                    best = (d, v)
            w = best[1] if best else None
        typ = ao.get("type", "door") if ao.get("type") in ("door", "window", "opening") else "door"
        if w is None:
            log["skipped"].append(f"added_openings {k}: no wall there")
            continue
        o = Opening(id=f"manual_{typ}_{k + 1:04d}", type=typ, subtype="manual", p0=tuple(a), p1=tuple(b), depth=w.thickness)
        _rehost(o, w)
        if o.width < min_piece:
            w.openings = [i for i in w.openings if i != o.id]
            log["skipped"].append(f"added_openings {k}: the two points are not along wall {w.id}")
            continue
        _manual(o, f"{typ} added by manual correction")
        o.set_evidence("host_wall", 1.0)
        (doors if typ == "door" else windows if typ == "window" else generic).append(o)
        ops[o.id] = o
        log["openings_added"] += 1
    return log


def floor_polygons(corrections, key) -> list:
    """Valid polygons of ``added_floors`` / ``floor_cuts``."""
    out = []
    for f in corrections.get(key) or []:
        try:
            p = Polygon(f["polygon"]).buffer(0)
        except Exception:  # noqa: BLE001
            continue
        if not p.is_empty and p.area > 0:
            out.append(p)
    return out
