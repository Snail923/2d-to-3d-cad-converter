"""Manual corrections, stored separately from the DXF source data.

Format (JSON, model coordinates in metres)::

    {
      "version": 1,
      "status_overrides": {"wall_0012": "REJECTED", "door_0003": "ACCEPTED", "room_0004": "ACCEPTED"},
      "added_walls": [{"start": [x, y], "end": [x, y], "thickness": 0.15, "note": "..."}],
      "room_overrides": [{"polygon": [[x, y], ...], "name": "Kitchen"}],
      "opening_type_overrides": {"opening_0002": "door"}
    }

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

EMPTY = {"version": 1, "status_overrides": {}, "added_walls": [], "room_overrides": [], "opening_type_overrides": {}}


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
