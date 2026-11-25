"""Room classification from labels first, geometry second."""
from __future__ import annotations

from ..topology.label_rules import label_types, room_label_type
from ..geometry.geometric_utils import mrr_dims


def classify_rooms(rooms, openings_by_id, cfg):
    cc = cfg["classification"]
    for r in rooms:
        rtype = room_label_type(r, cc["room_keywords"])
        if len(set(label_types(r, cc["room_keywords"]))) > 1:
            n = "several room names in one room: typed by an indoor name, a wall between them may be missing"
            if n not in r.notes:
                r.notes.append(n)
        if rtype:
            r.classification = rtype
            continue
        if r.classification == "shaft_or_void":
            continue
        w, L, _, _ = mrr_dims(r.polygon)
        width = r.boundary_support.get("inscribed_width", w)
        n_doors = sum(1 for o in r.openings if o in openings_by_id and openings_by_id[o].type in ("door", "opening"))
        if (L / max(width, 1e-9) >= cc["corridor_min_aspect"] and width <= cc["corridor_max_width_m"]) or n_doors >= 4:
            r.classification = "corridor"
        elif r.area < 4.0:
            r.classification = "service_room"
        else:
            r.classification = "room"
