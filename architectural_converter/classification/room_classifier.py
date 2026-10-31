"""Room classification from labels first, geometry second."""
from __future__ import annotations

from ..detectors.text_detector import room_type_of
from ..geometry.geometric_utils import mrr_dims


def classify_rooms(rooms, openings_by_id, cfg):
    cc = cfg["classification"]
    for r in rooms:
        rtype = None
        for t in r.label_texts:
            rtype = room_type_of(t, cc["room_keywords"])
            if rtype:
                break
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
