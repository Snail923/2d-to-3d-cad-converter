"""Select the decisions worth a second opinion and describe each one for a vision model.

Every item carries the engine's claim, a closed set of possible answers, the area of the
drawing to show (``bbox``, metres) with the element to highlight, and the context a person
would read off the drawing (layers, block names, room names on both sides, wall type, the
engine's evidence and notes). The web app renders the crop, asks the model and turns the
answer into a correction (see ``ANSWERS``), so a model can only choose between the engine's
own interpretations: it can never create or move geometry.

Priority: decisions changed by the plausibility rules / access reasoning, then uncertain
openings, then windows without a window symbol (furniture risk), then short uncertain walls."""
from __future__ import annotations

from ..models.base_entity import Status

# answer -> correction. "same" answers confirm the engine (status ACCEPTED).
ANSWERS = {
    "opening": {
        "window": {"type": "window", "status": "ACCEPTED"},
        "door": {"type": "door", "status": "ACCEPTED"},
        "sliding_or_closet_door": {"type": "sliding_or_closet", "status": "ACCEPTED"},
        "open_passage": {"type": "opening", "status": "ACCEPTED"},
        "not_an_opening": {"status": "REJECTED"},
        "unsure": {},
    },
    "wall": {"wall": {"status": "ACCEPTED"}, "not_a_wall": {"status": "REJECTED"}, "unsure": {}},
}


def _claim_of(o):
    if o.type == "door" and o.subtype in ("sliding_or_closet", "access_inferred"):
        return "sliding_or_closet_door" if o.subtype == "sliding_or_closet" else "door"
    return {"window": "window", "door": "door", "opening": "open_passage"}.get(o.type, "unsure")


def _bbox(geom, vc):
    x0, y0, x1, y1 = geom.bounds
    m = vc["crop_margin_m"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = min(max((x1 - x0) / 2 + m, (y1 - y0) / 2 + m, vc["min_crop_m"] / 2), vc["max_crop_m"] / 2)
    return [round(cx - half, 3), round(cy - half, 3), round(cx + half, 3), round(cy + half, 3)]


def _layers(e, drawing):
    raw = drawing.raw_entities
    lay, blocks = set(), set()
    for rid in e.source_entities:
        r = raw.get(rid)
        if r is None:
            continue
        lay.add(r.layer)
        if r.block_name:
            blocks.add(r.block_name)
        if r.dxftype == "INSERT" and getattr(r, "attributes", None):
            nm = r.attributes.get("name")
            if nm:
                blocks.add(nm)
    return sorted(lay)[:6], sorted(blocks)[:6]


def build_review_queue(st, cfg) -> list[dict]:
    vc = cfg["verification"]
    b = st.building
    decided = set(st.corrections.get("status_overrides", {})) | set(st.corrections.get("opening_type_overrides", {}))
    rooms = b.rooms
    items = []

    def room_desc(ids):
        out = []
        for rid in ids:
            r = rooms.get(rid)
            if r is not None and r.status != Status.REJECTED:
                out.append(f"{r.name or r.classification or 'unnamed room'} ({r.area:.1f} m²)")
        return out

    from shapely.geometry import LineString
    for o in b.openings.values():
        if o.id in decided or o.status == Status.REJECTED:
            continue
        reason = o.status_reason or ""
        changed = reason.startswith("plausibility") or reason.startswith("access reasoning")
        weak_window = o.type == "window" and o.evidence.get("symbol", 0.0) < 0.5
        if changed:
            prio = 0
        elif o.status == Status.NEEDS_REVIEW:
            prio = 1
        elif weak_window:
            prio = 2
        else:
            continue
        host = b.walls.get(o.host_wall)
        lay, blocks = _layers(o, st.drawing)
        geom = LineString([o.p0, o.p1])
        items.append({
            "id": o.id, "kind": "opening", "priority": prio, "claim": _claim_of(o),
            "answers": list(ANSWERS["opening"]),
            "question": (f"The engine reads the highlighted element as a {_claim_of(o).replace('_', ' ')}"
                         f" ({o.width:.2f} m wide) in a{'n' if host is not None and host.classification and host.classification[0] in 'aeiou' else ''} "
                         f"{(host.classification or 'unclassified wall').replace('_', ' ') if host is not None else 'NO wall'}. "
                         "What is it?"),
            "highlight": {"segment": [list(o.p0), list(o.p1)]},
            "bbox": _bbox(geom, vc),
            "context": {"rooms_on_sides": room_desc(o.rooms), "layers": lay, "blocks": blocks,
                        "width_m": round(o.width, 2), "engine_confidence": round(o.confidence, 2),
                        "evidence": {k: round(v, 2) for k, v in o.evidence.items()}, "notes": o.notes[-3:],
                        "level": o.level},
        })
    for w in b.walls.values():
        if w.id in decided or w.status != Status.NEEDS_REVIEW or w.axis is None:
            continue
        if w.length > vc["wall_review_max_length_m"]:
            continue
        lay, blocks = _layers(w, st.drawing)
        poly = w.polygon(0.05)
        items.append({
            "id": w.id, "kind": "wall", "priority": 3, "claim": "wall", "answers": list(ANSWERS["wall"]),
            "question": (f"The engine is unsure whether the highlighted element ({w.length:.2f} m long, "
                         f"{(w.thickness or 0) * 100:.0f} cm thick) is a wall. Is it a wall, or furniture / "
                         "annotation / something else?"),
            "highlight": {"polygon": [list(p) for p in poly.exterior.coords] if poly is not None else None,
                          "segment": [list(w.axis.coords[0]), list(w.axis.coords[-1])]},
            "bbox": _bbox(w.axis, vc),
            "context": {"rooms_on_sides": room_desc(sorted(w.rooms)), "layers": lay, "blocks": blocks,
                        "engine_confidence": round(w.confidence, 2),
                        "evidence": {k: round(v, 2) for k, v in w.evidence.items()}, "notes": w.notes[-3:],
                        "level": w.level},
        })
    items.sort(key=lambda x: (x["priority"], x["context"]["engine_confidence"], x["id"]))
    return items[: vc["max_items"]]
