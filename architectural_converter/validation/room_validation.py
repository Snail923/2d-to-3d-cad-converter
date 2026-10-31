"""Room checks: invalid polygons, overlaps, size, boundary support, labels."""
from __future__ import annotations

from ..models.base_entity import Status


def validate_rooms(report, rooms, labels, tol, cfg):
    vc = cfg["validation"]
    live = [r for r in rooms if r.status != Status.REJECTED]
    for r in rooms:
        c = r.polygon.representative_point()
        loc = (c.x, c.y)
        if not r.polygon.is_valid:
            report.add("room", "invalid_polygon", "error", r, loc, f"Room {r.id} polygon is invalid.",
                       "Correct the room boundary manually.", 0.9)
        if r.status == Status.REJECTED:
            report.add("room", "rejected_region", "info", r, loc,
                       f"Enclosed region {r.id} was not accepted as a room: {r.status_reason}.",
                       "No action needed unless this is a real room.", 0.7)
            continue
        if r.area < vc["min_room_area_m2"]:
            report.add("room", "small_room", "info", r, loc, f"Room {r.id} is very small ({r.area:.2f} m²).",
                       "Probably a shaft or void; reclassify if needed.", 0.6)
        if r.area > vc["max_room_area_m2"]:
            report.add("room", "large_room", "warning", r, loc, f"Room {r.id} is unusually large ({r.area:.1f} m²).",
                       "Check for missing interior walls.", 0.6)
        sup = r.boundary_support.get("accepted_walls", 0) + r.boundary_support.get("openings", 0)
        if sup < vc["min_wall_support"]:
            report.add("room", "weak_boundary_support", "warning", r, loc,
                       f"Room {r.id} boundary is weakly supported by accepted walls ({sup:.0%}).",
                       "Review uncertain walls around this room.", 0.6)
        if len(r.label_texts) > 1:
            report.add("room", "multiple_labels", "warning", r, loc,
                       f"Room {r.id} contains several room labels: {', '.join(t.splitlines()[0] for t in r.label_texts)}.",
                       "A boundary is probably missing between the labelled spaces; see room hypotheses.", 0.7)
        if r.area_text is not None and abs(r.area_text - r.area) > cfg["classification"]["area_match_tol"] * r.area:
            report.add("room", "area_mismatch", "warning", r, loc,
                       f"Room {r.id} area {r.area:.2f} m² differs from the annotated {r.area_text:.2f} m².",
                       "Check the room boundary or the drawing scale.", 0.6)
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            if a.polygon.intersects(b.polygon):
                inter = a.polygon.intersection(b.polygon).area
                if inter > 0.05 * min(a.area, b.area):
                    c = a.polygon.intersection(b.polygon).representative_point()
                    report.add("room", "overlapping_rooms", "error", a, (c.x, c.y),
                               f"Rooms {a.id} and {b.id} overlap.", "Correct one of the boundaries.", 0.8,
                               a.source_entities | b.source_entities)
    from shapely.geometry import Point
    for lb in labels:
        if lb.category != "room_name":
            continue
        if not any(r.polygon.contains(Point(lb.position)) for r in live):
            report.add("room", "missing_room", "warning", None, lb.position,
                       f"Room label '{lb.text.splitlines()[0]}' is not inside any detected room.",
                       "The room boundary is probably incomplete; review the room hypotheses near this label.", 0.7,
                       {lb.raw_id})
