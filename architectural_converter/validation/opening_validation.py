"""Opening checks: host walls, location, conflicts, widths."""
from __future__ import annotations

from shapely.geometry import Point

from ..models.base_entity import Status


def validate_openings(report, openings, walls, tol, cfg):
    vc = cfg["validation"]
    wmap = {w.id: w for w in walls}
    live = [o for o in openings if o.status != Status.REJECTED]
    for o in live:
        loc = o.center
        if o.host_wall is None:
            report.add("opening", "no_host_wall", "warning", o, loc, f"{o.type.capitalize()} {o.id} has no host wall.",
                       "Check that the symbol sits in a wall, or add the missing wall.", 0.7)
            continue
        w = wmap.get(o.host_wall)
        if w is None:
            continue
        p = w.polygon(max(tol.snap * 2, 0.01))
        if p is not None and p.buffer(max(0.05, tol.snap * 4)).distance(Point(loc)) > 0:
            report.add("opening", "outside_wall", "warning", o, loc, f"{o.id} lies outside its host wall {w.id}.",
                       "Verify the opening position.", 0.6)
        rng = vc["door_width_m"] if o.type in ("door", "opening") else vc["window_width_m"]
        if not (rng[0] <= o.width <= rng[1]):
            report.add("opening", "implausible_width", "warning", o, loc,
                       f"{o.id} width {o.width:.2f} m is outside the expected range {rng}.",
                       "Check the opening interpretation.", 0.6)
        if o.type == "opening" and o.subtype == "unidentified":
            report.add("opening", "unidentified_opening", "info", o, loc,
                       f"Gap of {o.width:.2f} m in wall {w.id} has no door/window symbol.",
                       "Classify as door, window or passage, or reject if the wall should be closed.", o.confidence)
        if any("conflicting door/window" in n for n in o.notes):
            report.add("opening", "door_window_conflict", "warning", o, loc,
                       f"{o.id} has conflicting door/window evidence.", "Choose the correct opening type.", 0.6)
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            if a.host_wall and a.host_wall == b.host_wall and a.host_interval and b.host_interval:
                if min(a.host_interval[1], b.host_interval[1]) - max(a.host_interval[0], b.host_interval[0]) > tol.snap:
                    report.add("opening", "overlapping_openings", "warning", a, a.center,
                               f"Openings {a.id} and {b.id} overlap in wall {a.host_wall}.",
                               "Keep only the correct opening.", 0.7, a.source_entities | b.source_entities)
