"""Architectural plausibility rules (deterministic "common sense" checks).

These run after walls have been classified (exterior / interior / partition) and rooms are
related to walls and openings, because they reason about the *context* of a candidate rather
than its own linework:

1. A window must sit in a wall. A window-like line group with no host wall is furniture,
   a shelf or annotation drawn near a wall and is rejected. A window *symbol* recognised by
   geometry only (no window name / layer) must also lie inside its host wall band; one that
   lies beside the wall is furniture placed against it and is rejected.
2. A window in an interior or partition wall, with no window symbol / name / layer to back
   it, is far more likely a sliding or wardrobe (closet) door drawn with thin parallel lines:
   it is re-typed as ``door`` / ``sliding_or_closet`` and sent to review.
   When the wall bounds a small space or one named closet / wardrobe / store, the note says so.

Every change is recorded in the candidate's notes and returned for the stage log; nothing is
changed for candidates that carry a manual status."""
from __future__ import annotations

from ..models.base_entity import Status

# a window *symbol* recognised by geometry alone is not strong evidence: furniture blocks have
# the same signature (equal parallel lines). Only names and layers that say "window" are.
STRONG_WINDOW_EVIDENCE = ("name_hint", "layer_match")


def band_share(o, host, tol) -> float | None:
    """Share of a window symbol's footprint that lies inside its host wall body.

    A window is drawn *in* the wall band; furniture placed against a wall (wardrobe, kitchen
    counter, TV unit, bath tub) lies beside it and only touches the band."""
    fp = getattr(o, "footprint", None)
    if fp is None or fp.is_empty or fp.area <= 0:
        return None
    poly = host.polygon(max(tol.snap * 2, 0.01)) if host is not None else None
    if poly is None:
        return None
    from ..geometry.geometric_utils import robust_intersection
    return float(robust_intersection(fp, poly.buffer(tol.snap)).area / fp.area)


def apply_plausibility(b, cfg, locked=frozenset(), tol=None) -> dict:
    pc = cfg["classification"].get("plausibility", {})
    if not pc.get("enabled", True):
        return {"skipped": "disabled"}
    th = cfg["detection"]["status_thresholds"]
    strong_min = pc.get("strong_window_evidence_min", 0.5)
    small_room = pc.get("closet_max_area_m2", 3.0)
    closet_words = tuple(pc.get("closet_words", ["closet", "wardrobe", "dressing", "store", "storage", "wic"]))
    log = {"unhosted_windows_rejected": 0, "interior_windows_to_doors": 0, "closet_windows_to_doors": 0,
           "interior_windows_kept": 0, "symbols_beside_wall_rejected": 0}
    min_band = cfg["detection"]["window"].get("symbol_min_band_share", 0.5)
    for o in b.openings.values():
        if o.type != "window" or o.id in locked or o.manual_status is not None or o.status == Status.REJECTED:
            continue
        host = b.walls.get(o.host_wall) if o.host_wall else None
        if host is None or host.status == Status.REJECTED:
            o.veto("window is not in any wall (furniture, shelf or annotation near a wall?)")
            log["unhosted_windows_rejected"] += 1
            continue
        strong = any(o.evidence.get(k, 0.0) >= strong_min for k in STRONG_WINDOW_EVIDENCE)
        if o.symbol_insert and not strong and tol is not None:
            share = band_share(o, host, tol)
            if share is not None:
                o.add_evidence("in_wall_band", share)
                if share < min_band:
                    o.veto(f"window-like symbol lies beside the wall, not in it ({share:.0%} inside): "
                           "furniture / fixture against a wall")
                    log["symbols_beside_wall_rejected"] += 1
                    continue
        rooms = [b.rooms[r] for r in host.rooms if r in b.rooms and b.rooms[r].status != Status.REJECTED]
        closet = [r for r in rooms if r.area <= small_room or
                  any(w in (r.name or "").lower() or w in (r.classification or "") for w in closet_words)]
        interior = host.classification in ("interior_wall", "partition_wall")
        if not interior:          # exterior walls: windows are expected (also towards stores / closets)
            continue
        if strong:
            o.notes.append("window in an interior wall (kept: window symbol / layer evidence)")
            log["interior_windows_kept"] += 1
            continue
        o.type = "door"
        o.subtype = "sliding_or_closet"
        o.sill_height = None
        why = "towards a small space / closet" if closet else "in an interior wall"
        o.notes.append(f"window-like lines {why} without window symbol or layer: read as sliding / closet door")
        o.classification = "interior_door"
        o.confidence = min(o.confidence, round(th["accept"] - 0.01, 4))
        o.status = Status.NEEDS_REVIEW if o.confidence >= th["review"] else Status.REJECTED
        o.status_reason = "plausibility: window re-typed as sliding / closet door"
        log["closet_windows_to_doors" if closet else "interior_windows_to_doors"] += 1
    return log
