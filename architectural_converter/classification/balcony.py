"""Balconies, terraces and loggias (open outdoor spaces attached to the building).

A balcony is drawn like a small room, but its outer edges are *railings* or parapets, not
full-height walls. Read as a room, it closes the building envelope too far out: the real
facade between the flat and the balcony is classified as an interior wall, its windows are
re-typed as closet doors, and the 3D model shows a closed box with full-height walls.

The decision is made from evidence that any drawing provides, never from coordinates:

* **railing elements** - walls on the building outline (exterior) recognised as railings by a
  detector (``railing_label`` evidence, e.g. SymPointV2's railing class), or clearly lighter than
  the drawing's own facade walls (thickness <= ``railing_max_rel_thickness`` x the length-
  weighted exterior wall thickness, or drawn as a single line), or whose source layers say
  balcony / terrace / railing / parapet (``layer_semantics.json`` category ``balcony``);
* a **balcony room** - a room whose boundary lies on the building outline for at least
  ``min_outline_share`` of its perimeter, where railing elements make up at least
  ``min_railing_share`` of that outline part, and which is attached to the building by at
  least one non-railing wall (the facade it hangs on). A room label (balcony, terrace...)
  decides on its own; a label naming another room type vetoes it.

Consequences, all recorded in notes / evidence:

* the room is classified ``balcony`` (open: no ceiling, its floor is outside);
* its railing walls become ``railing``: Phase 2 builds a balcony wall as a low solid parapet
  (``balcony_parapet_height_m``) with glass above it up to ``railing_height_m``;
* walls between the balcony and an indoor room are the real facade: ``exterior_wall``. So the
  windows in them stay windows and the doors to the balcony are exterior doors.
"""
from __future__ import annotations


from ..detectors.text_detector import room_type_of
from ..geometry.geometric_utils import robust_intersection, robust_union
from ..models.base_entity import Status
from ..preprocessing.layer_analyzer import category_scores
from .wall_classifier import FACADE_LINE_NOTE
from ..geometry.geometric_utils import safe_buffer as _sbuf

IGNORED_LAYER_CATEGORIES = ("grid", "dimension", "annotation")


def _live(e):
    return e is not None and e.status != Status.REJECTED


def _layer_balcony_score(w, drawing, sem) -> float:
    """Balcony / railing semantics of the wall's own source layers (grid lines ignored)."""
    best, seen = 0.0, set()
    for rid in w.source_entities:
        raw = drawing.raw_entities.get(rid) if drawing is not None else None
        if raw is None or raw.layer in seen:
            continue
        seen.add(raw.layer)
        sc = category_scores(raw.layer, sem)
        if sc and max(sc, key=sc.get) in IGNORED_LAYER_CATEGORIES:
            continue
        best = max(best, sc.get("balcony", 0.0))
    return best


def exterior_reference_thickness(walls) -> float | None:
    """Length-weighted median thickness of the exterior walls (the drawing's facade)."""
    vals = sorted((w.thickness, w.length) for w in walls
                  if _live(w) and w.classification == "exterior_wall" and w.thickness and w.length > 0
                  and FACADE_LINE_NOTE not in w.notes)
    if not vals:
        return None
    total = sum(L for _, L in vals)
    acc = 0.0
    for t, L in vals:
        acc += L
        if acc >= total / 2:
            return float(t)
    return float(vals[-1][0])


def _contact(room, w, tol) -> float:
    """Length of the room's edge along the wall's body."""
    poly = w.polygon(max(tol.snap * 2, 0.01))
    if poly is None or room.polygon is None:
        return 0.0
    return float(robust_intersection(room.polygon.exterior, _sbuf(poly, max(tol.snap * 3, 0.02))).length)


def _free_part(seg, w, walls, tol):
    """Length of a railing contact that is not inside the body of a heavier wall: a thin line
    running inside a thick facade wall is a line of that facade (render, cladding), not a rail.
    The room edge is offset from the walls' axes, so the heavier bodies are probed sideways."""
    if seg is None or seg.is_empty:
        return seg
    reach = (w.thickness or tol.snap * 2) + tol.snap * 2
    probe = _sbuf(seg, reach)
    cover = None
    for v in walls:
        if v is w or not v.thickness or (w.thickness and v.thickness <= 1.25 * w.thickness):
            continue
        poly = v.polygon()
        if poly is None or not poly.intersects(probe):
            continue
        cover = poly if cover is None else robust_union([cover, poly])
    if cover is None:
        return seg
    from ..geometry.geometric_utils import robust_difference
    return robust_difference(seg, _sbuf(cover, reach))


def detect_balconies(b, cfg, tol, drawing=None) -> dict:
    bc = cfg["classification"].get("balcony", {})
    log = {"balconies": [], "railings": 0, "facade_walls": 0}
    if not bc.get("enabled", True):
        return {"skipped": "disabled"}
    sem = cfg["layer_semantics"]["categories"]
    walls = {w.id: w for w in b.walls.values() if _live(w)}
    t_ext = exterior_reference_thickness(walls.values())
    rel = bc.get("railing_max_rel_thickness", 0.6)
    pad = max(tol.snap * 3, 0.02)

    railing_ev: dict[str, float] = {}
    explicit: set[str] = set()                        # named a railing (layer / detector), not only thin
    ops = getattr(b, "openings", {}) or {}
    log["walls_with_doors_not_railing"] = 0
    for w in walls.values():
        if w.classification != "exterior_wall" or FACADE_LINE_NOTE in w.notes:
            continue                                  # a line inside the facade is not a rail
        if any(oid in ops and _live(ops[oid]) and ops[oid].type == "door" for oid in w.openings):
            log["walls_with_doors_not_railing"] += 1
            continue                                  # a railing never has a door: this is a wall
        ev = 0.0
        lay = _layer_balcony_score(w, drawing, sem)
        if lay >= 0.5:
            ev = max(ev, lay)
            explicit.add(w.id)
        if w.evidence.get("railing_label", 0.0) >= 0.5:
            ev = max(ev, w.evidence["railing_label"])     # a detector recognised the railing itself
            explicit.add(w.id)
        if not w.thickness or w.kind == "centerline":
            ev = max(ev, 0.7)                         # single line on the outline: rail / parapet line
        elif t_ext and w.thickness <= rel * t_ext:
            ev = max(ev, 0.8)
        if ev > 0:
            railing_ev[w.id] = ev
    extra = rel * t_ext if t_ext else tol.snap * 4
    junction = max(t_ext or 0.0, tol.snap * 4) + pad      # end overlap of a wall running into another
    outline_zone = b.footprint.boundary if getattr(b, "footprint", None) is not None else None

    kw = cfg["classification"]["room_keywords"]
    for r in b.rooms.values():
        if not _live(r) or r.polygon is None or r.is_hypothesis:
            continue
        ltype = None
        for t in r.label_texts:
            ltype = room_type_of(t, kw)
            if ltype:
                break
        if ltype and ltype != "balcony":
            continue
        ring = r.polygon.exterior
        perim = max(ring.length, 1e-9)
        contact: dict[str, float] = {}
        on_outline: dict[str, float] = {}
        rail_segs: dict = {}
        for wid in r.bounding_walls:
            w = walls.get(wid)
            poly = w.polygon(max(tol.snap * 2, 0.01)) if w is not None else None
            if poly is None:
                continue
            seg = robust_intersection(ring, _sbuf(poly, pad))
            if seg.length <= tol.snap:
                continue
            contact[wid] = seg.length
            # the part of this edge that is on the building outline: the room edge is the inner
            # face of the wall, the outline its outer face (one wall thickness away)
            if outline_zone is not None:
                # (+ one more light element: balconies often show a slab edge / planter line
                # just outside the railing)
                reach = (w.thickness or tol.snap * 2) + extra + pad
                part = robust_intersection(seg, _sbuf(outline_zone, reach))
                on_outline[wid] = part.length
                rail_segs[wid] = part
        # union, not sum: overlapping duplicate walls and corner pieces must not count twice
        outline_geom = robust_union([g for g in rail_segs.values() if not g.is_empty]) if rail_segs else None
        rail_geom = robust_union([_free_part(rail_segs[wid], walls[wid], walls.values(), tol)
                                 for wid in rail_segs if wid in railing_ev]) if rail_segs else None
        outline = outline_geom.length if outline_geom is not None else 0.0
        rail = rail_geom.length if rail_geom is not None else 0.0
        attached = any(wid not in railing_ev for wid in contact)
        outline_share = outline / perim
        rail_share = rail / outline if outline > 0 else 0.0
        geometric = (outline_share >= bc.get("min_outline_share", 0.25) and rail_share >= bc.get("min_railing_share", 0.6)
                     and attached and r.area <= bc.get("max_area_m2", 40.0))
        if outline_share > 0:
            log.setdefault("measured", {})[r.id] = {
                "outline_share": round(outline_share, 3), "railing_share": round(rail_share, 3),
                "attached": attached, "area": round(r.area, 2),
                "edges": {wid: [round(contact[wid], 2), round(on_outline.get(wid, 0.0), 2),
                                walls[wid].classification, wid in railing_ev] for wid in sorted(contact)}}
        if not (geometric or ltype == "balcony"):
            continue
        r.classification = "balcony"
        r.add_evidence("outline_share", min(1.0, outline_share))
        r.add_evidence("railing_share", rail_share)
        why = "labelled balcony / terrace" if ltype == "balcony" else (
            f"{outline_share:.0%} of its edge is on the building outline, {rail_share:.0%} of that is railing / "
            f"parapet (lighter than the {t_ext or 0:.2f} m facade or on a balcony layer)")
        r.notes.append(f"balcony: {why}")
        log["balconies"].append(r.id)
        for wid in contact:
            w = walls[wid]
            if wid in railing_ev:
                # a railing bounds only outdoor space: never the side of an indoor room (a sliver
                # where its end runs into the facade is a junction, not a side)
                indoor = [x for x in w.rooms if x in b.rooms and x != r.id and _live(b.rooms[x])
                          and b.rooms[x].classification != "balcony"
                          and _contact(b.rooms[x], w, tol) > junction]
                if not indoor and w.classification != "railing":
                    w.classification = "railing"
                    w.notes.append(f"railing / parapet of balcony {r.id}")
                    log["railings"] += 1
            elif w.classification != "exterior_wall":
                indoor = [x for x in w.rooms if x in b.rooms and x != r.id and _live(b.rooms[x])]
                if indoor:
                    w.classification = "exterior_wall"
                    w.notes.append(f"facade towards balcony {r.id}")
                    log["facade_walls"] += 1
    # a railing does not need a closed balcony room: an outline element named a railing (balcony /
    # railing layer, detector label) that bounds no indoor room is an outdoor edge (open balconies,
    # galleries, terraces drawn without a closing line)
    balconies = {rid for rid in log["balconies"]}
    log["open_railings"] = 0
    for wid in explicit:
        w = walls[wid]
        if w.classification != "exterior_wall":
            continue
        indoor = [x for x in w.rooms if x in b.rooms and x not in balconies and _live(b.rooms[x])
                  and b.rooms[x].classification != "balcony" and _contact(b.rooms[x], w, tol) > junction]
        if not indoor:
            w.classification = "railing"
            w.notes.append("railing / parapet: named a railing and bounds no indoor room (open balcony edge)")
            log["railings"] += 1
            log["open_railings"] += 1
    # a wall bounding a balcony on one side and an indoor room on the other is facade
    for w in walls.values():
        if w.classification == "railing":
            if any(x in b.rooms and x not in balconies and _live(b.rooms[x]) and _contact(b.rooms[x], w, tol) > junction
                   for x in w.rooms):
                w.classification = "exterior_wall"
                log["railings"] -= 1
    if log["facade_walls"]:
        from .wall_classifier import inherit_facade  # noqa: PLC0415
        log["facade_walls"] += inherit_facade(list(walls.values()), tol)
    log["exterior_reference_thickness_m"] = round(t_ext, 4) if t_ext else None
    return log
