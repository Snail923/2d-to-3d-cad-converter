"""Approach 2: type the gaps left in walls from the model's labels of the lines inside them.

After reconstruction, a gap in a wall with no door / window placed in it is an *unidentified
opening*. The lines drawn inside the gap say what it is: when most of the labelled length there is
door (or window / curtain wall), the gap is that opening.

* If a door / window object owns those lines (shares DXF entities with them) but was placed
  elsewhere - typically across the gap along the wall its open leaf rests against - it is moved
  into the gap (the gap is the strongest evidence of where an opening is).
* Otherwise the gap itself becomes the door / window.

Every decision is recorded in the element's notes with the label shares."""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from architectural_converter.geometry.geometric_utils import dot, rect_polygon
from architectural_converter.models.base_entity import Status

from .classes import CLASSES, GROUP_OF

KIND_GROUPS = {"door": ("door", "opening"), "window": ("window", "curtain_wall")}
IGNORE_GROUPS = ("wall",)                 # jamb / face lines of the wall itself


def _live(e):
    return e is not None and e.status != Status.REJECTED


def type_gaps(generic, doors, windows, walls, prims, labels, cfg) -> dict:
    gc = cfg["build"]
    log = {"gaps": 0, "doors_moved": 0, "windows_moved": 0, "gaps_to_doors": 0, "gaps_to_windows": 0}
    plab = {int(k): v for k, v in (labels.get("primitives") or {}).items()}
    if not generic or not plab:
        return log
    idx = [k for k in plab if k < len(prims) and len(prims[k][5]) >= 2]
    if not idx:
        return log
    geoms = [LineString(prims[k][5]) for k in idx]
    tree = STRtree(geoms)
    walls_by_id = {w.id: w for w in walls}
    objects = {"door": [o for o in doors if _live(o)], "window": [o for o in windows if _live(o)]}
    min_share = gc.get("gap_min_share", 0.5)
    min_len = gc.get("gap_min_labelled_m", 0.3)
    for g in generic:
        if not _live(g) or g.type != "opening" or not g.host_wall or not g.host_interval:
            continue
        w = walls_by_id.get(g.host_wall)
        if w is None or w.kind not in ("straight", "centerline"):
            continue
        log["gaps"] += 1
        a, b = sorted(g.host_interval)
        band = rect_polygon(w.u, w.c, a, b, (w.thickness or gc["nominal_thickness_m"]) + 0.2)
        by_group: dict[str, float] = {}
        raw_in: set[str] = set()
        for i in tree.query(band):
            ls = geoms[int(i)]
            if not band.contains(ls.centroid):
                continue
            k = idx[int(i)]
            grp = GROUP_OF.get(CLASSES[plab[k][0]], "other")
            if grp in IGNORE_GROUPS:
                continue
            by_group[grp] = by_group.get(grp, 0.0) + prims[k][4]
            raw_in.add(prims[k][0])
        total = sum(by_group.values())
        if total < min_len:
            continue
        for kind, groups in KIND_GROUPS.items():
            share = sum(by_group.get(x, 0.0) for x in groups) / total
            if share < min_share:
                continue
            # the object drawn with these lines, if it was placed somewhere else
            owner = None
            best = 0
            for o in objects[kind]:
                if o.host_wall == g.host_wall and o.host_interval and \
                        min(b, max(o.host_interval)) - max(a, min(o.host_interval)) > 0.5 * (b - a):
                    owner = None
                    best = -1
                    break                                     # already in this gap
                n = len(o.source_entities & raw_in)
                if n > best:
                    owner, best = o, n
            if best < 0:
                break
            why = f"SymPointV2: {share:.0%} of the labelled lines in the gap are {kind}"
            if owner is not None and best > 0:
                old = walls_by_id.get(owner.host_wall)
                if old is not None and owner.id in old.openings:
                    old.openings.remove(owner.id)
                owner.p0, owner.p1 = g.p0, g.p1
                owner.width = g.width
                owner.depth = w.thickness or owner.depth
                owner.host_wall, owner.host_interval = g.host_wall, g.host_interval
                owner.alt_chords = []
                owner.notes.append(f"moved into the wall gap {g.id}: {why}")
                owner.add_evidence("gap_match", 1.0)
                owner.add_evidence("host_wall", 1.0)
                if owner.id not in w.openings:
                    w.openings.append(owner.id)
                g.veto(f"the gap holds {owner.id} ({why})")
                log[f"{kind}s_moved"] += 1
            else:
                g.type = kind
                g.subtype = "from_gap"
                g.add_evidence("sympoint_label", share)
                g.notes.append(f"gap typed as {kind}: {why}")
                g.status_reason = f"gap typed as {kind} from the model's labels"
                log[f"gaps_to_{kind}s"] += 1
            break
    return log
