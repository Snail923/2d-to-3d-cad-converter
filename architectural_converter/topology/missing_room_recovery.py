"""Missing-room recovery (hypotheses only - never silently added).

1. collect free wall ends and room labels that are not inside any room (or rooms holding
   several labels);
2. propose closing segments: collinear extension of a free end until it meets another wall,
   or a straight link between two free ends aligned with one of the walls;
3. a proposal is kept only if adding it encloses a new plausible room; it gains confidence
   when the new room captures an orphan label or separates two labels of one region;
4. kept proposals become hypothesis walls (low confidence, NEEDS_REVIEW) and the rooms they
   would create become room hypotheses. Phase 2 receives them only if accepted by a user.
"""
from __future__ import annotations

import math

from shapely.geometry import LineString, Point

from ..geometry.geometric_utils import canonical_dir, dot, normal, robust_intersection
from ..models.base_entity import Status
from ..models.room import Room
from ..models.wall import Wall
from .graph_builder import free_ends
from .room_detector import holes_of, wall_solid


def recover_missing_rooms(walls, columns, rooms, labels, tol, cfg, ids, room_detector):
    rc = cfg["reconstruction"]
    vc = cfg["validation"]
    live = [w for w in walls if w.status != Status.REJECTED]
    ends = free_ends(live, tol)
    room_polys = [r.polygon for r in rooms if r.status != Status.REJECTED]
    orphan = [lb for lb in labels if lb.category in ("room_name", "room_number")
              and not any(p.contains(Point(lb.position)) for p in room_polys)]
    multi = [r for r in rooms if len(r.label_texts) > 1]
    if not ends:
        return [], [], {"free_ends": 0, "proposals": 0, "kept": 0, "orphan_labels": len(orphan)}
    maxlen = rc["hypothesis_max_length_m"]
    proposals = []
    bodies = [(w, w.polygon(tol.snap * 2)) for w in live]
    for w, e, p in ends:
        out = w.u if e else (-w.u[0], -w.u[1])
        ray = LineString([p, (p[0] + out[0] * maxlen, p[1] + out[1] * maxlen)])
        best = None
        for v, poly in bodies:
            if v is w or poly is None or not poly.intersects(ray):
                continue
            d = Point(p).distance(robust_intersection(poly, ray))
            if d > tol.snap and (best is None or d < best[0]):
                best = (d, v)
        if best is not None:
            q = (p[0] + out[0] * best[0], p[1] + out[1] * best[0])
            proposals.append((p, q, w, best[1].id, "collinear extension"))
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            (wa, ea, pa), (wb, eb, pb) = ends[i], ends[j]
            d = math.dist(pa, pb)
            if d <= tol.snap or d > maxlen:
                continue
            u = canonical_dir(pb[0] - pa[0], pb[1] - pa[1])
            if max(abs(dot(u, wa.u)), abs(dot(u, wb.u))) < 0.985 and min(abs(dot(u, wa.u)), abs(dot(u, wb.u))) > 0.17:
                continue
            proposals.append((pa, pb, wa, wb.id, "link free ends"))
    base_holes = [h for h in holes_of(wall_solid(live, columns, tol, cfg)) if h.area >= vc["min_room_area_m2"]]
    kept = []
    new_rooms = []
    seen = set()
    for p, q, w, other, why in proposals[: 4 * rc["hypothesis_max_count"]]:
        key = (round(p[0], 2), round(p[1], 2), round(q[0], 2), round(q[1], 2))
        if key in seen:
            continue
        seen.add(key)
        u = canonical_dir(q[0] - p[0], q[1] - p[1])
        n = normal(u)
        s0, s1 = sorted((dot(p, u), dot(q, u)))
        t = w.thickness
        hyp = Wall(id="tmp", kind="straight" if t else "centerline", u=u, c=dot(p, n), s0=s0, s1=s1, thickness=t,
                   thickness_source=w.thickness_source if t else "unknown", is_hypothesis=True)
        hp = hyp.polygon(max(tol.snap * 2, 0.01))
        if hp is None:
            continue
        holes = holes_of(wall_solid(live, columns, tol, cfg, extra=[hp]))
        created = [h for h in holes if h.area >= vc["min_room_area_m2"]
                   and not any(abs(h.area - b.area) < 0.01 * b.area and robust_intersection(h, b).area > 0.98 * h.area for b in base_holes)]
        if not created:
            continue
        conf = rc["hypothesis_confidence"]
        captured = [lb for lb in orphan if any(h.contains(Point(lb.position)) for h in created)]
        if captured:
            conf += 0.1
        for r in multi:
            inside = [h for h in created if r.polygon.intersects(h)]
            if inside:
                conf += 0.1
                break
        hyp.id = ids.new("hypothesis_wall")
        hyp.confidence = min(conf, 0.5)
        hyp.status = Status.NEEDS_REVIEW
        hyp.status_reason = f"missing-boundary hypothesis ({why}) from {w.id} to {other}"
        hyp.detection_methods.add("missing_room_recovery")
        hyp.evidence = {"closes_room": 1.0, "orphan_label": 1.0 if captured else 0.0}
        hyp.reconstruction["hypothesis"] = {"from_wall": w.id, "to": other, "reason": why,
                                            "captured_labels": [lb.text for lb in captured]}
        kept.append(hyp)
        for h in created:
            r = Room(id=ids.new("room_hypothesis"), polygon=h, is_hypothesis=True, hypothesis_walls=[hyp.id])
            room_detector._labels(r, labels)
            r.confidence = hyp.confidence
            r.status = Status.NEEDS_REVIEW
            r.status_reason = "room enclosed only if hypothesis wall is accepted"
            new_rooms.append(r)
        if len(kept) >= rc["hypothesis_max_count"]:
            break
    log = {"free_ends": len(ends), "proposals": len(proposals), "kept": len(kept), "orphan_labels": len(orphan)}
    return kept, new_rooms, log
