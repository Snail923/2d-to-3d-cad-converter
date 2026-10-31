"""Wall checks: free ends, unexpected intersections, overlaps, thickness, duplicates, support."""
from __future__ import annotations

from ..classification.candidate_resolver import same_band
from ..models.base_entity import Status
from ..topology.graph_builder import free_ends


def validate_walls(report, walls, columns, tol, cfg):
    vc = cfg["validation"]
    th = cfg["thickness"]
    live = [w for w in walls if w.status != Status.REJECTED and not w.is_hypothesis]
    for w, e, p in free_ends(live, tol):
        report.add("wall", "unconnected_endpoint", "warning", w, p,
                   f"Wall {w.id} has an unconnected {'end' if e else 'start'} point.",
                   "Check for a missing wall segment or junction; connect the endpoint manually if a wall continues.",
                   0.6 if w.openings else 0.7)
    polys = [(w, w.polygon(tol.snap * 2)) for w in live]
    polys = [(w, p) for w, p in polys if p is not None]
    for i, (a, pa) in enumerate(polys):
        for b, pb in polys[i + 1:]:
            if not pa.intersects(pb):
                continue
            inter = pa.intersection(pb)
            if inter.area <= 0:
                continue
            small = min(pa.area, pb.area)
            if same_band(a, b, tol) and inter.area > vc["overlap_fraction"] * small:
                report.add("wall", "duplicate_wall", "warning", a, (inter.centroid.x, inter.centroid.y),
                           f"Walls {a.id} and {b.id} overlap along the same axis.",
                           "Merge or reject one of the duplicates.", 0.8, a.source_entities | b.source_entities)
            elif inter.area > vc["overlap_fraction"] * small and a.kind == b.kind == "straight" and abs(a.u[0] * b.u[0] + a.u[1] * b.u[1]) > 0.95:
                report.add("wall", "overlapping_walls", "warning", a, (inter.centroid.x, inter.centroid.y),
                           f"Parallel walls {a.id} and {b.id} overlap.", "Check thickness / offset of both walls.",
                           0.6, a.source_entities | b.source_entities)
            elif a.kind == b.kind == "straight" and abs(a.u[0] * b.u[1] - a.u[1] * b.u[0]) > 0.3:
                # crossing in the middle of both walls (X) without being a recorded junction
                ca, cb = inter.centroid, inter.centroid
                da = min(abs(a.u[0] * ca.x + a.u[1] * ca.y - s) for s in (a.s0, a.s1))
                db = min(abs(b.u[0] * cb.x + b.u[1] * cb.y - s) for s in (b.s0, b.s1))
                if da > (b.thickness or 0) + tol.snap * 4 and db > (a.thickness or 0) + tol.snap * 4:
                    report.add("wall", "wall_crossing", "info", a, (ca.x, ca.y),
                               f"Walls {a.id} and {b.id} cross each other (X-junction).",
                               "Confirm the crossing is intended (X junction) and not an overlay error.", 0.5,
                               a.source_entities | b.source_entities)
    for w in live:
        c = w.axis.centroid
        loc = (c.x, c.y)
        if w.thickness and not (th["plausible_min_m"] * 0.9 <= w.thickness <= th["plausible_max_m"] * 1.1):
            report.add("wall", "implausible_thickness", "warning", w, loc,
                       f"Wall {w.id} thickness {w.thickness:.3f} m is outside the plausible range.",
                       "Verify units and the pairing of the wall boundaries.", 0.6)
        if not w.thickness:
            report.add("wall", "unknown_thickness", "info", w, loc,
                       f"Wall {w.id} has no measurable thickness (single-line representation).",
                       "Phase 2 will apply the configured default thickness; set it manually if known.", 0.9)
        if w.status == Status.NEEDS_REVIEW:
            report.add("wall", "uncertain_wall", "info", w, loc,
                       f"Wall {w.id} is uncertain (confidence {w.confidence:.2f}).",
                       "Inspect the evidence and accept or reject the candidate.", w.confidence)
        fs = w.reconstruction.get("face_support")
        if fs and w.kind == "straight" and w.thickness_source == "measured" and min(fs.values()) < 0.3 and not w.openings:
            report.add("wall", "unsupported_wall", "warning", w, loc,
                       f"Wall {w.id} is weakly supported by drawn boundaries ({fs}).",
                       "Check whether the candidate is a real wall.", 0.5)
        for g in w.reconstruction.get("hypothesis_gaps", []):
            a, b = g["interval"]
            p = w.point_at((a + b) / 2)
            report.add("wall", "small_gap_closed", "info", w, p,
                       f"A small gap ({b - a:.3f} m) in wall {w.id} was closed by reconstruction.",
                       "Confirm that the gap is a drafting error and not an intentional opening.", g["confidence"])
