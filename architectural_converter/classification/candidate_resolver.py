"""Unified candidate resolver.

Combines candidates of all detectors:
* merges duplicate wall candidates (same axis band and thickness, overlapping) - evidence
  is merged per feature with ``max`` so the same underlying evidence is never counted twice;
* attaches layer-linework evidence to geometric walls sharing boundary segments;
* adds context evidence (hatch fill coverage, dimension support);
* absorbs polygon walls that are explained by straight candidates;
* resolves conflicts between walls and openings / columns (glazing lines, door leaves and
  column outlines are not walls) and between overlapping incompatible walls.
All decisions are recorded (status_reason, notes)."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from ..geometry.geometric_utils import angle_diff, dir_angle, dot, frame_point, normal
from ..models.base_entity import Status
from ..models.wall import Wall


def aligned(w: Wall, u):
    """Return (c, s0, s1) of straight wall ``w`` expressed in direction ``u``."""
    if dot(w.u, u) >= 0:
        return w.c, w.s0, w.s1
    return -w.c, -w.s1, -w.s0


def same_band(a: Wall, b: Wall, tol, t_rel=0.2) -> bool:
    if a.kind not in ("straight", "centerline") or b.kind not in ("straight", "centerline"):
        return False
    if angle_diff(dir_angle(a.u), dir_angle(b.u)) > tol.angle_max:
        return False
    cb, _, _ = aligned(b, a.u)
    ta, tb = a.thickness, b.thickness
    lim = max(2 * tol.snap, 0.25 * min([t for t in (ta, tb) if t] or [0.0]))
    if abs(cb - a.c) > lim:
        return False
    if ta and tb and abs(ta - tb) > max(2 * tol.snap, t_rel * max(ta, tb)):
        return False
    return True


class CandidateResolver:
    def __init__(self, cfg):
        self.cfg = cfg

    def resolve(self, ctx):
        tol = ctx.tol
        walls: list[Wall] = []
        for name in ("parallel_wall_detector", "polyline_wall_detector", "hatch_wall_detector",
                     "single_line_wall_detector"):
            walls += list(ctx.candidates.get(name, []))
        layer_lw = list(ctx.candidates.get("layer_wall_detector", []))
        doors = list(ctx.candidates.get("door_detector", []))
        windows = list(ctx.candidates.get("window_detector", []))
        columns = list(ctx.candidates.get("column_detector", []))
        log = {"merged_walls": 0, "absorbed_polygons": 0, "layer_evidence_attached": 0,
               "walls_rejected_by_openings": 0, "walls_rejected_by_columns": 0, "conflicts": 0}

        walls = self._merge_straight(walls, tol, log)
        self._attach_layer(walls, layer_lw, log)
        self._context_evidence(walls, ctx)
        walls = self._absorb_polygons(walls, tol, log)
        self._opening_conflicts(walls, doors, windows, tol, log)
        self._element_faces(walls, doors, windows, log)
        self._column_conflicts(walls, columns, tol, log)
        self._door_window_conflicts(doors, windows, tol)
        self._arc_door_conflicts(walls, doors, tol)
        w_weights = self.cfg["detection"]["weights"]["wall"]
        th = self.cfg["detection"]["status_thresholds"]
        gate = self.cfg["detection"].get("gates", {}).get("wall")
        for w in walls:
            w.score(w_weights, th, gate)
        self._overlap_conflicts(walls, tol, log)
        ctx.stats["resolver"] = log
        return walls, doors, windows, columns

    # ------------------------------------------------------------------
    def _merge_straight(self, walls, tol, log):
        straight = [w for w in walls if w.kind in ("straight", "centerline")]
        other = [w for w in walls if w.kind not in ("straight", "centerline")]
        straight.sort(key=lambda w: (0 if w.thickness else 1, -w.length))
        out: list[Wall] = []
        for w in straight:
            for m in out:
                if not same_band(m, w, tol):
                    continue
                c, s0, s1 = aligned(w, m.u)
                if min(m.s1, s1) - max(m.s0, s0) <= tol.snap:
                    continue
                # merge w into m
                Lm, Lw = m.length, s1 - s0
                if m.thickness and w.thickness:
                    m.thickness = (m.thickness * Lm + w.thickness * Lw) / (Lm + Lw)
                elif w.thickness and not m.thickness:
                    m.thickness, m.thickness_source, m.kind = w.thickness, w.thickness_source, "straight"
                if m.thickness_source == "measured" and w.thickness_source in ("mline", "polyline_width"):
                    m.thickness_source = w.thickness_source
                    m.thickness = w.thickness
                if (w.thickness is not None) == (m.thickness is not None):
                    m.c = (m.c * Lm + c * Lw) / (Lm + Lw)
                m.s0, m.s1 = min(m.s0, s0), max(m.s1, s1)
                di = m.detected_interval or (m.s0, m.s1)
                m.detected_interval = (min(di[0], s0), max(di[1], s1))
                m.absorb(w)
                m.face_segments |= w.face_segments
                log["merged_walls"] += 1
                break
            else:
                out.append(w)
        return out + other

    def _attach_layer(self, walls, layer_lw, log):
        by_seg: dict[str, list[Wall]] = {}
        for w in walls:
            for s in w.face_segments:
                by_seg.setdefault(s, []).append(w)
        for lw in layer_lw:
            hit = set()
            for s in lw.face_segments:
                for w in by_seg.get(s, []):
                    if id(w) not in hit:
                        hit.add(id(w))
                        w.add_evidence("layer_match", lw.evidence.get("layer_match", 0))
                        w.detection_methods.add("layer_wall_detector")
                        log["layer_evidence_attached"] += 1

    def _context_evidence(self, walls, ctx):
        tol = ctx.tol
        fills = [h.polygon for h in ctx.hatch_regions if h.cls == "wall_fill"]
        fill_u = unary_union(fills).buffer(tol.snap * 2) if fills else None
        cov_min = self.cfg["detection"]["hatch"]["coverage_for_fill_evidence"]
        for w in walls:
            p = w.polygon()
            if p is None or p.area <= 0:
                continue
            if fill_u is not None:
                cov = fill_u.intersection(p).area / p.area
                if cov >= cov_min:
                    w.add_evidence("hatch_fill", cov)
                    for h in ctx.hatch_regions:
                        if h.cls == "wall_fill" and h.polygon.intersects(p):
                            w.source_entities.add(h.raw_id)
            if w.kind == "straight" and w.thickness:
                for d in ctx.dimensions:
                    dp = d["defpoints"]
                    a, b = dp.get("defpoint2"), dp.get("defpoint3")
                    if a is None or b is None:
                        continue
                    if abs(d["measurement"] - w.thickness) > max(2 * tol.snap, 0.05 * w.thickness):
                        continue
                    n = normal(w.u)
                    ca, cb = dot(a, n), dot(b, n)
                    sa, sb = dot(a, w.u), dot(b, w.u)
                    faces = sorted((ca, cb))
                    if abs(faces[0] - (w.c - w.thickness / 2)) <= 2 * tol.snap and abs(faces[1] - (w.c + w.thickness / 2)) <= 2 * tol.snap \
                            and w.s0 - tol.snap <= sa <= w.s1 + tol.snap:
                        w.add_evidence("dimension_support", 1.0)
                        w.source_entities.add(d["raw_id"])

    def _absorb_polygons(self, walls, tol, log):
        straight = [w for w in walls if w.kind in ("straight", "arc") and w.polygon() is not None]
        su = unary_union([w.polygon() for w in straight]).buffer(tol.snap * 3) if straight else None
        out = []
        for w in walls:
            if w.kind == "polygon" and su is not None and w.shape is not None:
                cov = su.intersection(w.shape).area / max(w.shape.area, 1e-12)
                if cov >= 0.85:
                    for s in straight:
                        if s.polygon().intersects(w.shape):
                            s.absorb(w)
                    log["absorbed_polygons"] += 1
                    continue
            out.append(w)
        return out

    def _opening_conflicts(self, walls, doors, windows, tol, log):
        regions = []
        for o in doors + windows:
            if o.footprint is not None and o.confidence >= self.cfg["detection"]["status_thresholds"]["review"]:
                regions.append((o, o.footprint.buffer(tol.snap * 2)))
        for w in walls:
            p = w.polygon(fallback_thickness=tol.snap * 2)
            if p is None or p.area <= 0:
                continue
            for o, fp in regions:
                if fp.intersection(p).area >= 0.8 * p.area:
                    if o.type == "window" and o.depth and (not w.thickness or w.thickness > o.depth * 0.9):
                        continue  # the wall band contains the window: only thinner glazing pairs are vetoed
                    w.veto(f"inside {o.type} {o.id}")
                    log["walls_rejected_by_openings"] += 1
                    break

    def _element_faces(self, walls, doors, windows, log):
        """A wall cannot use a door leaf / swing arc / glazing line as one of its whole faces."""
        th = self.cfg["detection"]["status_thresholds"]["review"]
        elem = set()
        for o in doors:
            if o.confidence >= th and not o.symbol_insert:
                elem |= o.source_entities
        for o in windows:
            if o.confidence >= th and "glazing_lines" in o.detection_methods:
                elem |= o.source_entities
        for w in walls:
            faces = getattr(w, "face_raw", None)
            if faces and any(f and f <= elem for f in faces):
                w.veto("boundary is a door/window element")
                log["walls_rejected_by_openings"] += 1

    def _column_conflicts(self, walls, columns, tol, log):
        cols = [(c, c.polygon.buffer(tol.snap * 2)) for c in columns if c.status != Status.REJECTED]
        for w in walls:
            p = w.polygon(fallback_thickness=tol.snap * 2)
            if p is None or p.area <= 0:
                continue
            for c, cp in cols:
                if cp.intersection(p).area >= 0.7 * p.area:
                    w.veto(f"inside column {c.id}")
                    log["walls_rejected_by_columns"] += 1
                    break

    def _door_window_conflicts(self, doors, windows, tol):
        for d in doors:
            if d.footprint is None:
                continue
            for w in windows:
                if w.footprint is None:
                    continue
                if w.footprint.intersection(d.footprint.buffer(tol.snap)).area > 0.5 * w.footprint.area:
                    loser = w if d.confidence >= w.confidence else d
                    loser.notes.append("conflicting door/window interpretation")
                    loser.add_evidence("width_plausibility", 0.0)

    def _arc_door_conflicts(self, walls, doors, tol):
        arcs = [w for w in walls if w.kind == "arc"]
        for d in doors:
            for w in arcs:
                if w.source_entities & d.source_entities:
                    d.veto(f"arc belongs to curved wall {w.id}")

    def _overlap_conflicts(self, walls, tol, log):
        live = [w for w in walls if w.status != Status.REJECTED and w.polygon() is not None]
        live.sort(key=lambda w: -w.confidence)
        kept = []
        for w in live:
            p = w.polygon()
            for k in kept:
                if k.kind == w.kind == "straight" and abs(dot(k.u, w.u)) < 0.95:
                    continue  # crossing walls at junctions are expected
                inter = k.polygon().intersection(p).area
                if inter >= 0.6 * p.area:
                    w.veto(f"overlaps stronger candidate {k.id}")
                    log["conflicts"] += 1
                    break
            else:
                kept.append(w)
