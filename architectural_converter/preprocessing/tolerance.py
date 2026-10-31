"""Adaptive tolerance estimation.

Tolerances are not fixed numbers. They are estimated per drawing from
* units / numeric precision (coordinate magnitude),
* the distribution of near-miss endpoint gaps (drafting noise vs. intentional gaps),
* the thinnest plausible wall (snapping must never bridge a wall thickness),
and bounded by configurable priors. Separate values are kept for duplicates, endpoint
snapping, intersections, parallel comparison and thickness estimation."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ..geometry.endpoint_processor import nearest_endpoint_distances


@dataclass
class Tolerances:
    precision: float
    duplicate: float
    snap: float
    intersection: float
    angle: float              # radians, base parallel tolerance
    angle_max: float
    parallel_offset_frac: float
    thickness_min: float
    thickness_max: float
    thickness_clusters: list[dict] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def pair_angle_tol(self, len_a: float, len_b: float) -> float:
        L = max(min(len_a, len_b), 1e-9)
        return min(self.angle_max, max(self.angle, math.atan2(2 * self.snap, L)))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["angle_deg"] = math.degrees(self.angle)
        d["angle_max_deg"] = math.degrees(self.angle_max)
        return d


def estimate_tolerances(drawing, segments, cfg) -> Tolerances:
    tc = cfg["tolerances"]
    th = cfg["thickness"]
    # numeric precision from the original coordinate magnitude
    max_abs = 0.0
    for g in drawing.entities:
        if g.points is not None and len(g.points):
            max_abs = max(max_abs, float(np.max(np.abs(g.points))))
    off = np.abs(np.asarray(drawing.transform.offset)) * drawing.transform.scale
    magnitude = max_abs + float(np.max(off)) if len(off) else max_abs
    precision = max(magnitude * tc["precision_relative"], 1e-9)
    duplicate = max(tc["duplicate_tol_m"], 10 * precision)

    cap = min(tc["snap_max_m"], tc["snap_max_fraction_of_min_thickness"] * th["min_m"])
    ev: dict = {"cap": cap}
    live = [s for s in segments if s.dup_of is None and s.role in ("linework", "assembly")]
    snap = tc["snap_prior_m"]
    if len(live) >= 4:
        P = np.array([p for s in live for p in (s.p0, s.p1)])
        owner = np.repeat(np.arange(len(live)), 2)
        dists = nearest_endpoint_distances(P, owner, cap * 1.5)
        finite = dists[np.isfinite(dists)]
        near = np.sort(finite[(finite > duplicate) & (finite <= cap * 1.5)])
        ev["endpoints"] = int(len(P))
        ev["coincident_share"] = float(np.mean(finite <= duplicate)) if len(finite) else 0.0
        ev["near_miss_count"] = int(len(near))
        if len(near) >= 3:
            # natural break in log-space separates drafting noise from real gaps
            logs = np.log(near)
            jumps = np.diff(logs)
            upper = near[-1]
            for k, jmp in enumerate(jumps):
                if jmp > math.log(2.5) and k >= 1:
                    upper = near[k]
                    break
            snap = 1.5 * float(upper)
            ev["noise_upper"] = float(upper)
    snap = float(min(cap, max(tc["snap_min_m"], snap, 3 * duplicate)))
    return Tolerances(
        precision=precision,
        duplicate=duplicate,
        snap=snap,
        intersection=snap * tc["intersection_tol_factor"],
        angle=math.radians(tc["angle_tol_deg"]),
        angle_max=math.radians(tc["angle_tol_max_deg"]),
        parallel_offset_frac=tc["parallel_offset_tol_fraction"],
        thickness_min=th["min_m"],
        thickness_max=th["max_m"],
        evidence=ev,
    )
