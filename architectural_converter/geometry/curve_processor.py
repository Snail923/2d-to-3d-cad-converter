"""Arc / curve helpers (angles always radians, arcs stored CCW)."""
from __future__ import annotations

import math

TWO_PI = 2 * math.pi


def norm_angle(a: float) -> float:
    return a % TWO_PI


def angle_in_arc(a: float, start: float, sweep: float, tol: float = 1e-9) -> bool:
    return (norm_angle(a - start)) <= sweep + tol


def arc_overlap(start1: float, sweep1: float, start2: float, sweep2: float) -> tuple[float, float] | None:
    """Angular overlap of two CCW arcs: returns (start, sweep) of the overlap (largest piece)."""
    best = None
    for s_a, w_a, s_b, w_b in ((start1, sweep1, start2, sweep2), (start2, sweep2, start1, sweep1)):
        d = norm_angle(s_b - s_a)
        if d <= w_a:
            w = min(w_a - d, w_b)
            if w > 0 and (best is None or w > best[1]):
                best = (norm_angle(s_b), w)
    return best


def chord_points(cx, cy, r, a):
    return (cx + r * math.cos(a), cy + r * math.sin(a))
