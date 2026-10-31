"""Degenerate / invalid geometry identification. Invalid entities are flagged, never deleted."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from ..models.drawing import Drawing


def flag_degenerate(d: Drawing, tol: float) -> dict[str, int]:
    counts: dict[str, int] = {}

    def bad(g, why):
        g.valid = False
        g.flags.add(why)
        counts[why] = counts.get(why, 0) + 1

    for g in d.entities:
        if g.kind == "line":
            if np.hypot(*(g.points[1] - g.points[0])) <= tol:
                bad(g, "zero_length")
        elif g.kind in ("polyline", "curve", "mline"):
            if g.points is None or len(g.points) < 2:
                bad(g, "too_few_points")
            elif float(np.max(np.ptp(g.points, axis=0))) <= tol:
                bad(g, "zero_extent")
        elif g.kind == "arc":
            if g.data.get("radius", 0) <= tol:
                bad(g, "zero_radius")
        elif g.kind == "circle":
            if g.data.get("radius", 0) <= tol:
                bad(g, "zero_radius")
        elif g.kind == "polygon":
            if len(g.points) < 3 or Polygon(g.points).area <= tol * tol:
                bad(g, "zero_area")
        elif g.kind == "hatch":
            loops = [lp for lp in g.data.get("loops", []) if len(lp) >= 3 and abs(Polygon(lp).area) > tol * tol]
            if not loops:
                bad(g, "empty_hatch")
            g.data["loops"] = loops
        elif g.kind == "text":
            if not str(g.data.get("text", "")).strip():
                bad(g, "empty_text")
        if g.points is not None and not np.all(np.isfinite(g.points)):
            bad(g, "non_finite")
    if counts:
        d.issues.append({"stage": "preprocessing", "severity": "info",
                         "message": "Degenerate geometry flagged (excluded from detection)", "details": counts})
    return counts
