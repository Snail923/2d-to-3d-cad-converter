"""Configurable architectural rules shared by the classifiers (no geometry detection here)."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from ..geometry.geometric_utils import polygons_of, robust_union


def building_footprint(solid):
    """Filled outline of the wall solid (all parts, holes removed)."""
    if solid is None or solid.is_empty:
        return None
    return robust_union([Polygon(p.exterior) for p in polygons_of(solid)])


def partition_threshold(walls, cfg) -> float:
    """Thickness at or below which an interior wall is a partition.

    Uses the drawing's own thickness classes when they exist: if interior walls form >= 2
    clusters, the thinner ones (<= ratio x thickest interior class) are partitions;
    otherwise the configured absolute limit applies."""
    cc = cfg["classification"]
    ts = sorted({round(w.thickness, 3) for w in walls if w.thickness and w.classification == "interior_wall"})
    limit = cc["partition_max_thickness_m"]
    if len(ts) >= 2:
        thick = max(ts)
        thin = [t for t in ts if t <= cc["partition_cluster_ratio"] * thick]
        if thin:
            limit = max(limit, max(thin) * 1.02) if max(thin) <= 0.16 else limit
    return limit
