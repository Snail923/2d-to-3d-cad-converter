"""Endpoint snapping for centre-line walls: endpoints that lie within the snapping tolerance
of each other are moved to their common cluster centre (recorded per wall)."""
from __future__ import annotations

import numpy as np

from ..geometry.endpoint_processor import cluster_centers, cluster_points
from ..geometry.geometric_utils import dot


def snap_centerline_endpoints(walls, tol) -> int:
    cl = [w for w in walls if w.kind == "centerline"]
    if len(cl) < 2:
        return 0
    P = np.array([w.point_at(s) for w in cl for s in (w.s0, w.s1)])
    labels = cluster_points(P, tol.snap)
    centers = cluster_centers(P, labels)
    moved = 0
    for k, w in enumerate(cl):
        for e in (0, 1):
            i = 2 * k + e
            if (labels == labels[i]).sum() < 2:
                continue
            c = centers[labels[i]]
            s = dot(c, w.u)
            if abs(s - (w.s1 if e else w.s0)) > 1e-9:
                if e:
                    w.s1 = max(w.s1, s)
                else:
                    w.s0 = min(w.s0, s)
                w.reconstruction.setdefault("snapped", []).append({"end": e, "to": [round(c[0], 4), round(c[1], 4)]})
                moved += 1
    return moved
