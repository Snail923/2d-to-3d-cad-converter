"""Endpoint statistics and clustering."""
from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPoint, Point
from shapely.strtree import STRtree


def nearest_endpoint_distances(P: np.ndarray, owner: np.ndarray, cap: float) -> np.ndarray:
    """For each endpoint, distance to the nearest endpoint of a *different* segment (<= cap)."""
    if len(P) < 2:
        return np.zeros(0)
    pts = [Point(p) for p in P]
    tree = STRtree(pts)
    out = np.full(len(P), np.inf)
    for i, p in enumerate(P):
        idx = tree.query(Point(p).buffer(cap, 4))
        best = np.inf
        for j in idx:
            j = int(j)
            if owner[j] == owner[i]:
                continue
            dd = float(np.hypot(*(P[j] - p)))
            if dd < best:
                best = dd
        out[i] = best
    return out


def cluster_points(P: np.ndarray, tol: float) -> np.ndarray:
    """Greedy single-link clustering; returns label per point."""
    n = len(P)
    labels = -np.ones(n, dtype=int)
    if n == 0:
        return labels
    tree = STRtree([Point(p) for p in P])
    k = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        stack = [i]
        labels[i] = k
        while stack:
            a = stack.pop()
            for j in tree.query(Point(P[a]).buffer(tol, 4)):
                j = int(j)
                if labels[j] < 0 and np.hypot(*(P[j] - P[a])) <= tol:
                    labels[j] = k
                    stack.append(j)
        k += 1
    return labels


def cluster_centers(P: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.zeros((labels.max() + 1 if len(labels) else 0, 2))
    for k in range(len(out)):
        out[k] = P[labels == k].mean(axis=0)
    return out
