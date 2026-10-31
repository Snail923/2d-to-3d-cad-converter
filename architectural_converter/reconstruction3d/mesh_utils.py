"""Small mesh helpers (numpy only): oriented boxes, polygon triangulation, extrusion."""
from __future__ import annotations

import numpy as np


def box(u, c, s0, s1, t, z0, z1):
    """Oriented box along direction u, offset c (normal), [s0, s1] x [c - t/2, c + t/2] x [z0, z1]."""
    u = np.asarray(u, float)
    n = np.array([-u[1], u[0]])
    h = t / 2.0
    base = [s0 * u + (c - h) * n, s1 * u + (c - h) * n, s1 * u + (c + h) * n, s0 * u + (c + h) * n]
    V = np.array([[p[0], p[1], z0] for p in base] + [[p[0], p[1], z1] for p in base])
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return V, F


def _area2(P):
    x, y = P[:, 0], P[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _point_in_tri(p, a, b, c):
    def s(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def earclip(ring) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple polygon ring (no repeated closing point).
    Returns triangles (CCW) as index triples into ``ring``."""
    P = np.asarray(ring, float)
    n = len(P)
    if n < 3:
        return []
    idx = list(range(n))
    if _area2(P) < 0:
        idx.reverse()
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10 * n * n:
        guard += 1
        m = len(idx)
        clipped = False
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = P[i0], P[i1], P[i2]
            cr = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cr <= 1e-12:
                continue
            if any(_point_in_tri(P[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            clipped = True
            break
        if not clipped:  # degenerate remainder: fan it
            break
    if len(idx) >= 3:
        for k in range(1, len(idx) - 1):
            tris.append((idx[0], idx[k], idx[k + 1]))
    return tris


def extrude(ring, z0, z1):
    """Prism from a simple polygon ring."""
    P = np.asarray(ring, float)
    if len(P) > 1 and np.allclose(P[0], P[-1]):
        P = P[:-1]
    n = len(P)
    if n < 3:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    if _area2(P) < 0:
        P = P[::-1]
    tris = earclip(P)
    V = np.vstack([np.column_stack([P, np.full(n, z0)]), np.column_stack([P, np.full(n, z1)])])
    F = []
    for a, b, c in tris:
        F.append([a, c, b])              # bottom (facing down)
        F.append([a + n, b + n, c + n])  # top
    for i in range(n):
        j = (i + 1) % n
        F.append([i, j, j + n])
        F.append([i, j + n, i + n])
    return V, np.array(F, int)
