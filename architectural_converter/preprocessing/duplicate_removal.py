"""Exact / contained duplicate identification and partial-overlap reporting.

Duplicates are *marked* (``Segment.dup_of``) and excluded from detection, but kept with
their source references; the surviving segment's evidence lists the duplicates."""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree

from ..geometry.geometric_utils import angle_diff, dir_angle, normal


def mark_duplicates(segments, dup_tol: float, angle_tol: float) -> dict:
    if not segments:
        return {"exact": 0, "contained": 0, "partial_overlaps": 0, "overlap_pairs": []}
    lines = [s.line() for s in segments]
    tree = STRtree(lines)
    exact = contained = 0
    overlaps = []
    for i, s in enumerate(segments):
        if s.dup_of is not None:
            continue
        u = s.u
        n = normal(u)
        a0, a1 = sorted((float(np.dot(s.p0, u)), float(np.dot(s.p1, u))))
        minx, miny, maxx, maxy = lines[i].bounds
        for j in tree.query(LineString([(minx - dup_tol, miny - dup_tol), (maxx + dup_tol, maxy + dup_tol)]).envelope):
            j = int(j)
            if j == i:
                continue
            t = segments[j]
            if t.dup_of is not None:
                continue
            if angle_diff(dir_angle(u), dir_angle(t.u)) > angle_tol:
                continue
            o0 = abs(float(np.dot(t.p0 - s.p0, n)))
            o1 = abs(float(np.dot(t.p1 - s.p0, n)))
            if o0 > dup_tol or o1 > dup_tol:
                continue
            b0, b1 = sorted((float(np.dot(t.p0, u)), float(np.dot(t.p1, u))))
            if abs(a0 - b0) <= dup_tol and abs(a1 - b1) <= dup_tol:
                if j > i:
                    t.dup_of = s.id
                    t.flags.add("exact_duplicate")
                    exact += 1
            elif b0 >= a0 - dup_tol and b1 <= a1 + dup_tol:
                t.dup_of = s.id
                t.flags.add("contained_duplicate")
                contained += 1
            elif min(a1, b1) - max(a0, b0) > dup_tol and j > i:
                overlaps.append((s.id, t.id, round(min(a1, b1) - max(a0, b0), 5)))
    return {"exact": exact, "contained": contained, "partial_overlaps": len(overlaps),
            "overlap_pairs": overlaps[:500]}
