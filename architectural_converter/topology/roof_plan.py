"""Roof outline from the roof plan of the drawing.

A roof plan shows the pitched roof as areas hatched with a roof-tile pattern (AutoCAD's AR-RSHKE,
AR-RROOF, or any pattern / layer named roof, shake, shingle ...), one hatch per roof face, with
what is not tiled left out: a flat deck or an open core in the middle is a hole in the hatched
area. The union of those hatches is the roof's outline, as drawn.

Roof plans are drawn on their own sheet. The sheet is brought onto the reference floor plan with
the same linework vote that aligns the floor plans (the plot lines, the building outline and the
grid repeat on every sheet)."""
from __future__ import annotations

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon, box

from ..geometry.geometric_utils import polygons_of, robust_union, safe_buffer


def is_roof_hatch(region, keys) -> bool:
    name = f"{getattr(region, 'pattern', '') or ''} {getattr(region, 'layer', '') or ''}".upper()
    return any(k.upper() in name for k in keys)


def roof_areas(hatch_regions, keys, min_area_m2: float = 10.0, min_width_m: float = 2.0) -> list:
    """Tiled areas drawn in plan: unions of roof hatches wide enough to be a roof seen from above
    (a roof seen in an elevation is a thin band)."""
    polys = [h.polygon for h in hatch_regions if is_roof_hatch(h, keys) and h.polygon is not None and not h.polygon.is_empty]
    if not polys:
        return []
    merged = safe_buffer(safe_buffer(robust_union(polys), 0.08, join_style=2), -0.08, join_style=2)
    out = []
    for p in polygons_of(merged):
        solid = Polygon(p.exterior)
        core = safe_buffer(solid, -min_width_m / 2)
        if p.area >= min_area_m2 and core is not None and not core.is_empty:
            out.append(p)
    return out


def _points(strokes, bbox, pad):
    reg = box(bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    P = []
    for st in strokes:
        if reg.contains(st.line()):
            P.append(st.p0)
            P.append(st.p1)
    return np.array(P, dtype=float) if P else np.zeros((0, 2))


def align_roof(area, strokes, ref_bbox, tol, pad: float = 12.0):
    """(translation onto the reference floor plan, votes, points) for a roof area on its sheet."""
    from .level_analysis import _align  # noqa: PLC0415
    grid = max(tol.snap * 4, 0.02)
    d, votes, n = _align(_points(strokes, ref_bbox, pad), _points(strokes, area.bounds, pad), grid)
    return d, votes, n


def roof_for_level(areas, strokes, ref_bbox, tol, target, below=None, min_overlap: float = 0.6, min_area_m2: float = 10.0,
                   ref_translation=(0.0, 0.0)):
    """The drawn roof that belongs over ``target`` (the attic storey's outline in the stacking
    frame): the aligned roof area that lies on it, cut to ``below`` (the storey under it: a roof is
    not wider than the building). ``ref_bbox`` is the sheet of a placed floor plan the roof sheet is
    aligned onto, ``ref_translation`` that floor plan's own translation into the stacking frame.
    Returns (polygons with their holes, info) or (None, info)."""
    best = None
    for a in areas:
        d, votes, n = align_roof(a, strokes, ref_bbox, tol)
        if d is None or votes < max(8, 0.05 * n):
            continue
        d = (d[0] + ref_translation[0], d[1] + ref_translation[1])
        moved = translate(a, d[0], d[1])
        solid = Polygon(moved.exterior)
        ov = solid.intersection(target).area / max(solid.area, 1e-9)
        if ov >= min_overlap and (best is None or votes > best[1]):
            best = (moved, votes, n, d, ov)
    if best is None:
        return None, {"roof_areas": len(areas), "matched": False}
    moved, votes, n, d, ov = best
    if below is not None and not below.is_empty:
        moved = moved.intersection(below)
    parts = [p for p in polygons_of(moved) if p.area >= min_area_m2]
    return parts or None, {"roof_areas": len(areas), "matched": bool(parts), "votes": votes, "points": n,
                           "translation": [round(d[0], 3), round(d[1], 3)], "overlap": round(ov, 3)}
