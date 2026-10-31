"""Block-reference (symbol) analysis.

Each INSERT gets a role (door, window, column, furniture, assembly, unknown) from
* name hints of the block and of the insert layer (configurable keyword tables),
* the geometric signature of its exploded content in world space.

Large blocks containing wall linework (``assembly``, e.g. a hotel-room module) are
transparent: their content is treated as ordinary linework. Symbol content is excluded
from wall boundary detection but used by the opening / column detectors."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import MultiPoint

from ..geometry.geometric_utils import angle_diff, canonical_dir, dir_angle, mrr_dims
from ..preprocessing.layer_analyzer import category_scores

SYMBOL_ROLES = ("door", "window", "column", "furniture", "symbol")


def door_arcs(ents, cfg_door, tol):
    lo, hi = cfg_door["leaf_width_m"]
    s_lo, s_hi = cfg_door["sweep_deg"]
    out = []
    for g in ents:
        if g.kind != "arc":
            continue
        r = g.data["radius"]
        sw = math.degrees(g.data["sweep"])
        if lo <= r <= hi and s_lo <= sw <= s_hi:
            out.append(g)
    return out


def leaf_line_for(arc_g, lines, tol, rel):
    """Line from the arc centre with length ~ radius ending at one arc endpoint."""
    c = np.array(arc_g.data["center"])
    r = arc_g.data["radius"]
    best = None
    for ln in lines:
        p0, p1 = ln.points[0], ln.points[1]
        for a, b in ((p0, p1), (p1, p0)):
            if np.hypot(*(a - c)) <= max(4 * tol, 0.03 * r):
                L = float(np.hypot(*(b - a)))
                if abs(L - r) <= rel * r:
                    best = (ln, b)
    return best


def analyze_symbols(drawing, cfg, tol) -> dict[str, dict]:
    sem = cfg["layer_semantics"]["categories"]
    det = cfg["detection"]
    sym = det["symbols"]
    geo = drawing.geo_by_id()
    refs = sorted(drawing.inserts.values(), key=lambda r: -r.depth)
    roles: dict[str, str] = {}
    for ref in refs:
        nested_symbol_children = set()
        for other in drawing.inserts.values():
            if other.parent_path[-1:] == (ref.id,) and roles.get(other.id) in SYMBOL_ROLES:
                nested_symbol_children.update(other.child_geo_ids)
        ents = [geo[g] for g in ref.child_geo_ids if g in geo and geo[g].valid and g not in nested_symbol_children]
        sig = signature(ents, det, tol)
        ref.signature = sig
        scores = {k: 0.0 for k in ("door", "window", "column", "furniture", "assembly")}
        name_sc = category_scores(ref.name, sem)
        lay_sc = category_scores(ref.layer, sem)
        for k in ("door", "window", "column", "furniture"):
            scores[k] += 0.6 * name_sc.get(k, 0.0) + 0.3 * lay_sc.get(k, 0.0)
        if name_sc.get("wall", 0) > 0:
            scores["assembly"] += 0.4
        scores["door"] += sig["door_score"]
        scores["window"] += sig["window_score"]
        scores["column"] += sig["column_score"]
        ext = sig["extent_length"]
        if ext >= sym["assembly_min_extent_m"] and sig["n_lines"] >= 4:
            scores["assembly"] += 0.5 + (0.1 if sig["n_nested"] else 0.0)
        if 0 < ext <= sym["max_symbol_extent_m"] and max(scores["door"], scores["window"], scores["column"]) < 0.4:
            scores["furniture"] += 0.3
        best = max(scores, key=scores.get)
        role = best if scores[best] >= sym["role_min_score"] else (
            "symbol" if ext <= sym["max_symbol_extent_m"] else "assembly")
        ref.role = role
        ref.role_scores = {k: round(v, 3) for k, v in scores.items()}
        roles[ref.id] = role
    # effective role of every primitive: innermost non-assembly insert wins
    for g in drawing.entities:
        if not g.block_path:
            continue
        r = "linework"
        for rid in reversed(g.block_path):
            rr = roles.get(rid)
            if rr in SYMBOL_ROLES:
                r = rr
                break
        g.role = r
    return roles


def signature(ents, det, tol) -> dict:
    pts = [g.points for g in ents if g.points is not None and g.kind not in ("text", "dimension")]
    lines = [g for g in ents if g.kind == "line"]
    arcs = door_arcs(ents, det["door"], tol.snap)
    sig = {"n_entities": len(ents), "n_lines": len(lines), "n_arcs": sum(1 for g in ents if g.kind == "arc"),
           "n_door_arcs": len(arcs), "n_circles": sum(1 for g in ents if g.kind == "circle"),
           "n_fills": sum(1 for g in ents if g.kind in ("hatch",) or (g.kind == "polygon" and g.data.get("filled"))),
           "n_closed": sum(1 for g in ents if g.closed and g.kind in ("polyline", "polygon", "circle")),
           "n_nested": len({g.block_path[-1] for g in ents if len(g.block_path) > 1}),
           "door_score": 0.0, "window_score": 0.0, "column_score": 0.0,
           "extent_length": 0.0, "extent_width": 0.0}
    if not pts:
        return sig
    allp = np.vstack(pts)
    if len(allp) >= 2:
        w, L, u, cen = mrr_dims(MultiPoint([tuple(p) for p in allp]))
        sig.update({"extent_length": float(L), "extent_width": float(w), "axis": list(u), "center": list(cen)})
    # door: swing arc (+ leaf)
    if arcs:
        s = 0.5
        if any(leaf_line_for(a, lines, tol.snap, det["door"]["leaf_length_tol"]) for a in arcs):
            s += 0.3
        sig["door_score"] = s
    # window: >= 2 equal parallel lines, thin overall, no door arcs
    wlo, whi = det["window"]["width_m"]
    if not arcs and len(lines) >= 2:
        groups: dict[int, list[float]] = {}
        for ln in lines:
            d = ln.points[1] - ln.points[0]
            L = float(np.hypot(*d))
            if not (wlo * 0.9 <= L <= whi):
                continue
            a = dir_angle(canonical_dir(float(d[0]), float(d[1])))
            key = None
            for k in groups:
                if angle_diff(k / 1e4, a) <= tol.angle * 2:
                    key = k
                    break
            groups.setdefault(key if key is not None else int(a * 1e4), []).append(L)
        best = max((sorted(v) for v in groups.values()), key=len, default=[])
        if len(best) >= 2:
            med = float(np.median(best))
            equal = [L for L in best if abs(L - med) <= det["window"]["length_match_tol"] * med]
            if len(equal) >= 2 and sig["extent_width"] <= tol.thickness_max * 1.3:
                sig["window_score"] = 0.5 + (0.1 if len(equal) >= 3 else 0.0)
    # column: compact, small, closed / filled
    clo, chi = det["column"]["size_m"]
    W, L = sig["extent_width"], sig["extent_length"]
    if L > 0 and clo <= W and L <= chi and L / max(W, 1e-9) <= det["column"]["max_aspect"] and sig["n_door_arcs"] == 0:
        if sig["n_closed"] or sig["n_fills"] or sig["n_circles"]:
            sig["column_score"] = 0.45 + (0.1 if sig["n_fills"] else 0.0)
    return sig
