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


class LineTable:
    """Line endpoints as arrays, so 'which lines start at this point' is one vector operation."""

    def __init__(self, lines):
        self.lines = list(lines)
        if self.lines:
            self.a = np.array([ln.points[0][:2] for ln in self.lines], dtype=float)
            self.b = np.array([ln.points[-1][:2] for ln in self.lines], dtype=float)
            self.L = np.hypot(*(self.b - self.a).T)
        else:
            self.a = self.b = np.zeros((0, 2))
            self.L = np.zeros(0)


def leaf_candidates(arc_g, lines, tol, rel):
    """Radial lines that can be the drawn door leaf of a swing arc.

    A leaf starts at the hinge (arc centre), has the length of the radius and its tip lands on
    one of the two arc end points. Door symbols also hold frame / threshold / jamb lines of a
    similar length that start near the hinge, so every candidate is returned with the end point
    it reaches (0 = arc start, 1 = arc end) and its *tip error* (distance from the tip to that
    end point): the leaf is the line whose tip meets the arc, not the last line found.

    Returns ``[(tip_error, endpoint, line, tip)]`` sorted by error."""
    table = lines if isinstance(lines, LineTable) else LineTable(lines)
    if not table.lines:
        return []
    c = np.array(arc_g.data["center"][:2], dtype=float)
    r = arc_g.data["radius"]
    ends = np.array([[c[0] + r * math.cos(arc_g.data["start"]), c[1] + r * math.sin(arc_g.data["start"])],
                     [c[0] + r * math.cos(arc_g.data["end"]), c[1] + r * math.sin(arc_g.data["end"])]])
    near = max(4 * tol, 0.03 * r)
    tip_near = max(4 * tol, 0.06 * r)
    ok_len = np.abs(table.L - r) <= rel * r
    out = []
    for start, end in ((table.a, table.b), (table.b, table.a)):
        hit = np.nonzero(ok_len & (np.hypot(*(start - c).T) <= near))[0]
        for k in hit:
            tip = end[k]
            d = np.hypot(*(ends - tip).T)
            e = int(np.argmin(d))
            if d[e] <= tip_near:
                out.append((float(d[e]), e, table.lines[int(k)], tip.copy()))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def leaf_line_for(arc_g, lines, tol, rel):
    """Best drawn leaf of a swing arc: ``(line, tip)`` or None (see ``leaf_candidates``)."""
    cands = leaf_candidates(arc_g, lines, tol, rel)
    if not cands:
        return None
    return cands[0][2], cands[0][3]


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
        semantic = {}
        for k in ("door", "window", "column", "furniture"):
            semantic[k] = 0.6 * name_sc.get(k, 0.0) + 0.3 * lay_sc.get(k, 0.0)
            scores[k] += semantic[k]
        if name_sc.get("wall", 0) > 0:
            scores["assembly"] += 0.4
        # the geometric signatures are generic (two equal parallel lines, a compact closed
        # shape...) and furniture matches them too: wardrobes, counters, TV units, bath tubs.
        # When the block's own name or layer says furniture / fixture and nothing says
        # door / window / column, geometry alone may not turn it into a building element.
        furn_sem = max(name_sc.get("furniture", 0.0), lay_sc.get("furniture", 0.0))
        disc = sym.get("furniture_semantic_discount", 0.3)
        for k, key in (("door", "door_score"), ("window", "window_score"), ("column", "column_score")):
            g = sig[key]
            if furn_sem >= 0.5 and max(name_sc.get(k, 0.0), lay_sc.get(k, 0.0)) < 0.5:
                g *= disc
            scores[k] += g
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
        ref.semantic_scores = {k: round(v, 3) for k, v in semantic.items()}
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
                # purity: a window symbol is frame + glass lines. A block made largely of
                # curves (sinks, cushions, hangers, appliances) or whose parallel lines do not
                # span the symbol from jamb to jamb is furniture drawn with parallel edges.
                n_curves = sig["n_arcs"] + sig["n_circles"]
                curve_share = n_curves / max(1, n_curves + sig["n_lines"])
                span = med / max(sig["extent_length"], 1e-9)
                wcfg = det["window"]
                if curve_share > wcfg.get("symbol_max_curve_share", 0.25) or span < wcfg.get("symbol_min_span", 0.8):
                    sig["window_score"] *= 0.5
                sig["window_curve_share"] = round(curve_share, 3)
                sig["window_span"] = round(span, 3)
    # column: compact, small, closed / filled
    clo, chi = det["column"]["size_m"]
    W, L = sig["extent_width"], sig["extent_length"]
    if L > 0 and clo <= W and L <= chi and L / max(W, 1e-9) <= det["column"]["max_aspect"] and sig["n_door_arcs"] == 0:
        if sig["n_closed"] or sig["n_fills"] or sig["n_circles"]:
            sig["column_score"] = 0.45 + (0.1 if sig["n_fills"] else 0.0)
    return sig
