"""DXF -> SymPointV2 input, and SymPointV2 output -> per-primitive / per-entity labels.

SymPointV2 (Liu et al. 2024, trained on FloorPlanCAD) reads a floor plan as a set of primitives
(line, arc, circle, ellipse). Each primitive is four points sampled at 0, 1/3, 2/3 and 1 of its
length (quarter points for circles / ellipses), its length, its type, its CAD layer and its stroke
width: the ``*_s2.json`` fields written by the repository's ``parse_svg_v5.py`` and read by
``svgnet/data/svg3.py`` (``SVGDataset.load``). They are written here directly from the normalised
DXF geometry (metres, y flipped to the SVG orientation), together with the RawEntity id and the
global index of every primitive, so each prediction can be traced back to its DXF handle.

Large drawings are cut into overlapping square tiles. Each primitive is *owned* by the tile whose
core contains its centre; the overlap only gives context, and labels are taken from the owner.
Tiles with too many primitives are split into quarters (GPU memory)."""
from __future__ import annotations

import math

import numpy as np

from .classes import BG, CLASSES

LINE, ARC, CIRCLE, ELLIPSE = 0, 1, 2, 3
COMMAND_NAMES = {LINE: "line", ARC: "arc", CIRCLE: "circle", ELLIPSE: "ellipse"}


def load_drawing(path: str, units: str | None = None):
    """Approach 1's front end only: read the DXF (DWG via ODA), detect units, normalise to metres.
    No detector or rule of approach 1 runs here."""
    from architectural_converter.ingestion import ingest
    from architectural_converter.preprocessing.coordinate_normalizer import normalize
    from architectural_converter.preprocessing.unit_detection import detect_units
    d = ingest(path)
    d.units = detect_units(d, units)
    normalize(d)
    return d


def _resample(P: np.ndarray, fr=(0.0, 1 / 3, 2 / 3, 1.0)):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(cum[-1])
    out = []
    for f in fr:
        t = f * L
        k = int(np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(seg) - 1))
        a = (t - cum[k]) / seg[k] if seg[k] > 0 else 0.0
        out.append(P[k] + a * (P[k + 1] - P[k]))
    return np.array(out), L


def _ring(P):
    Q = P[:-1] if len(P) > 1 and np.allclose(P[0], P[-1]) else P
    n = len(Q)
    L = float(np.linalg.norm(np.diff(np.vstack([Q, Q[:1]]), axis=0), axis=1).sum())
    return Q[[0, n // 4, n // 2, (3 * n) // 4]], L


def primitives(drawing, cfg: dict | None = None):
    """[(raw_id, layer, command, (4,2) points, length, (N,2) polyline for display)] from the
    normalised geometry. Polylines / MLINEs are represented by their exploded lines and arcs."""
    cfg = cfg or {}
    min_len = float(cfg.get("min_primitive_length_m", 1e-4))
    hatches = bool(cfg.get("include_hatch_boundaries", False))
    out = []
    for g in drawing.entities:
        if not g.valid:
            continue
        if g.kind == "hatch" and hatches:
            for loop in g.data.get("loops") or []:
                P = np.asarray(loop, float)
                Q = np.vstack([P, P[:1]]) if len(P) > 2 and not np.allclose(P[0], P[-1]) else P
                for a, b in zip(Q[:-1], Q[1:]):
                    pts, L = _resample(np.array([a, b]))
                    if L >= min_len:
                        out.append((g.raw_id, g.layer, LINE, pts, L, np.array([a, b])))
            continue
        if g.points is None or len(g.points) < 2:
            continue
        P = np.asarray(g.points, float)
        if g.kind == "line":
            pts, L = _resample(P[[0, -1]])
            cmd, disp = LINE, P[[0, -1]]
        elif g.kind == "arc":
            pts, L = _resample(P)
            cmd, disp = ARC, P
        elif g.kind == "circle":
            pts, L = _ring(P)
            cmd, disp = CIRCLE, np.vstack([P, P[:1]])
        elif g.kind == "curve":
            if g.closed and len(P) >= 8:
                pts, L = _ring(P)
                cmd, disp = ELLIPSE, P
            else:
                pts, L = _resample(P)
                cmd, disp = ARC, P
        else:
            continue
        if L < min_len:
            continue
        out.append((g.raw_id, g.layer, cmd, pts, L, disp))
    return out


def _make_tile(tid, core, sel, own, prims, C, ov, layers, lid):
    ox0, oy1 = core[0] - ov, core[3] + ov
    W = (core[2] - core[0]) + 2 * ov
    H = (core[3] - core[1]) + 2 * ov
    args, cmds, lens, lids = [], [], [], []
    for k in sel:
        pts = prims[k][3]
        q = np.column_stack([pts[:, 0] - ox0, oy1 - pts[:, 1]])     # SVG frame: y down
        args.append([round(float(v), 5) for v in q.reshape(-1)])
        cmds.append(prims[k][2])
        lens.append(round(float(prims[k][4]), 5))
        lids.append(lid[prims[k][1]])
    n = len(sel)
    return {
        "id": tid, "core": [round(v, 4) for v in core], "pidx": [int(k) for k in sel],
        "ids": [prims[k][0] for k in sel], "own": [bool(v) for v in own], "layer_names": layers,
        "sympoint": {"commands": cmds, "args": args, "lengths": lens,
                     "semanticIds": [BG] * n, "instanceIds": [-1] * n,
                     "width": round(W, 5), "height": round(H, 5), "obj_cts": [], "boxes": [],
                     "rgb": [[0, 0, 0]] * n, "layerIds": lids, "widths": [1.0] * n},
    }


def export_tiles(drawing, cfg: dict, prims=None) -> dict:
    """SymPointV2 request: {"version", "classes", "tiles": [{"id", "core", "pidx", "ids", "own", "sympoint"}]}."""
    prims = primitives(drawing, cfg) if prims is None else prims
    T, ov = float(cfg["tile_m"]), float(cfg["overlap_m"])
    max_n = int(cfg.get("max_primitives_per_tile", 40000))
    base = {"version": 2, "model": "SymPointV2", "classes": CLASSES, "primitives": len(prims),
            "tile_m": T, "overlap_m": ov}
    if not prims:
        return {**base, "tiles": []}
    C = np.array([p[3].mean(axis=0) for p in prims])
    x0, y0 = C.min(axis=0)
    x1, y1 = C.max(axis=0)
    nx, ny = max(1, math.ceil((x1 - x0 + 1e-9) / T)), max(1, math.ceil((y1 - y0 + 1e-9) / T))
    layers = sorted({p[1] for p in prims})
    lid = {ly: k + 1 for k, ly in enumerate(layers)}
    tiles = []

    def add(tid, core, last_x, last_y, depth=0):
        ox0, oy0, ox1, oy1 = core[0] - ov, core[1] - ov, core[2] + ov, core[3] + ov
        sel = np.nonzero((C[:, 0] >= ox0) & (C[:, 0] <= ox1) & (C[:, 1] >= oy0) & (C[:, 1] <= oy1))[0]
        own = (C[sel, 0] >= core[0]) & (C[sel, 1] >= core[1]) & \
              ((C[sel, 0] < core[2]) | last_x) & ((C[sel, 1] < core[3]) | last_y)
        if not own.any():
            return
        if len(sel) > max_n and depth < 6:
            mx, my = (core[0] + core[2]) / 2, (core[1] + core[3]) / 2
            for qi, (a, b, c, d, lx, ly) in enumerate(((core[0], core[1], mx, my, False, False),
                                                       (mx, core[1], core[2], my, last_x, False),
                                                       (core[0], my, mx, core[3], False, last_y),
                                                       (mx, my, core[2], core[3], last_x, last_y))):
                add(f"{tid}q{qi}", (a, b, c, d), lx, ly, depth + 1)
            return
        tiles.append(_make_tile(tid, core, sel, own, prims, C, ov, layers, lid))

    for j in range(ny):
        for i in range(nx):
            cx0, cy0 = x0 + i * T, y0 + j * T
            add(f"t{j:02d}_{i:02d}", (cx0, cy0, cx0 + T, cy0 + T), i == nx - 1, j == ny - 1)
    return {**base, "tiles": tiles}


def labels_from_response(request: dict, response: dict, min_score: float = 0.0, min_mass: float = 0.0) -> dict:
    """Labels of the owned primitives, aggregated per DXF entity.

    Returns {"model", "primitives": {pidx: [class index, score]}, "entities": {raw_id: {"label",
    "score", "length"}} (length-weighted vote of its primitives; score = share of the class in the
    model's answer for the line, 0-1; lines whose answer mass is below ``min_mass`` are unlabelled), "instances": [{"label", "score",
    "members": [raw ids], "pidx": [...]}]}. Unlabelled / below ``min_score`` primitives are omitted."""
    req = {t["id"]: t for t in request.get("tiles", [])}
    votes: dict[str, dict[int, float]] = {}
    lens: dict[str, float] = {}
    prim: dict[int, list] = {}
    instances = []
    for rt in response.get("tiles", []):
        t = req.get(rt["id"])
        if t is None:
            continue
        L = t["sympoint"]["lengths"]
        pidx = t.get("pidx") or list(range(len(t["ids"])))
        for k, ans in enumerate(rt.get("semantic", [])):
            cls, sc = ans[0], ans[1]
            mass = ans[2] if len(ans) > 2 else None        # older service: [class, score]
            if mass is None and sc is not None and sc > 1:   # older service: raw summed score
                sc = 1.0
            if k >= len(t["ids"]) or not t["own"][k] or cls is None or cls >= BG or sc < min_score \
                    or (mass is not None and mass < min_mass):
                continue
            prim[int(pidx[k])] = [int(cls), float(sc)]
            rid = t["ids"][k]
            v = votes.setdefault(rid, {})
            v[cls] = v.get(cls, 0.0) + sc * L[k]
            lens[rid] = lens.get(rid, 0.0) + L[k]
        for inst in rt.get("instances", []):
            ks = [k for k in inst.get("members", []) if 0 <= k < len(t["ids"]) and t["own"][k]]
            if ks and 0 <= inst.get("label", BG) < BG:
                instances.append({"label": CLASSES[inst["label"]], "score": float(inst.get("score", 0.0)),
                                  "members": sorted({t["ids"][k] for k in ks}),
                                  "pidx": sorted(int(pidx[k]) for k in ks), "tile": t["id"]})
    ents = {}
    for rid, v in votes.items():
        cls = max(v, key=v.get)
        ents[rid] = {"label": CLASSES[cls], "score": round(v[cls] / max(lens[rid], 1e-12), 4),
                     "length": round(lens[rid], 4)}
    return {"model": response.get("model", "SymPointV2"), "primitives": prim, "entities": ents,
            "instances": instances}
