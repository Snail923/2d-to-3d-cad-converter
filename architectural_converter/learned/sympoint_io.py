"""DXF -> SymPointV2 input, and SymPointV2 output -> per-entity labels.

SymPointV2 (Liu et al. 2024, trained on FloorPlanCAD) reads a floor plan as a set of primitives
(Line, Arc, circle, ellipse). Each primitive is four points sampled at 0, 1/3, 2/3 and 1 of its
length (quarter points for circles / ellipses), its length, its type, the CAD layer it is on and
its stroke width - exactly the ``*_s2.json`` files written by the repository's
``parse_svg_v5.py`` and read by ``svgnet/data/svg3.py``. This module writes those fields
directly from the normalised DXF geometry (metres, y flipped to SVG orientation) plus our own
``ids`` (RawEntity id per primitive) so the predictions can be traced back to DXF handles.

Large drawings are cut into overlapping square tiles. Each primitive is *owned* by the tile whose
core contains its centre; the overlap only gives context, and labels are taken from the owner."""
from __future__ import annotations

import math

import numpy as np

# FloorPlanCAD / SymPointV2 classes (semantic id - 1); index 35 is background.
CLASSES = ["single door", "double door", "sliding door", "folding door", "revolving door", "rolling door",
           "window", "bay window", "blind window", "opening symbol", "sofa", "bed", "chair", "table",
           "TV cabinet", "Wardrobe", "cabinet", "gas stove", "sink", "refrigerator", "airconditioner", "bath",
           "bath tub", "washing machine", "squat toilet", "urinal", "toilet", "stairs", "elevator", "escalator",
           "row chairs", "parking spot", "wall", "curtain wall", "railing"]
BG = len(CLASSES)
LINE, ARC, CIRCLE, ELLIPSE = 0, 1, 2, 3


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


def primitives(drawing, min_len: float = 1e-4):
    """[(raw_id, layer, command, (4,2) points, length)] from the normalised geometry."""
    out = []
    for g in drawing.entities:
        if not g.valid or g.points is None or len(g.points) < 2:
            continue
        P = np.asarray(g.points, float)
        if g.kind == "line":
            pts, L = _resample(P[[0, -1]])
            cmd = LINE
        elif g.kind == "arc":
            pts, L = _resample(P)
            cmd = ARC
        elif g.kind == "circle":
            n = len(P)
            pts = P[[0, n // 4, n // 2, (3 * n) // 4]]
            L = float(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum())
            cmd = CIRCLE
        elif g.kind == "curve":
            if g.closed and len(P) >= 8:
                Q = P[:-1] if np.allclose(P[0], P[-1]) else P
                n = len(Q)
                pts = Q[[0, n // 4, n // 2, (3 * n) // 4]]
                L = float(np.linalg.norm(np.diff(np.vstack([Q, Q[:1]]), axis=0), axis=1).sum())
                cmd = ELLIPSE
            else:
                pts, L = _resample(P)
                cmd = ARC
        else:
            continue                 # polylines are represented by their exploded lines / arcs
        if L < min_len:
            continue
        out.append((g.raw_id, g.layer, cmd, pts, L))
    return out


def export_tiles(drawing, cfg) -> dict:
    """SymPointV2 request: {"version", "classes", "tiles": [{"id", "core", "sympoint": {...}, "ids", "own"}]}."""
    lc = cfg["learned"]["sympoint"]
    prims = primitives(drawing)
    if not prims:
        return {"version": 1, "classes": CLASSES, "tiles": []}
    C = np.array([p[3].mean(axis=0) for p in prims])
    x0, y0 = C.min(axis=0)
    x1, y1 = C.max(axis=0)
    T, ov = float(lc["tile_m"]), float(lc["overlap_m"])
    nx, ny = max(1, math.ceil((x1 - x0 + 1e-9) / T)), max(1, math.ceil((y1 - y0 + 1e-9) / T))
    layers = sorted({p[1] for p in prims})
    lid = {ly: k + 1 for k, ly in enumerate(layers)}
    tiles = []
    for j in range(ny):
        for i in range(nx):
            cx0, cy0 = x0 + i * T, y0 + j * T
            core = (cx0, cy0, cx0 + T, cy0 + T)
            ox0, oy0, ox1, oy1 = cx0 - ov, cy0 - ov, cx0 + T + ov, cy0 + T + ov
            sel = np.nonzero((C[:, 0] >= ox0) & (C[:, 0] <= ox1) & (C[:, 1] >= oy0) & (C[:, 1] <= oy1))[0]
            own = (C[sel, 0] >= core[0]) & (C[sel, 1] >= core[1]) & \
                  ((C[sel, 0] < core[2]) | (i == nx - 1)) & ((C[sel, 1] < core[3]) | (j == ny - 1))
            if not own.any():
                continue
            W = H = T + 2 * ov
            args, cmds, lens, lids = [], [], [], []
            for k in sel:
                rid, ly, cmd, pts, L = prims[k]
                q = np.column_stack([pts[:, 0] - ox0, oy1 - pts[:, 1]])     # SVG frame: y down
                args.append([round(float(v), 5) for v in q.reshape(-1)])
                cmds.append(cmd)
                lens.append(round(float(L), 5))
                lids.append(lid[ly])
            n = len(sel)
            tiles.append({
                "id": f"t{j:02d}_{i:02d}", "core": [round(v, 4) for v in core],
                "ids": [prims[k][0] for k in sel], "own": [bool(v) for v in own],
                "layer_names": layers,
                "sympoint": {"commands": cmds, "args": args, "lengths": lens,
                             "semanticIds": [BG] * n, "instanceIds": [-1] * n,
                             "width": W, "height": H, "obj_cts": [], "boxes": [],
                             "rgb": [[0, 0, 0]] * n, "layerIds": lids, "widths": [1.0] * n},
            })
    return {"version": 1, "model": "SymPointV2", "classes": CLASSES, "tiles": tiles,
            "primitives": len(prims), "tile_m": T, "overlap_m": ov}


def labels_from_response(request: dict, response: dict, min_score: float = 0.0) -> dict:
    """Per RawEntity id: {"label", "score", "length"} (length-weighted vote of its primitives),
    plus instances as lists of RawEntity ids. Only primitives owned by a tile are used."""
    req = {t["id"]: t for t in request.get("tiles", [])}
    votes: dict[str, dict[int, float]] = {}
    lens: dict[str, float] = {}
    instances = []
    for rt in response.get("tiles", []):
        t = req.get(rt["id"])
        if t is None:
            continue
        L = t["sympoint"]["lengths"]
        for k, (cls, sc) in enumerate(rt.get("semantic", [])):
            if not t["own"][k] or cls is None or cls >= BG or sc < min_score:
                continue
            rid = t["ids"][k]
            v = votes.setdefault(rid, {})
            v[cls] = v.get(cls, 0.0) + sc * L[k]
            lens[rid] = lens.get(rid, 0.0) + L[k]
        for inst in rt.get("instances", []):
            mem = [t["ids"][k] for k in inst.get("members", []) if 0 <= k < len(t["ids"]) and t["own"][k]]
            if mem and 0 <= inst.get("label", BG) < BG:
                instances.append({"label": CLASSES[inst["label"]], "score": float(inst.get("score", 0.0)),
                                  "members": sorted(set(mem))})
    out = {}
    for rid, v in votes.items():
        cls = max(v, key=v.get)
        out[rid] = {"label": CLASSES[cls], "score": round(v[cls] / max(lens[rid], 1e-12), 4),
                    "length": round(lens[rid], 4)}
    return {"model": response.get("model", "SymPointV2"), "entities": out, "instances": instances}
