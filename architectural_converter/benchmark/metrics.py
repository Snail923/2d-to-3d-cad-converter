"""Benchmark metrics against reference geometry.

Walls are compared by *length*, not by count, so that different (equally valid)
segmentations are not punished: centre lines are sampled every 5 cm; a sample is correct
when it lies within ``max(0.6 t, 8 cm)`` of a reference axis with a compatible direction.
Rooms are matched by IoU >= 0.5; openings and columns by centre distance."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree

STEP = 0.05


def ref_to_model(ref, model):
    """Function mapping reference world metres to model coordinates."""
    fr = ref["frame"]
    to_m = fr["to_m"]
    off_m = fr["offset_m"]
    tr = model["source"]["transform"]
    sc = tr["scale_to_m"]
    od = tr["offset_drawing_units"]

    def f(p):
        dx = (p[0] + off_m[0]) / to_m
        dy = (p[1] + off_m[1]) / to_m
        return ((dx - od[0]) * sc, (dy - od[1]) * sc)
    return f


def _samples(line: LineString):
    if line.length <= 0:
        return []
    n = max(2, int(math.ceil(line.length / STEP)) + 1)
    out = []
    for d in np.linspace(0, line.length, n):
        p = line.interpolate(d)
        q = line.interpolate(min(line.length, d + 0.01))
        r = line.interpolate(max(0.0, d - 0.01))
        ang = math.atan2(q.y - r.y, q.x - r.x) % math.pi
        out.append((p, ang))
    return out


def _angle_ok(a, b, tol=math.radians(15)):
    d = abs(a - b) % math.pi
    return min(d, math.pi - d) <= tol


def wall_metrics(model, ref, statuses=("ACCEPTED", "NEEDS_REVIEW")):
    f = ref_to_model(ref, model)
    refs = []
    for w in ref["walls"]:
        pts = w.get("axis") or [w["start"], w["end"]]
        refs.append((LineString([f(p) for p in pts]), w["thickness"]))
    dets = []
    for w in model["walls"]:
        if w["status"] not in statuses or w.get("is_hypothesis"):
            continue
        g = w["geometry"]
        line = LineString([g["start"], g["end"]]) if g.get("start") else LineString(g["axis"])
        dets.append((line, w["thickness"], w["id"]))
    rtree = STRtree([r[0] for r in refs])
    det_total = det_ok = 0
    fp = 0
    th_err = []
    for line, t, wid in dets:
        smp = _samples(line)
        ok = 0
        for p, ang in smp:
            hit = False
            for k in rtree.query(p.buffer(0.5)):
                rl, rt = refs[int(k)]
                if rl.distance(p) <= max(0.6 * rt, 0.08):
                    q = rl.interpolate(rl.project(p))
                    q2 = rl.interpolate(min(rl.length, rl.project(p) + 0.01))
                    q1 = rl.interpolate(max(0.0, rl.project(p) - 0.01))
                    rang = math.atan2(q2.y - q1.y, q2.x - q1.x) % math.pi
                    if _angle_ok(ang, rang):
                        hit = True
                        if t:
                            th_err.append(abs(t - rt))
                        break
            ok += hit
        det_total += len(smp)
        det_ok += ok
        if smp and ok / len(smp) < 0.5:
            fp += 1
    dtree = STRtree([d[0] for d in dets]) if dets else None
    ref_total = ref_ok = 0
    missed = 0
    for rl, rt in refs:
        smp = _samples(rl)
        ok = 0
        for p, ang in smp:
            hit = False
            if dtree is not None:
                for k in dtree.query(p.buffer(0.5)):
                    dl = dets[int(k)][0]
                    if dl.distance(p) <= max(0.6 * rt, 0.08):
                        q2 = dl.interpolate(min(dl.length, dl.project(p) + 0.01))
                        q1 = dl.interpolate(max(0.0, dl.project(p) - 0.01))
                        if _angle_ok(ang, math.atan2(q2.y - q1.y, q2.x - q1.x) % math.pi):
                            hit = True
                            break
            ok += hit
        ref_total += len(smp)
        ref_ok += ok
        if smp and ok / len(smp) < 0.5:
            missed += 1
    return {"wall_precision": det_ok / det_total if det_total else 0.0,
            "wall_recall": ref_ok / ref_total if ref_total else 0.0,
            "false_positive_walls": fp, "missed_walls": missed,
            "thickness_mae_m": float(np.mean(th_err)) if th_err else None,
            "detected_walls": len(dets), "reference_walls": len(refs)}


def room_metrics(model, ref, statuses=("ACCEPTED", "NEEDS_REVIEW")):
    f = ref_to_model(ref, model)
    rpolys = [(Polygon([f(p) for p in r["polygon"]]).buffer(0), r.get("name")) for r in ref["rooms"]]
    dpolys = []
    for r in model["rooms"]:
        if r["status"] not in statuses or r.get("is_hypothesis"):
            continue
        pg = r["polygon"]
        dpolys.append((Polygon(pg["exterior"], pg["holes"]).buffer(0), r.get("name")))
    pairs = []
    for i, (dp, _) in enumerate(dpolys):
        for j, (rp, _) in enumerate(rpolys):
            if dp.intersects(rp):
                inter = dp.intersection(rp).area
                iou = inter / max(dp.union(rp).area, 1e-12)
                if iou >= 0.5:
                    pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_d, used_r, ious, names = set(), set(), [], 0
    for iou, i, j in pairs:
        if i in used_d or j in used_r:
            continue
        used_d.add(i)
        used_r.add(j)
        ious.append(iou)
        if dpolys[i][1] and rpolys[j][1] and dpolys[i][1].strip().upper() == rpolys[j][1].strip().upper():
            names += 1
    nd, nr = len(dpolys), len(rpolys)
    return {"room_precision": len(ious) / nd if nd else (1.0 if nr == 0 else 0.0),
            "room_recall": len(ious) / nr if nr else 1.0,
            "room_boundary_iou": float(np.mean(ious)) if ious else 0.0,
            "room_name_match": names / nr if nr else 1.0,
            "detected_rooms": nd, "reference_rooms": nr,
            "false_positive_rooms": nd - len(ious), "missed_rooms": nr - len(ious)}


def _match_points(det, ref, radius_fn):
    pairs = []
    for i, (dp, dw) in enumerate(det):
        for j, (rp, rw) in enumerate(ref):
            d = math.dist(dp, rp)
            if d <= radius_fn(rw):
                pairs.append((d, i, j))
    pairs.sort()
    ud, ur = set(), set()
    for d, i, j in pairs:
        if i in ud or j in ur:
            continue
        ud.add(i)
        ur.add(j)
    return len(ud)


def opening_metrics(model, ref, statuses=("ACCEPTED", "NEEDS_REVIEW")):
    f = ref_to_model(ref, model)
    out = {}
    det_all = [o for o in model["openings"] if o["status"] in statuses]
    for kind, rkey in (("door", "doors"), ("window", "windows")):
        det = [(tuple(o["position"]["center"]), o["width"]) for o in det_all if o["type"] == kind]
        rf = [(f(r["center"]), r["width"]) for r in ref[rkey]]
        m = _match_points(det, rf, lambda w: max(0.5 * w, 0.3))
        out[f"{kind}_precision"] = m / len(det) if det else (1.0 if not rf else 0.0)
        out[f"{kind}_recall"] = m / len(rf) if rf else 1.0
        out[f"{kind}_fp"] = len(det) - m
        out[f"{kind}_missed"] = len(rf) - m
    det = [(tuple(o["position"]["center"]), o["width"]) for o in det_all]
    rf = [(f(r["center"]), r["width"]) for k in ("doors", "windows", "openings") for r in ref[k]]
    m = _match_points(det, rf, lambda w: max(0.5 * w, 0.3))
    out["opening_localization_recall"] = m / len(rf) if rf else 1.0
    out["opening_localization_precision"] = m / len(det) if det else (1.0 if not rf else 0.0)
    return out


def column_metrics(model, ref, statuses=("ACCEPTED", "NEEDS_REVIEW")):
    f = ref_to_model(ref, model)
    det = [(tuple(c["center"]), max(c["size"])) for c in model["columns"] if c["status"] in statuses]
    rf = [(f(c["center"]), max(c["size"])) for c in ref["columns"]]
    m = _match_points(det, rf, lambda s: max(s, 0.3))
    return {"column_precision": m / len(det) if det else (1.0 if not rf else 0.0),
            "column_recall": m / len(rf) if rf else 1.0, "column_fp": len(det) - m, "column_missed": len(rf) - m}


def evaluate(model, ref) -> dict:
    out = {}
    out.update(wall_metrics(model, ref))
    out.update(room_metrics(model, ref))
    out.update(opening_metrics(model, ref))
    out.update(column_metrics(model, ref))
    acc = wall_metrics(model, ref, statuses=("ACCEPTED",))
    out["wall_precision_accepted_only"] = acc["wall_precision"]
    out["wall_recall_accepted_only"] = acc["wall_recall"]
    review = sum(1 for k in ("walls", "openings", "rooms") for x in model[k] if x["status"] == "NEEDS_REVIEW")
    out["needs_review_items"] = review
    out["manual_corrections_required"] = (out["false_positive_walls"] + out["missed_walls"] + out["false_positive_rooms"]
                                          + out["missed_rooms"] + out["door_fp"] + out["door_missed"] + out["window_fp"]
                                          + out["window_missed"] + out["column_fp"] + out["column_missed"])
    return out
