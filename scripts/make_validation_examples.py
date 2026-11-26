"""Build web/validation_examples.json: the built-in example images of the room check (web/validator.js).

Each example is a crop of samples/real/apartment-building-plan.dxf around one element approach 1 is sure of
(all its doors and windows matched the drawing's own door blocks and window layer), stored as the drawing's
lines in the crop plus the box to frame in red. The browser renders them like its own crops.

    python scripts/make_validation_examples.py
"""
from __future__ import annotations

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline  # noqa: E402
from architectural_converter.visualization.feature_overlay import view_payload  # noqa: E402

SRC = os.path.join(ROOT, "samples", "real", "apartment-building-plan.dxf")
OUT = os.path.join(ROOT, "web", "validation_examples.json")


def box_of(pts, pad=0.0):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]


def crop_around(hl, least=2.6):
    cx, cy = (hl[0] + hl[2]) / 2, (hl[1] + hl[3]) / 2
    h = max(least, (hl[2] - hl[0]) * 1.8, (hl[3] - hl[1]) * 1.8) / 2
    return [cx - h, cy - h, cx + h, cy + h]


def opening_box(o, t, kind):
    a, b = o["position"]["start"], o["position"]["end"]
    L = math.dist(a, b) or 1e-9
    n = [-(b[1] - a[1]) / L, (b[0] - a[0]) / L]
    d = max(o.get("width") or L, t) if kind == "door" else t / 2 + 0.05
    return box_of([(p[0] + s * n[0] * d, p[1] + s * n[1] * d) for p in (a, b) for s in (1, -1)], 0.05)


def lines_in(view, bb):
    out = []
    for ln in view["lines"]:
        p = ln["p"]
        if max(q[0] for q in p) >= bb[0] and min(q[0] for q in p) <= bb[2] and max(q[1] for q in p) >= bb[1] and min(q[1] for q in p) <= bb[3]:
            out.append([[round(x, 3), round(y, 3)] for x, y in p])
    return out


def main():
    st = ExtractionPipeline().process(SRC)
    m, view = st.model, view_payload(st)
    walls = {w["id"]: w for w in m["walls"]}
    sure = lambda e: e["status"] == "ACCEPTED" and (e.get("confidence") or 0) >= 0.85  # noqa: E731
    specs = []

    def add(kind, ref, hl, least=2.6):
        bb = crop_around(hl, least)
        specs.append({"kind": kind, "ref": ref, "hl": [round(v, 3) for v in hl], "bb": [round(v, 3) for v in bb], "lines": lines_in(view, bb)})

    for kind, test in (("window", lambda o: o["type"] == "window"), ("door", lambda o: o["type"] == "door" and "swing" in (o.get("subtype") or ""))):
        cands = sorted((o for o in m["openings"] if sure(o) and test(o) and o.get("position")), key=lambda o: -o["confidence"])
        if cands:
            o = cands[0]
            add(kind, o["id"], opening_box(o, (walls.get(o["host_wall"]) or {}).get("thickness") or 0.2, kind))
    ws = sorted((w for w in m["walls"] if sure(w) and w.get("classification") == "exterior_wall" and not w.get("openings")
                 and w["geometry"].get("start") and math.dist(w["geometry"]["start"], w["geometry"]["end"]) >= 2.5), key=lambda w: -w["confidence"])
    def wall_box(w):
        a, b = w["geometry"]["start"], w["geometry"]["end"]
        L = math.dist(a, b)
        u = [(b[0] - a[0]) / L, (b[1] - a[1]) / L]
        mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
        n, h = [-u[1], u[0]], (w.get("thickness") or 0.2) / 2 + 0.05
        ends = [(mid[0] - u[0], mid[1] - u[1]), (mid[0] + u[0], mid[1] + u[1])]
        return box_of([(p[0] + s * n[0] * h, p[1] + s * n[1] * h) for p in ends for s in (1, -1)])

    def crossings(hl):                      # lines entering the box, or furniture / doors next to it
        near = [hl[0] - 0.4, hl[1] - 0.4, hl[2] + 0.4, hl[3] + 0.4]
        inside = lambda b, ln: any(b[0] < x < b[2] and b[1] < y < b[3] for x, y in ln["p"])  # noqa: E731
        return sum(1 for ln in view["lines"] if inside(hl, ln) or (ln["k"] in ("furniture", "door") and inside(near, ln)))
    if ws:
        best = min(ws[:20], key=lambda w: crossings(wall_box(w)))
        add("wall", best["id"], wall_box(best))
    stairs = [s for s in m.get("stairs") or [] if s["status"] == "ACCEPTED" and s.get("polygon")]
    if stairs:
        add("stair", stairs[0]["id"], box_of(stairs[0]["polygon"]["exterior"], 0.1), 3.0)
    # a bed: a furniture block 1.8-2.3 m long and 0.8-1.9 m wide (a counter-example: furniture is not a wall)
    import ezdxf
    from ezdxf import bbox
    tr = m["source"]["transform"]
    f = lambda x, y: ((x - tr["offset_drawing_units"][0]) * tr["scale_to_m"], (y - tr["offset_drawing_units"][1]) * tr["scale_to_m"])  # noqa: E731
    for e in ezdxf.readfile(SRC).modelspace().query("INSERT"):
        if e.dxf.layer.lower() != "furniture":
            continue
        bx = bbox.extents([e])
        lo, hi = f(bx.extmin.x, bx.extmin.y), f(bx.extmax.x, bx.extmax.y)
        dx, dy = sorted((hi[0] - lo[0], hi[1] - lo[1]))
        if 0.8 <= dx <= 1.9 and 1.8 <= dy <= 2.3:
            add("furniture", e.dxf.handle, [lo[0] - 0.05, lo[1] - 0.05, hi[0] + 0.05, hi[1] + 0.05], 3.0)
            break
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(specs, fh, separators=(",", ":"))
    print(f"{len(specs)} examples ({', '.join(s['kind'] for s in specs)}) -> {OUT} ({os.path.getsize(OUT) // 1024} KB)")


if __name__ == "__main__":
    main()
