import json, sys, math
import ezdxf, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as P
dxf, model, level, out = sys.argv[1:5]
level = None if level == "none" else level
m = json.load(open(model)); tr = m["source"]["transform"]; sc = tr["scale_to_m"]; od = tr["offset_drawing_units"]
f = lambda x, y: ((x - od[0]) * sc, (y - od[1]) * sc)
rooms = [r for r in m["rooms"] if r["level"] == level and r["status"] != "REJECTED"]
xs = [p[0] for r in rooms for p in r["polygon"]["exterior"]]; ys = [p[1] for r in rooms for p in r["polygon"]["exterior"]]
X0, X1, Y0, Y1 = min(xs) - 3, max(xs) + 3, min(ys) - 3, max(ys) + 3
fig, ax = plt.subplots(figsize=(16, 16 * (Y1 - Y0) / (X1 - X0)), dpi=110)
doc = ezdxf.readfile(dxf)
from ezdxf.disassemble import recursive_decompose, to_primitives
for prim in to_primitives(recursive_decompose(doc.modelspace())):
    try: vs = list(prim.vertices())
    except Exception: continue
    pts = [f(v.x, v.y) for v in vs]
    if len(pts) < 2: continue
    if all(not (X0 <= x <= X1 and Y0 <= y <= Y1) for x, y in pts): continue
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#bbbbbb", lw=0.4, zorder=1)
for r in rooms:
    ax.add_patch(P(r["polygon"]["exterior"], closed=True, fc="#ffe9a8", ec="#e0b100", alpha=0.45, lw=0.6, zorder=2))
    c = r["centroid"]; ax.text(c[0], c[1], f'{r.get("classification")}\n{r["area"]:.1f}', fontsize=5, ha="center", va="center", zorder=6)
for w in m["walls"]:
    if w.get("level") != level or w["status"] == "REJECTED" or w.get("is_hypothesis"): continue
    g = w["geometry"]; a = g.get("start") and [g["start"], g["end"]] or g["axis"]
    col = {"exterior_wall": "#1f4e96", "railing": "#777"}.get(w.get("classification"), "#2e7d32")
    if w["status"] == "NEEDS_REVIEW": col = "#e000a0"
    ax.plot([p[0] for p in a], [p[1] for p in a], color=col, lw=max(1.0, (w["thickness"] or 0.1) * 12), alpha=0.6, solid_capstyle="butt", zorder=3)
for o in m["openings"]:
    if o.get("level") != level or o["status"] == "REJECTED": continue
    c = o["position"]["center"]
    col = {"door": "#ff7f00", "window": "#00a0ff"}.get(o["type"], "#ff0000")
    mk = "s" if o["status"] == "ACCEPTED" else "x"
    ax.plot(c[0], c[1], mk, color=col, ms=5, zorder=5)
for s in m.get("stairs", []):
    if s.get("level") == level and s.get("status") != "REJECTED":
        pg = s.get("polygon") or s.get("outline")
        if pg: ax.add_patch(P(pg["exterior"] if isinstance(pg, dict) else pg, closed=True, fc="#f44", alpha=0.25, zorder=2))
ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect("equal"); ax.axis("off")
ax.set_title(f"{level}: rooms={len(rooms)}  legend: blue=ext wall, green=int wall, magenta=review wall, orange=door, cyan=window, red=unidentified, x=review", fontsize=8)
plt.savefig(out, bbox_inches="tight"); print("saved", out)
