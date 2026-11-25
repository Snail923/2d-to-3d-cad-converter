"""Doors / windows of the real apartment scored against the DXF's own door blocks (q1 on 'Door') and window groups
('winows' + 'Profile' layers, clustered). Rooms: counted against a manual count."""
import ezdxf, json, math, sys
from ezdxf import bbox
from shapely.geometry import box
from shapely.ops import unary_union
d = ezdxf.readfile("/home/user/latest/samples/real/apartment-building-plan.dxf"); msp = d.modelspace()
def truth(model):
    tr = model["source"]["transform"]; sc = tr["scale_to_m"]; od = tr["offset_drawing_units"]
    f = lambda x, y: ((x - od[0]) * sc, (y - od[1]) * sc)
    doors = [f((b.extmin.x + b.extmax.x) / 2, (b.extmin.y + b.extmax.y) / 2) for b in (bbox.extents([e]) for e in msp.query('INSERT[name=="q1"]'))]
    bx = [box(b.extmin.x - 50, b.extmin.y - 50, b.extmax.x + 50, b.extmax.y + 50) for b in (bbox.extents([e]) for e in msp.query('*[layer=="winows"]'))]
    u = unary_union(bx); wins = [f(g.centroid.x, g.centroid.y) for g in getattr(u, "geoms", [u])]
    return doors, wins
def match(det, ref, r):
    pairs = sorted((math.dist(a, b), i, j) for i, a in enumerate(det) for j, b in enumerate(ref) if math.dist(a, b) <= r)
    ud, ur = set(), set()
    for _, i, j in pairs:
        if i not in ud and j not in ur: ud.add(i); ur.add(j)
    return len(ud)
out = {}
for v in sys.argv[1:]:
    m = json.load(open(f"/home/user/analysis/real_models/apartment-building-plan.{v}.json"))
    doors, wins = truth(m)
    live = [o for o in m["openings"] if o["status"] != "REJECTED"]
    D = [tuple(o["position"]["center"]) for o in live if o["type"] == "door"]
    W = [tuple(o["position"]["center"]) for o in live if o["type"] == "window"]
    kd, kw = match(D, doors, 1.0), match(W, wins, 1.5)
    out[v] = {"door_P": kd / max(len(D), 1), "door_R": kd / len(doors), "window_P": kw / max(len(W), 1), "window_R": kw / len(wins),
              "doors": len(D), "windows": len(W), "rooms": sum(1 for r in m["rooms"] if r["status"] != "REJECTED")}
    print(v, {k: round(x, 2) if isinstance(x, float) else x for k, x in out[v].items()})
json.dump(out, open("/home/user/analysis/apt_scores.json", "w"), indent=1)
