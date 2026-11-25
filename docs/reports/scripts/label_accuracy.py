"""Line-level accuracy of the raw SymPointV2 answer (before any rule), length weighted.
Truth = the layer's role (benchmark layer schemes; real drawings: their layer names)."""
import json, os, sys, collections as C
from architectural_converter.benchmark.dxf_writer import LAYER_SCHEMES
from sympoint_converter import load_config
from sympoint_converter.classes import GROUP_OF
from sympoint_converter.sympoint_io import load_drawing, primitives

GPU = sys.argv[1]; OUT = sys.argv[2]
BENCH = "/home/user/latest/architectural_converter/benchmark/drawings"
ROLE2G = {"wall": "wall", "column": "wall", "door": "door", "window": "window", "furniture": "furniture"}
T = {}
for sch, m in LAYER_SCHEMES.items():
    if sch == "poor": continue
    for role, ly in m.items():
        T[ly] = ROLE2G.get(role, "background")
REAL = {"apartment-building-plan": {"Wall": "wall", "Door": "door", "winows": "window", "furniture": "furniture", "WC": "furniture",
                                    "floor_Stair": "stair", "Balcony": "railing", "canel": "furniture"},
        "SNOOBAR_545_Permit": {"A-WALL-BLWO": "wall", "L-wall": "wall", "A-DOOR": "door", "A--GLAZ": "window", "A-FURN-FIXT": "furniture",
                               "A-STRS": "stair"}}
GMAP = {"curtain_wall": "window", "opening": "door"}      # how the build uses them
cfg = load_config()
res = {}
for f in sorted(os.listdir(GPU)):
    if not f.endswith(".sympoint.labels.json"): continue
    name = f.split(".sympoint")[0]
    if name.startswith("09"): continue
    dxf = os.path.join(BENCH, name + ".dxf") if name[:2].isdigit() else f"/home/user/latest/samples/real/{name}.dxf"
    lab = json.load(open(os.path.join(GPU, f)))["entities"]
    d = load_drawing(dxf)
    L = C.defaultdict(float); layer_of = {}
    for rid, ly, cmd, pts, ln, _ in primitives(d, cfg):
        L[rid] += ln; layer_of[rid] = ly
    truth = (lambda ly: T.get(ly, "background")) if name[:2].isdigit() else (lambda ly, m=REAL[name]: m.get(ly, "background"))
    conf = C.defaultdict(float)
    for rid, ln in L.items():
        e = lab.get(rid)
        p = GROUP_OF.get(e["label"], "other") if e else "none"
        p = GMAP.get(p, p)
        conf[(truth(layer_of[rid]), p)] += ln
    res[name] = {f"{a}|{b}": round(v, 3) for (a, b), v in conf.items()}
    tot = C.defaultdict(float)
    for (a, b), v in conf.items(): tot[a] += v
    print(name, {a: f"{conf[(a, a)] / tot[a]:.2f} of {tot[a]:.0f}m" for a in sorted(tot) if a != "background"},
          "bg->wall %.2f" % (conf[("background", "wall")] / max(tot["background"], 1e-9)), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
