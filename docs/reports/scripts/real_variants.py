"""Build approach 2 from the REAL SymPointV2 answers in several variants and score them.
Variants: rules (as shipped), raw (rules off), and on the layer-stripped drawing: rules, rules + approach-1 fallback.
Run from the repo root: python real_variants.py GPU_DIR OUT.json [names]"""
import json, os, sys, time, traceback
sys.path.insert(0, "/home/user/latest")
import scratch_bench_strip as S
from scratch_bench import KEEP
from architectural_converter.benchmark.metrics import evaluate
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.pipeline.pipeline_config import PipelineConfig
from sympoint_converter import CLASSES, load_config
from sympoint_converter.pipeline import SymPointPipeline
from sympoint_converter.sympoint_io import export_tiles, load_drawing

GPU, OUT = sys.argv[1], sys.argv[2]; names = sys.argv[3:] or None
BENCH = "/home/user/latest/architectural_converter/benchmark/drawings"
EMPTY = [CLASSES.index("wall"), 0.0, 0.0]

def check(req, resp):
    rt = {t["id"]: t for t in resp["tiles"]}
    bad = [t["id"] for t in req["tiles"] if t["id"] not in rt or len(rt[t["id"]]["semantic"]) != len(t["ids"])]
    return bad

def by_id(path, resp, cfg):
    req = export_tiles(load_drawing(path), cfg); bad = check(req, resp)
    if bad: raise RuntimeError(f"response does not match the request tiling: {bad[:5]}")
    rt = {t["id"]: t for t in resp["tiles"]}
    return {rid: s for t in req["tiles"] for rid, s in zip(t["ids"], rt[t["id"]]["semantic"])}

def remap(path, ids, cfg, votes=None):
    req = export_tiles(load_drawing(path), cfg); tiles = []; filled = 0
    for t in req["tiles"]:
        sem = []
        for rid in t["ids"]:
            s = ids.get(rid, EMPTY)
            if votes and rid in votes and (s[1] < cfg["min_label_score"] or s[2] < cfg["min_label_mass"]):
                s = [CLASSES.index(votes[rid]), 0.6, 0.6]; filled += 1
            sem.append(s)
        tiles.append({"id": t["id"], "semantic": sem, "instances": []})
    return {"model": "SymPointV2 (remapped)", "tiles": tiles}, filled

def a1_votes(model):
    v = {}
    for o in model["openings"]:
        if o["status"] == "REJECTED": continue
        c = {"door": "single door", "window": "window"}.get(o["type"])
        if c:
            for s in o.get("source_entities", []): v[s["id"]] = c
    for w in model["walls"]:
        if w["status"] == "REJECTED" or w.get("is_hypothesis"): continue
        for s in w.get("source_entities", []): v.setdefault(s["id"], "wall")
    return v

def build(path, resp, reasoning=True):
    cfg = load_config({"use_reasoning": reasoning, "cache_labels": False})
    return SymPointPipeline(cfg, response=resp, labels_progress=lambda m: None).process(path)

res = {}
p1 = ExtractionPipeline(PipelineConfig())
for f in sorted(os.listdir(BENCH)):
    if not f.endswith(".dxf"): continue
    name = f[:-4]
    if names and name not in names: continue
    path = os.path.join(BENCH, f); ref = json.load(open(path.replace(".dxf", ".reference.json")))
    resp = json.load(open(os.path.join(GPU, name + ".sympoint.response.json")))
    cfg = load_config({"cache_labels": False})
    res[name] = {}
    def score(k, fn):
        t0 = time.perf_counter()
        try:
            m = evaluate(fn().model, ref); res[name][k] = {x: m[x] for x in KEEP} | {"s": round(time.perf_counter() - t0, 1)}
        except Exception as e:
            res[name][k] = {"error": f"{type(e).__name__}: {e}"}; traceback.print_exc()
        print(name, k, res[name][k].get("error") or {x: round(res[name][k][x], 2) for x in ("wall_recall", "room_recall", "door_recall", "window_recall", "manual_corrections_required")}, flush=True)
    score("A2_real_rules", lambda: build(path, resp, True))
    score("A2_real_raw", lambda: build(path, resp, False))
    if name.startswith("09"):
        json.dump(res, open(OUT, "w"), indent=1); continue
    ids = by_id(path, resp, cfg); sp = S.strip(path)
    score("A2_real_rules_nolayers", lambda: build(sp, remap(sp, ids, cfg)[0], True))
    votes = a1_votes(p1.process(sp).model)
    score("A2_real_rules_fallback_nolayers", lambda: build(sp, remap(sp, ids, cfg, votes)[0], True))
    score("A2_real_rules_fallback", lambda: build(path, remap(path, ids, cfg, a1_votes(p1.process(path).model))[0], True))
    json.dump(res, open(OUT, "w"), indent=1)
