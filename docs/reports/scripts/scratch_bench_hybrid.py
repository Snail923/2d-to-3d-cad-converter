"""Hybrid: where the (damaged) model is silent, approach 1's accepted elements vote on the line's label.
Layer-stripped drawings, so no layer-name crutch. Run: python scratch_bench_hybrid.py out.json seeds"""
import json, os, sys, time, traceback
import scratch_bench_strip as S
from scratch_bench import answer, KEEP, CLASSES
from architectural_converter.benchmark.generate import DEFAULT_DIR
from architectural_converter.benchmark.metrics import evaluate
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.pipeline.pipeline_config import PipelineConfig
from sympoint_converter import load_config
from sympoint_converter.pipeline import SymPointPipeline
from sympoint_converter.sympoint_io import export_tiles, load_drawing

def a1_votes(model):
    v = {}
    for o in model["openings"]:
        if o["status"] == "REJECTED": continue
        c = {"door": "single door", "window": "window"}.get(o["type"])
        if c:
            for s in o.get("source_entities", []): v[s["id"] if isinstance(s, dict) else s] = c
    for w in model["walls"]:
        if w["status"] == "REJECTED" or w.get("is_hypothesis"): continue
        for s in w.get("source_entities", []): v.setdefault(s["id"] if isinstance(s, dict) else s, "wall")
    return v

def run(orig, stripped, seed, votes):
    cfg = load_config({"use_reasoning": True, "cache_labels": False})
    ans = answer(export_tiles(load_drawing(orig), cfg), True, seed)
    req0 = export_tiles(load_drawing(orig), cfg)
    by_id = {rid: s for t, a in zip(req0["tiles"], ans["tiles"]) for rid, s in zip(t["ids"], a["semantic"])}
    req = export_tiles(load_drawing(stripped), cfg)
    tiles, filled = [], 0
    for t in req["tiles"]:
        sem = []
        for rid in t["ids"]:
            s = by_id.get(rid, [CLASSES.index("wall"), 0.05, 0.02])
            if s[1] < 0.35 and rid in votes:           # model silent -> approach 1 votes
                s = [CLASSES.index(votes[rid]), 0.6, 0.6]; filled += 1
            sem.append(s)
        tiles.append({"id": t["id"], "semantic": sem, "instances": []})
    return SymPointPipeline(cfg, response={"model": "sim", "tiles": tiles}, labels_progress=lambda m: None).process(stripped), filled

out, seeds = sys.argv[1], int(sys.argv[2]); res = {}
p1 = ExtractionPipeline(PipelineConfig())
for f in sorted(os.listdir(DEFAULT_DIR)):
    if not f.endswith(".dxf") or f.startswith("09"): continue
    name = f[:-4]; path = os.path.join(DEFAULT_DIR, f); ref = json.load(open(path.replace(".dxf", ".reference.json")))
    sp = S.strip(path); votes = a1_votes(p1.process(sp).model); res[name] = {}
    for s in range(seeds):
        try:
            st, filled = run(path, sp, s, votes); m = evaluate(st.model, ref)
            res[name][f"A2_damaged_reasoning_plus_A1_fallback_nolayers_s{s}"] = {x: m[x] for x in KEEP}
            print(name, s, "filled", filled, {x: round(m[x], 2) for x in ("wall_recall", "room_recall", "door_recall", "window_recall")}, flush=True)
        except Exception as e:
            res[name][f"err_s{s}"] = {"error": str(e)}; traceback.print_exc()
    json.dump(res, open(out, "w"), indent=1)
