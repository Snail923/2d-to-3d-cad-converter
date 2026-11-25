"""Same drawings with every layer renamed to '0' (no layer names to lean on). The simulated model's labels
are taken from the original drawing by DXF entity id, so the model is as good as before; only the
layer-name crutch is gone. Run: python scratch_bench_strip.py out.json seeds [names]"""
import json, os, sys, time, traceback, tempfile
import ezdxf
from scratch_bench import answer, KEEP, CLASSES
from architectural_converter.benchmark.generate import DEFAULT_DIR
from architectural_converter.benchmark.metrics import evaluate
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.pipeline.pipeline_config import PipelineConfig
from sympoint_converter import load_config
from sympoint_converter.pipeline import SymPointPipeline
from sympoint_converter.sympoint_io import export_tiles, load_drawing

TMP = tempfile.mkdtemp(dir="/tmp/claude-0")

def strip(path):
    doc = ezdxf.readfile(path)
    for blk in doc.blocks:
        for e in blk:
            try: e.dxf.layer = "0"
            except Exception: pass
    for e in doc.modelspace():
        try: e.dxf.layer = "0"
        except Exception: pass
    out = os.path.join(TMP, os.path.basename(path)); doc.saveas(out); return out

def a2(orig, stripped, damage, seed, reasoning):
    cfg = load_config({"use_reasoning": reasoning, "cache_labels": False})
    ans = answer(export_tiles(load_drawing(orig), cfg), damage, seed)
    req0 = export_tiles(load_drawing(orig), cfg)
    by_id = {}
    for t, a in zip(req0["tiles"], ans["tiles"]):
        for rid, s in zip(t["ids"], a["semantic"]):
            by_id[rid] = s
    req = export_tiles(load_drawing(stripped), cfg)
    miss = 0; tiles = []
    for t in req["tiles"]:
        sem = []
        for rid in t["ids"]:
            s = by_id.get(rid)
            if s is None: miss += 1; s = [CLASSES.index("wall"), 0.05, 0.02]
            sem.append(s)
        tiles.append({"id": t["id"], "semantic": sem, "instances": []})
    if miss: print("  unmatched primitives:", miss, flush=True)
    return SymPointPipeline(cfg, response={"model": "sim", "tiles": tiles}, labels_progress=lambda m: None).process(stripped)

def main():
    out, seeds = sys.argv[1], int(sys.argv[2]); names = sys.argv[3:] or None
    res = {}; p1 = ExtractionPipeline(PipelineConfig())
    for f in sorted(os.listdir(DEFAULT_DIR)):
        if not f.endswith(".dxf") or (names and f[:-4] not in names): continue
        name = f[:-4]; path = os.path.join(DEFAULT_DIR, f); ref = json.load(open(path.replace(".dxf", ".reference.json")))
        sp = strip(path)
        runs = {"A1_rules_nolayers": lambda: p1.process(sp), "A2_perfect_nolayers": lambda: a2(path, sp, False, 0, True)}
        for s in range(seeds):
            runs[f"A2_damaged_reasoning_nolayers_s{s}"] = (lambda s=s: a2(path, sp, True, s, True))
            runs[f"A2_damaged_raw_nolayers_s{s}"] = (lambda s=s: a2(path, sp, True, s, False))
        res[name] = {}
        for k, fn in runs.items():
            t0 = time.perf_counter()
            try:
                m = evaluate(fn().model, ref); res[name][k] = {x: m[x] for x in KEEP} | {"s": round(time.perf_counter() - t0, 1)}
            except Exception as e:
                res[name][k] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
            print(name, k, res[name][k].get("error") or {x: round(res[name][k][x], 2) for x in ("wall_recall", "room_recall", "door_recall", "window_recall")}, flush=True)
        json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
