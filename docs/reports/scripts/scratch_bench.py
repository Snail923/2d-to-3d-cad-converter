"""Benchmark: approach 1 vs approach 2 (perfect labels / damaged labels, with and without reasoning).

Damage profile = the one the sympoint_converter README says the real model shows on the permit set:
10-15 % wall lines unlabelled, 60-70 % window lines unlabelled, 8 % whole tiles unlabelled,
25 % furniture and 15 % door lines called wall, 3 % stray wall labels.
Run:  python scratch_bench.py out.json [seeds]
"""
import json
import os
import random
import sys
import time
import traceback

from architectural_converter.benchmark.dxf_writer import LAYER_SCHEMES
from architectural_converter.benchmark.generate import DEFAULT_DIR
from architectural_converter.benchmark.metrics import evaluate
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.pipeline.pipeline_config import PipelineConfig
from sympoint_converter import CLASSES, load_config
from sympoint_converter.pipeline import SymPointPipeline
from sympoint_converter.sympoint_io import export_tiles, load_drawing

os.environ.setdefault("SYMPOINT_STATE_DIR", "/tmp/claude-0/sp_state")
ROLE = {"wall": "wall", "door": "single door", "window": "window", "column": "wall", "furniture": "bed", "hatch": "wall"}
ORACLE = {}
for sch in LAYER_SCHEMES.values():
    for role, ly in sch.items():
        if role in ROLE and ly != "0":
            ORACLE.setdefault(ly, ROLE[role])


def answer(req, damage=False, seed=0):
    rnd = random.Random(seed)
    tiles = []
    for t in req["tiles"]:
        dead_tile = damage and rnd.random() < 0.08
        sem = []
        for lid in t["sympoint"]["layerIds"]:
            c = ORACLE.get(t["layer_names"][lid - 1])
            if damage:
                r = rnd.random()
                if dead_tile:
                    c = None
                elif c == "wall" and r < 0.125:
                    c = None
                elif c == "window" and r < 0.65:
                    c = None
                elif c == "bed" and r < 0.25:
                    c = "wall"
                elif c == "single door" and r < 0.15:
                    c = "wall"
                elif c is None and r < 0.03:
                    c = "wall"
            sem.append([CLASSES.index(c), 0.9, 1.0] if c else [CLASSES.index("wall"), 0.05, 0.02])
        tiles.append({"id": t["id"], "semantic": sem, "instances": []})
    return {"model": "sim", "tiles": tiles}


def a2(path, damage, seed, reasoning):
    cfg = load_config({"use_reasoning": reasoning, "cache_labels": False})
    req = export_tiles(load_drawing(path), cfg)
    return SymPointPipeline(cfg, response=answer(req, damage, seed), labels_progress=lambda m: None).process(path)


KEEP = ["wall_precision", "wall_recall", "room_precision", "room_recall", "room_boundary_iou", "door_precision",
        "door_recall", "window_precision", "window_recall", "column_precision", "column_recall",
        "manual_corrections_required", "needs_review_items"]


def main():
    out = sys.argv[1]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    names = sys.argv[3:] or None
    res = {}
    p1 = ExtractionPipeline(PipelineConfig())
    for f in sorted(os.listdir(DEFAULT_DIR)):
        if not f.endswith(".dxf"):
            continue
        name = f[:-4]
        if names and name not in names:
            continue
        path = os.path.join(DEFAULT_DIR, f)
        ref = json.load(open(path.replace(".dxf", ".reference.json")))
        runs = {"A1_rules": lambda: p1.process(path),
                "A2_perfect_labels": lambda: a2(path, False, 0, True)}
        for s in range(seeds):
            runs[f"A2_damaged_reasoning_s{s}"] = (lambda s=s: a2(path, True, s, True))
            runs[f"A2_damaged_raw_s{s}"] = (lambda s=s: a2(path, True, s, False))
        res[name] = {}
        for k, fn in runs.items():
            t0 = time.perf_counter()
            try:
                m = evaluate(fn().model, ref)
                res[name][k] = {x: m[x] for x in KEEP} | {"s": round(time.perf_counter() - t0, 1)}
            except Exception as e:
                res[name][k] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
            print(name, k, res[name][k].get("error") or {x: round(v, 2) for x, v in res[name][k].items()
                                                         if x in ("wall_recall", "room_recall", "door_recall", "window_recall")},
                  flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
