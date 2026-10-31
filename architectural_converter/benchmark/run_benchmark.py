"""Run the pipeline on every benchmark drawing and score it against the reference.

    python -m architectural_converter.benchmark.run_benchmark [--out results.json] [--md report.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
import traceback

from ..pipeline.extraction_pipeline import ExtractionPipeline
from ..pipeline.pipeline_config import PipelineConfig
from .generate import DEFAULT_DIR, generate
from .metrics import evaluate

KEYS = ["wall_precision", "wall_recall", "room_precision", "room_recall", "room_boundary_iou", "door_precision",
        "door_recall", "window_precision", "window_recall", "false_positive_walls", "missed_walls",
        "manual_corrections_required"]


def run(drawings_dir: str = DEFAULT_DIR, config: PipelineConfig | None = None, names=None) -> dict:
    if not glob.glob(os.path.join(drawings_dir, "*.dxf")):
        generate(drawings_dir)
    pipe = ExtractionPipeline(config or PipelineConfig())
    results = {}
    for dxf in sorted(glob.glob(os.path.join(drawings_dir, "*.dxf"))):
        name = os.path.splitext(os.path.basename(dxf))[0]
        if names and name not in names:
            continue
        ref_path = dxf.replace(".dxf", ".reference.json")
        if not os.path.exists(ref_path):
            continue
        with open(ref_path, encoding="utf-8") as fh:
            ref = json.load(fh)
        t0 = time.perf_counter()
        try:
            st = pipe.process(dxf)
            m = evaluate(st.model, ref)
            m["seconds"] = round(time.perf_counter() - t0, 2)
            m["units"] = st.drawing.units.unit
            m["description"] = ref.get("description", "")
        except Exception as exc:  # report, do not hide
            m = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
        results[name] = m
    return results


def to_markdown(results: dict) -> str:
    head = "| drawing | " + " | ".join(KEYS) + " |\n|" + "---|" * (len(KEYS) + 1) + "\n"
    rows = []
    for name, m in results.items():
        if "error" in m:
            rows.append(f"| {name} | ERROR: {m['error']} |")
            continue
        vals = []
        for k in KEYS:
            v = m.get(k)
            vals.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        rows.append(f"| {name} | " + " | ".join(vals) + " |")
    return head + "\n".join(rows) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("names", nargs="*")
    a = ap.parse_args(argv)
    res = run(a.dir, names=a.names or None)
    md = to_markdown(res)
    print(md)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as fh:
            fh.write(md)
    return res


if __name__ == "__main__":
    main()
