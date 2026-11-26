"""Side-by-side comparison of approach 1 (rules) and approach 2 (SymPointV2) on the same drawings.

For every drawing both pipelines run on the same file. The report gives:

* counts and timings of both models,
* scores against a reference (``<name>.reference.json`` next to the DXF, e.g. the benchmark
  drawings) with approach 1's benchmark metrics, when a reference exists,
* the agreement between the two approaches (approach 1's model used as the reference for
  approach 2), which needs no reference at all: low agreement shows where to look,
* both 2D overlays next to each other."""
from __future__ import annotations

import html
import json
import os
import time
import traceback

from architectural_converter.benchmark.metrics import evaluate
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.pipeline.pipeline_config import PipelineConfig
from architectural_converter.visualization.feature_overlay import render_overlay

from .pipeline import SymPointPipeline

SCORE_KEYS = ["wall_precision", "wall_recall", "room_precision", "room_recall", "door_precision", "door_recall",
              "window_precision", "window_recall", "column_precision", "column_recall", "manual_corrections_required"]
OK = ("ACCEPTED", "NEEDS_REVIEW")


def counts(model) -> dict:
    live = lambda xs: [x for x in xs if x["status"] in OK and not x.get("is_hypothesis")]  # noqa: E731
    walls = live(model["walls"])
    ops = live(model["openings"])
    return {"walls": len(walls), "wall_length_m": round(sum(w["geometry"].get("length", 0) for w in walls), 1),
            "doors": sum(1 for o in ops if o["type"] == "door"), "windows": sum(1 for o in ops if o["type"] == "window"),
            "other_openings": sum(1 for o in ops if o["type"] == "opening"), "rooms": len(live(model["rooms"])),
            "columns": len(live(model["columns"])), "stairs": len(live(model.get("stairs", []))),
            "needs_review": sum(1 for k in ("walls", "openings", "rooms") for x in model[k] if x["status"] == "NEEDS_REVIEW"),
            "validation_issues": model["validation"]["summary"].get("total", 0)}


def model_as_reference(model) -> dict:
    """Approach 1's model in the benchmark reference format (frame = identity on the model)."""
    tr = model["source"]["transform"]
    sc, od = tr["scale_to_m"], tr["offset_drawing_units"]
    live = lambda xs: [x for x in xs if x["status"] in OK and not x.get("is_hypothesis")]  # noqa: E731
    walls = []
    for w in live(model["walls"]):
        g = w["geometry"]
        t = w["thickness"] or 0.15
        if g.get("start"):
            walls.append({"start": g["start"], "end": g["end"], "thickness": t, "kind": "straight"})
        elif g.get("axis"):
            walls.append({"start": g["axis"][0], "end": g["axis"][-1], "axis": g["axis"], "thickness": t, "kind": "arc"})
    ops = live(model["openings"])
    op = lambda o: {"center": o["position"]["center"], "width": o["width"]}  # noqa: E731
    return {"name": "approach 1", "frame": {"units": "m", "to_m": sc, "offset_m": [od[0] * sc, od[1] * sc]},
            "walls": walls,
            "rooms": [{"polygon": r["polygon"]["exterior"], "name": r.get("name")} for r in live(model["rooms"])],
            "doors": [op(o) for o in ops if o["type"] == "door"],
            "windows": [op(o) for o in ops if o["type"] == "window"],
            "openings": [op(o) for o in ops if o["type"] == "opening"],
            "columns": [{"center": c["center"], "size": c["size"]} for c in live(model["columns"])]}


def _short(m):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k in SCORE_KEYS}


def compare_one(path, sp_cfg, url=None, response=None, progress=print) -> dict:
    out = {"file": path}
    ref_path = os.path.splitext(path)[0] + ".reference.json"
    ref = None
    if os.path.exists(ref_path):
        with open(ref_path, encoding="utf-8") as fh:
            ref = json.load(fh)
    t0 = time.perf_counter()
    st1 = ExtractionPipeline(PipelineConfig()).process(path)
    out["approach1"] = {"seconds": round(time.perf_counter() - t0, 2), "counts": counts(st1.model)}
    t0 = time.perf_counter()
    st2 = SymPointPipeline(sp_cfg, url=url, response=response, labels_progress=progress).process(path)
    out["approach2"] = {"seconds": round(time.perf_counter() - t0, 2), "counts": counts(st2.model),
                        "labels_source": st2.logs["detect"].get("labels_source"),
                        "extract": {k: v for k, v in st2.logs["detect"].items() if k != "approach"}}
    if ref is not None:
        out["approach1"]["scores"] = _short(evaluate(st1.model, ref))
        out["approach2"]["scores"] = _short(evaluate(st2.model, ref))
    agree = evaluate(st2.model, model_as_reference(st1.model))
    out["agreement"] = {"wall_length_also_in_approach1": round(agree["wall_precision"], 3),
                        "approach1_wall_length_found": round(agree["wall_recall"], 3),
                        "rooms_matching": round(min(agree["room_precision"], agree["room_recall"]), 3),
                        "doors_matching": round(min(agree["door_precision"], agree["door_recall"]), 3),
                        "windows_matching": round(min(agree["window_precision"], agree["window_recall"]), 3)}
    out["_states"] = (st1, st2)
    return out


def run(paths, sp_cfg, url=None, response=None, progress=print) -> list:
    res = []
    for p in paths:
        progress(f"== {os.path.basename(p)}")
        try:
            res.append(compare_one(p, sp_cfg, url, response if len(paths) == 1 else None, progress))
        except Exception as e:  # noqa: BLE001  (report, do not hide)
            res.append({"file": p, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})
    return res


def to_markdown(res) -> str:
    lines = []
    for r in res:
        name = os.path.basename(r["file"])
        if "error" in r:
            lines.append(f"### {name}\nERROR: {r['error']}\n")
            continue
        a, b = r["approach1"], r["approach2"]
        lines.append(f"### {name}\n")
        lines.append("| | approach 1 (rules) | approach 2 (SymPointV2) |\n|---|---|---|")
        for k in a["counts"]:
            lines.append(f"| {k} | {a['counts'][k]} | {b['counts'][k]} |")
        lines.append(f"| seconds | {a['seconds']} | {b['seconds']} (labels: {b['labels_source']}) |")
        if "scores" in a:
            for k in SCORE_KEYS:
                fa, fb = a["scores"].get(k), b["scores"].get(k)
                fmt = (lambda v: f"{v:.2f}" if isinstance(v, float) else str(v))
                lines.append(f"| **{k}** | {fmt(fa)} | {fmt(fb)} |")
        lines.append("\nAgreement (approach 1 as reference): " +
                     ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in r["agreement"].items()) + "\n")
    return "\n".join(lines)


def to_html(res) -> str:
    css = """:root{--bg:#fbfbfa;--panel:#fff;--fg:#1d1d1b;--muted:#6b6b66;--line:#e4e3df;--good:#2f7d4f;--bad:#b4442c}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--panel:#1f1f1d;--fg:#ecebe6;--muted:#9c9b95;--line:#33332f;--good:#6fcf97;--bad:#f08c74}}
body{margin:0;padding:16px;font:13px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
h1{font-size:18px}h2{font-size:15px;margin-top:28px}table{border-collapse:collapse;margin:8px 0}
td,th{padding:3px 10px;border-bottom:1px solid var(--line);text-align:left}td.n{text-align:right;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}
.pane{background:#fff;border:1px solid var(--line);border-radius:6px;padding:6px;overflow:hidden}.pane svg{width:100%;height:auto}
.cap{color:var(--muted);margin:4px 0}.b{color:var(--good)}.w{color:var(--bad)}"""
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Approach comparison</title><style>{css}</style></head><body><h1>Approach 1 (rules) vs approach 2 (SymPointV2)</h1>"]
    for r in res:
        name = html.escape(os.path.basename(r["file"]))
        if "error" in r:
            parts.append(f"<h2>{name}</h2><pre>{html.escape(r['error'])}</pre>")
            continue
        a, b = r["approach1"], r["approach2"]
        rows = "".join(f"<tr><td>{k.replace('_', ' ')}</td><td class=n>{a['counts'][k]}</td><td class=n>{b['counts'][k]}</td></tr>"
                       for k in a["counts"])
        rows += f"<tr><td>seconds</td><td class=n>{a['seconds']}</td><td class=n>{b['seconds']}</td></tr>"
        if "scores" in a:
            for k in SCORE_KEYS:
                va, vb = a["scores"].get(k), b["scores"].get(k)
                if isinstance(va, float):
                    better = "b" if vb > va + 0.005 else ("w" if vb < va - 0.005 else "")
                    rows += f"<tr><td><b>{k.replace('_', ' ')}</b></td><td class=n>{va:.2f}</td><td class='n {better}'>{vb:.2f}</td></tr>"
                elif va is not None:
                    better = "b" if vb < va else ("w" if vb > va else "")
                    rows += f"<tr><td><b>{k.replace('_', ' ')}</b></td><td class=n>{va}</td><td class='n {better}'>{vb}</td></tr>"
        agree = ", ".join(f"{k.replace('_', ' ')} <b>{v:.0%}</b>" for k, v in r["agreement"].items())
        st1, st2 = r.get("_states", (None, None))
        svgs = ""
        if st1 is not None:
            try:
                svgs = (f"<div class=grid><div><div class=cap>approach 1 (rules)</div><div class=pane>{render_overlay(st1)}</div></div>"
                        f"<div><div class=cap>approach 2 (SymPointV2)</div><div class=pane>{render_overlay(st2)}</div></div></div>")
            except Exception as e:  # noqa: BLE001
                svgs = f"<p class=cap>overlay failed: {html.escape(str(e))}</p>"
        parts.append(f"<h2>{name}</h2><table><tr><th></th><th>approach 1</th><th>approach 2</th></tr>{rows}</table>"
                     f"<p class=cap>Agreement, approach 1 as reference: {agree}. Scores in bold need a reference "
                     f"drawing ({name.rsplit('.', 1)[0]}.reference.json); green / red = approach 2 better / worse.</p>{svgs}")
    parts.append("</body></html>")
    return "".join(parts)


def strip_states(res):
    return [{k: v for k, v in r.items() if k != "_states"} for r in res]
