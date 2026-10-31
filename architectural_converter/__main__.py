"""Command line entry point.

    python -m architectural_converter building.dxf -o out/
    python -m architectural_converter building.dwg -o out/ --corrections corrections.json
    python -m architectural_converter --benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .pipeline.database_exporter import export_json, export_sqlite
from .pipeline.extraction_pipeline import STAGES, ExtractionPipeline
from .pipeline.pipeline_config import PipelineConfig


def main(argv=None):
    ap = argparse.ArgumentParser(prog="architectural_converter",
                                 description="Deterministic DWG/DXF architectural feature extraction (Phase 1) + 3D (Phase 2)")
    ap.add_argument("input", nargs="?", help="DXF or DWG file")
    ap.add_argument("-o", "--out", default="out", help="output directory")
    ap.add_argument("--config", help="JSON file with configuration overrides")
    ap.add_argument("--corrections", help="manual corrections JSON (from the review UI)")
    ap.add_argument("--until", choices=STAGES, default="export", help="stop after this stage (debugging)")
    ap.add_argument("--units", choices=["mm", "cm", "dm", "m", "in", "ft"], help="force drawing units")
    ap.add_argument("--no-3d", action="store_true", help="skip the Phase 2 GLB export")
    ap.add_argument("--benchmark", action="store_true", help="run the benchmark suite instead")
    a = ap.parse_args(argv)
    if a.benchmark:
        from .benchmark.run_benchmark import main as bench
        bench([])
        return 0
    if not a.input:
        ap.error("input file required")
    overrides = {}
    if a.config:
        with open(a.config, encoding="utf-8") as fh:
            overrides = json.load(fh)
    if a.units:
        overrides.setdefault("runtime", {})["forced_units"] = a.units
    cfg = PipelineConfig(overrides)
    corr = None
    if a.corrections:
        with open(a.corrections, encoding="utf-8") as fh:
            corr = json.load(fh)
    pipe = ExtractionPipeline(cfg)
    st = pipe.process(a.input, corrections=corr, until=a.until)
    os.makedirs(a.out, exist_ok=True)
    if st.model is None:
        print(f"Stopped after stage '{a.until}'. Timings: {st.timings}")
        return 0
    base = os.path.join(a.out, os.path.splitext(os.path.basename(a.input))[0])
    export_json(st.model, base + ".model.json")
    export_sqlite(st.model, base + ".model.sqlite")
    from .visualization.feature_overlay import render_overlay
    from .visualization.validation_viewer import html_report
    svg = render_overlay(st)
    with open(base + ".overlay.svg", "w", encoding="utf-8") as fh:
        fh.write(svg)
    with open(base + ".report.html", "w", encoding="utf-8") as fh:
        fh.write(html_report(st, svg))
    if not a.no_3d:
        from .reconstruction3d import build_3d
        m3 = build_3d(st.model, cfg)
        m3.export_glb(base + ".glb")
        with open(base + ".3d_metadata.json", "w", encoding="utf-8") as fh:
            json.dump(m3.metadata, fh, indent=1)
    c = st.model["statistics"]["counts"]
    print(f"units={st.drawing.units.unit} walls={c['walls']} openings={c['openings']} rooms={c['rooms']} "
          f"columns={c['columns']} issues={st.model['validation']['summary'].get('total', 0)}")
    print(f"outputs: {base}.*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
