"""Generate the benchmark DXF drawings and their reference geometry.

    python -m architectural_converter.benchmark.generate [out_dir]
"""
from __future__ import annotations

import json
import os
import sys

from .dxf_writer import render_plan
from .plans import PLANS

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "drawings")


def generate(out_dir: str = DEFAULT_DIR) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fn in PLANS:
        spec, style = fn()
        doc, ref = render_plan(spec, style)
        dxf = os.path.join(out_dir, f"{spec.name}.dxf")
        doc.saveas(dxf)
        with open(os.path.join(out_dir, f"{spec.name}.reference.json"), "w", encoding="utf-8") as fh:
            json.dump(ref, fh, indent=1)
        paths.append(dxf)
    return paths


if __name__ == "__main__":
    for p in generate(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR):
        print(p)
