"""Regression: every benchmark drawing must stay at or above its recorded baseline."""
import json
import os

import pytest

from architectural_converter.benchmark.metrics import evaluate
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "architectural_converter", "benchmark")
with open(os.path.join(BASE, "baseline.json"), encoding="utf-8") as fh:
    BASELINE = json.load(fh)["drawings"]


@pytest.mark.parametrize("name", sorted(BASELINE))
def test_benchmark_drawing(name):
    dxf = os.path.join(BASE, "drawings", name + ".dxf")
    with open(dxf.replace(".dxf", ".reference.json"), encoding="utf-8") as fh:
        ref = json.load(fh)
    st = ExtractionPipeline().process(dxf)
    m = evaluate(st.model, ref)
    b = BASELINE[name]
    for k, v in b.items():
        if k.startswith("max_"):
            assert m[k[4:]] <= v, f"{name}: {k[4:]}={m[k[4:]]} > {v}"
        else:
            assert m[k] >= v, f"{name}: {k}={m[k]:.3f} < baseline {v}"
