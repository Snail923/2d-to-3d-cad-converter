"""Scale regression: a 4 x 4 block of apartments (~1,800 entities) must stay fast and exact.

Before spatial indexing, resolution / reconstruction compared every candidate pair, so time grew
quadratically (a 6 x 6 block took ~250 s). This guards against re-introducing that."""
import os
import tempfile
import time

from architectural_converter.benchmark.dxf_writer import Style, render_plan
from architectural_converter.benchmark.metrics import evaluate
from architectural_converter.benchmark.plan_spec import PlanSpec
from architectural_converter.benchmark.plans import apartment
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline


def test_many_apartments_fast_and_exact():
    spec = PlanSpec("scale", "4x4 apartments")
    for i in range(4):
        for j in range(4):
            apartment(spec, i * 14.0, j * 11.0)
    doc, ref = render_plan(spec, Style(units="mm", layers="aia", door_repr="block", window_repr="block",
                                       furniture_repr="block"))
    path = os.path.join(tempfile.mkdtemp(), "scale.dxf")
    doc.saveas(path)
    msgs = []
    t0 = time.perf_counter()
    st = ExtractionPipeline(progress=msgs.append).process(path)
    elapsed = time.perf_counter() - t0
    m = evaluate(st.model, ref)
    assert m["room_recall"] == 1.0 and m["room_precision"] == 1.0
    assert m["wall_recall"] >= 0.99 and m["wall_precision"] >= 0.95
    assert m["door_recall"] == 1.0 and m["window_recall"] == 1.0
    assert elapsed < 60, f"pipeline too slow: {elapsed:.1f} s"
    assert any("Step 4/10" in s for s in msgs) and any("entities" in s for s in msgs)
