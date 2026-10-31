"""Small DXF builders for unit tests (coordinates in metres unless scaled)."""
from __future__ import annotations

import os
import tempfile

import ezdxf

from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.pipeline.pipeline_config import PipelineConfig


def new_doc(insunits=6):
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = insunits
    return doc, doc.modelspace()


def save(doc, name="t.dxf") -> str:
    d = tempfile.mkdtemp(prefix="acv_")
    p = os.path.join(d, name)
    doc.saveas(p)
    return p


def rect_lines(msp, x0, y0, x1, y1, layer="A-WALL", k=1.0):
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for a, b in zip(pts, pts[1:] + pts[:1]):
        msp.add_line((a[0] * k, a[1] * k), (b[0] * k, b[1] * k), dxfattribs={"layer": layer})


def box_room(msp, w=6.0, h=4.0, t=0.2, layer="A-WALL", k=1.0):
    """Closed room drawn as outer + inner rectangles (double lines)."""
    rect_lines(msp, 0, 0, w, h, layer, k)
    rect_lines(msp, t, t, w - t, h - t, layer, k)


def run(path, until="export", **overrides):
    return ExtractionPipeline(PipelineConfig(overrides or None)).process(path, until=until)
