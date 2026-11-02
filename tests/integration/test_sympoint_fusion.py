"""SymPointV2 integration: input format, label mapping and evidence fusion.

The real model needs a CUDA GPU (tools/sympoint_service); here its answers are simulated per DXF
entity so the engine side (export, id mapping, fusion rules) is tested exactly."""
import json
import os
import sys
import tempfile
import types

import numpy as np
import pytest

from architectural_converter.benchmark.dxf_writer import Style, render_plan
from architectural_converter.benchmark.plan_spec import PlanSpec
from architectural_converter.learned import CLASSES, export_tiles, labels_from_response
from architectural_converter.models.base_entity import Status
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.pipeline.pipeline_config import PipelineConfig


def simulated_model(request, label_of):
    """Answer like the service: per primitive [class index, score]."""
    tiles = []
    for t in request["tiles"]:
        sem = []
        for rid in t["ids"]:
            name = label_of(rid)
            sem.append([CLASSES.index(name), 0.9] if name else [CLASSES.index("wall"), 0.05])
        tiles.append({"id": t["id"], "semantic": sem, "instances": []})
    return {"model": "SymPointV2", "tiles": tiles}


def _bedroom_with_wardrobe_and_shelf():
    s = PlanSpec("wardrobe", "bedroom, wardrobe front drawn like glazing, shelf near a wall")
    s.chain([(0, 0), (6, 0), (6, 4), (0, 4)], 0.3, closed=True)
    s.wall((0, 1.6), (6, 1.6), 0.12)
    s.wall((4.6, 1.6), (4.6, 4), 0.12)
    s.opening_at((3, 1.6), 0.9, "door")
    s.opening_at((4.6, 2.8), 1.6, "window")      # wardrobe front
    s.opening_at((2, 4), 1.5, "window")
    s.opening_at((6, 0.8), 0.9, "door")
    doc, _ = render_plan(s, Style(layers="aia", window_repr="loose"))
    msp = doc.modelspace()
    for dy in (0, 60, 120):                      # shelf: three parallel lines in front of the north wall (mm)
        msp.add_line((1150, 3550 + dy), (2550, 3550 + dy), dxfattribs={"layer": "A-WALL"})
    p = os.path.join(tempfile.mkdtemp(prefix="acv_"), "wardrobe.dxf")
    doc.saveas(p)
    return p


def _labeller(st):
    """Simulated model answer per DXF entity (DXF coordinates are millimetres here)."""
    raw = st.drawing.raw_entities

    def label_of(rid):
        r = raw.get(rid)
        if r is None or not r.raw_coords:
            return None
        c = np.mean(np.asarray(r.raw_coords, float)[:, :2], axis=0) / 1000.0
        x, y = c
        if r.layer == "A-GLAZ" and abs(x - 4.6) < 0.2 and 1.9 < y < 3.7:
            return "Wardrobe"
        if r.layer == "A-WALL" and 1.0 < x < 2.7 and 3.5 < y < 3.7:
            return "bed"                         # the shelf lines (any furniture class)
        return {"A-WALL": "wall", "A-GLAZ": "window", "A-DOOR": "single door"}.get(r.layer)
    return label_of


def test_request_matches_sympoint_format():
    st = ExtractionPipeline(PipelineConfig()).process(_bedroom_with_wardrobe_and_shelf(), until="preprocess")
    req = export_tiles(st.drawing, PipelineConfig())
    t = req["tiles"][0]
    sp = t["sympoint"]
    n = len(t["ids"])
    assert n == req["primitives"] and all(len(a) == 8 for a in sp["args"])
    for k in ("commands", "lengths", "semanticIds", "instanceIds", "rgb", "layerIds", "widths"):
        assert len(sp[k]) == n
    assert set(sp["commands"]) <= {0, 1, 2, 3} and sp["semanticIds"][0] == len(CLASSES)
    a = np.array(sp["args"])
    assert a.min() >= 0 and a[:, 0::2].max() <= sp["width"] and a[:, 1::2].max() <= sp["height"]


def test_loads_with_sympoint_own_loader():
    """If a SymPointV2 checkout is available, our JSON goes through its data loader unchanged."""
    root = os.environ.get("SYMPOINT_DIR", "/tmp/ext/SymPointV2")
    src = os.path.join(root, "svgnet", "data", "svg3.py")
    if not os.path.exists(src):
        pytest.skip("no SymPointV2 checkout")
    saved = {m: sys.modules.get(m) for m in ("torch", "torch.utils", "torch.utils.data")}
    try:
        for m in saved:
            sys.modules[m] = types.ModuleType(m)
        sys.modules["torch.utils.data"].Dataset = object
        ns = {}
        exec(compile(open(src).read().replace("from .aug_utils import *", ""), src, "exec"), ns)
    finally:
        for m, v in saved.items():
            if v is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = v
    st = ExtractionPipeline(PipelineConfig()).process(_bedroom_with_wardrobe_and_shelf(), until="preprocess")
    t = export_tiles(st.drawing, PipelineConfig())["tiles"][0]
    p = os.path.join(tempfile.mkdtemp(), "x_s2.json")
    json.dump(t["sympoint"], open(p, "w"))
    coord, feat, label, lengths, layers = ns["SVGDataset"].load(p, idx=1)
    n = len(t["ids"])
    assert feat.shape[1] == 7 and coord[:n, :2].min() >= 0 and coord[:n, :2].max() <= 1


def test_fusion_rejects_furniture_and_retypes_wardrobe_front():
    pipe = ExtractionPipeline(PipelineConfig())
    st = pipe.process(_bedroom_with_wardrobe_and_shelf())
    shelf_walls = [w for w in st.building.walls.values() if w.status != Status.REJECTED
                   and w.thickness and w.thickness < 0.09]
    assert shelf_walls, "without labels the shelf is (wrongly) read as a thin wall"
    req = export_tiles(st.drawing, pipe.cfg)
    labels = labels_from_response(req, simulated_model(req, _labeller(st)))
    st = pipe.rerun(st, symbol_labels=labels)
    lg = st.logs["learned"]
    assert lg["resolve"]["walls_vetoed"] >= 1
    for w in shelf_walls:
        assert st.building.walls[w.id].status == Status.REJECTED
        assert "furniture" in st.building.walls[w.id].status_reason
    doors = [o for o in st.building.openings.values() if o.subtype == "sliding_or_closet" and o.status != Status.REJECTED]
    assert len(doors) == 1 and "SymPointV2" in doors[0].status_reason
    wins = [o for o in st.building.openings.values() if o.type == "window" and o.status != Status.REJECTED]
    assert len(wins) == 1 and wins[0].evidence.get("learned_label", 0) > 0.9
    real_walls = [w for w in st.building.walls.values() if w.status == Status.ACCEPTED]
    assert real_walls and all(w.evidence.get("learned_label", 1.0) > 0.5 for w in real_walls)
    assert len([r for r in st.building.rooms.values() if r.status != Status.REJECTED]) >= 3
