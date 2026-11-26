"""Combined approach: SymPointV2 labels + approach 1's detectors filling what the model misses.

Two kinds of tests: the real model's saved answers (``real_answers/``, GPU run) rebuilt without a GPU,
and small unit tests of the fusion steps."""
import json
import math
import os

import pytest

from architectural_converter.benchmark.generate import DEFAULT_DIR
from architectural_converter.benchmark.metrics import evaluate
from architectural_converter.models.base_entity import Status
from architectural_converter.models.opening import Opening
from sympoint_converter import CLASSES, load_config
from sympoint_converter.combined import CombinedPipeline, fill_unlabelled, merge_duplicate_openings
from sympoint_converter.pipeline import SymPointPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
ANSWERS = os.path.join(HERE, "real_answers")
ROOT = os.path.dirname(os.path.dirname(HERE))
BENCH = sorted(f[:-4] for f in os.listdir(DEFAULT_DIR) if f.endswith(".dxf"))


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SYMPOINT_STATE_DIR", str(tmp_path / "state"))


def _answer(name):
    with open(os.path.join(ANSWERS, f"{name}.sympoint.response.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _build(path, resp, combined=True, cls=CombinedPipeline):
    cfg = load_config({"cache_labels": False, "combined": {"enabled": combined}})
    return cls(cfg, response=resp, labels_progress=lambda m: None).process(path)


@pytest.mark.parametrize("name", BENCH)
def test_real_model_benchmark(name):
    """The real model's answers + approach 1: every benchmark drawing at most one correction away
    (measured 25 Nov 2026: 0 on 12 drawings, 1 on 09 with everything on layer 0)."""
    path = os.path.join(DEFAULT_DIR, f"{name}.dxf")
    with open(path.replace(".dxf", ".reference.json"), encoding="utf-8") as fh:
        ref = json.load(fh)
    st = _build(path, _answer(name))
    m = evaluate(st.model, ref)
    assert m["manual_corrections_required"] <= 1, m
    assert m["wall_recall"] >= 0.98 and m["room_recall"] >= 0.98, m
    assert st.logs["detect"]["approach"] == "combined"


def test_combined_fixes_what_the_model_alone_misses():
    """Single-line walls (the model leaves them unlabelled) and columns (the model has no column class)."""
    for name, key in (("11_single_line", "wall_recall"), ("10_blocks_hatches_overlaps", "column_recall")):
        path = os.path.join(DEFAULT_DIR, f"{name}.dxf")
        with open(path.replace(".dxf", ".reference.json"), encoding="utf-8") as fh:
            ref = json.load(fh)
        alone = evaluate(_build(path, _answer(name), combined=False).model, ref)
        both = evaluate(_build(path, _answer(name)).model, ref)
        assert both[key] >= 0.99 and alone[key] <= 0.5, (name, key, alone[key], both[key])


def test_model_only_is_approach_2_unchanged():
    path = os.path.join(DEFAULT_DIR, "03_commercial.dxf")
    a = _build(path, _answer("03_commercial"), combined=False)
    b = _build(path, _answer("03_commercial"), cls=SymPointPipeline)
    assert a.logs["detect"]["approach"] == "sympoint"
    assert json.dumps(a.model["walls"], sort_keys=True) == json.dumps(b.model["walls"], sort_keys=True)
    assert len(a.model["openings"]) == len(b.model["openings"])


def test_real_apartment_doors_and_windows():
    """Real drawing: doors and windows checked against the drawing's own door blocks and window layer."""
    import ezdxf
    from ezdxf import bbox
    from shapely.geometry import box
    from shapely.ops import unary_union
    path = os.path.join(ROOT, "samples", "real", "apartment-building-plan.dxf")
    st = _build(path, _answer("apartment-building-plan"))
    m = st.model
    tr = m["source"]["transform"]
    f = lambda x, y: ((x - tr["offset_drawing_units"][0]) * tr["scale_to_m"], (y - tr["offset_drawing_units"][1]) * tr["scale_to_m"])
    msp = ezdxf.readfile(path).modelspace()
    doors = [f((b.extmin.x + b.extmax.x) / 2, (b.extmin.y + b.extmax.y) / 2)
             for b in (bbox.extents([e]) for e in msp.query('INSERT[name=="q1"]'))]
    u = unary_union([box(b.extmin.x - 50, b.extmin.y - 50, b.extmax.x + 50, b.extmax.y + 50)
                     for b in (bbox.extents([e]) for e in msp.query('*[layer=="winows"]'))])
    wins = [f(g.centroid.x, g.centroid.y) for g in getattr(u, "geoms", [u])]

    def matched(det, refs, r):
        pairs = sorted((math.dist(a, b), i, j) for i, a in enumerate(det) for j, b in enumerate(refs) if math.dist(a, b) <= r)
        ud, ur = set(), set()
        for _, i, j in pairs:
            if i not in ud and j not in ur:
                ud.add(i), ur.add(j)
        return len(ud)
    live = [o for o in m["openings"] if o["status"] != "REJECTED"]
    D = [tuple(o["position"]["center"]) for o in live if o["type"] == "door"]
    W = [tuple(o["position"]["center"]) for o in live if o["type"] == "window"]
    kd, kw = matched(D, doors, 1.0), matched(W, wins, 1.5)
    assert kd / len(doors) >= 0.97 and kd / len(D) >= 0.95, (kd, len(D), len(doors))
    assert kw / len(wins) >= 0.95 and kw / len(W) >= 0.95, (kw, len(W), len(wins))      # model alone: 30 % of windows


def test_fill_only_where_the_model_is_silent():
    cfg = load_config()
    req = {"tiles": [{"id": "t", "ids": ["a", "b", "c"]}]}
    resp = {"tiles": [{"id": "t", "semantic": [[CLASSES.index("bed"), 0.9, 1.0],          # confident: kept
                                               [CLASSES.index("wall"), 0.05, 0.02],        # silent: filled
                                               [CLASSES.index("wall"), 0.05, 0.02]]}]}     # silent, no vote: kept
    out, n = fill_unlabelled(req, resp, {"a": "wall", "b": "window"}, cfg)
    sem = out["tiles"][0]["semantic"]
    assert n == 1
    assert sem[0][0] == CLASSES.index("bed") and sem[1][0] == CLASSES.index("window") and sem[2][1] == 0.05
    assert resp["tiles"][0]["semantic"][1][0] == CLASSES.index("wall")                  # the model's answer is not changed


def test_duplicate_openings_keep_approach_1_backed_one():
    def door(i, x, conf, methods):
        o = Opening(id=i, type="door", p0=(x - 0.45, 0.0), p1=(x + 0.45, 0.0), width=0.9)
        o.confidence, o.status, o.detection_methods = conf, Status.ACCEPTED, set(methods)
        return o
    a = door("door_0001", 1.0, 0.95, {"sympoint_door"})
    b = door("door_0002", 1.15, 0.80, {"approach1_geometry", "door_detector"})
    c = door("door_0003", 3.0, 0.90, {"sympoint_door"})
    assert merge_duplicate_openings([a, b, c], 0.4) == 1
    assert a.status == Status.REJECTED and "duplicate of door_0002" in a.status_reason
    assert b.status == Status.ACCEPTED and c.status == Status.ACCEPTED


def test_approach_1_failure_does_not_stop_the_run(monkeypatch):
    import sympoint_converter.combined as comb

    def boom(*a, **k):
        raise RuntimeError("detector exploded")
    monkeypatch.setattr(comb, "approach1_elements", boom)
    path = os.path.join(DEFAULT_DIR, "01_residential_apartment.dxf")
    st = _build(path, _answer("01_residential_apartment"))
    assert "detector exploded" in st.logs["detect"]["combined"]["error"]
    assert any("approach 1 skipped" in i["message"] for i in st.drawing.issues)
    assert len([w for w in st.model["walls"] if w["status"] != "REJECTED"]) > 5
