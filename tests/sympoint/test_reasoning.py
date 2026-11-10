"""Approach 2 reasoning modules: the drawing's structure corrects the model's line-by-line answer.

The model's answer is simulated: an oracle that reads the layer names, then damaged the way the
real model fails (lines left unlabelled, symbols called wall, stray wall labels)."""
import random

import pytest
from shapely.geometry import Polygon, box

from architectural_converter.models.base_entity import Status
from architectural_converter.topology.ground import unassigned_floor
from sympoint_converter import CLASSES, load_config
from sympoint_converter.pipeline import SymPointPipeline
from sympoint_converter.reasoning.label_consensus import refine_labels
from sympoint_converter.sympoint_io import export_tiles, labels_from_response, load_drawing, primitives
from tests.helpers import box_room, new_doc, rect_lines, run, save

KIND = {"A-WALL": "wall", "A-DOOR": "single door", "A-GLAZ": "window", "A-FURN": "bed", "L12": "bed", "X1": "window"}


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SYMPOINT_STATE_DIR", str(tmp_path / "state"))


def answer(req, damage=None, seed=3, pick=None):
    """Oracle answer; ``damage(layer, rnd) -> class name | None | 'keep'`` changes single lines,
    ``pick(raw id, layer)`` likewise for chosen lines."""
    rnd = random.Random(seed)
    tiles = []
    for t in req["tiles"]:
        sem = []
        for k, lid in enumerate(t["sympoint"]["layerIds"]):
            ly = t["layer_names"][lid - 1]
            c = KIND.get(ly)
            if damage is not None:
                d = damage(ly, rnd)
                c = c if d == "keep" else d
            if pick is not None:
                d = pick(t["ids"][k], ly)
                c = c if d == "keep" else d
            sem.append([CLASSES.index(c), 0.9, 1.0] if c else [CLASSES.index("wall"), 0.05, 0.02])
        tiles.append({"id": t["id"], "semantic": sem, "instances": []})
    return {"model": "simulated", "tiles": tiles}


def build(path, damage=None, pick=None, **over):
    cfg = load_config(over)
    d = load_drawing(path)
    req = export_tiles(d, cfg)
    by_id = (lambda rid, ly: pick(d.raw_entities[rid], ly)) if pick is not None else None
    pipe = SymPointPipeline(cfg, response=answer(req, damage, pick=by_id), labels_progress=lambda m: None)
    return pipe.process(path)


def live(xs):
    return [x for x in xs if x.status != Status.REJECTED]


def _house():
    """Two rooms (double-line walls), a window drawn as 4 glazing lines on its own layer in a wall
    gap, a kitchen counter (wall-like) on a furniture layer, and annotation lines."""
    doc, msp = new_doc()
    for ly in ("A-WALL", "A-GLAZ", "A-FURN", "A-ANNO-DIMS", "L12"):
        doc.layers.add(ly)
    W = dict(dxfattribs={"layer": "A-WALL"})
    t = 0.2
    # outer walls with a 1.6 m window gap in the bottom wall (x 2.0-3.6)
    for (a, b) in (((0, 0), (2.0, 0)), ((3.6, 0), (10, 0)), ((10, 0), (10, 6)), ((10, 6), (0, 6)), ((0, 6), (0, 0)),
                   ((t, t), (2.0, t)), ((3.6, t), (10 - t, t)), ((10 - t, t), (10 - t, 6 - t)), ((10 - t, 6 - t), (t, 6 - t)),
                   ((t, 6 - t), (t, t)), ((2.0, 0), (2.0, t)), ((3.6, 0), (3.6, t))):
        msp.add_line(a, b, **W)
    # partition with a door gap (y 2.5-3.4)
    for (a, b) in (((5, t), (5, 2.5)), ((5.12, t), (5.12, 2.5)), ((5, 2.5), (5.12, 2.5)),
                   ((5, 3.4), (5, 6 - t)), ((5.12, 3.4), (5.12, 6 - t)), ((5, 3.4), (5.12, 3.4))):
        msp.add_line(a, b, **W)
    for y in (0.0, 0.07, 0.13, t):
        msp.add_line((2.0, y), (3.6, y), dxfattribs={"layer": "A-GLAZ"})
    # more glazing of the same kind elsewhere on the layer (so the layer has enough lines to vote)
    for i in range(8):
        msp.add_line((20 + i, 0), (20 + i, 1.2), dxfattribs={"layer": "A-GLAZ"})
    # a kitchen counter along the top wall: its front line runs 0.6 m from the wall face, like a
    # second wall face; plus appliances
    msp.add_line((5.12, 5.2), (10 - t, 5.2), dxfattribs={"layer": "A-FURN"})
    # a wardrobe run across the room, wall to wall: two parallel lines 0.6 m apart
    # (on a layer whose name says nothing; the same layer holds the other furniture of the plan)
    msp.add_line((5.12, 4.0), (10 - t, 4.0), dxfattribs={"layer": "L12"})
    msp.add_line((5.12, 4.6), (10 - t, 4.6), dxfattribs={"layer": "L12"})
    for i in range(30):
        msp.add_line((40 + 0.3 * i, 0), (40 + 0.3 * i, 2.5), dxfattribs={"layer": "L12"})
    for x0 in (6.0, 7.5, 8.6):
        rect_lines(msp, x0, 5.3, x0 + 0.5, 5.7, "A-FURN")
    for i in range(16):
        msp.add_line((0, -1 - 0.1 * i), (10, -1 - 0.1 * i), dxfattribs={"layer": "A-ANNO-DIMS"})
    msp.add_text("BEDROOM", dxfattribs={"insert": (2, 3), "height": 0.2})
    msp.add_text("KITCHEN", dxfattribs={"insert": (7, 3), "height": 0.2})
    return save(doc)


def test_layers_vote_on_unlabelled_and_stray_lines():
    path = _house()
    cfg = load_config()
    d = load_drawing(path)
    prims = primitives(d, cfg)
    req = export_tiles(d, cfg, prims)

    def damage(ly, rnd):
        if ly == "A-GLAZ":
            return "window" if rnd.random() < 0.4 else None          # most glazing lines unlabelled
        if ly == "A-FURN":
            return "wall" if rnd.random() < 0.3 else "bed"           # wardrobe sides called wall
        if ly == "A-ANNO-DIMS":
            return "wall" if rnd.random() < 0.1 else None            # stray wall labels on dimensions
        return "keep"
    labels = labels_from_response(req, answer(req, damage), cfg["min_label_score"], cfg["min_label_mass"])
    st = run(path, until="context")
    out, log = refine_labels(d, st.ctx.layers, prims, labels, cfg)
    by_layer = {}
    for rid, ly, *_ in prims:
        by_layer.setdefault(ly, set()).add(out["entities"].get(rid, {}).get("label"))
    assert by_layer["A-GLAZ"] == {"window"}                          # filled from the layer's own majority
    assert by_layer["A-FURN"] <= {"bed", None} and "bed" in by_layer["A-FURN"]   # no wall on a furniture layer
    assert by_layer["A-ANNO-DIMS"] == {None}                         # stray wall lines dropped
    assert by_layer["A-WALL"] == {"wall"}
    assert log["layers"]["consistent"]["A-GLAZ"][0] == "glazing" and log["changed"]
    assert any(e.get("origin") == "layer_vote" and "model_label" not in e for e in out["entities"].values())
    assert labels["entities"] != out["entities"] and all("origin" not in e for e in labels["entities"].values())
    # primitives follow the entities (extraction reads both)
    lab = {prims[k][0]: CLASSES[v[0]] for k, v in out["primitives"].items()}
    assert all(lab[rid] == e["label"] for rid, e in out["entities"].items())


def test_layer_name_labels_silent_lines_and_blocks_vote():
    """A layer the model says nothing about still has a name; the same block line gets the label
    most of its inserts got."""
    doc, msp = new_doc()
    doc.layers.add("A-WALL")
    doc.layers.add("Fenster")
    doc.layers.add("X1")
    box_room(msp, 8, 5, 0.2)
    for i in range(12):
        msp.add_line((1 + 0.5 * i, 0.07), (1.4 + 0.5 * i, 0.07), dxfattribs={"layer": "Fenster"})
    blk = doc.blocks.new("W1")
    for y in (0.0, 0.1, 0.2):
        blk.add_line((0, y), (1.2, y), dxfattribs={"layer": "X1"})
    for i in range(6):
        msp.add_blockref("W1", (12 + 2 * i, 0))
    path = save(doc)
    cfg = load_config()
    d = load_drawing(path)
    prims = primitives(d, cfg)
    req = export_tiles(d, cfg, prims)
    labels = labels_from_response(req, answer(req, lambda ly, r: (None if ly == "Fenster" else (
        "window" if r.random() < 0.7 else None) if ly == "X1" else "keep")), cfg["min_label_score"], cfg["min_label_mass"])
    out, log = refine_labels(d, run(path, until="context").ctx.layers, prims, labels, cfg)
    lab = {ly: set() for ly in ("Fenster", "X1")}
    for rid, ly, *_ in prims:
        if ly in lab:
            lab[ly].add(out["entities"].get(rid, {}).get("label"))
    assert lab["Fenster"] == {"window"} and log["names"]["layers"] == {"Fenster": "window"}
    assert lab["X1"] == {"window"} and log["blocks"]["lines_from_other_inserts"] > 0
    off, log2 = refine_labels(d, None, prims, labels, load_config({"use_reasoning": False}))
    assert off is labels and log2 == {"skipped": "disabled"}


def test_reasoning_recovers_the_window_and_drops_furniture_walls():
    path = _house()

    def damage(ly, rnd):
        if ly == "A-GLAZ":
            return "window" if rnd.random() < 0.35 else None
        return "keep"

    def pick(raw, ly):                                               # the wardrobe run called wall
        return "wall" if ly == "L12" and raw.raw_coords and raw.raw_coords[0][0] < 20 else "keep"
    on = build(path, damage, pick)
    off = build(path, damage, pick, use_reasoning=False)
    wins = lambda st: [o for o in live(st.building.openings.values()) if o.type == "window"]  # noqa: E731
    assert len(wins(on)) == 1 and abs(wins(on)[0].width - 1.6) < 0.1
    # without the votes the wardrobe is a wall that cuts the kitchen in two; its layer is furniture
    furn = lambda st: [w for w in live(st.building.walls.values())  # noqa: E731
                       if any(e["layer"] == "L12" for e in st.drawing.raw_refs(w.source_entities))]
    assert furn(off) and not furn(on)
    assert len(live(on.building.rooms.values())) == 2
    assert len(live(off.building.rooms.values())) == 3
    assert on.logs["detect"]["label_consensus"]["layers"]["wall_lines_following_symbol_layer"] == 2
    assert off.logs["detect"]["label_consensus"] == {"skipped": "disabled"}


def test_wall_groups_that_enclose_nothing_are_rejected():
    doc, msp = new_doc()
    doc.layers.add("A-WALL")
    box_room(msp, 6, 4, 0.2)
    msp.add_text("OFFICE", dxfattribs={"insert": (2, 2), "height": 0.2})
    # a detail drawn 30 m away: two parallel lines 0.2 m apart (looks like a wall, closes nothing)
    msp.add_line((30, 0), (33, 0), dxfattribs={"layer": "A-WALL"})
    msp.add_line((30, 0.2), (33, 0.2), dxfattribs={"layer": "A-WALL"})
    path = save(doc)
    on, off = build(path), build(path, use_reasoning=False)
    far = lambda st: [w for w in st.building.walls.values() if w.axis.centroid.x > 20]  # noqa: E731
    assert far(off) and all(w.status != Status.REJECTED for w in far(off))
    assert far(on) and all(w.status == Status.REJECTED and "not part of a building" in w.status_reason for w in far(on))
    assert on.logs["topology"]["wall_structure"]["rejected_walls"] == len(far(on))
    assert len(live(on.building.rooms.values())) == 1
    assert len([w for w in live(on.building.walls.values())]) == len([w for w in live(off.building.walls.values())]) - len(far(on))


def _two_names(texts):
    doc, msp = new_doc()
    box_room(msp, 10, 8, 0.2)
    # partition from the bottom wall that stops 4.5 m short of the top wall
    msp.add_line((5.0, 0.2), (5.0, 3.3))
    msp.add_line((5.15, 0.2), (5.15, 3.3))
    msp.add_line((5.0, 3.3), (5.15, 3.3))
    for x, t in zip((2, 8), texts):
        msp.add_text(t, dxfattribs={"insert": (x, 2), "height": 0.2})
    for e in msp:
        if e.dxftype() == "LINE":
            e.dxf.layer = "A-WALL"
    return save(doc)


def test_one_room_one_name_adds_the_missing_wall():
    path = _two_names(("BEDROOM", "BALCONY"))
    st = build(path)
    rooms = live(st.building.rooms.values())
    assert sorted(r.classification for r in rooms) == ["balcony", "bedroom"]
    added = [w for w in live(st.building.walls.values()) if "label_implied" in w.detection_methods]
    assert len(added) == 1 and added[0].status == Status.NEEDS_REVIEW and "one room has one name" in added[0].status_reason
    assert st.logs["topology"]["room_names"]["split_rooms"]
    # without the rule the room keeps both names; it is typed by the indoor one, never 'balcony'
    off = build(path, use_reasoning=False)
    rooms = live(off.building.rooms.values())
    assert len(rooms) == 1 and rooms[0].classification == "bedroom"
    assert any("several room names" in n for n in rooms[0].notes)
    assert not [w for w in off.building.walls.values() if "label_implied" in w.detection_methods]


def test_open_plan_names_share_a_room():
    st = build(_two_names(("LIVING", "DINING")))
    assert len(live(st.building.rooms.values())) == 1
    assert not [w for w in st.building.walls.values() if "label_implied" in w.detection_methods]
    assert st.building.hypotheses                         # still proposed, not added


def test_approach_1_keeps_hypotheses_as_proposals():
    st = run(_two_names(("BEDROOM", "BALCONY")))
    assert len([r for r in st.building.rooms.values() if r.status == Status.ACCEPTED]) == 1
    assert not [w for w in st.building.walls.values() if "label_implied" in w.detection_methods]
    assert [r.classification for r in live(st.building.rooms.values())] == ["bedroom"]


def test_floor_of_an_area_mostly_surrounded_by_walls():
    # a U of 1 m thick "building" around a 5 m wide, 6 m deep court open on one side
    u = Polygon([(0, 0), (7, 0), (7, 7), (6, 7), (6, 1), (1, 1), (1, 7), (0, 7)])
    assert unassigned_floor([u], [u], close_m=2.0) == []                        # mouth wider than 2 x 2 m
    got = unassigned_floor([u], [u], close_m=2.0, wide_close_m=(4.0, 6.0), min_enclosure=0.6)
    assert len(got) == 1 and abs(got[0].area - 30.0) < 1.5                      # walls on 3 of 4 sides
    # the inner corner of an L is bounded on two sides only: outside
    ell = box(0, 0, 10, 1).union(box(0, 0, 1, 10))
    assert unassigned_floor([ell], [ell], close_m=2.0, wide_close_m=(4.0, 6.0), min_enclosure=0.6) == []
