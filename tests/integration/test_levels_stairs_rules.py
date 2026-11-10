"""Multi-storey stacking, stair flights and plausibility rules."""
import os
import tempfile

from architectural_converter.benchmark.dxf_writer import Style, render_plan
from architectural_converter.benchmark.permit_set import build_permit_set
from architectural_converter.benchmark.plan_spec import PlanSpec
from architectural_converter.models.base_entity import Status
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.pipeline.pipeline_config import PipelineConfig
from architectural_converter.reconstruction3d import build_3d


def _run(doc, name):
    p = os.path.join(tempfile.mkdtemp(prefix="acv_"), name)
    doc.saveas(p)
    return ExtractionPipeline(PipelineConfig()).process(p)


def test_permit_set_levels_and_stairs():
    doc, ref = build_permit_set()
    st = _run(doc, "permit.dxf")
    b = st.building
    plans = [lv for lv in b.levels if lv.kind == "floor_plan"]
    assert [lv.elevation for lv in plans] == [705.40, 708.75, 712.10, 715.45]
    assert all(abs(lv.height - 3.35) < 1e-6 for lv in plans)
    assert {lv.kind for lv in b.levels} >= {"duplicate", "elevation_or_section"}
    ref_off = {r["elevation"]: r["sheet_offset"] for r in ref["levels"]}
    base = next(lv for lv in plans if lv.translation == [0.0, 0.0] or tuple(lv.translation) == (0.0, 0.0))
    for lv in plans:      # aligned onto the reference sheet
        exp = ref_off[base.elevation][0] - ref_off[lv.elevation][0]
        assert abs(lv.translation[0] - exp) < 0.05 and abs(lv.translation[1]) < 0.05
    # one accepted flight per floor plan; their treads never become walls
    stairs = [s for s in b.stairs.values() if s.status == Status.ACCEPTED]
    assert sorted(s.level for s in stairs) == sorted(lv.id for lv in plans)
    assert all(s.treads == 12 and abs(s.going - 0.3) < 0.01 for s in stairs)
    tread_ids = set().union(*(s.source_entities for s in stairs))
    assert not [w for w in b.walls.values() if w.status != Status.REJECTED and w.source_entities
                and w.source_entities <= tread_ids]
    # elements of the elevation / duplicate sheets are not modelled
    for w in b.walls.values():
        lv = next((x for x in b.levels if x.id == w.level), None)
        if lv is not None and lv.kind != "floor_plan":
            assert w.status == Status.REJECTED
    m3 = build_3d(st.model, PipelineConfig())
    lo, hi = m3.metadata["bounds"]
    assert hi[0] - lo[0] < 30 and hi[2] > 10.05 + 2.5      # stacked, not side by side
    assert m3.metadata["counts"]["stair"] == 4


def test_no_stairs_in_plain_plans():
    from architectural_converter.benchmark.plans import PLANS
    spec, style = PLANS[0]() if callable(PLANS[0]) else PLANS[0]
    doc, _ = render_plan(spec, style)
    st = _run(doc, "p.dxf")
    assert not st.building.stairs and not st.building.levels


def test_window_in_closet_partition_becomes_sliding_door():
    s = PlanSpec("closet", "bedroom with a wardrobe front drawn like glazing")
    s.chain([(0, 0), (6, 0), (6, 4), (0, 4)], 0.3, closed=True)
    s.wall((0, 1.6), (6, 1.6), 0.12)
    s.wall((4.6, 1.6), (4.6, 4), 0.12)
    s.opening_at((3, 1.6), 0.9, "door")
    s.opening_at((4.6, 2.8), 1.6, "window")
    s.opening_at((2, 4), 1.5, "window")
    s.opening_at((6, 0.8), 0.9, "door")
    s.label("BEDROOM", (2, 2.8))
    doc, _ = render_plan(s, Style(layers="poor", window_repr="loose"))
    st = _run(doc, "closet.dxf")
    ops = [o for o in st.building.openings.values() if o.status != Status.REJECTED]
    sliding = [o for o in ops if o.subtype == "sliding_or_closet"]
    assert len(sliding) == 1 and sliding[0].type == "door" and sliding[0].status == Status.NEEDS_REVIEW
    windows = [o for o in ops if o.type == "window"]
    assert len(windows) == 1 and windows[0].classification == "exterior_window"


def _bedroom_door_as_window_block():
    s = PlanSpec("access", "bedroom whose door is drawn with a window block")
    s.chain([(0, 0), (8, 0), (8, 5), (0, 5)], 0.3, closed=True)
    s.wall((4, 0), (4, 5), 0.12)
    s.opening_at((8, 2.5), 1.0, "door")
    s.opening_at((4, 2.5), 0.9, "window")
    s.opening_at((2, 5), 1.5, "window")
    s.opening_at((6, 5), 1.5, "window")
    s.label("BEDROOM", (2, 2.5))
    s.label("LIVING", (6, 2.5))
    doc, _ = render_plan(s, Style(layers="aia", window_repr="block"))
    p = os.path.join(tempfile.mkdtemp(prefix="acv_"), "access.dxf")
    doc.saveas(p)
    return p


def test_unreachable_room_gets_its_door():
    pipe = ExtractionPipeline(PipelineConfig())
    st = pipe.process(_bedroom_door_as_window_block())
    acc = st.logs["classify"]["access"]
    assert acc["unreachable_before"] == 1 and acc["unreachable_after"] == 0
    oid = acc["windows_to_doors"][0]["opening"]
    o = st.building.openings[oid]
    assert o.type == "door" and o.subtype == "access_inferred" and o.status == Status.NEEDS_REVIEW
    assert not [i for i in st.report.issues if i.code == "no_access"]
    windows = [x for x in st.building.openings.values() if x.type == "window" and x.status != Status.REJECTED]
    assert len(windows) == 2 and all(x.classification == "exterior_window" for x in windows)
    # a manual correction re-types the opening and locks it against the automatic rules
    corr = {"status_overrides": {oid: "ACCEPTED"}, "opening_type_overrides": {oid: "sliding_or_closet"}}
    st2 = pipe.rerun(st, corr)
    o2 = st2.building.openings[oid]
    assert o2.type == "door" and o2.subtype == "sliding_or_closet" and o2.status == Status.ACCEPTED
    assert o2.status_reason == "manual correction"


def test_floors_can_stay_side_by_side_in_3d():
    """stack_levels false: every floor plan stays where it is drawn, at ground level (used by
    approach 2 and the web app's 'stack floors' switch)."""
    from architectural_converter.reconstruction3d import build_3d
    import numpy as np
    doc, _ = build_permit_set()
    st = _run(doc, "permit_flat.dxf")
    stacked = build_3d(st.model, PipelineConfig())
    flat = build_3d(st.model, PipelineConfig({"phase2": {"stack_levels": False}}))
    assert stacked.metadata["stacked_in_3d"] and not flat.metadata["stacked_in_3d"]
    zmax = lambda m3: max(float(np.max(m["vertices"][:, 2])) for m in m3.meshes)  # noqa: E731
    assert zmax(stacked) > 9.0 and zmax(flat) < 3.5                    # 4 floors vs one storey high
    xs = lambda m3: np.ptp(np.vstack([m["vertices"] for m in m3.meshes])[:, 0])  # noqa: E731
    assert xs(flat) > xs(stacked) * 2                                   # the plans stay side by side
