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


def test_every_stacked_storey_has_a_base_and_a_roof():
    """Stacking: each floor-plan level gets a base slab and a roof slab over its closed outline;
    side by side there are none; a storey named attic keeps its top open."""
    import numpy as np
    doc, _ = build_permit_set()
    st = _run(doc, "permit_slabs.dxf")
    plans = sorted((lv for lv in st.building.levels if lv.kind == "floor_plan"), key=lambda lv: lv.z)
    assert all(lv.outline and lv.outline_source == "plan" for lv in plans)
    assert all(not p.interiors for lv in plans for p in lv.outline)
    assert all(lv["outline"] for lv in st.model["building"]["levels"] if lv["kind"] == "floor_plan")
    m3 = build_3d(st.model, PipelineConfig())
    by = {m["id"]: m for m in m3.meshes}
    slab = 0.15
    for k, lv in enumerate(plans):
        base, roof = by[f"{lv.id}_base"], by[f"{lv.id}_roof"]
        assert base["type"] == "floor" and roof["type"] == "roof" and base["level"] == roof["level"] == lv.id
        zb, zr = base["vertices"][:, 2], roof["vertices"][:, 2]
        assert lv.z - slab < zb.min() and zb.max() < lv.z                      # inside the room floors
        if k < len(plans) - 1:                                                 # = the floor zone of the storey above
            assert plans[k + 1].z - slab < zr.min() and zr.max() < plans[k + 1].z
        else:                                                                  # on top of the highest walls
            assert abs(zr.min() - (lv.z + lv.height)) < 1e-6 and abs(zr.max() - (lv.z + lv.height + slab)) < 1e-6
        # the slab covers the whole storey: every wall of the level stands on it
        xy = np.vstack([m["vertices"][:, :2] for m in m3.meshes if m.get("level") == lv.id and m["type"] == "wall"])
        assert base["vertices"][:, 0].min() <= xy[:, 0].min() + 0.06 and base["vertices"][:, 0].max() >= xy[:, 0].max() - 0.06
        assert base["vertices"][:, 1].min() <= xy[:, 1].min() + 0.06 and base["vertices"][:, 1].max() >= xy[:, 1].max() - 0.06
    assert m3.metadata["counts"]["roof"] == len(plans)
    assert [s["base"] and s["roof"] is True for s in m3.metadata["level_slabs"]] == [True] * len(plans)
    assert "level" in m3.to_json()["meshes"][0]
    flat = build_3d(st.model, PipelineConfig({"phase2": {"stack_levels": False}}))
    assert not [m for m in flat.meshes if m["type"] == "roof" or m["id"].endswith("_base")]
    # an attic keeps its top open (its roof is not a flat slab)
    model = __import__("copy").deepcopy(st.model)
    top = max((lv for lv in model["building"]["levels"] if lv["kind"] == "floor_plan"), key=lambda lv: lv["z"])
    top["name"] = "Attic Floor Plan"
    m4 = build_3d(model, PipelineConfig())
    ids = {m["id"] for m in m4.meshes}
    assert f"{top['id']}_base" in ids and f"{top['id']}_roof" not in ids
    assert m4.metadata["counts"]["roof"] == len(plans) - 1


def test_outline_connects_the_wall_vertices_when_the_plan_is_not_closed():
    from shapely.geometry import box
    from architectural_converter.topology.level_slabs import level_outline
    # a closed plan: its own shape, voids filled
    ring = box(0, 0, 10, 8).difference(box(0.2, 0.2, 9.8, 7.8))
    out, src = level_outline([ring], [(0, 0), (10, 0), (10, 8), (0, 8)])
    assert src == "plan" and len(out) == 1 and abs(out[0].area - 80) < 0.5 and not out[0].interiors
    # three separate walls around a 10 x 8 m space, the gaps between them too wide to close:
    # no shape of its own, so the outer wall vertices are connected
    walls = [box(0, 0, 4, 0.2), box(9.8, 2, 10, 8), box(0, 7.8, 3, 8)]
    pts = [(0, 0.1), (4, 0.1), (9.9, 2), (9.9, 8), (0, 7.9), (3, 7.9)]
    out, src = level_outline(walls, pts)
    assert src == "wall_vertices" and len(out) == 1 and out[0].area > 55 and not out[0].interiors
    assert all(out[0].buffer(0.2).contains(w) for w in walls)
    assert level_outline([], []) == ([], None)


def _areas(mesh):
    import numpy as np
    V, F = mesh
    if not len(F):
        return 0.0, 0.0
    t = V[F]
    a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum()), float(0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum())


def test_hip_roof_surface():
    import numpy as np
    from architectural_converter.reconstruction3d.roof import hip_roof, roof_height_at
    ring = [(0, 0), (10, 0), (10, 6), (0, 6)]
    r = hip_roof(ring, 45.0)
    (s0, p0), (s1, p1) = _areas(r["courses"][0]), _areas(r["courses"][1])
    V = np.vstack([r["courses"][0][0], r["courses"][1][0]])
    assert abs(r["rise"] - 3.0) < 0.01 and abs(V[:, 2].max() - 3.0) < 0.01 and V[:, 2].min() == 0.0
    assert abs(p0 + p1 - 60.0) < 0.01                                       # covers the outline, no overlap
    assert abs(s0 + s1 - 60 / np.cos(np.radians(45))) < 0.05                # four planes at 45 degrees
    assert p0 > 20 and p1 > 20 and not len(r["deck"][1])                    # rows of tiles alternate
    # the ridge runs along the middle of the long sides; the short sides carry the triangles (hips)
    top = V[V[:, 2] > 2.99]
    assert top[:, 0].min() < 3.2 and top[:, 0].max() > 6.8 and np.allclose(top[:, 1], 3.0, atol=0.02)
    assert abs(roof_height_at(5, 1, [ring], 45.0, None, 0.0) - 1.0) < 1e-6
    assert abs(roof_height_at(0.5, 3, [ring], 45.0, None, 0.5) - 1.0) < 1e-6 and roof_height_at(11, 3, [ring], 45.0, None, 0) is None
    # tiles: every course stands proud at its lower edge (a sawtooth over the plane)
    t = hip_roof(ring, 45.0, course_m=0.5, thickness_m=0.05)
    Vt = np.vstack([t["courses"][0][0], t["courses"][1][0]])
    eave = Vt[np.isclose(Vt[:, 1], 0.0) & (Vt[:, 0] > 1) & (Vt[:, 0] < 9)]
    assert len(eave) and np.allclose(eave[:, 2], 0.05)
    # a height limit gives a flat deck; a hole (the untiled middle of the roof plan) stays open
    h = hip_roof({"exterior": [(0, 0), (20, 0), (20, 17), (0, 17)], "holes": [[(8, 3), (13.5, 3), (13.5, 8.5), (8, 8.5)]]},
                 30.0, max_rise=2.5, eave_z=0.5)
    tiles = _areas(h["courses"][0])[1] + _areas(h["courses"][1])[1]
    deck = _areas(h["deck"])[1]
    assert abs(tiles + deck - (340 - 30.25)) < 0.1 and deck > 20 and abs(h["rise"] - 2.5) < 1e-9
    assert abs(h["deck"][0][:, 2].max() - 3.0) < 1e-9
    fp = {"exterior": [(0, 0), (20, 0), (20, 17), (0, 17)], "holes": [[(8, 3), (13.5, 3), (13.5, 8.5), (8, 8.5)]]}
    assert roof_height_at(10, 5, [fp], 30.0, 2.5, 0.5) is None and roof_height_at(10, 1, [fp], 30.0, 2.5, 0.5) > 0.5
    # an L-shaped outline has a valley and splits towards two ridges: still one covering surface
    ell = hip_roof([(0, 0), (12, 0), (12, 5), (5, 5), (5, 12), (0, 12)], 30.0)
    assert abs(_areas(ell["courses"][0])[1] + _areas(ell["courses"][1])[1] - 95.0) < 0.05
    assert abs(ell["rise"] - 2.5 * np.tan(np.radians(30))) < 0.02
    assert not len(hip_roof([(0, 0), (1, 0)], 30.0)["courses"][0][1])


def test_attic_storey_gets_a_pitched_roof_and_its_walls_are_trimmed():
    import copy
    import numpy as np
    doc, _ = build_permit_set()
    st = _run(doc, "permit_attic.dxf")
    model = copy.deepcopy(st.model)
    top = max((lv for lv in model["building"]["levels"] if lv["kind"] == "floor_plan"), key=lambda lv: lv["z"])
    top["name"] = "Attic"
    top["roof"] = {"kind": "pitched", "source": "storey outline", "footprint": [{"exterior": r, "holes": []} for r in top["outline"]]}
    m3 = build_3d(model, PipelineConfig())
    roofs = [m for m in m3.meshes if m["classification"] == "pitched_roof"]
    assert {m["id"] for m in roofs} == {f"{top['id']}_roof", f"{top['id']}_roof_b"} and roofs[0]["color"] != roofs[1]["color"]
    assert all(m["type"] == "roof" and m["level"] == top["id"] for m in roofs)
    info = m3.metadata["pitched_roofs"][0]
    z = np.concatenate([m["vertices"][:, 2] for m in m3.meshes if m["id"].startswith(f"{top['id']}_roof")]) - top["z"]
    assert info["ridge_m"] <= top["height"] + 1e-6 and info["ridge_m"] > 1.5 and z.max() <= info["ridge_m"] + 0.06
    assert abs(z.min() - info["eave_m"]) < 0.06 and info["eave_m"] == 1.0          # starts on the knee wall, no overhang
    # the roof is not wider than the storey
    xy = np.vstack([m["vertices"][:, :2] for m in roofs])
    base = next(m for m in m3.meshes if m["id"] == f"{top['id']}_base")["vertices"][:, :2]
    assert (xy.min(0) >= base.min(0) - 1e-6).all() and (xy.max(0) <= base.max(0) + 1e-6).all()
    # every wall of the attic ends under the roof; the storeys below keep their height
    walls = [m for m in m3.meshes if m["type"] == "wall" and m.get("level") == top["id"]]
    assert walls and max(m["vertices"][:, 2].max() for m in walls) - top["z"] <= info["ridge_m"] + 1e-6
    outer = min(walls, key=lambda m: m["vertices"][:, 1].min())                # a facade wall: knee-wall high
    assert outer["vertices"][:, 2].max() - top["z"] < 1.6
    below = [m for m in m3.meshes if m["type"] == "wall" and m.get("level") not in (None, top["id"])]
    flat = build_3d(st.model, PipelineConfig())
    assert max(m["vertices"][:, 2].max() for m in below) == max(
        m["vertices"][:, 2].max() for m in flat.meshes if m["type"] == "wall" and m.get("level") not in (None, top["id"]))
    off = build_3d(model, PipelineConfig({"phase2": {"pitched_roof": {"enabled": False}}}))
    assert not [m for m in off.meshes if m["classification"] == "pitched_roof"]


def test_roof_outline_is_read_from_the_roof_plan_hatches():
    from types import SimpleNamespace
    from shapely.geometry import Polygon, box
    from architectural_converter.topology.roof_plan import roof_areas, roof_for_level
    H = lambda poly, pat, layer="L1": SimpleNamespace(polygon=poly, pattern=pat, layer=layer)  # noqa: E731
    # the roof plan sheet, 100 m to the right of the floor plan: four faces around an untiled middle
    faces = [Polygon([(100, 0), (120, 0), (114, 4), (106, 4)]), Polygon([(120, 0), (120, 16), (114, 12), (114, 4)]),
             Polygon([(120, 16), (100, 16), (106, 12), (114, 12)]), Polygon([(100, 16), (100, 0), (106, 4), (106, 12)])]
    regions = [H(f, "AR-RSHKE") for f in faces] + [H(box(100, 30, 120, 31), "AR-RSHKE"),      # a roof in an elevation
                                                    H(box(0, 0, 20, 16), "ANSI31")]             # another hatch
    areas = roof_areas(regions, ["AR-RSHKE", "ROOF"])
    assert len(areas) == 1 and abs(areas[0].area - (320 - 64)) < 0.5 and len(areas[0].interiors) == 1
    # linework that repeats on both sheets (plot lines) brings the roof sheet onto the floor plan
    def stroke(a, b):
        from shapely.geometry import LineString
        return SimpleNamespace(p0=a, p1=b, line=lambda a=a, b=b: LineString([a, b]))
    plot = [((-5, -5), (25, -5)), ((25, -5), (25, 21)), ((25, 21), (-5, 21)), ((-5, 21), (-5, -5)),
            ((0, 0), (20, 0)), ((20, 0), (20, 16)), ((20, 16), (0, 16)), ((0, 16), (0, 0)), ((3, -5), (3, -2)), ((9, 21), (9, 18))]
    strokes = [stroke(a, b) for a, b in plot] + [stroke((a[0] + 100, a[1]), (b[0] + 100, b[1])) for a, b in plot]
    tol = SimpleNamespace(snap=0.01)
    target = box(0, 0, 20, 16)
    parts, info = roof_for_level(areas, strokes, (0, 0, 20, 16), tol, target, below=box(1, 0, 20, 16))
    assert info["matched"] and abs(info["translation"][0] + 100) < 0.01 and abs(info["translation"][1]) < 0.01
    assert len(parts) == 1 and len(parts[0].interiors) == 1                 # the open middle is kept
    assert parts[0].bounds == (1.0, 0.0, 20.0, 16.0)                         # cut to the storey below
    assert roof_for_level(areas, strokes, (0, 0, 20, 16), tol, box(200, 200, 220, 216))[0] is None
