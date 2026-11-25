"""Manual editing tools: cut / move walls, add / move / close openings, add / cut floor."""
import numpy as np

from architectural_converter.models.base_entity import Status
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.pipeline.pipeline_config import PipelineConfig
from architectural_converter.reconstruction3d import build_3d
from tests.helpers import box_room, new_doc, save


def _plan():
    """10 x 6 m house, a partition at x = 5 with a 0.9 m door gap, two named rooms."""
    doc, msp = new_doc()
    box_room(msp, 10, 6, 0.2)
    for (a, b) in (((5, 0.2), (5, 2.5)), ((5.12, 0.2), (5.12, 2.5)), ((5, 2.5), (5.12, 2.5)),
                   ((5, 3.4), (5, 5.8)), ((5.12, 3.4), (5.12, 5.8)), ((5, 3.4), (5.12, 3.4))):
        msp.add_line(a, b, dxfattribs={"layer": "A-WALL"})
    msp.add_text("BEDROOM", dxfattribs={"insert": (2, 3), "height": 0.2})
    msp.add_text("KITCHEN", dxfattribs={"insert": (7, 3), "height": 0.2})
    return save(doc)


def _run(path, corr=None):
    return ExtractionPipeline(PipelineConfig()).process(path, corrections=corr)


def live(xs):
    return [x for x in xs if x.status != Status.REJECTED]


def _wall(st, pred):
    return next(w for w in live(st.building.walls.values()) if pred(w))


def test_cut_a_part_out_of_a_wall_and_add_openings():
    path = _plan()
    st = _run(path)
    assert len(live(st.building.rooms.values())) == 2
    bottom = _wall(st, lambda w: abs(w.u[0]) > 0.99 and w.axis.centroid.y < st.building.footprint.centroid.y - 2)
    (x0, y), (x1, _) = sorted(bottom.axis.coords)
    n_walls = len(live(st.building.walls.values()))
    corr = {"wall_cuts": [{"wall": bottom.id, "start": [x0 + 1.0, y], "end": [x0 + 4.0, y]}],
            "added_openings": [{"type": "window", "start": [x0 + 6.0, y], "end": [x0 + 7.5, y]},
                               {"type": "door", "start": [x0 + 8.0, y + 0.3], "end": [x0 + 8.9, y + 0.3], "wall": bottom.id}]}
    st2 = _run(path, corr)
    log = st2.logs["reconstruct"]["manual_edits"]
    assert log["wall_cuts"] == 1 and log["wall_pieces_added"] == 1 and log["openings_added"] == 2 and not log["skipped"]
    a, b = st2.building.walls[bottom.id], st2.building.walls[bottom.id + "_2"]
    assert abs(a.length - 1.0) < 0.01 and abs(b.length - (bottom.length - 4.0)) < 0.01     # split in two
    assert len(live(st2.building.walls.values())) == n_walls + 1
    # 3 m of the facade are gone: the bedroom is open, only the kitchen remains a room
    assert [r.classification for r in live(st2.building.rooms.values())] == ["kitchen"]
    win, door = st2.building.openings["manual_window_0001"], st2.building.openings["manual_door_0002"]
    assert win.host_wall == b.id and abs(win.width - 1.5) < 1e-6 and win.status == Status.ACCEPTED
    assert door.host_wall == b.id and abs(door.width - 0.9) < 1e-6 and abs(door.p0[1] - y) < 1e-6   # snapped into the wall
    m3 = build_3d(st2.model, PipelineConfig())
    assert any(m["id"] == win.id and m["type"] == "window" for m in m3.meshes)
    assert st2.model["corrections"]["wall_cuts"] and st2.model["corrections"]["added_floors"] == []


def test_move_a_wall_end_and_close_an_opening():
    path = _plan()
    st = _run(path)
    part = _wall(st, lambda w: abs(w.u[1]) > 0.99 and abs(w.axis.centroid.x - st.building.footprint.centroid.x) < 1)
    gap = next(o for o in live(st.building.openings.values()) if o.host_wall == part.id)
    assert abs(gap.width - 0.9) < 0.05
    # close the door gap: the opening is rejected and the wall is solid there
    st2 = _run(path, {"status_overrides": {gap.id: "REJECTED"}})
    assert st2.building.openings[gap.id].status == Status.REJECTED
    assert len(live(st2.building.rooms.values())) == 2
    m3 = build_3d(st2.model, PipelineConfig())
    wall = next(m for m in m3.meshes if m["id"] == part.id)
    assert len(wall["vertices"]) == 8                                    # one solid box, no void
    # move the upper end of the partition 2 m down and make it 0.3 m thick: the rooms merge
    (xa, ya), (xb, yb) = sorted(part.axis.coords, key=lambda p: p[1])
    st3 = _run(path, {"wall_edits": {part.id: {"start": [xa, ya], "end": [xb, yb - 2.5], "thickness": 0.3}},
                      "opening_edits": {gap.id: {"start": [xa, gap.p0[1]], "end": [xa, gap.p0[1] + 0.5]}}})
    w = st3.building.walls[part.id]
    assert abs(w.length - (part.length - 2.5)) < 0.01 and w.thickness == 0.3 and w.status == Status.ACCEPTED
    assert st3.logs["reconstruct"]["manual_edits"]["walls_edited"] == 1
    g = st3.building.openings[gap.id]
    assert abs(g.width - 0.5) < 0.01 and g.host_wall == part.id
    assert len(live(st3.building.rooms.values())) == 1


def test_add_and_cut_floor():
    path = _plan()
    st = _run(path)
    fp = st.building.footprint
    x0, y0, x1, y1 = fp.bounds
    room = next(r for r in live(st.building.rooms.values()) if r.classification == "kitchen")
    cx, cy = room.polygon.centroid.coords[0]
    corr = {"added_floors": [{"polygon": [[x1 + 1, y0], [x1 + 4, y0], [x1 + 4, y0 + 3], [x1 + 1, y0 + 3]]}],
            "floor_cuts": [{"polygon": [[cx - 0.5, cy - 0.5], [cx + 0.5, cy - 0.5], [cx + 0.5, cy + 0.5], [cx - 0.5, cy + 0.5]]}]}
    st2 = _run(path, corr)
    assert any(abs(p.area - 9.0) < 0.01 for p, _ in st2.ground)                  # the added terrace floor
    k = next(r for r in st2.model["rooms"] if r["id"] == room.id)
    assert k["floor_parts"] and len(k["floor_parts"]) >= 2                       # the hole splits the floor in parts
    m3 = build_3d(st2.model, PipelineConfig())
    floor = next(m for m in m3.meshes if m["id"] == room.id and m["type"] == "floor")
    V, F = floor["vertices"], floor["faces"]
    top = [f for f in F if np.allclose(V[f][:, 2], 0.0)]
    area = sum(0.5 * abs(np.cross(V[f][1, :2] - V[f][0, :2], V[f][2, :2] - V[f][0, :2])) for f in top)
    assert abs(area - (room.area - 1.0)) < 0.05                                  # 1 m² hole in the floor
    assert any(m["classification"] == "ground" and abs(np.ptp(m["vertices"][:, 0]) - 3.0) < 0.01 for m in m3.meshes)
    st0 = _run(path)
    assert next(r for r in st0.model["rooms"] if r["id"] == room.id)["floor_parts"] is None


def test_added_and_cut_pieces_can_be_deleted_again():
    path = _plan()
    st = _run(path)
    bottom = _wall(st, lambda w: abs(w.u[0]) > 0.99 and w.axis.centroid.y < st.building.footprint.centroid.y - 2)
    (x0, y), _ = sorted(bottom.axis.coords)
    corr = {"wall_cuts": [{"wall": bottom.id, "start": [x0 + 1.0, y], "end": [x0 + 2.0, y]}],
            "added_openings": [{"type": "door", "start": [x0 + 6.0, y], "end": [x0 + 6.9, y]}],
            "status_overrides": {bottom.id + "_2": "REJECTED", "manual_door_0001": "REJECTED"}}
    st2 = _run(path, corr)
    assert st2.building.walls[bottom.id + "_2"].status == Status.REJECTED
    assert st2.building.openings["manual_door_0001"].status == Status.REJECTED
    # corrections written by an older version (without the new keys) still load
    st3 = _run(path, {"version": 1, "status_overrides": {}})
    assert st3.model["corrections"]["wall_cuts"] == [] and len(live(st3.building.rooms.values())) == 2
