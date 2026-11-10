"""Phase 1 entry point.

    pipeline = ExtractionPipeline(PipelineConfig())
    result = pipeline.process("building.dwg")          # or .dxf
    model = result.model                               # architectural model (dict)
    result = pipeline.rerun(result, corrections)       # after manual review

Stages (each can be run on its own for debugging with ``run_stage``):
 ingest -> preprocess -> context -> detect -> resolve -> reconstruct -> topology ->
 classify -> validate -> export
"""
from __future__ import annotations

from ..geometry.geometric_utils import robust_intersection

import copy
import time
from dataclasses import dataclass, field
from typing import Any

from .. import __version__
from ..classification.candidate_resolver import CandidateResolver
from ..classification.room_classifier import classify_rooms
from ..classification.wall_classifier import classify_columns, classify_openings, classify_walls
from ..detectors import DETECTORS, run_order
from ..detectors.context import DrawingContext, build_context
from ..geometry.geometry_adapter import build_segments
from ..ingestion import ingest, ingest_document
from ..models.base_entity import IdFactory, Status
from ..models.building import Building
from ..models.drawing import Drawing
from ..preprocessing.coordinate_normalizer import normalize
from ..preprocessing.duplicate_removal import mark_duplicates
from ..preprocessing.geometry_cleaner import flag_degenerate
from ..preprocessing.layer_analyzer import layer_parallel_shares
from ..preprocessing.tolerance import estimate_tolerances
from ..preprocessing.unit_detection import detect_units
from ..reconstruction.wall_reconstruction import reconstruct
from ..topology.graph_builder import build_axis_graph
from ..topology.missing_room_recovery import recover_missing_rooms
from ..topology.room_adjacency import relate_rooms
from ..topology.room_detector import RoomDetector
from ..topology.wall_connectivity import topology_evidence
from ..validation.geometry_validation import validate_geometry
from ..validation.opening_validation import validate_openings
from ..validation.room_validation import validate_rooms
from ..validation.validation_report import ValidationReport
from ..validation.wall_validation import validate_walls
from .corrections import (OPENING_TYPES, apply_status_overrides, manual_rooms, manual_walls, normalize_corrections)
from .pipeline_config import PipelineConfig

def _share_geometry(x, memo):
    return x


try:  # shapely 2 geometries are immutable: deep copies (re-run cache) can share them
    import shapely.geometry as _sg
    for _cls in (_sg.Point, _sg.LineString, _sg.LinearRing, _sg.Polygon, _sg.MultiPoint, _sg.MultiLineString,
                 _sg.MultiPolygon, _sg.GeometryCollection):
        copy._deepcopy_dispatch[_cls] = _share_geometry
except Exception:  # pragma: no cover
    pass

STAGES = ["ingest", "preprocess", "context", "detect", "resolve", "reconstruct", "topology", "classify",
          "validate", "export"]


@dataclass
class PipelineState:
    input_file: str
    drawing: Drawing | None = None
    segments: list = field(default_factory=list)
    dup_stats: dict = field(default_factory=dict)
    ctx: DrawingContext | None = None
    detected: dict | None = None          # pristine detector output (for re-runs)
    ids_after_detect: IdFactory | None = None
    corrections: dict = field(default_factory=dict)
    walls: list = field(default_factory=list)
    doors: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    stairs: list = field(default_factory=list)
    building: Building | None = None
    solid: Any = None
    report: ValidationReport | None = None
    model: dict | None = None
    timings: dict = field(default_factory=dict)
    logs: dict = field(default_factory=dict)
    completed: list = field(default_factory=list)


class ExtractionPipeline:
    STAGE_LABELS = {"ingest": "Reading DXF entities and blocks", "preprocess": "Units, tolerances, cleaning",
                    "context": "Layer semantics, block symbols, texts", "detect": "Detecting walls, doors, windows, columns",
                    "resolve": "Resolving candidates", "reconstruct": "Reconstructing walls and openings",
                    "topology": "Topology and rooms", "classify": "Classifying", "validate": "Validating",
                    "export": "Building the model"}

    def __init__(self, config: PipelineConfig | dict | None = None, progress=None):
        if config is None:
            config = PipelineConfig()
        elif isinstance(config, dict):
            config = PipelineConfig(config)
        self.cfg = config
        self.progress = progress  # optional callable(message: str) for live progress reporting

    def _report(self, msg: str) -> None:
        if self.progress is not None:
            try:
                self.progress(msg)
            except Exception:
                pass

    # ------------------------------------------------------------------ public API
    def process(self, input_file: str, corrections: dict | None = None, until: str = "export",
                doc=None) -> PipelineState:
        st = PipelineState(input_file=input_file)
        st.corrections = normalize_corrections(corrections)
        st._doc = doc
        for stage in STAGES:
            self.run_stage(st, stage)
            if stage == until:
                break
        return st

    def rerun(self, st: PipelineState, corrections: dict | None = None, from_stage: str = "resolve") -> PipelineState:
        """Re-run from ``resolve`` (default) using cached detection results."""
        if corrections is not None:
            st.corrections = normalize_corrections(corrections)
        start = STAGES.index(from_stage)
        if start <= STAGES.index("detect"):
            return self.process(st.input_file, st.corrections)
        st.ctx.candidates = copy.deepcopy(st.detected)
        st.ctx.ids = copy.deepcopy(st.ids_after_detect)
        st.completed = st.completed[:start]
        for stage in STAGES[start:]:
            self.run_stage(st, stage)
        return st

    def run_stage(self, st: PipelineState, stage: str) -> PipelineState:
        n = STAGES.index(stage) + 1
        extra = ""
        if stage != "ingest" and st.drawing is not None:
            extra = f" · {len(st.drawing.raw_entities):,} entities"
            if st.segments:
                extra += f", {len(st.segments):,} segments"
        self._report(f"Step {n}/{len(STAGES)}: {self.STAGE_LABELS.get(stage, stage)}{extra}")
        t0 = time.perf_counter()
        getattr(self, f"_stage_{stage}")(st)
        st.timings[stage] = round(time.perf_counter() - t0, 4)
        st.completed.append(stage)
        return st

    # ------------------------------------------------------------------ stages
    def _stage_ingest(self, st):
        inc = self.cfg["runtime"]["include_paperspace"]
        if getattr(st, "_doc", None) is not None:
            st.drawing = ingest_document(st._doc, st.input_file, include_paperspace=inc)
        else:
            st.drawing = ingest(st.input_file, include_paperspace=inc)

    def _stage_preprocess(self, st):
        d = st.drawing
        d.units = detect_units(d, self.cfg["runtime"].get("forced_units"))
        normalize(d)
        tol0 = self.cfg["tolerances"]["duplicate_tol_m"]
        degenerate = flag_degenerate(d, tol0 * 0.1)
        st.segments = build_segments(d)
        tol = estimate_tolerances(d, st.segments, self.cfg)
        st.dup_stats = mark_duplicates(st.segments, tol.duplicate, tol.angle)
        st._tol = tol
        st.logs["preprocess"] = {"units": d.units.__dict__, "transform": d.transform.to_dict(),
                                 "degenerate": degenerate, "duplicates": {k: v for k, v in st.dup_stats.items() if k != "overlap_pairs"},
                                 "tolerances": tol.to_dict()}

    def _stage_context(self, st):
        st.ctx = build_context(st.drawing, st.segments, st._tol, self.cfg)

    def _stage_detect(self, st):
        ctx = st.ctx
        disabled = set(self.cfg["runtime"].get("disabled_detectors") or [])
        enabled = [n for n in self.cfg["detection"]["enabled_detectors"] if n in DETECTORS and n not in disabled]
        for n in sorted(disabled):
            st.drawing.issues.append({"stage": "detect", "severity": "warning",
                                      "message": f"Detector {n} was disabled for this run"})
        for name in run_order(enabled):
            if name == "text_detector":
                continue  # labels are produced while building the context
            det = DETECTORS[name](self.cfg)
            self._report(f"Step 4/{len(STAGES)}: detector {name.replace('_', ' ')}")
            t0 = time.perf_counter()
            try:
                ctx.candidates[name] = det.detect(st.drawing, ctx)
            except Exception as exc:  # a failing detector must not stop the other independent detectors
                ctx.candidates[name] = []
                st.drawing.issues.append({"stage": "detect", "severity": "warning",
                                          "message": f"Detector {name} failed and was skipped: {type(exc).__name__}: {exc}"})
            st.timings[f"detect:{name}"] = round(time.perf_counter() - t0, 4)
        shares = layer_parallel_shares(ctx.strokes, ctx.pairs)
        for k, v in shares.items():
            if k in ctx.layers:
                ctx.layers[k].parallel_share = v
        st.detected = copy.deepcopy(ctx.candidates)
        st.ids_after_detect = copy.deepcopy(ctx.ids)
        st.logs["detect"] = {k: len(v) for k, v in ctx.candidates.items()}

    def _stage_resolve(self, st):
        walls, doors, windows, columns = CandidateResolver(self.cfg).resolve(st.ctx)
        corr = st.corrections
        n = apply_status_overrides(walls + doors + windows + columns, corr)
        walls += manual_walls(corr, st.ctx.ids)
        st.walls, st.doors, st.windows, st.columns = walls, doors, windows, columns
        st.logs["resolve"] = dict(st.ctx.stats.get("resolver", {}), manual_overrides=n,
                                  manual_walls=len(corr.get("added_walls", [])))
        # stair flights: their tread lines must not survive as wall candidates
        from ..detectors.stair_detector import (assemble_flights, detect_stairs, veto_stair_outline_walls,
                                                veto_tread_walls)
        if "stair_detector" in self.cfg["runtime"].get("disabled_detectors", []):
            st.stairs, slog = [], {"skipped": "disabled"}
        else:
            self._report("Step 5/10: stair flights")
            try:
                ids = st.ctx.ids
                st.stairs, slog = detect_stairs(st.ctx, self.cfg, ids, st.ctx.labels)
                apply_status_overrides(st.stairs, corr)
                slog["walls_vetoed"] = veto_tread_walls(st.stairs, walls, self.cfg)
                raw = st.drawing.raw_entities
                slog["outline_walls_vetoed"] = veto_stair_outline_walls(
                    st.stairs, walls, self.cfg, lambda r: raw[r].layer if r in raw else None, st.ctx.layer_score)
                slog["assembly"] = assemble_flights(st.stairs, walls, self.cfg, st.ctx.tol)
            except Exception as e:  # noqa: BLE001  (an optional detector must never stop the pipeline)
                st.stairs, slog = [], {"error": str(e)}
                st.drawing.issues.append({"stage": "resolve", "severity": "warning",
                                          "message": f"stair detector failed: {e}"})
        st.logs["stairs"] = slog

    def _stage_reconstruct(self, st):
        walls, generic, log = reconstruct(st.walls, st.doors, st.windows, st.ctx, self.cfg)
        st.walls = walls
        corr = st.corrections
        for o in generic:
            t = corr.get("opening_type_overrides", {}).get(o.id)
            if t in OPENING_TYPES:
                o.type = OPENING_TYPES[t][0]
                o.subtype = OPENING_TYPES[t][1] or "manual"
        apply_status_overrides(generic, corr)
        ow = self.cfg["detection"]["weights"]
        th = self.cfg["detection"]["status_thresholds"]
        for o in generic:
            if o.width < max(st.ctx.tol.snap * 2, 0.05) and o.manual_status is None:
                o.veto("degenerate gap (no width)")
                continue
            o.score({**ow["opening"], "manual": 10.0}, th)
            if o.manual_status is None and o.status != Status.REJECTED:
                o.confidence = min(o.confidence, self.cfg["reconstruction"]["unidentified_opening_confidence"])
                o.status = Status.NEEDS_REVIEW if o.confidence >= th["review"] else Status.REJECTED
        st._generic = generic
        st.logs["reconstruct"] = log

    def _stage_topology(self, st):
        cfg = self.cfg
        tol = st.ctx.tol
        walls = st.walls
        live = [w for w in walls if w.status != Status.REJECTED]
        G_walls = topology_evidence(live, st.columns, tol)
        ww = cfg["detection"]["weights"]["wall"]
        th = cfg["detection"]["status_thresholds"]
        gate = cfg["detection"].get("gates", {}).get("wall")
        for w in live:
            if w.detection_methods & {"door_implied", "label_implied"} and w.manual_status is None:
                continue                      # added by the door / room-name rules: keeps its own confidence
            w.score(ww, th, gate)
        for kind, coll in (("door", st.doors), ("window", st.windows)):
            for o in coll:
                if o.host_wall is None and o.manual_status is None:
                    o.add_evidence("host_wall", 0.0)
                o.score(cfg["detection"]["weights"][kind], th)
        openings = [o for o in st.doors + st.windows + st._generic if o.status != Status.REJECTED]
        # a door is always in a wall: complete the wall on the line of every door
        from ..topology.door_rules import complete_door_walls, promote_door_rooms
        try:
            dlog = complete_door_walls(walls, st.doors, tol, cfg, st.ctx.ids, st.windows)
            for o in st.windows:                       # hosted by the rule: score again with its host
                if o.host_wall and any("hosted by the door / window rule" in n for n in o.notes):
                    o.score(cfg["detection"]["weights"]["window"], th)
            openings = [o for o in st.doors + st.windows + st._generic if o.status != Status.REJECTED]
        except Exception as e:  # noqa: BLE001  (an optional rule must never stop the pipeline)
            dlog = {"error": f"{type(e).__name__}: {e}"}
            st.drawing.issues.append({"stage": "topology", "severity": "warning",
                                      "message": f"door rules (door line) skipped: {dlog['error']}"})
        live = [w for w in walls if w.status != Status.REJECTED]
        # walls whose openings were all rejected stay closed; rejected openings detach from hosts
        for w in walls:
            w.openings = [oid for oid in w.openings if any(o.id == oid for o in openings)]
        rd = RoomDetector(cfg, st.ctx.ids)
        self._report("Step 7/10: room detection")
        rooms, solid, rlog = rd.detect(live, st.columns, openings, st.ctx.labels, tol)
        mrooms = manual_rooms(st.corrections)
        if mrooms:
            keep = []
            for r in rooms:
                if any(robust_intersection(m.polygon, r.polygon).area > 0.5 * min(m.polygon.area, r.polygon.area) for m in mrooms):
                    r.veto("replaced by manual room boundary")
                keep.append(r)
            from ..topology.boundary_validator import boundary_support
            for m in mrooms:
                m.boundary_support = boundary_support(m.polygon, live, tol)
            rooms = keep + mrooms
        apply_status_overrides(rooms, st.corrections)
        live_rooms = [r for r in rooms if r.status != Status.REJECTED]
        G_rooms = relate_rooms(live_rooms, live, openings, tol)
        from ..topology.door_rules import drop_unused
        if "error" not in dlog:
            dlog["unused_rejected"] = drop_unused(walls, live_rooms)
        live = [w for w in walls if w.status != Status.REJECTED]
        if cfg["runtime"].get("safe_mode"):
            hyps, hrooms, hlog = [], [], {"skipped": "safe mode"}
            st.drawing.issues.append({"stage": "topology", "severity": "warning",
                                      "message": "Safe mode: missing-room hypotheses were not computed"})
        else:
            self._report("Step 7/10: missing-room recovery")
            hyps, hrooms, hlog = recover_missing_rooms(walls, st.columns, live_rooms, st.ctx.labels, tol, cfg, st.ctx.ids, rd)
            if not getattr(st, "_door_room_pass", False):
                try:
                    promoted = promote_door_rooms(hyps, hrooms, openings, tol, cfg, st.ctx.ids, live_rooms, walls)
                except Exception as e:  # noqa: BLE001
                    promoted = []
                    st.drawing.issues.append({"stage": "topology", "severity": "warning",
                                              "message": f"door rules (rooms with a door) skipped: {type(e).__name__}: {e}"})
                try:
                    from ..topology.label_rules import promote_label_rooms
                    named, nlog = promote_label_rooms(hyps, hrooms, tol, cfg, st.ctx.ids, live_rooms, walls)
                    promoted = promoted + named
                    st.logs["label_rules_first_pass"] = nlog
                except Exception as e:  # noqa: BLE001
                    st.drawing.issues.append({"stage": "topology", "severity": "warning",
                                              "message": f"room-name rules skipped: {type(e).__name__}: {e}"})
                if promoted:
                    # a room with a door needs all its walls: the closing hypotheses become walls,
                    # and rooms are detected again with them
                    st.walls = walls + promoted
                    st._door_room_pass = True
                    st.logs["door_rules_first_pass"] = {"door_line": dlog, "room_walls_promoted": [h.id for h in promoted]}
                    try:
                        return self._stage_topology(st)
                    finally:
                        st._door_room_pass = False
        G_axis = build_axis_graph(live, tol)
        max_t = max([w.thickness or 0.0 for w in live] + [tol.snap * 4])  # shorter dangling pieces are junction overshoots
        b = Building()
        b.walls = {w.id: w for w in walls}
        b.openings = {o.id: o for o in st.doors + st.windows + st._generic}
        b.rooms = {r.id: r for r in rooms}
        b.columns = {c.id: c for c in st.columns}
        b.stairs = {s_.id: s_ for s_ in st.stairs}
        b.hypotheses = {h.id: h for h in hyps}
        b.room_hypotheses = {r.id: r for r in hrooms}
        b.labels = st.ctx.labels
        b.graphs = {"walls": G_walls, "rooms": G_rooms, "axes": G_axis}
        self._report("Step 7/10: floor levels")
        self._assign_levels(st, b, tol)
        from ..detectors.stair_detector import assign_rise
        p2 = self.cfg["phase2"]
        assign_rise(b.stairs.values(), {lv.id: lv for lv in b.levels}, self.cfg,
                    p2["wall_height_m"] + p2["slab_thickness_m"])
        st.building = b
        st.solid = solid
        st.logs["topology"] = {"rooms": rlog, "recovery": hlog, "door_rules": dlog,
                               "room_names": st.logs.get("label_rules_first_pass", {}),
                               "door_room_walls": st.logs.get("door_rules_first_pass", {}).get("room_walls_promoted", []),
                               "wall_graph_components": _ncomp(G_walls),
                               "axis_graph": {"nodes": G_axis.number_of_nodes(), "edges": G_axis.number_of_edges(),
                                              "dangling_nodes": sum(1 for n in G_axis.nodes if G_axis.degree(n) == 1
                                                                   and next(iter(G_axis.edges(n, data=True)))[2]["length"] > max_t)}}

    def _assign_levels(self, st, b, tol):
        from shapely.geometry import Point, box as _box
        from ..topology.level_analysis import analyze_levels
        levels, wall_level, llog = analyze_levels(list(b.walls.values()), st.ctx.labels, st.ctx.strokes, tol, self.cfg)
        st.logs["levels"] = llog
        if not llog.get("stacked"):
            b.levels = []
            return
        b.levels = levels
        lv_by_id = {lv.id: lv for lv in levels}
        boxes = [(lv, _box(*lv.bbox).buffer(0.5)) for lv in levels]

        def locate(geom):
            if geom is None:
                return None
            c = geom.representative_point() if hasattr(geom, "representative_point") else Point(geom)
            hits = [(lv.kind != "floor_plan", bx.area, lv.id) for lv, bx in boxes if bx.contains(c)]
            return min(hits)[2] if hits else None          # tightest floor plan first
        for w in b.walls.values():
            w.level = wall_level.get(w.id) or locate(w.polygon(0.05) or w.axis)
        for o in b.openings.values():
            host = b.walls.get(o.host_wall) if o.host_wall else None
            o.level = host.level if host is not None else locate(Point(o.center))
        for coll in (b.rooms, b.room_hypotheses):
            for r in coll.values():
                r.level = locate(r.polygon)
        for c in b.columns.values():
            c.level = locate(c.polygon)
        for h in b.hypotheses.values():
            h.level = locate(h.axis)
        for s_ in b.stairs.values():
            s_.level = locate(s_.polygon)
        reasons = {"elevation_or_section": "drawn in an elevation / section view, not a floor plan",
                   "duplicate": "duplicate view of a level already modelled (e.g. structural / sanitary plan)",
                   "roof_plan": "roof plan (roof outline / parapets), not stacked as a storey",
                   "sheet_frame": "sheet border / title block lines, not a wall"}
        for e in list(b.walls.values()) + list(b.openings.values()) + list(b.rooms.values()) + list(b.columns.values()) + list(b.stairs.values()):
            lv = lv_by_id.get(e.level)
            if lv is not None and lv.kind in reasons and e.manual_status is None:
                e.veto(reasons[lv.kind])

    def _stage_classify(self, st):
        b = st.building
        tol = st.ctx.tol
        walls = list(b.walls.values())
        fp, limit = classify_walls(walls, st.solid, tol, self.cfg)
        b.footprint = fp
        classify_openings(list(b.openings.values()), walls)
        classify_columns(list(b.columns.values()), walls, tol)
        classify_rooms([r for r in b.rooms.values()], b.openings, self.cfg)
        classify_rooms(list(b.room_hypotheses.values()), b.openings, self.cfg)
        for h in b.hypotheses.values():
            h.classification = "hypothesis"
        from ..classification.access_solver import solve_access
        from ..classification.balcony import detect_balconies
        from ..classification.plausibility import apply_plausibility
        balc = detect_balconies(b, self.cfg, tol, st.drawing)
        from .corrections import apply_opening_type_overrides
        locked = apply_opening_type_overrides(list(b.openings.values()), st.corrections)
        plaus = apply_plausibility(b, self.cfg, locked, tol)
        access = solve_access(b, self.cfg, locked)
        classify_openings(list(b.openings.values()), walls)
        st.logs["classify"] = {"partition_threshold_m": limit, "balconies": balc, "plausibility": plaus,
                               "access": access}

    def _stage_validate(self, st):
        b = st.building
        tol = st.ctx.tol
        rep = ValidationReport(st.drawing)
        validate_geometry(rep, st.drawing, st.ctx, st.dup_stats)
        validate_walls(rep, list(b.walls.values()), list(b.columns.values()), tol, self.cfg)
        validate_openings(rep, list(b.openings.values()), list(b.walls.values()), tol, self.cfg)
        validate_rooms(rep, list(b.rooms.values()), b.labels, tol, self.cfg, b.openings)
        for h in b.hypotheses.values():
            c = h.axis.centroid
            rep.add("room", "missing_boundary_hypothesis", "warning", h, (c.x, c.y),
                    f"Proposed missing wall {h.id}: {h.status_reason}.",
                    "Accept the hypothesis if the wall exists, otherwise ignore it.", h.confidence)
        self._check_ground(st, rep)
        st.report = rep

    def ground_pieces(self, st):
        """[(polygon, level id)] of areas inside the building (outline closed over small notches)
        without a room floor. Computed once here; Phase 2 only extrudes them (model building.ground)."""
        from shapely.geometry import box as _box
        from ..topology.ground import unassigned_floor
        gc = self.cfg["phase2"].get("ground", {})
        b = st.building
        if not gc.get("enabled", True) or b is None or getattr(b, "footprint", None) is None:
            return []
        from ..geometry.geometric_utils import polygons_of
        live = lambda e: e.status != Status.REJECTED  # noqa: E731
        plans = [(lv.id, _box(*lv.bbox)) for lv in b.levels if lv.kind == "floor_plan"] if b.levels else []

        def on_plans(g):
            c = g.representative_point()
            return not plans or any(bx.contains(c) for _, bx in plans)
        # only the floor-plan views take part (elevations, sections, duplicates have no floors)
        cover = [r.polygon for r in b.rooms.values() if live(r) and r.polygon is not None]
        cover += [w.polygon(0.05) for w in b.walls.values() if live(w) and not w.is_hypothesis]
        cover += [c.polygon for c in b.columns.values() if live(c) and c.polygon is not None]
        outline = [g for g in polygons_of(b.footprint) if on_plans(g)]
        try:
            pieces = unassigned_floor(outline, [g for g in cover if g is not None and on_plans(g)],
                                      gc.get("close_m", 2.0), gc.get("min_area_m2", 1.0), gc.get("min_width_m", 0.4),
                                      gc.get("wide_close_m", ()), gc.get("min_enclosure", 0.6))
        except Exception as e:  # noqa: BLE001  (an optional check must never stop the pipeline)
            st.drawing.issues.append({"stage": "validate", "severity": "warning",
                                      "message": f"ground coverage check skipped: {type(e).__name__}: {e}"})
            return []
        out = []
        for p in pieces:
            c = p.representative_point()
            lid = next((i for i, bx in plans if bx.contains(c)), None)
            if plans and lid is None:
                continue
            out.append((p, lid))
        return out

    def _check_ground(self, st, rep):
        pieces = self.ground_pieces(st)
        st.ground = pieces
        for p, _ in pieces:
            c = p.representative_point()
            rep.add("room", "floor_without_room", "warning", None, (c.x, c.y),
                    f"{p.area:.1f} m² inside the building is not part of any room (no floor from a room): "
                    "a wall may be missing, or the space was not recognised as a room.",
                    "Add the missing wall or draw the room; the 3D model fills it with a ground slab.", 0.6, set())
        st.logs["ground"] = {"unassigned_floor_pieces": len(pieces), "area_m2": round(sum(p.area for p, _ in pieces), 2)}
        self._level_outlines(st, pieces)

    def _level_outlines(self, st, pieces):
        """Closed outline of every floor-plan level (stacked drawings): its base and roof slab."""
        b = st.building
        lc = self.cfg["phase2"].get("level_slabs", {})
        if b is None or not b.levels or not lc.get("enabled", True):
            return
        from ..topology.level_slabs import level_outline
        gc = self.cfg["phase2"].get("ground", {})
        live = lambda e: e.status != Status.REJECTED  # noqa: E731
        log = {}
        bound = {wid for r in b.rooms.values() if live(r) and not r.is_hypothesis for wid in r.bounding_walls}
        hosts = {o.host_wall for o in b.openings.values() if live(o) and o.host_wall}
        for lv in b.levels:
            if lv.kind != "floor_plan":
                continue
            walls = [w for w in b.walls.values() if live(w) and not w.is_hypothesis and w.level == lv.id]
            # the structure of the storey: walls that bound a room or carry an opening (a free-standing
            # group kept for review does not make the storey bigger)
            used = [w for w in walls if w.id in bound or w.id in hosts]
            # (the vertices of all its walls stay the reference: when the structure covers less than
            # half of what they span, the storey did not close and its wall vertices are connected)
            geoms = [w.polygon(0.05) for w in (used or walls)]
            geoms += [r.polygon for r in b.rooms.values() if live(r) and not r.is_hypothesis and r.level == lv.id]
            geoms += [c.polygon for c in b.columns.values() if live(c) and c.level == lv.id]
            geoms += [p for p, lid in pieces if lid == lv.id]
            pts = [xy for w in walls for xy in w.axis.coords]
            try:
                lv.outline, lv.outline_source = level_outline(
                    geoms, pts, gc.get("close_m", 2.0), gc.get("wide_close_m", ()), gc.get("min_enclosure", 0.6),
                    lc.get("min_hull_share", 0.5), lc.get("hull_ratio", 0.25), gc.get("min_area_m2", 1.0))
            except Exception as e:  # noqa: BLE001  (an optional step must never stop the pipeline)
                lv.outline, lv.outline_source = [], None
                st.drawing.issues.append({"stage": "validate", "severity": "warning",
                                          "message": f"outline of level {lv.name or lv.id} skipped: {type(e).__name__}: {e}"})
            log[lv.id] = {"source": lv.outline_source, "area_m2": round(sum(p.area for p in lv.outline), 2)}
        # the storeys repeat: cut what one sheet shows far outside all the others
        plans = sorted((lv for lv in b.levels if lv.kind == "floor_plan" and lv.outline), key=lambda lv: lv.z)
        over = lc.get("max_overhang_m", 3.0)
        if over is not None and len(plans) > 1:
            try:
                from ..topology.level_slabs import limit_overhang
                kept, cut = limit_overhang([(lv.outline, tuple(lv.translation)) for lv in plans], float(over),
                                           gc.get("min_area_m2", 1.0))
                for lv, polys, c in zip(plans, kept, cut):
                    if c > 0.5 and polys:
                        lv.outline = polys
                        log[lv.id]["cut_beyond_other_storeys_m2"] = c
                        log[lv.id]["area_m2"] = round(sum(p.area for p in polys), 2)
            except Exception as e:  # noqa: BLE001
                st.drawing.issues.append({"stage": "validate", "severity": "warning",
                                          "message": f"overhang limit of the level outlines skipped: {type(e).__name__}: {e}"})
        st.logs["level_outlines"] = log

    def _stage_export(self, st):
        from .database_exporter import build_model
        st.model = build_model(st, self.cfg)


def _ncomp(G):
    import networkx as nx
    return nx.number_connected_components(G) if G.number_of_nodes() else 0
