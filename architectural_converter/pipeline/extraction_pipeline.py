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
from .corrections import (apply_status_overrides, manual_rooms, manual_walls, normalize_corrections)
from .pipeline_config import PipelineConfig

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
        enabled = [n for n in self.cfg["detection"]["enabled_detectors"] if n in DETECTORS]
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

    def _stage_reconstruct(self, st):
        walls, generic, log = reconstruct(st.walls, st.doors, st.windows, st.ctx, self.cfg)
        st.walls = walls
        corr = st.corrections
        for o in generic:
            t = corr.get("opening_type_overrides", {}).get(o.id)
            if t:
                o.type = t
                o.subtype = "manual"
        apply_status_overrides(generic, corr)
        ow = self.cfg["detection"]["weights"]
        th = self.cfg["detection"]["status_thresholds"]
        for o in generic:
            o.score({**ow["opening"], "manual": 10.0}, th)
            if o.manual_status is None:
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
            w.score(ww, th, gate)
        for kind, coll in (("door", st.doors), ("window", st.windows)):
            for o in coll:
                if o.host_wall is None and o.manual_status is None:
                    o.add_evidence("host_wall", 0.0)
                o.score(cfg["detection"]["weights"][kind], th)
        openings = [o for o in st.doors + st.windows + st._generic if o.status != Status.REJECTED]
        live = [w for w in walls if w.status != Status.REJECTED]
        # walls whose openings were all rejected stay closed; rejected openings detach from hosts
        for w in walls:
            w.openings = [oid for oid in w.openings if any(o.id == oid for o in openings)]
        rd = RoomDetector(cfg, st.ctx.ids)
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
        hyps, hrooms, hlog = recover_missing_rooms(walls, st.columns, live_rooms, st.ctx.labels, tol, cfg, st.ctx.ids, rd)
        G_axis = build_axis_graph(live, tol)
        max_t = max([w.thickness or 0.0 for w in live] + [tol.snap * 4])  # shorter dangling pieces are junction overshoots
        b = Building()
        b.walls = {w.id: w for w in walls}
        b.openings = {o.id: o for o in st.doors + st.windows + st._generic}
        b.rooms = {r.id: r for r in rooms}
        b.columns = {c.id: c for c in st.columns}
        b.hypotheses = {h.id: h for h in hyps}
        b.room_hypotheses = {r.id: r for r in hrooms}
        b.labels = st.ctx.labels
        b.graphs = {"walls": G_walls, "rooms": G_rooms, "axes": G_axis}
        st.building = b
        st.solid = solid
        st.logs["topology"] = {"rooms": rlog, "recovery": hlog, "wall_graph_components": _ncomp(G_walls),
                               "axis_graph": {"nodes": G_axis.number_of_nodes(), "edges": G_axis.number_of_edges(),
                                              "dangling_nodes": sum(1 for n in G_axis.nodes if G_axis.degree(n) == 1
                                                                   and next(iter(G_axis.edges(n, data=True)))[2]["length"] > max_t)}}

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
        st.logs["classify"] = {"partition_threshold_m": limit}

    def _stage_validate(self, st):
        b = st.building
        tol = st.ctx.tol
        rep = ValidationReport(st.drawing)
        validate_geometry(rep, st.drawing, st.ctx, st.dup_stats)
        validate_walls(rep, list(b.walls.values()), list(b.columns.values()), tol, self.cfg)
        validate_openings(rep, list(b.openings.values()), list(b.walls.values()), tol, self.cfg)
        validate_rooms(rep, list(b.rooms.values()), b.labels, tol, self.cfg)
        for h in b.hypotheses.values():
            c = h.axis.centroid
            rep.add("room", "missing_boundary_hypothesis", "warning", h, (c.x, c.y),
                    f"Proposed missing wall {h.id}: {h.status_reason}.",
                    "Accept the hypothesis if the wall exists, otherwise ignore it.", h.confidence)
        st.report = rep

    def _stage_export(self, st):
        from .database_exporter import build_model
        st.model = build_model(st, self.cfg)


def _ncomp(G):
    import networkx as nx
    return nx.number_connected_components(G) if G.number_of_nodes() else 0
