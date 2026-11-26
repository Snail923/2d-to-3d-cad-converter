"""Approach 2 pipeline: approach 1's stages with only the extraction replaced.

    ingest -> preprocess -> context            (approach 1's front end: DXF, units, strokes, texts)
    detect  = SymPointV2 labels -> candidates  (extract.py; no approach-1 detector runs)
    resolve = candidates + manual corrections  (no approach-1 candidate resolver / vetoes)
    reconstruct -> topology -> classify -> validate -> export   (approach 1, unchanged)

The result is a normal ``PipelineState`` / archmodel, so approach 1's exporters, overlay, report,
3D builder and benchmark metrics work on it as they are."""
from __future__ import annotations

import copy
import time

from architectural_converter.pipeline.corrections import apply_status_overrides, manual_walls
from architectural_converter.pipeline.extraction_pipeline import STAGES, ExtractionPipeline
from architectural_converter.pipeline.pipeline_config import PipelineConfig

from . import load_config
from .extract import extract
from .labels import get_labels


class SymPointPipeline(ExtractionPipeline):
    STAGE_LABELS = dict(ExtractionPipeline.STAGE_LABELS,
                        detect="SymPointV2 labels -> walls, doors, windows, columns, stairs",
                        resolve="Applying manual corrections")

    def __init__(self, sp_config: dict | None = None, engine_overrides: dict | None = None, progress=None,
                 url: str | None = None, response: dict | None = None, labels_progress=print):
        self.sp = sp_config or load_config()
        from . import _merge
        ov = copy.deepcopy(self.sp.get("engine_overrides", {}))
        ov.pop("_doc", None)
        ov = _merge(ov, engine_overrides or {})
        super().__init__(PipelineConfig(ov), progress)
        self.url, self.response, self.labels_progress = url, response, labels_progress

    def _stage_detect(self, st):
        t0 = time.perf_counter()
        labels, prims, req, resp, source = get_labels(st.drawing, st.input_file, self.sp, self.url, self.response,
                                                      progress=self.labels_progress)
        st.timings["detect:sympoint_labels"] = round(time.perf_counter() - t0, 4)
        t0 = time.perf_counter()
        st.sp_labels_model = labels
        from .reasoning.label_consensus import refine_labels
        try:
            labels, clog = refine_labels(st.drawing, st.ctx.layers, prims, labels, self.sp)
        except Exception as e:  # noqa: BLE001  (a reasoning module must never stop the run)
            clog = {"error": f"{type(e).__name__}: {e}"}
            st.drawing.issues.append({"stage": "detect", "severity": "warning",
                                      "message": f"approach 2: label consensus skipped: {clog['error']}"})
        st.timings["detect:label_consensus"] = round(time.perf_counter() - t0, 4)
        st.sp_labels, st.sp_prims, st.sp_request, st.sp_response = labels, prims, req, resp
        t0 = time.perf_counter()
        cands, log = extract(st.drawing, st.ctx, prims, labels, self.sp, st.ctx.ids)
        for step, err in (log.get("step_errors") or {}).items():
            st.drawing.issues.append({"stage": "detect", "severity": "warning",
                                      "message": f"approach 2: {step} skipped after an error: {err}"})
        st.timings["detect:sympoint_extract"] = round(time.perf_counter() - t0, 4)
        st.ctx.candidates = cands
        st.detected = copy.deepcopy(cands)
        st.ids_after_detect = copy.deepcopy(st.ctx.ids)
        st.logs["detect"] = {"approach": "sympoint", "labels_source": source, "tiles": len(req["tiles"]),
                             "primitives": len(prims), "labelled_entities": len(labels["entities"]),
                             "model_seconds": resp.get("seconds"), "label_consensus": clog, **log}

    def _stage_reconstruct(self, st):
        super()._stage_reconstruct(st)
        from .gap_typing import type_gaps
        st.logs["reconstruct"]["gap_typing"] = type_gaps(st._generic, st.doors, st.windows, st.walls,
                                                         st.sp_prims, st.sp_labels, self.sp)

    def _stage_topology(self, st):
        super()._stage_topology(st)
        from .reasoning.wall_structure import prune_wall_groups
        try:
            wlog = prune_wall_groups(st.building, st.ctx.tol, self.sp)
        except Exception as e:  # noqa: BLE001  (a reasoning module must never stop the run)
            wlog = {"error": f"{type(e).__name__}: {e}"}
            st.drawing.issues.append({"stage": "topology", "severity": "warning",
                                      "message": f"approach 2: wall structure check skipped: {wlog['error']}"})
        prev = getattr(st, "_wall_structure_log", None)        # topology can run twice (door / name rules)
        if prev and "error" not in wlog and "skipped" not in wlog and "skipped" not in prev and "error" not in prev:
            wlog = {k: round(wlog[k] + prev[k], 2) for k in wlog}
        st._wall_structure_log = wlog
        st.logs["topology"]["wall_structure"] = wlog

    def _stage_resolve(self, st):
        st._wall_structure_log = None
        c = st.ctx.candidates
        walls = list(c.get("sympoint_walls", []))
        doors = list(c.get("sympoint_doors", []))
        windows = list(c.get("sympoint_windows", []))
        columns = list(c.get("sympoint_columns", []))
        stairs = list(c.get("sympoint_stairs", []))
        wt = self.cfg["detection"]["weights"]
        th = self.cfg["detection"]["status_thresholds"]
        for coll, key in ((walls, "wall"), (doors, "door"), (windows, "window"), (columns, "column"), (stairs, "stair")):
            for e in coll:
                e.score({**wt[key], "manual": 10.0}, th)
        corr = st.corrections
        n = apply_status_overrides(walls + doors + windows + columns + stairs, corr)
        walls += manual_walls(corr, st.ctx.ids)
        st.walls, st.doors, st.windows, st.columns, st.stairs = walls, doors, windows, columns, stairs
        st.logs["resolve"] = {"manual_overrides": n, "manual_walls": len(corr.get("added_walls", []))}
        st.logs["stairs"] = {"flights": len(stairs), "source": "sympoint"}


__all__ = ["SymPointPipeline", "STAGES"]
