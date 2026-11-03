"""Thin API used by the browser review UI (runs inside Pyodide, also usable from CPython).

Every function returns JSON strings or bytes so the JavaScript side does not need to know
Python objects. State (the last pipeline run) is kept in this module so corrections can be
re-applied quickly without re-running detection."""
from __future__ import annotations

import json
import os
import tempfile

from .models.base_entity import Status
from .pipeline.database_exporter import _jsonable, export_sqlite
from .pipeline.extraction_pipeline import ExtractionPipeline
from .pipeline.pipeline_config import PipelineConfig
from .reconstruction3d import build_3d
from .visualization.feature_overlay import render_overlay, view_payload
from .visualization.validation_viewer import html_report

_S: dict = {}


def _payload(st, cfg) -> str:
    _progress("Building the 3D model and the review view")
    m3 = build_3d(st.model, cfg)
    ctx = st.ctx
    out = {
        "model": st.model,
        "view": view_payload(st),
        "mesh": m3.to_json(),
        "context": {"tolerances": ctx.tol.to_dict(), "thickness_clusters": ctx.thickness_clusters,
                    "symbol_roles": ctx.summary()["symbol_roles"], "stats": ctx.stats,
                    "labels": [{"id": lb.id, "text": lb.text, "category": lb.category, "room_type": lb.room_type,
                                "position": list(lb.position)} for lb in ctx.labels]},
    }
    return json.dumps(out, default=_jsonable, separators=(",", ":"))


def _progress(msg: str) -> None:
    """Forward progress to the browser (a JS function registered by the worker), if present."""
    try:
        import js  # only available inside Pyodide
        js.pyProgress(msg)
    except Exception:
        pass


def process(path: str, corrections_json: str | None = None, config_json: str | None = None) -> str:
    cfg = PipelineConfig(json.loads(config_json) if config_json else None)
    pipe = ExtractionPipeline(cfg, progress=_progress)
    corr = json.loads(corrections_json) if corrections_json else None
    st = pipe.process(path, corrections=corr)
    _S.update(pipe=pipe, st=st, cfg=cfg)
    return _payload(st, cfg)


def rerun(corrections_json: str) -> str:
    pipe, st, cfg = _S["pipe"], _S["st"], _S["cfg"]
    st = pipe.rerun(st, json.loads(corrections_json))
    _S["st"] = st
    return _payload(st, cfg)


def export_model_json() -> str:
    return json.dumps(_S["st"].model, default=_jsonable, indent=1)


def export_sqlite_bytes() -> bytes:
    path = os.path.join(tempfile.gettempdir(), "archmodel.sqlite")
    if os.path.exists(path):
        os.remove(path)
    export_sqlite(_S["st"].model, path)
    with open(path, "rb") as fh:
        return fh.read()


def export_glb_bytes() -> bytes:
    return build_3d(_S["st"].model, _S["cfg"]).export_glb()


def export_report_html() -> str:
    st = _S["st"]
    return html_report(st, render_overlay(st))


def evaluate(reference_json: str) -> str:
    from .benchmark.metrics import evaluate as ev
    return json.dumps(ev(_S["st"].model, json.loads(reference_json)), default=_jsonable)


def config_json() -> str:
    return json.dumps(PipelineConfig().to_dict(), indent=1)


def summary() -> str:
    st = _S["st"]
    b = st.building
    return json.dumps({"walls": sum(1 for w in b.walls.values() if w.status != Status.REJECTED),
                       "rooms": sum(1 for r in b.rooms.values() if r.status != Status.REJECTED)})
