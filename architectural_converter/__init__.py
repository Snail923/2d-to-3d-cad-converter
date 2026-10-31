"""Modular, deterministic DWG/DXF architectural feature extraction engine.

Phase 1: ingestion -> preprocessing -> semantic context -> detection -> resolution ->
reconstruction -> topology -> classification -> validation -> export.
Phase 2 (reference implementation): ``reconstruction3d``.
"""
__version__ = "1.0.0"

from .pipeline.pipeline_config import PipelineConfig  # noqa: E402,F401


def __getattr__(name):  # lazy import to keep ``import architectural_converter`` light
    if name == "ExtractionPipeline":
        from .pipeline.extraction_pipeline import ExtractionPipeline
        return ExtractionPipeline
    raise AttributeError(name)
