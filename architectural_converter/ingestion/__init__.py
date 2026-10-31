"""Stage 1: DXF ingestion."""
from __future__ import annotations

from ..models.drawing import Drawing
from .block_extractor import read_block_definitions
from .drawing_metadata import read_layers, read_metadata
from .dxf_loader import load_document
from .entity_extractor import extract_entities


def ingest(path: str, include_paperspace: bool = False, dwg_converter: str | None = None) -> Drawing:
    """Read DXF (or DWG via ODA) into a ``Drawing`` with raw + un-normalised geometry."""
    doc, issues, dxf_path = load_document(path, dwg_converter=dwg_converter)
    return ingest_document(doc, dxf_path, issues, include_paperspace)


def ingest_document(doc, source: str, issues=None, include_paperspace: bool = False) -> Drawing:
    d = Drawing(source_file=source)
    d.issues.extend(issues or [])
    d.metadata = read_metadata(doc)
    d.layers = read_layers(doc)
    d.blocks = read_block_definitions(doc)
    extract_entities(doc, d, include_paperspace=include_paperspace)
    for raw in d.raw_entities.values():
        li = d.layers.get(raw.layer)
        if li is not None:
            li.entity_counts[raw.dxftype] = li.entity_counts.get(raw.dxftype, 0) + 1
    return d
