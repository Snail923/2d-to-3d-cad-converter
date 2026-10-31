"""Normalised drawing representation (Stage 1 deliverable).

``RawEntity``  - immutable record of the source DXF entity (handle, layer, type, original
                 world coordinates in drawing units, attributes, block path).
``GeoEntity``  - normalised geometry in metres (after scale + offset), referencing its RawEntity.
``Drawing``    - container with metadata, units, layers, raw + normalised entities and blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RawEntity:
    id: str
    handle: str | None
    dxftype: str
    layer: str
    space: str
    attributes: dict[str, Any] = field(default_factory=dict)
    raw_coords: list[list[float]] = field(default_factory=list)   # world coords, drawing units
    block_path: tuple[str, ...] = ()       # RawEntity ids of enclosing INSERTs (outer -> inner)
    block_name: str | None = None          # block containing this entity (innermost)
    source_block_handle: str | None = None  # handle of the entity inside the BLOCK definition
    parent_id: str | None = None

    def ref(self) -> dict[str, Any]:
        return {"id": self.id, "handle": self.handle, "type": self.dxftype, "layer": self.layer,
                "block": self.block_name, "block_path": list(self.block_path),
                "source_block_handle": self.source_block_handle}


@dataclass
class GeoEntity:
    id: str
    raw_id: str
    kind: str           # line | polyline | arc | circle | curve | polygon | hatch | text | dimension | point
    layer: str
    points: np.ndarray | None = None
    closed: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    block_path: tuple[str, ...] = ()
    parent: str | None = None          # GeoEntity id of the entity this was derived from
    valid: bool = True
    flags: set[str] = field(default_factory=set)
    role: str = "linework"             # effective symbol role (set by symbol analysis)

    @property
    def from_block(self) -> bool:
        return bool(self.block_path)


@dataclass
class BlockReference:
    id: str                 # RawEntity id of the INSERT
    name: str
    layer: str
    insert: tuple[float, float]
    rotation: float
    scale: tuple[float, float, float]
    depth: int
    parent_path: tuple[str, ...]
    child_geo_ids: list[str] = field(default_factory=list)   # all descendant GeoEntities
    role: str = "unknown"
    role_scores: dict[str, float] = field(default_factory=dict)
    signature: dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerInfo:
    name: str
    color: int | None = None
    linetype: str | None = None
    lineweight: int | None = None
    is_off: bool = False
    is_frozen: bool = False
    entity_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class UnitInfo:
    insunits: int
    header_unit: str
    unit: str
    to_m: float
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class Transform:
    """normalised = (raw - offset) * scale"""
    scale: float = 1.0
    offset: tuple[float, float] = (0.0, 0.0)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, dtype=float) - np.asarray(self.offset)) * self.scale

    def inverse(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, dtype=float) / self.scale + np.asarray(self.offset)

    def to_dict(self) -> dict[str, Any]:
        return {"scale_to_m": self.scale, "offset_drawing_units": list(self.offset),
                "formula": "model_m = (dxf_xy - offset) * scale_to_m"}


@dataclass
class Drawing:
    source_file: str
    metadata: dict[str, Any] = field(default_factory=dict)
    units: UnitInfo | None = None
    transform: Transform = field(default_factory=Transform)
    layers: dict[str, LayerInfo] = field(default_factory=dict)
    raw_entities: dict[str, RawEntity] = field(default_factory=dict)
    entities: list[GeoEntity] = field(default_factory=list)
    blocks: dict[str, dict[str, Any]] = field(default_factory=dict)
    inserts: dict[str, BlockReference] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)
    normalized: bool = False

    def geo_by_id(self) -> dict[str, GeoEntity]:
        return {g.id: g for g in self.entities}

    def raw_refs(self, raw_ids) -> list[dict[str, Any]]:
        return [self.raw_entities[r].ref() for r in sorted(raw_ids) if r in self.raw_entities]

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for r in self.raw_entities.values():
            counts[r.dxftype] = counts.get(r.dxftype, 0) + 1
        kinds: dict[str, int] = {}
        for g in self.entities:
            kinds[g.kind] = kinds.get(g.kind, 0) + 1
        return {"source_file": self.source_file, "raw_entity_counts": counts,
                "normalized_kinds": kinds, "layers": len(self.layers),
                "blocks": len(self.blocks), "inserts": len(self.inserts),
                "units": self.units.unit if self.units else None,
                "issues": len(self.issues)}
