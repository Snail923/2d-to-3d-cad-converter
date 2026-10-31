"""Room model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import Polygon

from ..geometry.geometric_utils import coords_list, round_pt
from .base_entity import ArchitecturalEntity


@dataclass
class Room(ArchitecturalEntity):
    type: str = "room"
    polygon: Polygon | None = None
    name: str | None = None
    label_texts: list[str] = field(default_factory=list)
    label_ids: list[str] = field(default_factory=list)
    area_text: float | None = None
    bounding_walls: set[str] = field(default_factory=set)
    openings: set[str] = field(default_factory=set)
    adjacent_rooms: set[str] = field(default_factory=set)
    connected_rooms: set[str] = field(default_factory=set)
    boundary_support: dict[str, float] = field(default_factory=dict)
    is_hypothesis: bool = False
    hypothesis_walls: list[str] = field(default_factory=list)

    @property
    def area(self) -> float:
        return float(self.polygon.area) if self.polygon is not None else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = self.base_dict()
        p = self.polygon
        d.update({
            "name": self.name,
            "polygon": coords_list(p),
            "area": round(self.area, 3),
            "perimeter": round(p.length, 3) if p is not None else None,
            "centroid": round_pt((p.representative_point().x, p.representative_point().y)) if p is not None else None,
            "label_texts": self.label_texts,
            "area_text": self.area_text,
            "bounding_walls": sorted(self.bounding_walls),
            "openings": sorted(self.openings),
            "adjacent_rooms": sorted(self.adjacent_rooms),
            "connected_rooms": sorted(self.connected_rooms),
            "boundary_support": {k: round(v, 3) for k, v in self.boundary_support.items()},
            "is_hypothesis": self.is_hypothesis,
            "hypothesis_walls": self.hypothesis_walls,
        })
        return d
