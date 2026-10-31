"""Column model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shapely.geometry import Polygon

from ..geometry.geometric_utils import coords_list, round_pt
from .base_entity import ArchitecturalEntity


@dataclass
class Column(ArchitecturalEntity):
    type: str = "column"
    polygon: Polygon | None = None
    shape: str = "rectangular"     # rectangular | circular | polygonal
    size: tuple[float, float] = (0.0, 0.0)
    height: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.base_dict()
        c = self.polygon.centroid if self.polygon is not None else None
        d.update({"shape": self.shape, "size": [round(self.size[0], 4), round(self.size[1], 4)],
                  "center": round_pt((c.x, c.y)) if c is not None else None,
                  "polygon": coords_list(self.polygon), "height": self.height})
        return d
