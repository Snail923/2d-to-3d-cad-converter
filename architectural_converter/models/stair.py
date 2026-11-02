"""Stair flight model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import Polygon

from ..geometry.geometric_utils import coords_list, round_pt
from .base_entity import ArchitecturalEntity


@dataclass
class Stair(ArchitecturalEntity):
    """One straight flight: a run of parallel, regularly spaced tread lines.

    ``u`` is the tread direction, ``run`` the walking direction (normal to the treads, pointing
    up the flight when known); ``tread_offsets`` are the tread line positions along ``run``
    measured from ``origin``."""
    type: str = "stair"
    polygon: Polygon | None = None
    origin: tuple[float, float] = (0.0, 0.0)
    u: tuple[float, float] = (1.0, 0.0)
    run: tuple[float, float] = (0.0, 1.0)
    width: float = 0.0
    going: float = 0.0
    tread_offsets: list[float] = field(default_factory=list)
    tread_span: tuple[float, float] = (0.0, 0.0)     # extent along u, relative to origin
    direction_known: bool = False
    stroke_ids: list[str] = field(default_factory=list)
    rise: float | None = None
    rise_source: str | None = None

    @property
    def treads(self) -> int:
        return len(self.tread_offsets)

    def to_dict(self) -> dict[str, Any]:
        d = self.base_dict()
        d.update({"polygon": coords_list(self.polygon), "origin": round_pt(self.origin), "u": round_pt(self.u, 6),
                  "run": round_pt(self.run, 6), "width": round(self.width, 3), "going": round(self.going, 3),
                  "treads": self.treads, "risers": self.treads + 1,
                  "tread_offsets": [round(x, 4) for x in self.tread_offsets],
                  "tread_span": [round(self.tread_span[0], 4), round(self.tread_span[1], 4)],
                  "direction_known": self.direction_known,
                  "rise": round(self.rise, 4) if self.rise else None, "rise_source": self.rise_source})
        return d
