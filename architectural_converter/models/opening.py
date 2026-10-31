"""Openings (doors, windows, generic openings)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..geometry.geometric_utils import round_pt
from .base_entity import ArchitecturalEntity


@dataclass
class Opening(ArchitecturalEntity):
    type: str = "opening"               # door | window | opening
    subtype: str = "unknown"            # single_swing | double_swing | sliding | fixed | passage ...
    p0: tuple[float, float] = (0.0, 0.0)   # chord end points across the opening (in wall direction)
    p1: tuple[float, float] = (0.0, 0.0)
    alt_chords: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    depth: float | None = None
    width: float = 0.0
    height: float | None = None
    sill_height: float | None = None
    host_wall: str | None = None
    host_interval: tuple[float, float] | None = None   # along host wall axis
    hinge: tuple[float, float] | None = None
    swing: dict[str, Any] | None = None
    symbol_insert: str | None = None
    footprint: Any = None                # shapely polygon of the symbol (for conflicts)
    rooms: list[str] = field(default_factory=list)
    is_hypothesis: bool = False

    @property
    def center(self) -> tuple[float, float]:
        return ((self.p0[0] + self.p1[0]) / 2, (self.p0[1] + self.p1[1]) / 2)

    def to_dict(self) -> dict[str, Any]:
        d = self.base_dict()
        d.update({
            "subtype": self.subtype,
            "position": {"center": round_pt(self.center), "start": round_pt(self.p0), "end": round_pt(self.p1)},
            "width": round(self.width, 4),
            "depth": round(self.depth, 4) if self.depth else None,
            "height": self.height,
            "sill_height": self.sill_height,
            "host_wall": self.host_wall,
            "host_interval": [round(x, 4) for x in self.host_interval] if self.host_interval else None,
            "hinge": round_pt(self.hinge) if self.hinge else None,
            "swing": self.swing,
            "symbol_insert": self.symbol_insert,
            "rooms": list(self.rooms),
            "is_hypothesis": self.is_hypothesis,
        })
        return d


def Door(**kw) -> Opening:  # noqa: N802 - factory kept for readability
    kw.setdefault("type", "door")
    return Opening(**kw)


def Window(**kw) -> Opening:  # noqa: N802
    kw.setdefault("type", "window")
    return Opening(**kw)
