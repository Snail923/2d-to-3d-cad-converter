"""Wall model (candidate and final object share one class).

Straight walls are parameterised in a local frame: direction ``u`` (canonical, angle in
[0, pi)), normal ``n = (-uy, ux)``, centre-line offset ``c = n . p`` and interval
``[s0, s1]`` along ``u``. ``detected_interval`` keeps the interval supported directly by
source geometry; ``s0/s1`` may be extended by reconstruction (recorded in
``reconstruction``)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import LineString, Polygon

from ..geometry.geometric_utils import coords_list, frame_point, rect_polygon, round_pt
from .base_entity import ArchitecturalEntity


@dataclass
class Wall(ArchitecturalEntity):
    type: str = "wall"
    kind: str = "straight"                    # straight | arc | polygon | centerline
    u: tuple[float, float] = (1.0, 0.0)
    c: float = 0.0
    s0: float = 0.0
    s1: float = 0.0
    thickness: float | None = None
    thickness_source: str = "measured"        # measured | polyline_width | mline | unknown | manual
    detected_interval: tuple[float, float] | None = None
    arc: dict[str, float] | None = None       # center_x, center_y, radius, start, end (rad, ccw)
    shape: Polygon | None = None              # for polygon walls
    face_segments: set[str] = field(default_factory=set)
    openings: list[str] = field(default_factory=list)
    reconstruction: dict[str, Any] = field(default_factory=dict)
    connections: set[str] = field(default_factory=set)
    rooms: set[str] = field(default_factory=set)
    height: float | None = None
    is_hypothesis: bool = False

    # ---- geometry -----------------------------------------------------------------
    @property
    def length(self) -> float:
        if self.kind in ("straight", "centerline"):
            return max(0.0, self.s1 - self.s0)
        return self.axis.length

    def point_at(self, s: float) -> tuple[float, float]:
        return frame_point(self.u, s, self.c)

    @property
    def start(self) -> tuple[float, float]:
        if self.kind in ("straight", "centerline"):
            return self.point_at(self.s0)
        return self.axis.coords[0]

    @property
    def end(self) -> tuple[float, float]:
        if self.kind in ("straight", "centerline"):
            return self.point_at(self.s1)
        return self.axis.coords[-1]

    @property
    def axis(self) -> LineString:
        if self.kind in ("straight", "centerline"):
            return LineString([self.point_at(self.s0), self.point_at(self.s1)])
        if self.kind == "arc" and self.arc:
            return LineString(arc_points(self.arc["center_x"], self.arc["center_y"], self.arc["radius"],
                                         self.arc["start"], self.arc["end"]))
        if self.shape is not None:
            from ..geometry.geometric_utils import mrr_dims
            w, L, u, cen = mrr_dims(self.shape)
            return LineString([(cen[0] - u[0] * L / 2, cen[1] - u[1] * L / 2),
                               (cen[0] + u[0] * L / 2, cen[1] + u[1] * L / 2)])
        return LineString()

    def polygon(self, fallback_thickness: float | None = None) -> Polygon | None:
        t = self.thickness if self.thickness else fallback_thickness
        if self.kind == "polygon":
            return self.shape
        if not t:
            return None
        if self.kind in ("straight", "centerline"):
            if self.s1 - self.s0 <= 0:
                return None
            return rect_polygon(self.u, self.c, self.s0, self.s1, t)
        if self.kind == "arc" and self.arc:
            a = self.arc
            outer = arc_points(a["center_x"], a["center_y"], a["radius"] + t / 2, a["start"], a["end"])
            inner = arc_points(a["center_x"], a["center_y"], a["radius"] - t / 2, a["start"], a["end"])
            return Polygon(outer + inner[::-1])
        return None

    # ---- serialisation ------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = self.base_dict()
        poly = self.polygon()
        geom: dict[str, Any] = {"kind": self.kind}
        if self.kind in ("straight", "centerline"):
            geom.update({"start": round_pt(self.start), "end": round_pt(self.end),
                         "direction": round_pt(self.u, 6), "length": round(self.length, 4)})
        else:
            geom["axis"] = coords_list(self.axis)
            geom["length"] = round(self.axis.length, 4)
        if self.kind == "arc" and self.arc:
            geom["arc"] = {k: round(v, 6) for k, v in self.arc.items()}
        geom["polygon"] = coords_list(poly) if poly is not None else None
        d.update({
            "geometry": geom,
            "thickness": round(self.thickness, 4) if self.thickness else None,
            "thickness_source": self.thickness_source,
            "height": self.height,
            "openings": list(self.openings),
            "connections": sorted(self.connections),
            "rooms": sorted(self.rooms),
            "face_segments": sorted(self.face_segments),
            "reconstruction": self.reconstruction,
            "detected_interval": [round(x, 4) for x in self.detected_interval] if self.detected_interval else None,
            "is_hypothesis": self.is_hypothesis,
        })
        return d


def arc_points(cx: float, cy: float, r: float, a0: float, a1: float, max_step_deg: float = 4.0) -> list[tuple[float, float]]:
    """Points along a CCW arc from a0 to a1 (radians)."""
    sweep = (a1 - a0) % (2 * math.pi)
    if sweep <= 1e-9:
        sweep = 2 * math.pi
    n = max(2, int(math.ceil(math.degrees(sweep) / max_step_deg)) + 1)
    angs = np.linspace(a0, a0 + sweep, n)
    return [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in angs]
