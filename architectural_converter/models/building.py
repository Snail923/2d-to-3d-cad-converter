"""Building container: the structured result of Phase 1."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .column import Column
from .opening import Opening
from .room import Room
from .wall import Wall


@dataclass
class TextLabel:
    id: str
    raw_id: str
    text: str
    position: tuple[float, float]
    height: float
    category: str            # room_name | area | tag | other
    room_type: str | None = None
    area_value: float | None = None
    layer: str = ""


@dataclass
class Building:
    walls: dict[str, Wall] = field(default_factory=dict)
    openings: dict[str, Opening] = field(default_factory=dict)
    rooms: dict[str, Room] = field(default_factory=dict)
    columns: dict[str, Column] = field(default_factory=dict)
    hypotheses: dict[str, Wall] = field(default_factory=dict)
    room_hypotheses: dict[str, Room] = field(default_factory=dict)
    labels: list[TextLabel] = field(default_factory=list)
    footprint: Any = None
    graphs: dict[str, Any] = field(default_factory=dict)
    relationships: list[dict[str, Any]] = field(default_factory=list)

    def all_entities(self):
        yield from self.walls.values()
        yield from self.openings.values()
        yield from self.rooms.values()
        yield from self.columns.values()
        yield from self.hypotheses.values()
        yield from self.room_hypotheses.values()

    def get(self, eid: str):
        for coll in (self.walls, self.openings, self.rooms, self.columns, self.hypotheses, self.room_hypotheses):
            if eid in coll:
                return coll[eid]
        return None
