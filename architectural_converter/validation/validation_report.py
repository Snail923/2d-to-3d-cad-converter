"""Validation issues and report."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..geometry.geometric_utils import round_pt


@dataclass
class Issue:
    id: str
    category: str          # wall | room | opening | geometry | column | ingestion
    code: str
    severity: str          # error | warning | info
    entity_id: str | None
    source_entities: list[dict]
    location: list[float] | None
    description: str
    suggestion: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "category": self.category, "code": self.code, "severity": self.severity,
                "entity_id": self.entity_id, "source_entities": self.source_entities,
                "location": self.location, "description": self.description, "suggestion": self.suggestion,
                "confidence": round(self.confidence, 3)}


class ValidationReport:
    def __init__(self, drawing):
        self.drawing = drawing
        self.issues: list[Issue] = []
        self._n = 0

    def add(self, category, code, severity, entity, location, description, suggestion, confidence, sources=None):
        self._n += 1
        src = sources if sources is not None else (getattr(entity, "source_entities", set()) if entity is not None else set())
        loc = round_pt(location) if location is not None else None
        self.issues.append(Issue(id=f"issue_{self._n:04d}", category=category, code=code, severity=severity,
                                 entity_id=getattr(entity, "id", entity if isinstance(entity, str) else None),
                                 source_entities=self.drawing.raw_refs(src)[:20], location=loc,
                                 description=description, suggestion=suggestion, confidence=confidence))

    def summary(self) -> dict[str, Any]:
        by_sev: dict[str, int] = {}
        by_code: dict[str, int] = {}
        for i in self.issues:
            by_sev[i.severity] = by_sev.get(i.severity, 0) + 1
            by_code[i.code] = by_code.get(i.code, 0) + 1
        return {"total": len(self.issues), "by_severity": by_sev, "by_code": by_code}

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary(), "issues": [i.to_dict() for i in self.issues]}
