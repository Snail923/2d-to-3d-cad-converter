"""Common base for architectural objects and detector candidates."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Status(str, enum.Enum):
    ACCEPTED = "ACCEPTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED = "REJECTED"


class IdFactory:
    """Deterministic, prefix-scoped identifiers (``wall_0001`` ...)."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def new(self, prefix: str) -> str:
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        return f"{prefix}_{n:04d}"


def weighted_confidence(evidence: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean of the evidence features that are present.

    Missing features do not penalise a candidate; each feature counts once no matter how
    many detectors reported it (features are merged with ``max``)."""
    num = 0.0
    den = 0.0
    for k, v in evidence.items():
        w = weights.get(k)
        if w is None or v is None:
            continue
        num += w * float(v)
        den += w
    return round(num / den, 4) if den > 0 else 0.0


def status_for(conf: float, thresholds: dict[str, float]) -> Status:
    if conf >= thresholds["accept"]:
        return Status.ACCEPTED
    if conf >= thresholds["review"]:
        return Status.NEEDS_REVIEW
    return Status.REJECTED


@dataclass
class ArchitecturalEntity:
    id: str
    type: str = "entity"
    confidence: float = 0.0
    status: Status = Status.NEEDS_REVIEW
    classification: str | None = None
    detection_methods: set[str] = field(default_factory=set)
    source_entities: set[str] = field(default_factory=set)   # RawEntity ids
    evidence: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    status_reason: str | None = None
    manual_status: Status | None = None

    def add_evidence(self, key: str, value: float) -> None:
        value = float(max(0.0, min(1.0, value)))
        if key in self.evidence:
            self.evidence[key] = max(self.evidence[key], value)
        else:
            self.evidence[key] = value

    def set_evidence(self, key: str, value: float) -> None:
        self.evidence[key] = float(max(0.0, min(1.0, value)))

    def absorb(self, other: "ArchitecturalEntity") -> None:
        """Merge another candidate's evidence without double counting."""
        self.detection_methods |= other.detection_methods
        self.source_entities |= other.source_entities
        for k, v in other.evidence.items():
            self.add_evidence(k, v)
        for n in other.notes:
            if n not in self.notes:
                self.notes.append(n)

    def score(self, weights: dict[str, float], thresholds: dict[str, float], gate: dict | None = None) -> None:
        self.confidence = weighted_confidence(self.evidence, weights)
        if gate and gate.get("applies_if") in self.evidence:
            strong = any(self.evidence.get(k, 0.0) >= gate["min"] for k in gate["any_of"])
            if not strong and self.evidence.get("layer_match", 0.0) < gate["layer_min"]:
                self.confidence = round(self.confidence * gate["penalty"], 4)
                if "no strong evidence (gate)" not in self.notes:
                    self.notes.append("no strong evidence (gate)")
        if self.manual_status is not None:
            self.status = self.manual_status
            self.status_reason = "manual correction"
            return
        if self.status == Status.REJECTED and self.status_reason and self.status_reason.startswith("veto"):
            return
        self.status = status_for(self.confidence, thresholds)

    def veto(self, reason: str) -> None:
        self.status = Status.REJECTED
        self.status_reason = f"veto: {reason}"

    def base_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "classification": self.classification,
            "confidence": round(float(self.confidence), 3),
            "status": self.status.value,
            "status_reason": self.status_reason,
            "detection_methods": sorted(self.detection_methods),
            "source_entities": sorted(self.source_entities),
            "evidence": {k: round(float(v), 3) for k, v in sorted(self.evidence.items())},
            "notes": list(self.notes),
        }
