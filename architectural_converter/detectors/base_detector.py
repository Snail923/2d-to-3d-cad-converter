"""Common detector interface and registry.

Every detector:
* receives the normalised ``Drawing`` and the ``DrawingContext``,
* returns standard candidates (``Wall`` / ``Opening`` / ``Column``) with confidence,
  evidence and source entity references,
* never modifies source geometry (it may add *context* such as thickness clusters),
* declares the detectors whose output it consumes in ``requires``.

New detectors are added with ``@register`` and enabled in ``config/detection.json``."""
from __future__ import annotations

from typing import Any

DETECTORS: dict[str, type["BaseDetector"]] = {}


def register(cls):
    DETECTORS[cls.name] = cls
    return cls


class BaseDetector:
    name = "base"
    produces = "wall"                # wall | door | window | column | label
    requires: tuple[str, ...] = ()

    def __init__(self, cfg):
        self.cfg = cfg

    def detect(self, drawing, context) -> list[Any]:
        raise NotImplementedError

    # helpers -----------------------------------------------------------------------
    def weights(self, kind: str | None = None) -> dict[str, float]:
        return self.cfg["detection"]["weights"][kind or self.produces]

    def thresholds(self) -> dict[str, float]:
        return self.cfg["detection"]["status_thresholds"]

    def finalize(self, cands, kind: str | None = None):
        w = self.weights(kind)
        th = self.thresholds()
        gate = self.cfg["detection"].get("gates", {}).get(kind or self.produces)
        for c in cands:
            c.detection_methods.add(self.name)
            c.score(w, th, gate)
        return cands


def run_order(enabled: list[str]) -> list[str]:
    """Topological order honouring ``requires`` (missing optional deps are ignored)."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(n, stack=()):
        if n in seen or n not in DETECTORS:
            return
        if n in stack:
            raise ValueError(f"Detector dependency cycle at {n}")
        for dep in DETECTORS[n].requires:
            if dep in enabled:
                visit(dep, stack + (n,))
        seen.add(n)
        order.append(n)

    for n in enabled:
        visit(n)
    return order
