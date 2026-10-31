"""Configuration system.

All tunable values live in ``architectural_converter/config/*.json``. A ``PipelineConfig``
is a deep-merged view of those files plus optional user overrides; modules read values
from it and never hard-code thresholds.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
SECTIONS = ("tolerances", "thickness", "detection", "reconstruction", "classification",
            "validation", "phase2", "layer_semantics")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class PipelineConfig:
    """Hierarchical configuration. Access with ``cfg["detection"]["door"]`` or
    ``cfg.get("detection.door.leaf_width_m")``."""

    def __init__(self, overrides: dict | None = None, config_dir: str | None = None):
        self.config_dir = config_dir or CONFIG_DIR
        data: dict[str, Any] = {}
        for name in SECTIONS:
            path = os.path.join(self.config_dir, f"{name}.json")
            with open(path, "r", encoding="utf-8") as fh:
                data[name] = json.load(fh)
        self.data = _deep_merge(data, overrides or {})
        # runtime options (not physical thresholds)
        self.data.setdefault("runtime", {})
        rt = self.data["runtime"]
        rt.setdefault("include_paperspace", False)
        rt.setdefault("forced_units", None)  # e.g. "mm" to override unit inference
        rt.setdefault("export_statuses", self.data["phase2"].get("export_statuses",
                                                                 ["ACCEPTED", "NEEDS_REVIEW"]))

    @classmethod
    def from_file(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)
