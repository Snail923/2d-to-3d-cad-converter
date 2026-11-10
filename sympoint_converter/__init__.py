"""Approach 2: learned extraction with SymPointV2 (panoptic symbol spotting on the DXF primitives).

Kept separate from ``architectural_converter`` (approach 1, rule-based) so the two can be compared.
The model replaces approach 1's extraction (detectors + resolver); approach 1's front end (DXF, units,
normalisation) and build stages (reconstruction, rooms, classification, validation, export, 3D) are
reused unchanged. The model runs in its own service (``sympoint_converter/service``) on a CUDA machine.

    python -m sympoint_converter compare drawing.dxf -o out/    # approach 1 vs approach 2
    python -m sympoint_converter build drawing.dxf -o out/      # approach 2 model, report, GLB
    python -m sympoint_converter label drawing.dxf -o out/      # raw labels viewer
(the local service starts automatically, see README.md)
"""
import json
import os

from .classes import BG, CLASSES, COLORS, GROUP_OF, GROUPS
from .sympoint_io import export_tiles, labels_from_response, load_drawing, primitives

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if v is None:
            continue
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# "use_reasoning": false = the model's answer as it is (to see what the reasoning modules change)
REASONING_OFF = {"reasoning": {"label_consensus": {"enabled": False}, "wall_structure": {"enabled": False}},
                 "engine_overrides": {"reconstruction": {"label_rules": {"enabled": False}},
                                      "phase2": {"ground": {"wide_close_m": []}}}}


def load_config(overrides: dict | None = None) -> dict:
    """config.json deep-merged with ``overrides`` (None values are ignored)."""
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if os.environ.get("SYMPOINT_URL"):
        cfg["service"]["url"] = os.environ["SYMPOINT_URL"]
    cfg = _merge(cfg, overrides or {})
    if cfg.get("use_reasoning") is False:
        cfg = _merge(cfg, REASONING_OFF)
    return cfg


__all__ = ["BG", "CLASSES", "COLORS", "GROUPS", "GROUP_OF", "export_tiles", "labels_from_response",
           "load_drawing", "primitives", "load_config"]
