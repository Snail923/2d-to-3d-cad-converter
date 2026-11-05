"""Approach 2: learned extraction with SymPointV2 (panoptic symbol spotting on the DXF primitives).

Kept separate from ``architectural_converter`` (approach 1, rule-based) so the two can be compared.
It reuses only approach 1's front end (DXF reading, units, coordinate normalisation); the model runs
in its own service (``sympoint_converter/service``) on a CUDA machine.

    python -m sympoint_converter health --url http://localhost:8790
    python -m sympoint_converter label drawing.dxf --url http://localhost:8790 -o out/
"""
import json
import os

from .classes import BG, CLASSES, COLORS, GROUP_OF, GROUPS
from .sympoint_io import export_tiles, labels_from_response, load_drawing, primitives

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config(overrides: dict | None = None) -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return cfg


__all__ = ["BG", "CLASSES", "COLORS", "GROUPS", "GROUP_OF", "export_tiles", "labels_from_response",
           "load_drawing", "primitives", "load_config"]
