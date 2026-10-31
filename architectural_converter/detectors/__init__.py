"""Detectors. Importing this package registers all built-in detectors."""
from . import (column_detector, door_detector, hatch_wall_detector, layer_wall_detector,  # noqa: F401
               parallel_wall_detector, polyline_wall_detector, single_line_wall_detector, text_detector,
               window_detector)
from .base_detector import DETECTORS, BaseDetector, register, run_order  # noqa: F401
