"""Optional learned evidence (SymPointV2 panoptic symbol spotting) fused into the rule-based engine.

The model itself runs outside the engine (``tools/sympoint_service`` on a CUDA machine); this
package only prepares its input from the DXF and turns its per-primitive labels into evidence
on the engine's candidates. Without labels the engine behaves exactly as before."""
from .sympoint_io import CLASSES, export_tiles, labels_from_response
from .fusion import fuse_resolved, fuse_openings

__all__ = ["CLASSES", "export_tiles", "labels_from_response", "fuse_resolved", "fuse_openings"]
