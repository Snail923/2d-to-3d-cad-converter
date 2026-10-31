"""Convert drawing-unit world coordinates to a local metric frame.

``model = (dxf - offset) * scale``. The offset (robust lower-left of the geometry) avoids
float precision loss on drawings placed at large survey coordinates. Raw coordinates stay
untouched in ``RawEntity.raw_coords``; the transform is stored for the round trip."""
from __future__ import annotations

import numpy as np

from ..models.drawing import Drawing, Transform

SCALAR_KEYS = ("radius", "width", "height", "measurement")
POINT_KEYS = ("center",)


def compute_offset(d: Drawing) -> tuple[float, float]:
    pts = [g.points for g in d.entities if g.points is not None and g.kind not in ("text", "dimension")]
    if not pts:
        return (0.0, 0.0)
    allp = np.vstack(pts)
    lo = np.percentile(allp, 1, axis=0)
    return (float(np.floor(lo[0])), float(np.floor(lo[1])))


def normalize(d: Drawing) -> Drawing:
    if d.normalized:
        return d
    scale = d.units.to_m if d.units else 1.0
    tr = Transform(scale=scale, offset=compute_offset(d))
    d.transform = tr
    for g in d.entities:
        if g.points is not None:
            g.points = tr.apply(g.points)
        data = g.data
        for k in SCALAR_KEYS:
            if k in data and isinstance(data[k], (int, float)) and data[k] is not None:
                data[k] = float(data[k]) * scale
        for k in POINT_KEYS:
            if k in data:
                data[k] = tuple(tr.apply(np.array([data[k]]))[0])
        if "loops" in data:
            data["loops"] = [tr.apply(lp) for lp in data["loops"]]
        if "defpoints" in data:
            data["defpoints"] = {k: tuple(tr.apply(np.array([v]))[0]) for k, v in data["defpoints"].items()}
    for ref in d.inserts.values():
        ref.insert = tuple(tr.apply(np.array([ref.insert]))[0])
    d.normalized = True
    return d
