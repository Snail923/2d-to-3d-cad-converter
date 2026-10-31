"""Drawing-level metadata: header variables, layers, layouts."""
from __future__ import annotations

from typing import Any

from ..models.drawing import LayerInfo

HEADER_VARS = ["$ACADVER", "$INSUNITS", "$MEASUREMENT", "$LUNITS", "$EXTMIN", "$EXTMAX",
               "$LIMMIN", "$LIMMAX", "$DWGCODEPAGE", "$LTSCALE", "$DIMSCALE"]


def _plain(v: Any) -> Any:
    try:
        if hasattr(v, "__len__") and not isinstance(v, str):
            return [float(x) for x in v]
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, float, str)) or v is None:
        return v
    return str(v)


def read_metadata(doc) -> dict[str, Any]:
    hdr = doc.header
    meta = {k: _plain(hdr.get(k)) for k in HEADER_VARS if k in hdr}
    meta["dxfversion"] = doc.dxfversion
    meta["layouts"] = [lay.name for lay in doc.layouts]
    meta["paperspace_entity_counts"] = {}
    for lay in doc.layouts:
        if lay.name.lower() == "model":
            continue
        meta["paperspace_entity_counts"][lay.name] = len(lay)
    return meta


def read_layers(doc) -> dict[str, LayerInfo]:
    out: dict[str, LayerInfo] = {}
    for layer in doc.layers:
        name = layer.dxf.name
        out[name] = LayerInfo(
            name=name,
            color=int(layer.dxf.get("color", 7)),
            linetype=str(layer.dxf.get("linetype", "Continuous")),
            lineweight=int(layer.dxf.get("lineweight", -3)),
            is_off=bool(layer.is_off()),
            is_frozen=bool(layer.is_frozen()),
        )
    return out
