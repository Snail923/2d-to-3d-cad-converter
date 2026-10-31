"""BLOCK definition inventory (definitions are reported even when not inserted)."""
from __future__ import annotations

from typing import Any


def read_block_definitions(doc) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for blk in doc.blocks:
        name = blk.name
        if name.lower().startswith("*model_space") or name.lower().startswith("*paper_space"):
            continue
        counts: dict[str, int] = {}
        for e in blk:
            counts[e.dxftype()] = counts.get(e.dxftype(), 0) + 1
        bp = blk.block.dxf.get("base_point", (0, 0, 0))
        out[name] = {
            "name": name,
            "base_point": [float(bp[0]), float(bp[1])],
            "entity_counts": counts,
            "is_anonymous": name.startswith("*"),
            "is_xref": bool(getattr(blk.block, "is_xref", False)),
            "insert_count": 0,
        }
    return out
