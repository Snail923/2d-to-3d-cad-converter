"""Thin wrapper around Shapely's STRtree."""
from __future__ import annotations

from typing import Sequence

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree


class SpatialIndex:
    def __init__(self, geoms: Sequence[BaseGeometry], items: Sequence | None = None):
        self.geoms = list(geoms)
        self.items = list(items) if items is not None else list(range(len(self.geoms)))
        self.tree = STRtree(self.geoms) if self.geoms else None

    def query_bbox(self, minx, miny, maxx, maxy):
        if self.tree is None:
            return []
        return [self.items[int(i)] for i in self.tree.query(box(minx, miny, maxx, maxy))]

    def query_geom(self, geom: BaseGeometry, pad: float = 0.0):
        minx, miny, maxx, maxy = geom.bounds
        return self.query_bbox(minx - pad, miny - pad, maxx + pad, maxy + pad)

    def query_intersecting(self, geom: BaseGeometry):
        if self.tree is None:
            return []
        return [self.items[int(i)] for i in self.tree.query(geom, predicate="intersects")]
