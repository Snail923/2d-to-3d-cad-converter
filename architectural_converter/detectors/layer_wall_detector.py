"""Detector D: layer semantics as supporting evidence.

Produces *linework* candidates for strokes on layers whose names indicate walls. They carry
only ``layer_match`` evidence; the resolver attaches that evidence to geometric
candidates sharing the same boundary segments and discards linework candidates that are
not confirmed by geometry (the single-line detector evaluates those lines separately)."""
from __future__ import annotations

from ..models.wall import Wall
from .base_detector import BaseDetector, register


@register
class LayerWallDetector(BaseDetector):
    name = "layer_wall_detector"
    produces = "wall"

    def detect(self, drawing, ctx):
        out = []
        for k, st in enumerate(ctx.strokes):
            ev = max(ctx.wall_evidence(l) for l in st.layers)
            if ev < 0.9:
                continue
            w = Wall(id=ctx.ids.new("wall"), kind="linework", u=st.u, c=st.c, s0=st.s0, s1=st.s1, thickness=None,
                     thickness_source="unknown")
            w.face_segments = set(st.members)
            w.source_entities = set(st.raw_ids)
            w.add_evidence("layer_match", ev)
            w.detection_methods.add(self.name)
            w.confidence = ev
            w.status_reason = "layer evidence only (needs geometric confirmation)"
            out.append(w)
        return out
