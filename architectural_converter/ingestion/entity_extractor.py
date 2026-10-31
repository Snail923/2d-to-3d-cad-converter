"""DXF entity extraction (Stage 1).

Walks model space (and optionally paper space). INSERTs are expanded recursively with
their full transformation (nested blocks, rotation, non-uniform scale, MINSERT arrays).
Every primitive keeps a reference to its RawEntity, which preserves handle, original
layer, type, attributes, world coordinates (drawing units) and the chain of INSERTs.

Output geometry is still in *drawing units*. ``preprocessing.coordinate_normalizer``
converts it to metres later.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from ..models.drawing import BlockReference, Drawing, GeoEntity, RawEntity

ARC_STEP_DEG = 3.0
MAX_RAW_COORDS = 4000
MAX_DEPTH = 16
SUPPORTED = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "SPLINE", "ELLIPSE", "HATCH",
             "SOLID", "TRACE", "3DFACE", "INSERT", "MLINE", "TEXT", "MTEXT", "ATTRIB", "DIMENSION",
             "POINT", "ARC_DIMENSION", "LARGE_RADIAL_DIMENSION"}


class EntityExtractor:
    def __init__(self, drawing: Drawing, include_paperspace: bool = False):
        self.d = drawing
        self.include_paperspace = include_paperspace
        self._raw_n = 0
        self._geo_n = 0
        self.unsupported: dict[str, int] = {}

    # ------------------------------------------------------------------ ids
    def _raw_id(self) -> str:
        self._raw_n += 1
        return f"e_{self._raw_n:06d}"

    def _geo_id(self) -> str:
        self._geo_n += 1
        return f"g_{self._geo_n:06d}"

    # ------------------------------------------------------------------ entry
    def extract(self, doc) -> None:
        self._walk(doc.modelspace(), space="model", path=(), inherited_layer=None, block_name=None, depth=0)
        if self.include_paperspace:
            for layout in doc.layouts:
                if layout.name.lower() == "model":
                    continue
                self._walk(layout, space=f"paper:{layout.name}", path=(), inherited_layer=None,
                           block_name=None, depth=0)
        if self.unsupported:
            self.d.issues.append({"stage": "ingestion", "severity": "info",
                                  "message": "Unsupported entity types kept as raw only",
                                  "details": dict(self.unsupported)})

    # ------------------------------------------------------------------ helpers
    def _layer_of(self, e, inherited: str | None) -> str:
        layer = e.dxf.get("layer", "0") or "0"
        if layer == "0" and inherited:
            return inherited
        return layer

    def _attrs(self, e) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in ("color", "linetype", "lineweight", "true_color", "ltscale", "thickness", "elevation"):
            if e.dxf.hasattr(k):
                v = e.dxf.get(k)
                out[k] = v if isinstance(v, (int, float, str)) else str(v)
        return out

    def _source_handle(self, e) -> tuple[str | None, str | None]:
        h = e.dxf.get("handle", None)
        src = getattr(e, "source_of_copy", None)
        src_h = src.dxf.get("handle", None) if src is not None else None
        return h, src_h

    def _new_raw(self, e, layer, space, path, block_name, parent_id=None, extra=None) -> RawEntity:
        h, src_h = self._source_handle(e)
        raw = RawEntity(id=self._raw_id(), handle=h, dxftype=e.dxftype(), layer=layer, space=space,
                        attributes=self._attrs(e), block_path=tuple(path), block_name=block_name,
                        source_block_handle=src_h, parent_id=parent_id)
        if extra:
            raw.attributes.update(extra)
        self.d.raw_entities[raw.id] = raw
        return raw

    def _add_geo(self, raw: RawEntity, kind: str, points=None, closed=False, data=None,
                 parent: str | None = None, layer: str | None = None) -> GeoEntity:
        pts = None if points is None else np.asarray(points, dtype=float)[:, :2]
        g = GeoEntity(id=self._geo_id(), raw_id=raw.id, kind=kind, layer=layer or raw.layer, points=pts,
                      closed=closed, data=data or {}, block_path=raw.block_path, parent=parent)
        self.d.entities.append(g)
        if pts is not None and not raw.raw_coords:
            raw.raw_coords = [[float(x), float(y)] for x, y in pts[:MAX_RAW_COORDS]]
        return g

    # ------------------------------------------------------------------ walk
    def _walk(self, entities: Iterable, space: str, path: tuple, inherited_layer, block_name, depth: int):
        for e in entities:
            try:
                self._entity(e, space, path, inherited_layer, block_name, depth)
            except Exception as exc:  # never lose the whole drawing because of one entity
                self.d.issues.append({"stage": "ingestion", "severity": "warning",
                                      "message": f"Failed to extract {e.dxftype()}: {exc}",
                                      "handle": e.dxf.get("handle", None)})

    def _entity(self, e, space, path, inherited_layer, block_name, depth):
        t = e.dxftype()
        layer = self._layer_of(e, inherited_layer)
        if t == "INSERT":
            self._insert(e, layer, space, path, block_name, depth)
            return
        if t not in SUPPORTED:
            self.unsupported[t] = self.unsupported.get(t, 0) + 1
            self._new_raw(e, layer, space, path, block_name)
            return
        if t == "LINE":
            raw = self._new_raw(e, layer, space, path, block_name)
            s, en = e.dxf.start, e.dxf.end
            self._add_geo(raw, "line", [(s.x, s.y), (en.x, en.y)])
        elif t in ("LWPOLYLINE", "POLYLINE"):
            self._polyline(e, layer, space, path, block_name)
        elif t == "ARC":
            raw = self._new_raw(e, layer, space, path, block_name,
                                extra={"radius": float(e.dxf.radius), "start_angle": float(e.dxf.start_angle),
                                       "end_angle": float(e.dxf.end_angle)})
            self._arc_geo(e, raw)
        elif t == "CIRCLE":
            raw = self._new_raw(e, layer, space, path, block_name, extra={"radius": float(e.dxf.radius)})
            c = e.ocs().to_wcs(e.dxf.center)
            r = float(e.dxf.radius)
            n = int(360 / ARC_STEP_DEG)
            pts = [(c.x + r * math.cos(2 * math.pi * i / n), c.y + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
            self._add_geo(raw, "circle", pts, closed=True, data={"center": (c.x, c.y), "radius": r})
        elif t in ("SPLINE", "ELLIPSE"):
            raw = self._new_raw(e, layer, space, path, block_name)
            pts = self._flatten_curve(e)
            if len(pts) >= 2:
                closed = bool(np.allclose(pts[0], pts[-1])) if len(pts) > 2 else False
                data = {}
                if t == "ELLIPSE":
                    data = {"ratio": float(e.dxf.ratio)}
                self._add_geo(raw, "curve", pts, closed=closed, data=data)
        elif t == "HATCH":
            self._hatch(e, layer, space, path, block_name)
        elif t in ("SOLID", "TRACE", "3DFACE"):
            raw = self._new_raw(e, layer, space, path, block_name)
            vs = []
            for k in ("vtx0", "vtx1", "vtx3", "vtx2"):
                if e.dxf.hasattr(k):
                    v = e.dxf.get(k)
                    if t != "3DFACE":
                        v = e.ocs().to_wcs(v)
                    vs.append((v[0], v[1]))
            uniq = []
            for v in vs:
                if not uniq or math.dist(uniq[-1], v) > 1e-12:
                    uniq.append(v)
            if len(uniq) >= 3:
                self._add_geo(raw, "polygon", uniq, closed=True, data={"filled": t != "3DFACE"})
        elif t == "MLINE":
            self._mline(e, layer, space, path, block_name)
        elif t in ("TEXT", "MTEXT", "ATTRIB"):
            self._text(e, layer, space, path, block_name)
        elif t in ("DIMENSION", "ARC_DIMENSION", "LARGE_RADIAL_DIMENSION"):
            self._dimension(e, layer, space, path, block_name)
        elif t == "POINT":
            raw = self._new_raw(e, layer, space, path, block_name)
            p = e.dxf.location
            self._add_geo(raw, "point", [(p.x, p.y)])

    # ------------------------------------------------------------------ INSERT
    def _insert(self, e, layer, space, path, block_name, depth):
        if depth >= MAX_DEPTH:
            self.d.issues.append({"stage": "ingestion", "severity": "warning",
                                  "message": "Block nesting too deep; truncated", "handle": e.dxf.get("handle")})
            return
        name = e.dxf.name
        if name in self.d.blocks:
            self.d.blocks[name]["insert_count"] = self.d.blocks[name].get("insert_count", 0) + 1
        # MINSERT arrays: expand into individual virtual inserts
        if getattr(e, "mcount", 1) > 1:
            raw_m = self._new_raw(e, layer, space, path, block_name,
                                  extra={"name": name, "minsert": True, "count": e.mcount})
            for vi in e.multi_insert():
                self._insert_single(vi, layer, space, path + (raw_m.id,), block_name, depth, parent_id=raw_m.id)
            return
        self._insert_single(e, layer, space, path, block_name, depth)

    def _insert_single(self, e, layer, space, path, block_name, depth, parent_id=None):
        name = e.dxf.name
        ins = e.dxf.insert
        try:
            ins_w = e.ocs().to_wcs(ins)
        except Exception:
            ins_w = ins
        sx, sy, sz = float(e.dxf.get("xscale", 1)), float(e.dxf.get("yscale", 1)), float(e.dxf.get("zscale", 1))
        rot = float(e.dxf.get("rotation", 0))
        raw = self._new_raw(e, layer, space, path, block_name, parent_id=parent_id,
                            extra={"name": name, "insert": [float(ins_w[0]), float(ins_w[1])],
                                   "rotation": rot, "scale": [sx, sy, sz]})
        raw.raw_coords = [[float(ins_w[0]), float(ins_w[1])]]
        ref = BlockReference(id=raw.id, name=name, layer=layer, insert=(float(ins_w[0]), float(ins_w[1])),
                             rotation=rot, scale=(sx, sy, sz), depth=depth, parent_path=tuple(path))
        self.d.inserts[raw.id] = ref
        before = len(self.d.entities)
        # attributes attached to the INSERT
        for att in getattr(e, "attribs", []):
            self._text(att, self._layer_of(att, layer), space, path + (raw.id,), name)
        try:
            children = list(e.virtual_entities())
        except Exception as exc:
            self.d.issues.append({"stage": "ingestion", "severity": "warning",
                                  "message": f"Could not explode INSERT of block '{name}': {exc}",
                                  "handle": e.dxf.get("handle")})
            children = []
        self._walk(children, space, path + (raw.id,), layer, name, depth + 1)
        ref.child_geo_ids = [g.id for g in self.d.entities[before:]]

    # ------------------------------------------------------------------ curves
    def _arc_geo(self, e, raw: RawEntity, parent: str | None = None, layer: str | None = None):
        c = e.ocs().to_wcs(e.dxf.center)
        r = float(e.dxf.radius)
        sag = max(r * (1 - math.cos(math.radians(ARC_STEP_DEG) / 2)), 1e-9)
        pts = [(p.x, p.y) for p in e.flattening(sag)]
        if len(pts) < 2:
            return None
        cx, cy = c.x, c.y
        mid = pts[len(pts) // 2]
        p0, p1 = pts[0], pts[-1]
        cr = (p0[0] - cx) * (mid[1] - cy) - (p0[1] - cy) * (mid[0] - cx)
        if cr < 0:  # clockwise in WCS (mirrored OCS) -> store CCW
            pts = pts[::-1]
            p0, p1 = p1, p0
        a0 = math.atan2(p0[1] - cy, p0[0] - cx) % (2 * math.pi)
        a1 = math.atan2(p1[1] - cy, p1[0] - cx) % (2 * math.pi)
        sweep = (a1 - a0) % (2 * math.pi)
        if sweep < 1e-9:
            sweep = 2 * math.pi
        return self._add_geo(raw, "arc", pts, data={"center": (cx, cy), "radius": r, "start": a0, "end": a1,
                                                    "sweep": sweep}, parent=parent, layer=layer)

    def _flatten_curve(self, e) -> list[tuple[float, float]]:
        try:
            box = None
            from ezdxf import bbox
            box = bbox.extents([e], fast=True)
            diag = box.size.magnitude if box.has_data else 1.0
        except Exception:
            diag = 1.0
        dist = max(diag * 0.002, 1e-9)
        return [(p.x, p.y) for p in e.flattening(dist)]

    def _polyline(self, e, layer, space, path, block_name):
        t = e.dxftype()
        if t == "POLYLINE" and (e.is_polygon_mesh or e.is_poly_face_mesh):
            self.unsupported["POLYLINE(mesh)"] = self.unsupported.get("POLYLINE(mesh)", 0) + 1
            self._new_raw(e, layer, space, path, block_name)
            return
        closed = bool(e.closed if t == "LWPOLYLINE" else e.is_closed)
        widths: list[float] = []
        extra: dict[str, Any] = {"closed": closed}
        if t == "LWPOLYLINE":
            cw = float(e.dxf.get("const_width", 0) or 0)
            pts_attr = list(e.get_points("xyseb"))
            extra["const_width"] = cw
            extra["n_vertices"] = len(pts_attr)
            widths = [cw] if cw > 0 else sorted({float(p[2]) for p in pts_attr} | {float(p[3]) for p in pts_attr})
            extra["has_bulge"] = any(abs(p[4]) > 1e-12 for p in pts_attr)
        else:
            dsw = float(e.dxf.get("default_start_width", 0) or 0)
            dew = float(e.dxf.get("default_end_width", 0) or 0)
            widths = sorted({dsw, dew})
            extra["n_vertices"] = len(e.vertices)
        raw = self._new_raw(e, layer, space, path, block_name, extra=extra)
        from ezdxf import path as ezpath
        try:
            p = ezpath.make_path(e)
            pts = [(v.x, v.y) for v in p.flattening(self._poly_sagitta(e))]
        except Exception:
            pts = []
        if len(pts) < 2:
            return
        width = max(widths) if widths else 0.0
        uniform_width = len([w for w in widths if w > 0]) <= 1 and width > 0
        poly = self._add_geo(raw, "polyline", pts, closed=closed,
                             data={"width": width if uniform_width else 0.0,
                                   "variable_width": bool(width > 0 and not uniform_width)})
        # explode into parts (LINE / ARC) that keep the link to the parent polyline
        try:
            parts = list(e.virtual_entities())
        except Exception:
            parts = []
        for part in parts:
            pt = part.dxftype()
            if pt == "LINE":
                s, en = part.dxf.start, part.dxf.end
                if math.dist((s.x, s.y), (en.x, en.y)) > 0:
                    self._add_geo(raw, "line", [(s.x, s.y), (en.x, en.y)], parent=poly.id)
            elif pt == "ARC":
                self._arc_geo(part, raw, parent=poly.id)

    def _poly_sagitta(self, e) -> float:
        try:
            from ezdxf import bbox
            box = bbox.extents([e], fast=True)
            diag = box.size.magnitude if box.has_data else 1.0
        except Exception:
            diag = 1.0
        return max(diag * 0.0005, 1e-9)

    # ------------------------------------------------------------------ hatch
    def _hatch(self, e, layer, space, path, block_name):
        pattern = e.dxf.get("pattern_name", "")
        solid = bool(e.dxf.get("solid_fill", 0))
        raw = self._new_raw(e, layer, space, path, block_name,
                            extra={"pattern_name": pattern, "solid_fill": solid,
                                   "associative": bool(e.dxf.get("associative", 0)),
                                   "n_paths": len(e.paths)})
        from ezdxf import path as ezpath
        loops = []
        has_curves = False
        for p in ezpath.from_hatch(e):
            if p.has_curves:
                has_curves = True
            try:
                cv = np.array([(v.x, v.y) for v in p.control_vertices()], dtype=float)
                diag = float(np.linalg.norm(cv.max(axis=0) - cv.min(axis=0))) if len(cv) else 1.0
            except Exception:
                diag = 1.0
            pts = [(v.x, v.y) for v in p.flattening(max(diag * 0.0005, 1e-9))]
            if len(pts) >= 3:
                loops.append(np.asarray(pts, dtype=float))
        if not loops:
            return
        raw.raw_coords = [[float(x), float(y)] for x, y in loops[0][:MAX_RAW_COORDS]]
        self._add_geo(raw, "hatch", None, closed=True,
                      data={"loops": loops, "pattern": pattern, "solid": solid, "has_curves": has_curves})

    # ------------------------------------------------------------------ MLINE
    def _mline(self, e, layer, space, path, block_name):
        verts = [(v.location.x, v.location.y) for v in e.vertices] if hasattr(e, "vertices") else []
        style_offsets = []
        try:
            style_offsets = [float(el.offset) for el in e.style.elements]
        except Exception:
            pass
        scale = float(e.dxf.get("scale_factor", 1.0))
        width = (max(style_offsets) - min(style_offsets)) * scale if len(style_offsets) >= 2 else 0.0
        raw = self._new_raw(e, layer, space, path, block_name,
                            extra={"style": e.dxf.get("style_name", ""), "scale": scale,
                                   "justification": int(e.dxf.get("justification", 0)),
                                   "style_offsets": style_offsets, "closed": bool(e.is_closed)})
        if len(verts) >= 2:
            ml = self._add_geo(raw, "mline", verts, closed=bool(e.is_closed), data={"width": abs(width)})
        else:
            return
        try:
            parts = list(e.virtual_entities())
        except Exception:
            parts = []
        for part in parts:
            if part.dxftype() == "LINE":
                s, en = part.dxf.start, part.dxf.end
                if math.dist((s.x, s.y), (en.x, en.y)) > 0:
                    self._add_geo(raw, "line", [(s.x, s.y), (en.x, en.y)], parent=ml.id,
                                  layer=self._layer_of(part, layer))

    # ------------------------------------------------------------------ text / dimension
    def _text(self, e, layer, space, path, block_name):
        t = e.dxftype()
        if t == "MTEXT":
            txt = e.plain_text()
            ins = e.dxf.insert
            h = float(e.dxf.get("char_height", 1.0))
            rot = float(e.dxf.get("rotation", 0))
        else:
            txt = e.plain_text() if hasattr(e, "plain_text") else e.dxf.text
            ins = e.dxf.insert
            h = float(e.dxf.get("height", 1.0))
            rot = float(e.dxf.get("rotation", 0))
            # aligned text: use align point centre when set
            if e.dxf.hasattr("align_point") and (e.dxf.get("halign", 0) or e.dxf.get("valign", 0)):
                ins = e.dxf.align_point
        try:
            p = e.ocs().to_wcs(ins) if t != "MTEXT" else ins
        except Exception:
            p = ins
        raw = self._new_raw(e, layer, space, path, block_name, extra={"text": txt, "height": h, "rotation": rot})
        self._add_geo(raw, "text", [(p[0], p[1])], data={"text": txt, "height": h, "rotation": rot})

    def _dimension(self, e, layer, space, path, block_name):
        try:
            meas = float(e.get_measurement()) if e.dxftype() == "DIMENSION" else None
        except Exception:
            meas = None
        defs = {}
        for k in ("defpoint", "defpoint2", "defpoint3", "defpoint4", "text_midpoint"):
            if e.dxf.hasattr(k):
                v = e.dxf.get(k)
                defs[k] = (float(v[0]), float(v[1]))
        dimtype = int(e.dxf.get("dimtype", 0)) & 0x0F
        raw = self._new_raw(e, layer, space, path, block_name,
                            extra={"measurement": meas, "dimtype": dimtype, "text": e.dxf.get("text", "")})
        pts = list(defs.values()) or [(0.0, 0.0)]
        self._add_geo(raw, "dimension", pts, data={"measurement": meas, "dimtype": dimtype, "defpoints": defs})


def extract_entities(doc, drawing: Drawing, include_paperspace: bool = False) -> Drawing:
    EntityExtractor(drawing, include_paperspace).extract(doc)
    return drawing
