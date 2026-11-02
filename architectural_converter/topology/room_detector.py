"""Room detection from reconstructed wall topology.

The architectural solid is the union of wall bodies (with openings bridged by their host
walls), centre-line walls (thin topological buffer) and accepted columns, closed by a tiny
morphological closing (snap tolerance) against numeric gaps. Every enclosed hole of the
solid is a *room candidate*; holes are then evaluated - not every enclosed polygon is a room:
tiny regions (shafts / drafting artefacts), narrow gaps (cavities between parallel walls),
unusually large regions and weakly supported regions are rejected or flagged."""
from __future__ import annotations

import re

from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from ..geometry.geometric_utils import inscribed_width, make_valid_polygonal, polygons_of, safe_buffer, robust_union, robust_difference, robust_intersection, bbox_pad, sample_points, near_mask
from ..models.base_entity import Status
from ..models.room import Room


def close_radius(tol, cfg) -> float:
    return max(tol.snap * cfg["tolerances"]["topology_close_factor"], 1e-4)


def wall_solid(walls, columns, tol, cfg, extra=None):
    """Union of wall bodies, each grown by the closing radius BEFORE the union.

    Growing every simple wall polygon individually closes numeric gaps at junctions exactly like
    a morphological closing of the union would, without ever buffering the merged building
    (which can fail with GEOS TopologyExceptions on large drawings). Room polygons are grown
    back by the same radius in ``holes_of``."""
    r = close_radius(tol, cfg)
    polys = []
    topo_w = max(tol.snap * 2, 0.01)
    for w in walls:
        p = w.polygon(topo_w)
        if p is not None and p.area > 0:
            polys.append(p)
    for c in columns:
        if c.status != Status.REJECTED and c.polygon is not None:
            polys.append(c.polygon)
    for p in extra or []:
        polys.append(p)
    if not polys:
        return None
    grown = []
    for p in polys:
        if isinstance(p, Polygon) and len(p.exterior.coords) <= 6 and not p.interiors:
            grown.append(p.buffer(r, join_style=2))            # simple quad: always safe
        else:
            grown.append(safe_buffer(make_valid_polygonal(p), r, join_style=2))
    solid = robust_union(grown)
    return solid


def holes_of(solid, grow: float = 0.0):
    """Enclosed holes of the solid (minus islands inside them), grown back by ``grow``."""
    out = []
    for poly in polygons_of(solid):
        for ring in poly.interiors:
            hp = Polygon(ring)
            inner = robust_intersection(solid, hp)
            if not inner.is_empty and inner.area > 0:
                hp = robust_difference(hp, solid)
            for q in polygons_of(hp):
                if grow > 0:
                    q = safe_buffer(q, grow, join_style=2)
                    q = polygons_of(q)[0] if not isinstance(q, Polygon) and polygons_of(q) else q
                out.append(q)
    return out


class RoomDetector:
    def __init__(self, cfg, ids):
        self.cfg = cfg
        self.ids = ids

    def detect(self, walls, columns, openings, labels, tol, extra=None, prefix="room"):
        vc = self.cfg["validation"]
        solid = wall_solid(walls, columns, tol, self.cfg, extra)
        if solid is None:
            return [], None, {"candidates": 0}
        accepted = [p for p in (w.polygon(max(tol.snap * 2, 0.01)) for w in walls if w.status == Status.ACCEPTED) if p is not None]
        review = [p for p in (w.polygon(max(tol.snap * 2, 0.01)) for w in walls if w.status == Status.NEEDS_REVIEW) if p is not None]
        acc_tree = STRtree(accepted) if accepted else None
        rev_tree = STRtree(review) if review else None
        near_d = tol.snap * 3 + close_radius(tol, self.cfg)
        op_polys = []
        wmap = {w.id: w for w in walls}
        for o in openings:
            if o.host_wall in wmap and o.host_interval:
                w = wmap[o.host_wall]
                from ..geometry.geometric_utils import rect_polygon
                op_polys.append(rect_polygon(w.u, w.c, o.host_interval[0], o.host_interval[1],
                                             (w.thickness or tol.snap * 2) + tol.snap * 4))
        op_tree = STRtree(op_polys) if op_polys else None
        rooms = []
        log = {"candidates": 0, "tiny": 0, "narrow": 0, "large": 0}
        lindex = self.label_index(labels)
        for hp in holes_of(solid, close_radius(tol, self.cfg)):
            log["candidates"] += 1
            area = hp.area
            if area < vc["shaft_min_area_m2"]:
                log["tiny"] += 1
                continue
            r = Room(id=self.ids.new(prefix), polygon=hp)
            width = inscribed_width(hp)
            bnd = hp.exterior
            L = bnd.length
            # how much of the room boundary runs along accepted / uncertain walls / openings
            pts = sample_points(bnd, 0.1)
            m_acc = near_mask(pts, accepted, near_d, acc_tree)
            m_rev = near_mask(pts, review, near_d, rev_tree) & ~m_acc
            m_op = near_mask(pts, op_polys, near_d, op_tree)
            n = max(len(pts), 1)
            f_acc, f_rev, f_op = m_acc.sum() / n, m_rev.sum() / n, m_op.sum() / n
            r.boundary_support = {"accepted_walls": f_acc, "review_walls": f_rev, "openings": f_op,
                                  "inscribed_width": width}
            r.add_evidence("wall_support", min(1.0, f_acc + 0.6 * f_rev + 0.3 * f_op))
            if width < vc["min_room_width_m"]:
                r.add_evidence("shape", 0.0)
                r.veto(f"narrow region ({width:.2f} m): cavity or drafting gap")
                log["narrow"] += 1
            else:
                r.add_evidence("shape", 1.0)
            if area < vc["min_room_area_m2"]:
                r.add_evidence("size_plausibility", 0.4)
                r.classification = "shaft_or_void"
            elif area > vc["max_room_area_m2"]:
                r.add_evidence("size_plausibility", 0.3)
                r.notes.append("unusually large enclosed region")
                log["large"] += 1
            else:
                r.add_evidence("size_plausibility", 1.0)
            self._labels(r, labels, lindex)
            rooms.append(r)
        w = self.cfg["detection"]["weights"]["room"]
        th = self.cfg["detection"]["status_thresholds"]
        for r in rooms:
            r.score(w, th)
        return rooms, solid, log

    @staticmethod
    def label_index(labels):
        pts = [Point(lb.position) for lb in labels]
        return (STRtree(pts), pts) if pts else (None, [])

    def _labels(self, r: Room, labels, index=None):
        names = []
        cc = self.cfg["classification"]
        if index is None:
            index = self.label_index(labels)
        tree, pts = index
        inside = sorted(int(k) for k in tree.query(r.polygon, predicate="contains")) if tree is not None else []
        for k in inside:
            lb = labels[k]
            if lb.category not in ("room_name", "room_number", "area"):
                continue
            if True:
                r.label_ids.append(lb.id)
                r.source_entities.add(lb.raw_id)
                if lb.category in ("room_name", "room_number"):
                    first = lb.text.splitlines()[0].strip()
                    names.append(first)
                    r.label_texts.append(lb.text)
                if lb.area_value is not None:
                    r.area_text = lb.area_value
        if names:
            r.name = names[0]
            r.add_evidence("label", 1.0)
            if len(names) > 1:
                r.notes.append(f"multiple room labels ({', '.join(names)}): possible missing boundary")
        if r.area_text:
            rel = abs(r.area_text - r.area) / max(r.area, 1e-9)
            r.add_evidence("area_text_match", max(0.0, 1.0 - rel / cc["area_match_tol"]))
