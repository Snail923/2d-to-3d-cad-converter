"""Approach 2 extraction: SymPointV2 labels -> wall / door / window / column / stair candidates.

The model decides WHAT every line is; this module only turns the labelled lines into the
parametric objects approach 1's build stages expect (no rule of approach 1's detectors runs):

* walls     : pairs of parallel wall-labelled strokes (straight) and concentric wall-labelled arcs
              (curved); the band between two faces is the wall, its width the thickness. Bands that
              touch side by side (a middle line inside a wall) are merged.
* columns   : SymPointV2 has no column class (FloorPlanCAD draws columns as wall). Small compact
              closed wall-labelled outlines, and isolated short wall pieces as thick as they are long,
              become columns.
* doors     : model instances of a door class (else clusters of door-labelled lines). The opening
              chord is the extent of the symbol along the nearest wall: a swing arc projects onto the
              wall from the hinge to the closed-leaf end, the open leaf onto the hinge.
* windows   : the same for window / bay window / blind window and curtain wall (on real drawings the
              model often calls window glazing in a wall "curtain wall").
* stairs    : clusters of stair-labelled lines; the treads are the parallel lines of the dominant
              direction, their spacing the going.

Every candidate keeps its source DXF entities and the model's class and score as evidence."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.strtree import STRtree

from architectural_converter.geometry.geometric_utils import canonical_dir, dot, frame_point, mrr_dims
from architectural_converter.models.column import Column
from architectural_converter.models.opening import Opening
from architectural_converter.models.stair import Stair
from architectural_converter.models.wall import Wall

from .classes import CLASSES, GROUP_OF

DOOR_SUBTYPE = {"single door": "single_swing", "double door": "double_swing", "sliding door": "sliding",
                "folding door": "folding", "revolving door": "revolving", "rolling door": "rolling"}
WINDOW_SUBTYPE = {"window": "window", "bay window": "bay_window", "blind window": "blind_window",
                  "curtain wall": "curtain_wall"}
WINDOW_GROUPS = ("window", "curtain_wall")
ELEMENT_LAYER_KINDS = ("wall", "balcony", "window", "door", "column", "stair", "furniture")


def _n(u):
    return (-u[1], u[0])


def _circle(P):
    """Centre and radius of the circle through the first, middle and last point of an arc."""
    (x1, y1), (x2, y2), (x3, y3) = P[0], P[len(P) // 2], P[-1]
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return None
    ux = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / d
    uy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / d
    return (ux, uy), math.hypot(x1 - ux, y1 - uy)


class Extractor:
    def __init__(self, drawing, ctx, prims, labels, cfg, ids):
        self.d, self.ctx, self.prims, self.labels = drawing, ctx, prims, labels
        self.c = cfg["build"]
        self.ids = ids
        self.tol = ctx.tol
        self.ent = labels.get("entities", {})
        lab = labels.get("primitives", {})
        self.plab = {int(k): v for k, v in lab.items()}
        self.log: dict = {}

    # ------------------------------------------------------------------ labels
    def group_of_raw(self, rid):
        e = self.ent.get(rid)
        if e is None:
            return None, 0.0
        return GROUP_OF.get(e["label"], "other"), float(e["score"])

    def stroke_label(self, st):
        """(share of the stroke's DXF entities labelled wall, mean wall score)."""
        n, w, sc = 0, 0, []
        for rid in st.raw_ids:
            n += 1
            g, s = self.group_of_raw(rid)
            if g == "wall":
                w += 1
                sc.append(s)
        return (w / n if n else 0.0), (float(np.mean(sc)) if sc else 0.0)

    # ------------------------------------------------------------------ main
    def run(self) -> dict:
        walls, wall_faces = self.straight_walls()
        walls += self.arc_walls()
        walls += self.railings()
        columns = self.columns(walls)
        doors, windows = self.openings(walls)
        stairs = self.stairs()
        self.log.update({"walls": len(walls), "columns": len(columns), "doors": len(doors),
                         "windows": len(windows), "stairs": len(stairs)})
        return {"sympoint_walls": walls, "sympoint_columns": columns, "sympoint_doors": doors,
                "sympoint_windows": windows, "sympoint_stairs": stairs}

    # ------------------------------------------------------------------ straight walls
    def straight_walls(self):
        c = self.c
        tmin, tmax = c["wall_thickness_m"]
        min_share = c["min_wall_share"]
        strokes = self.ctx.strokes
        W = []
        for k, st in enumerate(strokes):
            if st.length < c["min_face_length_m"]:
                continue
            share, sc = self.stroke_label(st)
            if share >= min_share:
                W.append((k, sc))
        self.log["wall_strokes"] = len(W)
        if not W:
            return [], {}
        score_of = dict(W)
        idx = [k for k, _ in W]
        self._wall_idx = idx
        tree = STRtree([strokes[k].line().buffer(tmax) for k in idx])
        ang_tol = math.radians(c["parallel_deg"])
        pairs = []
        for a_i, i in enumerate(idx):
            si = strokes[i]
            ni = _n(si.u)
            for b_j in tree.query(si.line()):
                j = idx[int(b_j)]
                if j <= i:
                    continue
                sj = strokes[j]
                if abs(si.u[0] * sj.u[1] - si.u[1] * sj.u[0]) > math.sin(ang_tol):
                    continue
                cj0, cj1 = dot(sj.p0, ni), dot(sj.p1, ni)
                d = abs((cj0 + cj1) / 2 - si.c)
                if not (tmin <= d <= tmax):
                    continue
                pj = sorted((dot(sj.p0, si.u), dot(sj.p1, si.u)))
                a, b = max(si.s0, pj[0]), min(si.s1, pj[1])
                if b - a < max(c["min_overlap_m"], 0.3 * d):
                    continue
                pairs.append((d, -(b - a), i, j, a, b, (cj0 + cj1) / 2))
        pairs.sort()
        used: dict[int, list] = {}
        walls = []

        def free(k, a, b, u):
            st = strokes[k]
            sg = 1.0 if dot(st.u, u) >= 0 else -1.0
            iv = [(min(sg * x, sg * y), max(sg * x, sg * y)) for x, y in used.get(k, [])]
            out, cur = [], a
            for x, y in sorted(iv):
                if y <= cur or x >= b:
                    continue
                if x > cur:
                    out.append((cur, x))
                cur = max(cur, y)
            if cur < b:
                out.append((cur, b))
            return out

        def mark(k, a, b, u):
            st = strokes[k]
            sg = 1.0 if dot(st.u, u) >= 0 else -1.0
            used.setdefault(k, []).append((min(sg * a, sg * b), max(sg * a, sg * b)))

        for d, _, i, j, a, b, cj in pairs:
            si = strokes[i]
            u = si.u
            for a2, b2 in free(i, a, b, u):
                for a3, b3 in free(j, a2, b2, u):
                    if b3 - a3 < max(c["min_overlap_m"], 0.3 * d):
                        continue
                    mark(i, a3, b3, u)
                    mark(j, a3, b3, u)
                    w = Wall(id="", kind="straight", u=u, c=(si.c + cj) / 2, s0=a3, s1=b3, thickness=d,
                             thickness_source="measured", detected_interval=(a3, b3))
                    for st in (si, strokes[j]):
                        sg = 1.0 if dot(st.u, u) >= 0 else -1.0
                        lo, hi = sorted((sg * a3, sg * b3))
                        for sid, rid in st.members_in(lo, hi):
                            w.face_segments.add(sid)
                            w.source_entities.add(rid)
                    w._faces = (i, j)
                    w._short = (b3 - a3) < c["short_piece_factor"] * d
                    w.add_evidence("sympoint_label", (score_of[i] + score_of[j]) / 2)
                    w.add_evidence("sympoint_faces", 1.0)
                    w.add_evidence("sympoint_thickness", 1.0 if c["plausible_thickness_m"][0] <= d <=
                                   c["plausible_thickness_m"][1] else 0.5)
                    w.detection_methods.add("sympoint_wall_faces")
                    walls.append(w)
        walls = self._merge_bands(walls)
        walls += self._single_line_walls(idx, used, score_of)
        for w in walls:
            w.id = self.ids.new("wall")
        unpaired = sum(1 for k in idx if k not in used)
        self.log["unpaired_wall_strokes"] = unpaired
        return walls, used

    def _single_line_walls(self, idx, used, score_of):
        """Wall-labelled lines without a second face: wide polylines (width = thickness) always;
        plain lines only when the drawing is a single-line one (most wall length unpaired)."""
        c = self.c
        strokes = self.ctx.strokes
        width_of = {}
        for g in self.d.entities:
            if g.kind == "polyline" and (g.data.get("width") or 0) >= c["wall_thickness_m"][0] \
                    and not g.data.get("variable_width"):
                width_of[g.raw_id] = float(g.data["width"])
        tot = sum(strokes[k].length for k in idx)
        paired = sum(min(strokes[k].length, sum(b - a for a, b in used.get(k, []))) for k in idx)
        single_line_drawing = tot > 0 and paired / tot < c["single_line_max_paired_share"]
        self.log["paired_wall_share"] = round(paired / tot, 3) if tot else 0.0
        out = []
        for k in idx:
            st = strokes[k]
            if k in used or st.length < c["single_line_min_length_m"]:
                continue
            wid = [width_of[r] for r in st.raw_ids if r in width_of]
            if wid:
                w = Wall(id="", kind="straight", u=st.u, c=st.c, s0=st.s0, s1=st.s1, thickness=float(np.median(wid)),
                         thickness_source="polyline_width", detected_interval=(st.s0, st.s1))
                w.add_evidence("sympoint_faces", 1.0)
                w.detection_methods.add("sympoint_wide_polyline")
            elif self._layer_kind(st) == "balcony":
                # the model calls it structure, its layer says balcony: a balcony edge drawn as one line
                w = Wall(id="", kind="centerline", u=st.u, c=st.c, s0=st.s0, s1=st.s1, thickness=None,
                         thickness_source="unknown", detected_interval=(st.s0, st.s1))
                w.add_evidence("sympoint_faces", 0.5)
                w.add_evidence("railing_label", score_of[k])
                w.detection_methods.add("sympoint_balcony_edge")
            elif single_line_drawing:
                w = Wall(id="", kind="centerline", u=st.u, c=st.c, s0=st.s0, s1=st.s1, thickness=None,
                         thickness_source="unknown", detected_interval=(st.s0, st.s1))
                w.add_evidence("sympoint_faces", 0.5)
                w.detection_methods.add("sympoint_single_line")
            else:
                continue
            w.face_segments = set(st.members)
            w.source_entities = set(st.raw_ids)
            w.add_evidence("sympoint_label", score_of[k])
            out.append(w)
        self.log["single_line_walls"] = len(out)
        return out

    def _layer_kind(self, st):
        """Building element a stroke's layer names (layer semantics), None when unclear."""
        kinds = set()
        for ly in st.layers or {st.layer}:
            p = self.ctx.layers.get(ly)
            k = p.primary if p is not None else None
            if k in ELEMENT_LAYER_KINDS:
                kinds.add(k)
        return kinds.pop() if len(kinds) == 1 else None

    def _merge_bands(self, walls):
        """Walls side by side (a middle line inside one wall) -> one band."""
        snap = max(self.tol.snap * 2, 0.01)
        changed = True
        while changed:
            changed = False
            out = []
            for w in walls:
                for v in out:
                    if abs(abs(dot(v.u, w.u)) - 1) > 1e-4:
                        continue
                    sg = 1.0 if dot(w.u, v.u) >= 0 else -1.0
                    wc = w.c * sg
                    ws0, ws1 = sorted((sg * w.s0, sg * w.s1))
                    gap = abs(wc - v.c) - (w.thickness + v.thickness) / 2
                    ov = min(ws1, v.s1) - max(ws0, v.s0)
                    if abs(gap) <= snap and ov >= 0.8 * min(ws1 - ws0, v.s1 - v.s0):
                        lo = min(wc - w.thickness / 2, v.c - v.thickness / 2)
                        hi = max(wc + w.thickness / 2, v.c + v.thickness / 2)
                        v.c, v.thickness = (lo + hi) / 2, hi - lo
                        v.s0, v.s1 = min(v.s0, ws0), max(v.s1, ws1)
                        v.detected_interval = (v.s0, v.s1)
                        v.absorb(w)
                        v.notes.append("merged with a parallel band (line inside the wall)")
                        changed = True
                        break
                else:
                    out.append(w)
            walls = out
        return walls

    # ------------------------------------------------------------------ balcony railings
    def railings(self):
        """Railing-labelled lines (balcony / gallery edges): pairs of parallel lines a few cm apart,
        else single lines. They carry ``railing_label`` evidence, which the shared balcony detection
        reads (a balcony wall = a low parapet with glass above it in 3D)."""
        c = self.c
        tmin, tmax = c["railing_thickness_m"]
        strokes = self.ctx.strokes
        R = []
        for k, st in enumerate(strokes):
            if st.length < c["railing_min_length_m"]:
                continue
            n, hit, sc = 0, 0, []
            for rid in st.raw_ids:
                n += 1
                g, s_ = self.group_of_raw(rid)
                if g == "railing":
                    hit += 1
                    sc.append(s_)
            if n and hit / n >= c["min_wall_share"]:
                R.append((k, float(np.mean(sc))))
        pairs = []
        for a in range(len(R)):
            for b in range(a + 1, len(R)):
                i, j = R[a][0], R[b][0]
                si, sj = strokes[i], strokes[j]
                if abs(si.u[0] * sj.u[1] - si.u[1] * sj.u[0]) > math.sin(math.radians(c["parallel_deg"])):
                    continue
                d = abs(dot(((sj.p0[0] + sj.p1[0]) / 2, (sj.p0[1] + sj.p1[1]) / 2), _n(si.u)) - si.c)
                pj = sorted((dot(sj.p0, si.u), dot(sj.p1, si.u)))
                lo, hi = max(si.s0, pj[0]), min(si.s1, pj[1])
                if tmin <= d <= tmax and hi - lo >= 0.5 * min(si.length, sj.length):
                    pairs.append((d, a, b, lo, hi))
        pairs.sort()
        used, out = set(), []
        score = dict(R)
        for d, a, b, lo, hi in pairs:
            if a in used or b in used:
                continue
            used |= {a, b}
            si, sj = strokes[R[a][0]], strokes[R[b][0]]
            cj = dot(((sj.p0[0] + sj.p1[0]) / 2, (sj.p0[1] + sj.p1[1]) / 2), _n(si.u))
            w = Wall(id="", kind="straight", u=si.u, c=(si.c + cj) / 2, s0=lo, s1=hi, thickness=d,
                     thickness_source="measured", detected_interval=(lo, hi))
            w.source_entities = set(si.raw_ids) | set(sj.raw_ids)
            w.add_evidence("sympoint_faces", 1.0)
            w.add_evidence("sympoint_label", (score[R[a][0]] + score[R[b][0]]) / 2)
            w.add_evidence("railing_label", (score[R[a][0]] + score[R[b][0]]) / 2)
            out.append(w)
        for a, (k, sc) in enumerate(R):
            if a in used:
                continue
            st = strokes[k]
            w = Wall(id="", kind="centerline", u=st.u, c=st.c, s0=st.s0, s1=st.s1, thickness=None,
                     thickness_source="unknown", detected_interval=(st.s0, st.s1))
            w.source_entities = set(st.raw_ids)
            w.add_evidence("sympoint_faces", 0.5)
            w.add_evidence("sympoint_label", sc)
            w.add_evidence("railing_label", sc)
            out.append(w)
        for w in out:
            w.id = self.ids.new("wall")
            w.detection_methods.add("sympoint_railing")
            w.notes.append("SymPointV2: railing (balcony / gallery edge)")
        self.log["railings"] = len(out)
        return out

    # ------------------------------------------------------------------ curved walls
    def arc_walls(self):
        c = self.c
        tmin, tmax = c["wall_thickness_m"]
        arcs = [a for a in self.ctx.arcs if self.group_of_raw(a.raw_id)[0] == "wall"
                and a.radius * a.sweep >= c["min_face_length_m"]]
        out, used = [], set()
        for i, A in enumerate(arcs):
            if i in used:
                continue
            best = None
            for j in range(i + 1, len(arcs)):
                B = arcs[j]
                if j in used or math.dist(A.center, B.center) > max(0.02, self.tol.snap * 2):
                    continue
                dr = abs(A.radius - B.radius)
                if not (tmin <= dr <= tmax):
                    continue
                b0 = (B.start - A.start) % (2 * math.pi)
                if b0 > A.sweep:                  # B starts before A (wraps through A's start)
                    b0 -= 2 * math.pi
                lo, hi = max(0.0, b0), min(A.sweep, b0 + B.sweep)
                if hi - lo <= 0.1:
                    continue
                if best is None or dr < best[0]:
                    best = (dr, j, lo, hi)
            if best is None:
                continue
            dr, j, lo, hi = best
            used |= {i, j}
            B = arcs[j]
            r = (A.radius + B.radius) / 2
            w = Wall(id=self.ids.new("wall"), kind="arc", thickness=dr, thickness_source="measured",
                     arc={"center_x": A.center[0], "center_y": A.center[1], "radius": r,
                          "start": (A.start + lo) % (2 * math.pi), "end": (A.start + hi) % (2 * math.pi)})
            w.source_entities |= {A.raw_id, B.raw_id}
            w.add_evidence("sympoint_label", (self.group_of_raw(A.raw_id)[1] + self.group_of_raw(B.raw_id)[1]) / 2)
            w.add_evidence("sympoint_faces", 1.0)
            w.add_evidence("sympoint_thickness", 1.0)
            w.detection_methods.add("sympoint_arc_faces")
            out.append(w)
        self.log["arc_walls"] = len(out)
        return out

    # ------------------------------------------------------------------ columns
    def columns(self, walls):
        c = self.c
        lo, hi = c["column_size_m"]
        cols = []
        for g in self.d.entities:
            if not g.valid or g.points is None or len(g.points) < 3:
                continue
            closed = g.closed or g.kind in ("circle", "hatch") or np.allclose(g.points[0], g.points[-1])
            if not closed or self.group_of_raw(g.raw_id)[0] != "wall":
                continue
            try:
                poly = Polygon(g.points).buffer(0)
            except Exception:  # noqa: BLE001
                continue
            if poly.is_empty or poly.area <= 0:
                continue
            w_, L, _, _ = mrr_dims(poly)
            if lo <= w_ and L <= hi and L <= 3 * w_ and poly.area >= 0.6 * w_ * L * (0.78 if g.kind == "circle" else 1):
                cols.append(self._column(poly, {g.raw_id}, "circular" if g.kind == "circle" else "rectangular"))
        cols = self._dedupe_columns(cols)
        # pieces shorter than they are thick: the four sides of a column outline drawn with lines
        # (closed by wall lines at both ends), or the two end caps of walls across an opening gap
        # (open at both ends) which are not a wall at all
        keep, caps = [], 0
        long_walls = [w for w in walls if w.kind == "straight" and not getattr(w, "_short", False)]
        # a closed outline that is part of a wall line (a pier between two openings, drawn as its own
        # closed polyline) is a wall, not a column
        piers = 0
        for cc in list(cols):
            inside = [w for w in long_walls if cc.polygon.buffer(0.01).contains(w.polygon())]
            if inside and any(self._in_wall_line(w, [v for v in long_walls if v is not w]) for w in inside):
                cols.remove(cc)
                piers += 1
        self.log["outline_piers_kept_as_walls"] = piers
        for w in walls:
            if w.kind == "straight" and getattr(w, "_short", False):
                ends = self._closed_ends(w)
                if ends == 0:                  # end caps of two walls facing each other across a gap
                    caps += 1
                    continue
                if ends == 2 and self._covers_pier(w, long_walls):
                    caps += 1                  # the cross-pairing of a pier's own outline
                    continue
                if ends == 2 and not self._in_wall_line(w, long_walls) and \
                        lo <= min(w.thickness, w.length) and max(w.thickness, w.length) <= hi:
                    poly = w.polygon()
                    if not any(cc.polygon.buffer(0.01).intersects(poly) for cc in cols):
                        cols.append(self._column(poly, set(w.source_entities), "rectangular"))
                    continue
                keep.append(w)                 # a short wall nib / pier
                continue
            if w.kind == "straight" and any(cc.polygon.buffer(0.01).contains(w.polygon()) for cc in cols):
                continue                     # the two sides of a column outline paired as a "wall"
            keep.append(w)
        self.log["end_cap_pairs_dropped"] = caps
        walls[:] = keep
        self.log["columns"] = len(cols)
        return cols

    def _closed_ends(self, w):
        """Wall-labelled lines across the band at both ends of a short piece."""
        strokes = self.ctx.strokes
        n = _n(w.u)
        tol = max(self.tol.snap * 2, 0.01)
        lo, hi = w.c - w.thickness / 2, w.c + w.thickness / 2
        ends = {0: [], 1: []}
        band = w.polygon().buffer(tol)
        for k in self._wall_idx:
            st = strokes[k]
            if abs(dot(st.u, w.u)) > 0.05 or not band.intersects(st.line()):
                continue
            s_mid = dot(((st.p0[0] + st.p1[0]) / 2, (st.p0[1] + st.p1[1]) / 2), w.u)
            a, b = sorted((dot(st.p0, n), dot(st.p1, n)))
            cover = max(0.0, min(b, hi) - max(a, lo)) / max(hi - lo, 1e-9)
            for e, s_end in ((0, w.s0), (1, w.s1)):
                if abs(s_mid - s_end) <= tol and cover >= 0.8:
                    ends[e].append(k)
        return int(bool(ends[0])) + int(bool(ends[1]))

    def _covers_pier(self, w, long_walls):
        poly = w.polygon()
        for v in long_walls:
            pv = v.polygon()
            if pv is not None and pv.intersection(poly).area >= 0.5 * poly.area and \
                    self._in_wall_line(v, [x for x in long_walls if x is not v]):
                return True
        return False

    def _in_wall_line(self, w, long_walls, reach=4.0):
        """A pier between two openings: longer walls continue in the same band."""
        tol = max(self.tol.snap * 2, 0.02)
        for v in long_walls:
            if abs(abs(dot(v.u, w.u)) - 1) > 1e-3:
                continue
            sg = 1.0 if dot(v.u, w.u) >= 0 else -1.0
            if abs(sg * v.c - w.c) > tol or abs(v.thickness - w.thickness) > max(tol, 0.1 * w.thickness):
                continue
            a, b = sorted((sg * v.s0, sg * v.s1))
            if max(a - w.s1, w.s0 - b) <= reach:
                return True
        return False

    def _column(self, poly, src, shape):
        w_, L, _, _ = mrr_dims(poly)
        col = Column(id="", polygon=poly, shape=shape, size=(w_, L))
        col.source_entities |= src
        col.add_evidence("sympoint_label", float(np.mean([self.group_of_raw(r)[1] for r in src] or [0.0])))
        col.add_evidence("size_plausibility", 1.0)
        col.detection_methods.add("sympoint_column")
        return col

    def _dedupe_columns(self, cols):
        out = []
        for cc in sorted(cols, key=lambda x: -x.polygon.area):
            if any(o.polygon.buffer(0.01).contains(cc.polygon.centroid) for o in out):
                for o in out:
                    if o.polygon.buffer(0.01).contains(cc.polygon.centroid):
                        o.source_entities |= cc.source_entities
                continue
            out.append(cc)
        for cc in out:
            cc.id = self.ids.new("column")
        return out

    # ------------------------------------------------------------------ openings
    def _clusters(self, groups, gap, inst_min):
        """[(primitive indices, class names by length)] for primitives whose class is in ``groups``:
        model instances first, then proximity clusters of the remaining labelled primitives."""
        cls_of = {k: CLASSES[v[0]] for k, v in self.plab.items()}
        mine = {k for k, cname in cls_of.items() if GROUP_OF.get(cname) in groups}
        out, taken = [], set()
        for inst in self.labels.get("instances", []):
            if GROUP_OF.get(inst["label"]) not in groups or inst["score"] < inst_min:
                continue
            ks = [k for k in inst.get("pidx", []) if k not in taken and k < len(self.prims)]
            if ks:
                taken |= set(ks)
                out.append(ks)
        rest = sorted(mine - taken)
        if rest:
            geoms = [LineString(self.prims[k][5]) if len(self.prims[k][5]) >= 2 else Point(self.prims[k][5][0])
                     for k in rest]
            tree = STRtree(geoms)
            parent = list(range(len(rest)))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for a, g in enumerate(geoms):
                for b in tree.query(g.buffer(gap)):
                    b = int(b)
                    if b > a and geoms[b].distance(g) <= gap:
                        parent[find(a)] = find(b)
            comp: dict[int, list] = {}
            for a in range(len(rest)):
                comp.setdefault(find(a), []).append(rest[a])
            out += list(comp.values())
        res = []
        for ks in out:
            by: dict[str, float] = {}
            for k in ks:
                if k in cls_of:
                    by[cls_of[k]] = by.get(cls_of[k], 0.0) + self.prims[k][4]
            res.append((ks, by))
        return res

    def _host_frame(self, pts, walls, tree, search):
        """Nearest straight wall to a symbol: (wall, distance) or (None, inf)."""
        hull = MultiPoint([tuple(p) for p in pts]).convex_hull
        best = (None, math.inf)
        for k in tree.query(hull.buffer(search)):
            w = walls[int(k)]
            d = w.polygon(self.c["nominal_thickness_m"]).distance(hull)
            if d < best[1]:
                best = (w, d)
        return best

    def _swing_arcs(self, ks):
        """[(primitive index, centre, radius, end a, end b)] of the swing arcs among ``ks``."""
        out = []
        lo, hi = self.c["door_width_m"]
        for k in ks:
            if self.prims[k][2] != 1 or len(self.prims[k][5]) < 3:
                continue
            P = np.asarray(self.prims[k][5], float)
            cr = _circle(P)
            if cr is None or not (0.5 * lo <= cr[1] <= hi):
                continue
            out.append((k, cr[0], cr[1], tuple(P[0]), tuple(P[-1])))
        return out

    def _split_doors(self, ks):
        """One door per swing arc (two arcs meeting leaf to leaf = one double door); every other
        line goes to the nearest arc."""
        arcs = self._swing_arcs(ks)
        if len(arcs) <= 1:
            return [(ks, arcs)]
        groups, used = [], set()
        for i, A in enumerate(arcs):
            if i in used:
                continue
            g = [A]
            used.add(i)
            for j in range(i + 1, len(arcs)):
                B = arcs[j]
                if j in used:
                    continue
                if abs(math.dist(A[1], B[1]) - (A[2] + B[2])) <= 0.15 * (A[2] + B[2]) and \
                        min(math.dist(p, q) for p in A[3:] for q in B[3:]) <= 0.1 * (A[2] + B[2]):
                    g.append(B)
                    used.add(j)
                    break
            groups.append(g)
        members = [[a[0] for a in g] for g in groups]
        arc_ids = {a[0] for a in arcs}
        for k in ks:
            if k in arc_ids:
                continue
            P = np.asarray(self.prims[k][5], float)
            best = min(range(len(groups)), key=lambda gi: min(
                float(np.min(np.linalg.norm(P - np.asarray(a[1]), axis=1))) for a in groups[gi]))
            members[best].append(k)
        return list(zip(members, groups))

    def _swing_chord(self, arcs, straight, tree):
        """(wall, s0, s1) of a swing door: hinge = arc centre, the closed leaf runs along the wall."""
        if len(arcs) == 2:
            (_, C1, r1, *_), (_, C2, r2, *_) = arcs
            mid = ((C1[0] + C2[0]) / 2, (C1[1] + C2[1]) / 2)
            d = (C2[0] - C1[0], C2[1] - C1[1])
            cands = [(C1, C2, d, Point(mid))]
        else:
            _, C, r, Ea, Eb = arcs[0]
            cands = [(C, E, (E[0] - C[0], E[1] - C[1]), Point(C)) for E in (Ea, Eb)]
        best = None
        for P0, P1, d, probe in cands:
            L = math.hypot(*d)
            if L <= 0:
                continue
            for k in tree.query(probe.buffer(self.c["host_search_m"])):
                w = straight[int(k)]
                if abs(dot(d, w.u)) / L < 0.95:
                    continue
                off = abs(dot(P0, _n(w.u)) - w.c)
                if off > (w.thickness or self.c["nominal_thickness_m"]) / 2 + 0.1:
                    continue
                s0, s1 = sorted((dot(P0, w.u), dot(P1, w.u)))
                dist = max(0.0, w.s0 - s1, s0 - w.s1)            # 0 when the chord touches the wall end
                # the closed leaf spans the opening (no wall under it); the open leaf usually rests
                # along a wall face: prefer the direction whose chord does not lie on wall material
                over = max(0.0, min(s1, w.s1) - max(s0, w.s0)) / max(s1 - s0, 1e-9)
                key = (over > 0.5, dist, off)
                if best is None or key < best[0]:
                    best = (key, w, s0, s1)
        return None if best is None else best[1:]

    def openings(self, walls):
        c = self.c
        straight = [w for w in walls if w.kind in ("straight", "centerline")]
        tree = STRtree([w.polygon(c["nominal_thickness_m"]) for w in straight]) if straight else None
        doors, windows = [], []
        skipped = {"no_wall_nearby": 0, "implausible_width": 0}
        for kind, groups, (wlo, whi) in (("door", ("door", "opening"), c["door_width_m"]),
                                         ("window", WINDOW_GROUPS, c["window_width_m"])):
            units = []
            for ks, by in self._clusters(groups, c["symbol_cluster_gap_m"], c["min_instance_score"]):
                if kind == "door":
                    units += [(sub, arcs) for sub, arcs in self._split_doors(ks)]
                else:
                    units.append((ks, []))
            for ks, arcs in units:
                by: dict[str, float] = {}
                for k in ks:
                    if k in self.plab:
                        cn = CLASSES[self.plab[k][0]]
                        by[cn] = by.get(cn, 0.0) + self.prims[k][4]
                if not by:
                    continue
                best_cls = max(by, key=by.get)
                pts = np.vstack([self.prims[k][5] for k in ks])
                host, chord = None, None
                if arcs and tree is not None:
                    sw = self._swing_chord(arcs, straight, tree)
                    if sw is not None:
                        host, a, b = sw
                        chord = (a, b)
                if host is None:
                    host, dist = (None, math.inf) if tree is None else self._host_frame(pts, straight, tree, c["host_search_m"])
                    if host is None or dist > c["host_search_m"]:
                        if kind == "window" or not c["unhosted_doors"]:
                            skipped["no_wall_nearby"] += 1
                            continue
                if host is not None:
                    u, cc, depth = host.u, host.c, host.thickness
                else:
                    wd, L, u, cen = mrr_dims(MultiPoint([tuple(p) for p in pts]).convex_hull)
                    cc, depth = dot(cen, _n(u)), wd
                if chord is None:
                    sv = pts @ np.asarray(u, float)
                    chord = (float(sv.min()), float(sv.max()))
                a, b = float(chord[0]), float(chord[1])
                width = b - a
                if not (wlo <= width <= whi):
                    skipped["implausible_width"] += 1
                    continue
                score = float(np.mean([self.plab[k][1] for k in ks if k in self.plab] or [0.0]))
                o = Opening(id="", type=kind,
                            subtype=(DOOR_SUBTYPE if kind == "door" else WINDOW_SUBTYPE).get(best_cls, best_cls.replace(" ", "_")),
                            p0=frame_point(u, a, cc), p1=frame_point(u, b, cc), width=width, depth=depth)
                if arcs:
                    o.hinge = tuple(arcs[0][1]) if len(arcs) == 1 else None
                    o.swing = {"leaves": len(arcs), "radius": round(float(arcs[0][2]), 4)}
                    if len(arcs) == 2:
                        o.subtype = "double_swing"
                o.footprint = MultiPoint([tuple(p) for p in pts]).convex_hull
                o.source_entities |= {self.prims[k][0] for k in ks}
                o.add_evidence("sympoint_label", score)
                o.add_evidence("symbol", score)               # the model recognised the symbol itself
                o.add_evidence("width_plausibility", 1.0)
                o.detection_methods.add(f"sympoint_{kind}")
                o.notes.append(f"SymPointV2: {best_cls} ({len(ks)} lines, score {score:.2f})")
                o._host_ref = host
                (doors if kind == "door" else windows).append(o)
        doors, md = self._merge_openings(doors)
        windows, mw = self._merge_openings(windows)
        for o in doors + windows:
            o.id = self.ids.new(o.type)
        skipped["merged_overlapping"] = md + mw
        self.log["openings_skipped"] = skipped
        return doors, windows

    def _merge_openings(self, ops):
        """Pieces of one opening (window glazing lines split into several clusters) -> one opening."""
        out, merged = [], 0
        for o in sorted(ops, key=lambda x: -x.width):
            w = o._host_ref
            for v in out:
                if w is None or v._host_ref is not w:
                    continue
                a0, a1 = sorted((dot(o.p0, w.u), dot(o.p1, w.u)))
                b0, b1 = sorted((dot(v.p0, w.u), dot(v.p1, w.u)))
                if min(a1, b1) - max(a0, b0) >= -0.02:
                    lo, hi = min(a0, b0), max(a1, b1)
                    v.p0, v.p1, v.width = frame_point(w.u, lo, w.c), frame_point(w.u, hi, w.c), float(hi - lo)
                    v.absorb(o)
                    merged += 1
                    break
            else:
                out.append(o)
        return out, merged

    # ------------------------------------------------------------------ stairs
    def stairs(self):
        c = self.c
        out = []
        for ks, by in self._clusters(("stair",), c["stair_cluster_gap_m"], c["min_instance_score"]):
            lines = [(self.prims[k][5][0], self.prims[k][5][-1], self.prims[k][4]) for k in ks
                     if self.prims[k][2] == 0 and self.prims[k][4] >= c["min_tread_m"]]
            if len(lines) < c["min_treads"]:
                continue
            hist: dict[int, float] = {}
            for p, q, L in lines:
                ang = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 180
                b = int(round(ang)) % 180
                hist[b] = hist.get(b, 0.0) + L
            dom = math.radians(max(hist, key=hist.get))
            u = canonical_dir(math.cos(dom), math.sin(dom))
            run = _n(u)
            treads = []
            for p, q, L in lines:
                d = np.subtract(q, p)
                if abs(d[0] * u[1] - d[1] * u[0]) / max(L, 1e-9) > math.sin(math.radians(5)):
                    continue
                treads.append((dot(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2), run), dot(p, u), dot(q, u)))
            treads.sort()
            offs, spans = [], []
            for y, x0, x1 in treads:
                if offs and y - offs[-1] < 0.05:
                    continue
                offs.append(y)
                spans.append(sorted((x0, x1)))
            if len(offs) < c["min_treads"]:
                continue
            going = float(np.median(np.diff(offs)))
            if not (c["going_m"][0] <= going <= c["going_m"][1]):
                continue
            w0 = float(np.median([s[0] for s in spans]))
            w1 = float(np.median([s[1] for s in spans]))
            o0, y0 = w0, offs[0]
            origin = (u[0] * o0 + run[0] * y0, u[1] * o0 + run[1] * y0)
            rel = [y - y0 for y in offs]
            st = Stair(id=self.ids.new("stair"), origin=origin, u=u, run=run, width=w1 - w0, going=going,
                       tread_offsets=rel, tread_span=(0.0, w1 - w0))
            pts = [(origin[0] + u[0] * x + run[0] * yy, origin[1] + u[1] * x + run[1] * yy)
                   for x, yy in ((0, -going), (w1 - w0, -going), (w1 - w0, rel[-1]), (0, rel[-1]))]
            st.polygon = Polygon(pts)
            st.source_entities |= {self.prims[k][0] for k in ks}
            score = float(np.mean([self.plab[k][1] for k in ks if k in self.plab] or [0.0]))
            st.add_evidence("sympoint_label", score)
            st.add_evidence("sympoint_regularity", 1.0 - min(1.0, float(np.std(np.diff(offs))) / max(going, 1e-9)))
            st.detection_methods.add("sympoint_stair")
            out.append(st)
        self.log["stairs"] = len(out)
        return out


def extract(drawing, ctx, prims, labels, cfg, ids):
    ex = Extractor(drawing, ctx, prims, labels, cfg, ids)
    cands = ex.run()
    return cands, ex.log
