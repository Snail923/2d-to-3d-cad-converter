"""Combined approach: approach 2 (SymPointV2 labels) with approach 1's geometry filling what the model misses.

    front end (shared) -> approach 1: context, detectors, resolver   (on a copy of the drawing)
                       -> SymPointV2 answer
                       -> fill: a line the model left unlabelled takes approach 1's class
                       -> approach 2 as before: label consensus, extraction, build
                       -> add: elements the model cannot produce (columns approach 1 accepted)

The model's own answers are never overridden: approach 1 only speaks where the model is silent
(below ``min_label_score`` / ``min_label_mass``), and only through elements approach 1 ACCEPTED.
Every filled line and added element is counted in ``logs["detect"]["combined"]``.
Settings: ``combined`` in config.json."""
from __future__ import annotations

import copy
import math
import time

from architectural_converter.models.base_entity import Status
from architectural_converter.pipeline.corrections import normalize_corrections
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline, PipelineState
from architectural_converter.pipeline.pipeline_config import PipelineConfig

from .classes import CLASSES, GROUP_OF
from .labels import get_labels
from .pipeline import SymPointPipeline
from .sympoint_io import labels_from_response

KIND_CLASS = {"wall": "wall", "door": "single door", "window": "window"}


def approach1_elements(st, progress=None):
    """Approach 1's context, detectors and resolver on a copy of the already normalised drawing."""
    p1 = ExtractionPipeline(PipelineConfig(), progress=progress)
    s1 = PipelineState(input_file=st.input_file)
    s1.corrections = normalize_corrections(None)
    s1.drawing = copy.deepcopy(st.drawing)
    s1.segments = copy.deepcopy(st.segments)
    s1._tol = st._tol
    for stage in ("context", "detect", "resolve"):
        p1.run_stage(s1, stage)
    return s1


def votes_of(s1, statuses=(Status.ACCEPTED,)) -> dict[str, str]:
    """raw DXF entity id -> model class, from approach 1's elements (openings win over walls)."""
    v: dict[str, str] = {}
    for kind, coll in (("door", s1.doors), ("window", s1.windows)):
        for e in coll:
            if e.status in statuses:
                for rid in e.source_entities:
                    v[rid] = KIND_CLASS[kind]
    for w in s1.walls:
        if w.status in statuses:
            for rid in w.source_entities:
                v.setdefault(rid, "wall")
    return v


def fill_unlabelled(request: dict, response: dict, votes: dict[str, str], cfg: dict) -> tuple[dict, int]:
    """Copy of the model's answer where each primitive the model left unlabelled and approach 1 has a
    vote for gets approach 1's class with score / mass ``combined.fill_score``."""
    s = float(cfg["combined"]["fill_score"])
    req = {t["id"]: t for t in request.get("tiles", [])}
    out, filled = {**response, "tiles": []}, 0
    for rt in response.get("tiles", []):
        t = req.get(rt["id"])
        sem = [list(a) for a in rt.get("semantic", [])]
        if t is not None:
            for k, rid in enumerate(t["ids"]):
                if k >= len(sem) or rid not in votes:
                    continue
                a = sem[k]
                sc, mass = a[1], (a[2] if len(a) > 2 else 1.0)
                if a[0] is None or sc is None or sc < cfg["min_label_score"] or (mass is not None and mass < cfg["min_label_mass"]):
                    sem[k] = [CLASSES.index(votes[rid]), s, s]
                    filled += 1
        out["tiles"].append({**rt, "semantic": sem})
    return out, filled


def _proj(v, w) -> tuple[float, float]:
    """Extent of wall ``v`` along wall ``w``'s direction (w's s coordinate)."""
    a, b = v.start, v.end
    return (a[0] * w.u[0] + a[1] * w.u[1], b[0] * w.u[0] + b[1] * w.u[1])


def merge_duplicate_openings(openings, dist_m: float) -> int:
    """Openings of the same type whose centres are closer than ``dist_m`` are one opening: the one backed
    by approach 1, else the more confident one, is kept; the others are rejected with a reason."""
    def rank(o):
        m = set(o.detection_methods)
        return (bool(m & {"approach1_geometry", "approach1_door", "approach1_window"}), o.confidence)
    live = sorted((o for o in openings if o.status != Status.REJECTED), key=rank, reverse=True)
    kept, n = [], 0
    for o in live:
        twin = next((k for k in kept if k.type == o.type and math.dist(k.center, o.center) < dist_m), None)
        if twin is None:
            kept.append(o)
            continue
        o.status = Status.REJECTED
        o.status_reason = f"combined: duplicate of {twin.id} ({math.dist(twin.center, o.center):.2f} m apart)"
        n += 1
    return n


class CombinedPipeline(SymPointPipeline):
    STAGE_LABELS = dict(SymPointPipeline.STAGE_LABELS,
                        detect="Approach 1 detectors + SymPointV2 labels -> walls, doors, windows, columns, stairs")

    def _labels(self, st):
        c = self.sp["combined"]
        t0 = time.perf_counter()
        log = {"enabled": bool(c.get("enabled", True))}
        st._a1 = None
        if log["enabled"]:
            try:
                st._a1 = approach1_elements(st, self.progress)
            except Exception as e:  # noqa: BLE001  (approach 1 failing must not stop approach 2)
                log["error"] = f"{type(e).__name__}: {e}"
                st.drawing.issues.append({"stage": "detect", "severity": "warning",
                                          "message": f"combined: approach 1 skipped after an error: {log['error']}"})
        st.timings["detect:approach1"] = round(time.perf_counter() - t0, 4)
        labels, prims, req, resp, source = get_labels(st.drawing, st.input_file, self.sp, self.url, self.response,
                                                      progress=self.labels_progress)
        if st._a1 is not None and c.get("fill_unlabelled", True):
            votes = votes_of(st._a1)
            resp_f, n = fill_unlabelled(req, resp, votes, self.sp)
            labels = labels_from_response(req, resp_f, self.sp["min_label_score"], self.sp["min_label_mass"])
            labels["model"] = f"{labels.get('model', 'SymPointV2')} + approach 1"
            log.update(approach1_votes=len(votes), filled_primitives=n)
        st._combined_log = log
        return labels, prims, req, resp, source

    def _stage_detect(self, st):
        super()._stage_detect(st)
        st.logs["detect"]["approach"] = "combined" if st._combined_log.get("enabled") else "sympoint"
        st.logs["detect"]["combined"] = st._combined_log

    def _stage_reconstruct(self, st):
        super()._stage_reconstruct(st)            # includes approach 2's gap typing, which can add doors
        c = self.sp["combined"]
        if getattr(st, "_a1", None) is not None and c.get("merge_duplicate_openings", True):
            n = merge_duplicate_openings(st.doors + st.windows + list(getattr(st, "_generic", []) or []),
                                         float(c["duplicate_opening_m"]))
            st.logs["detect"].setdefault("combined", {})["duplicate_openings_after_gaps"] = n

    def _stage_classify(self, st):
        c = self.sp["combined"]                   # last pass: later stages can move or re-open openings
        b = getattr(st, "building", None)
        if b is not None and getattr(b, "levels", None):
            n = 0
            for col in b.columns.values():
                if col.status != Status.REJECTED and "approach1_column" in col.detection_methods and col.level is None:
                    col.status = Status.REJECTED
                    col.status_reason = "combined: approach 1 column outside every plan view (sheet frame, title block, legend)"
                    n += 1
            st.logs["detect"].setdefault("combined", {})["columns_outside_views"] = n
        if c.get("drop_unsupported_doors", True):
            st.logs["detect"].setdefault("combined", {})["unsupported_doors_rejected"] = \
                self._drop_unsupported_doors(st, getattr(st, "_a1", None), c)
        if getattr(st, "_a1", None) is not None and c.get("merge_duplicate_openings", True):
            n = merge_duplicate_openings(st.doors + st.windows + list(getattr(st, "_generic", []) or []),
                                         float(c["duplicate_opening_m"]))
            st.logs["detect"].setdefault("combined", {})["duplicate_openings_before_classify"] = n
        super()._stage_classify(st)

    def _stage_resolve(self, st):
        super()._stage_resolve(st)
        c = self.sp["combined"]
        log = st.logs["detect"].setdefault("combined", {})
        a1 = getattr(st, "_a1", None)
        if a1 is None:
            return
        if c.get("resolve_wall_conflicts", True):
            log["walls_replaced"] = self._resolve_wall_conflicts(st, a1, c)
        if c.get("add_walls", True):
            log["walls_added"] = self._add_walls(st, a1, c)
            log["walls_not_attached"] = getattr(st, "_combined_detached_walls", 0)
            log["single_line_walls_skipped"] = getattr(self, "_skipped_single", 0)
        if c.get("add_columns", True):
            log["columns_added"] = self._add_columns(st, a1)
        if c.get("reconcile_openings", True):
            log["openings_reconciled"] = self._reconcile_openings(st, a1, c)
        if c.get("add_openings", True):
            log["doors_added"], log["windows_added"] = self._add_openings(st, a1, c)
        if c.get("merge_duplicate_openings", True):
            log["duplicate_openings"] = merge_duplicate_openings(st.doors + st.windows, float(c["duplicate_opening_m"]))

    def _type_support(self, st, o) -> float:
        """Share of the opening's source length the model itself labelled as that kind of opening."""
        ents = (getattr(st, "sp_labels_model", None) or {}).get("entities", {})
        want = {"door": ("door", "opening"), "window": ("window", "curtain_wall")}.get(o.type, ())
        tot = ok = 0.0
        for rid in o.source_entities:
            e = ents.get(rid)
            L = float(e.get("length") or 0.0) if e else 0.0
            tot += L
            if e and GROUP_OF.get(e["label"]) in want:
                ok += L
        return ok / tot if tot > 0 else 0.0

    def _reconcile_openings(self, st, a1, c) -> int:
        """An approach-2 door / window and approach-1 openings built from the same DXF lines, the same block
        insert, or at the same place: keep approach 2's when both agree on type, position and width; otherwise
        take approach 1's geometry and type (it splits merged symbols and finds the closed leaf). A second
        approach-2 opening from a symbol that already gave one is a duplicate and is rejected."""
        r = float(c["opening_match_m"])
        a1o = [o for o in a1.doors + a1.windows if o.status == Status.ACCEPTED]
        raw = st.drawing.raw_entities

        def inserts(o):          # the innermost block insert of each source line: one symbol = one opening
            out = set()
            for rid in o.source_entities:
                re_ = raw.get(rid)
                if re_ is not None and re_.block_path:
                    out.add(re_.block_path[-1])
            return out
        a1_ins = {id(a): inserts(a) for a in a1o}
        used, changed = set(), 0
        keep_d, keep_w = [], []
        for o in st.doors + st.windows:
            dst = keep_d if o in st.doors else keep_w
            if o.status == Status.REJECTED:
                dst.append(o)
                continue
            src, ins = set(o.source_entities), inserts(o)
            m = [a for a in a1o if id(a) not in used and (
                (src and len(src & set(a.source_entities)) >= 0.5 * min(len(src), len(a.source_entities) or 1))
                or (ins and ins & a1_ins[id(a)]) or math.dist(a.center, o.center) <= r)]
            if not m:
                dup = [a for a in a1o if id(a) in used and (
                    (src and len(src & set(a.source_entities)) >= 0.5 * min(len(src), len(a.source_entities) or 1))
                    or (ins and ins & a1_ins[id(a)]))]
                if dup:                      # the same symbol already gave an opening: this one is a second copy
                    o.status = Status.REJECTED
                    o.status_reason = f"combined: duplicate of the opening built from the same symbol as approach 1's {dup[0].id}"
                    changed += 1
                dst.append(o)
                continue
            agree = (len(m) == 1 and m[0].type == o.type and math.dist(m[0].center, o.center) <= float(c["opening_agree_m"])
                     and abs(m[0].width - o.width) <= 0.3)
            if agree:
                used.add(id(m[0]))
                o.evidence["approaches_agree"] = 1.0
                dst.append(o)
                continue
            for k, a in enumerate(m):
                used.add(id(a))
                e = copy.deepcopy(a)
                if a.type == o.type:
                    e.evidence["approaches_agree"] = 1.0
                e.id = o.id if k == 0 and a.type == o.type else st.ctx.ids.new(a.type)
                e.detection_methods = set(e.detection_methods) | {"approach1_geometry"}
                e.notes = list(getattr(e, "notes", [])) + [f"combined: replaces {o.id} ({o.type}, {o.width:.2f} m) from SymPointV2"]
                (keep_d if a.type == "door" else keep_w).append(e)
            changed += 1
        st.doors, st.windows = keep_d, keep_w
        return changed

    def _add_openings(self, st, a1, c) -> tuple[int, int]:
        """Doors / windows approach 1 accepted where approach 2 has no opening nearby."""
        r = float(c["opening_match_m"])

        def centre(o):
            g = getattr(o, "chord", None)
            if g is not None and hasattr(g, "centroid"):
                return (g.centroid.x, g.centroid.y)
            p = getattr(o, "position", None) or getattr(o, "center", None)
            if p is not None:
                return (p.x, p.y) if hasattr(p, "x") else tuple(p[:2])
            return None

        have = [centre(o) for o in st.doors + st.windows if o.status != Status.REJECTED]
        have = [p for p in have if p is not None]
        counts = []
        for src, dst, kind in ((a1.doors, st.doors, "door"), (a1.windows, st.windows, "window")):
            n = 0
            for o in src:
                p = centre(o)
                if o.status != Status.ACCEPTED or p is None or any(math.dist(p, q) <= r for q in have):
                    continue
                if self._model_says_not_opening(st, o, c):
                    continue
                e = copy.deepcopy(o)
                if self._type_support(st, o) >= float(c["agree_min_support"]):
                    e.evidence["approaches_agree"] = 1.0
                e.id = st.ctx.ids.new(kind)
                e.detection_methods = set(e.detection_methods) | {f"approach1_{kind}"}
                e.notes = list(getattr(e, "notes", [])) + [f"combined: {kind} from approach 1 where SymPointV2 found none"]
                dst.append(e)
                have.append(p)
                n += 1
            counts.append(n)
        return counts[0], counts[1]

    def _model_says_not_opening(self, st, o, c) -> bool:
        """The model confidently labelled most of the opening's lines as wall or furniture."""
        ents = (getattr(st, "sp_labels_model", None) or {}).get("entities", {})
        tot = other = 0.0
        for rid in o.source_entities:
            e = ents.get(rid)
            if e is None:
                continue
            L = float(e.get("length") or 0.0)
            tot += L
            if GROUP_OF.get(e["label"]) in ("wall", "furniture", "stair") and float(e.get("score") or 0.0) >= float(c["veto_min_score"]):
                other += L
        return tot > 0 and other / tot > float(c["wall_veto_share"])

    def _drop_unsupported_doors(self, st, a1, c) -> int:
        """An approach-2 door with no gap in its wall, no approach-1 opening near it and a weak model answer
        (one line, or a mean score below ``weak_door_score``) is furniture read as a door."""
        if a1 is None:
            return 0
        r = float(c["opening_match_m"])
        a1c = [o.center for o in a1.doors + a1.windows if o.status != Status.REJECTED]
        ents = (getattr(st, "sp_labels_model", None) or {}).get("entities", {})
        n = 0
        for o in st.doors:
            if o.status == Status.REJECTED or not o.evidence.get("hosted_without_gap") \
                    or set(o.detection_methods) & {"approach1_geometry", "approach1_door"} \
                    or any(math.dist(o.center, q) <= r for q in a1c):
                continue
            sc = [float(ents[x]["score"]) for x in o.source_entities if x in ents and GROUP_OF.get(ents[x]["label"]) in ("door", "opening")]
            if len(o.source_entities) <= 1 or not sc or sum(sc) / len(sc) < float(c["weak_door_score"]):
                o.status = Status.REJECTED
                o.status_reason = ("combined: door with no gap in its wall, no approach-1 door and a weak model answer "
                                   f"({len(o.source_entities)} line(s), mean door score {sum(sc) / len(sc) if sc else 0:.2f})")
                n += 1
        return n

    def _model_says_other(self, st, w, c) -> bool:
        """True when the model labelled most of the wall's source lines as something that is not a wall.
        Approach 1 still wins when it is sure and the wall closes a room (furniture does not enclose rooms)."""
        if w.confidence >= float(c["override_confidence"]) and w.evidence.get("cycle_support", 0.0) >= 0.9:
            return False
        ents = (getattr(st, "sp_labels_model", None) or {}).get("entities", {})
        tot = other = 0.0
        for rid in w.source_entities:
            e = ents.get(rid)
            if e is None:
                continue
            L = float(e.get("length") or 0.0)
            tot += L
            if GROUP_OF.get(e["label"]) in ("door", "window", "curtain_wall", "stair", "furniture", "opening") \
                    and float(e.get("score") or 0.0) >= float(c["veto_min_score"]):
                other += L
        return tot > 0 and other / tot > float(c["wall_veto_share"])

    def _resolve_wall_conflicts(self, st, a1, c) -> int:
        """An approach-2 wall that shares a face line with an approach-1 wall but pairs it differently
        (other thickness or centre line) is rejected: approach 1 pairs faces with geometric checks
        (empty band, material on one side), the model can call a door leaf or a furniture edge "wall".
        The approach-1 wall is then added where nothing covers it (``_add_walls``)."""
        a1w = [v for v in a1.walls if v.status == Status.ACCEPTED and v.kind == "straight" and v.thickness]
        by_src: dict[str, list] = {}
        for v in a1w:
            for rid in v.source_entities:
                by_src.setdefault(rid, []).append(v)
        n = 0
        for w in st.walls:
            if w.status == Status.REJECTED or w.kind != "straight" or not w.thickness \
                    or "approach1_wall" in w.detection_methods:
                continue
            vs = {id(v): v for rid in w.source_entities for v in by_src.get(rid, [])}.values()
            if not vs:
                continue
            def same(v):
                if abs(w.u[0] * v.u[0] + w.u[1] * v.u[1]) < 0.99:
                    return False
                sign = 1.0 if w.u[0] * v.u[0] + w.u[1] * v.u[1] > 0 else -1.0
                tt = max(0.03, float(c["wall_thickness_tol"]) * v.thickness)
                return abs(w.thickness - v.thickness) <= tt and abs(w.c - sign * v.c) <= max(0.05, 0.25 * v.thickness)
            if any(same(v) for v in vs):
                continue
            # only when approach 1's walls there span the wall: its resolve-stage walls can be fragments that
            # its own reconstruction would join later, and a long wall replaced by fragments leaves gaps
            ivs = sorted((max(w.s0, min(_proj(v, w))), min(w.s1, max(_proj(v, w)))) for v in vs)
            cov, end = 0.0, w.s0
            for a, b in ivs:
                a = max(a, end)
                if b > a:
                    cov += b - a
                    end = b
            if cov < float(c["wall_conflict_min_cover"]) * max(w.s1 - w.s0, 1e-9):
                continue
            v = max(vs, key=lambda x: x.confidence)
            w.status = Status.REJECTED
            w.status_reason = (f"combined: approach 1 pairs these lines differently ({v.id}, thickness {v.thickness:.2f} m "
                               f"instead of {w.thickness:.2f} m)")
            n += 1
        return n

    def _add_walls(self, st, a1, c) -> int:
        """Approach 1 walls where approach 2 built none: only the uncovered stretches of a straight wall."""
        from shapely.geometry import Point
        from shapely.strtree import STRtree
        live = [w for w in st.walls if w.status != Status.REJECTED and not w.is_hypothesis]
        axes = [w.axis for w in live]
        tree = STRtree(axes) if axes else None
        step, min_len = float(c["wall_sample_m"]), float(c["min_added_wall_m"])

        def covered(p, u, t):
            if tree is None:
                return False
            tol = max(0.6 * (t or 0.1), 0.08)
            P = Point(p)
            for k in tree.query(P.buffer(tol)):
                w = live[int(k)]
                if w.axis.distance(P) <= tol and (w.kind not in ("straight", "centerline")
                                                  or abs(u[0] * w.u[0] + u[1] * w.u[1]) >= 0.94):
                    return True
            return False

        added, cands = 0, []
        # drawing convention from approach 2's own walls: it builds single-line walls only where a wall line has
        # no second face (approach 1's resolve-stage walls include long site and sheet lines, which would skew it)
        own = [w for w in live if w.kind in ("straight", "centerline")]
        tot_len = sum(w.length for w in own)
        single_line_drawing = tot_len > 0 and \
            sum(w.length for w in own if w.kind == "centerline") / tot_len >= float(c["single_line_share"])
        self._skipped_single = 0
        for w in a1.walls:
            if w.status != Status.ACCEPTED or w.is_hypothesis or self._model_says_other(st, w, c):
                continue
            if w.kind == "centerline" and not single_line_drawing:
                # a double-line drawing: a lone line on the wall layer is a site / plot / detail line, not a wall
                self._skipped_single += 1
                continue
            if w.kind in ("straight", "centerline"):
                n = max(2, int((w.s1 - w.s0) / step) + 1)
                ss = [w.s0 + (w.s1 - w.s0) * i / (n - 1) for i in range(n)]
                free = [not covered(w.point_at(s), w.u, w.thickness) for s in ss]
                runs, i = [], 0
                while i < n:
                    if free[i]:
                        j = i
                        while j + 1 < n and free[j + 1]:
                            j += 1
                        runs.append((ss[max(i - 1, 0)] if i > 0 else ss[i], ss[min(j + 1, n - 1)] if j < n - 1 else ss[j]))
                        i = j + 1
                    else:
                        i += 1
                pieces = [(a, b) for a, b in runs if b - a >= min_len]
            else:
                ax = w.axis
                pts = [ax.interpolate(d) for d in [ax.length * i / 10 for i in range(11)]]
                share = sum(not covered((p.x, p.y), (1.0, 0.0), w.thickness) for p in pts) / len(pts)
                pieces = [None] if share > 0.8 else []
            for piece in pieces:
                e = copy.deepcopy(w)
                if piece is not None:
                    e.s0, e.s1 = piece
                    e.detected_interval = piece
                cands.append(e)
        # only walls that attach to the building: a piece within ``attach_m`` of a wall already there (or of a
        # piece attached before it). Plot limits, site lines and details away from the building stay out.
        d, gap = float(c["attach_m"]), float(c["attach_across_opening_m"])
        anchors = [w.axis for w in live]
        aw = list(live)

        def continues(e, w):          # same line, across a gap no wider than an opening (a door between them)
            if e.kind not in ("straight", "centerline") or w.kind not in ("straight", "centerline"):
                return False
            dp = e.u[0] * w.u[0] + e.u[1] * w.u[1]
            if abs(dp) < 0.99 or abs(e.c - (1 if dp > 0 else -1) * w.c) > max(0.1, 0.5 * (e.thickness or 0.1)):
                return False
            a0, a1 = sorted(((w.s0, w.s1) if dp > 0 else (-w.s1, -w.s0)))
            return max(a0 - e.s1, e.s0 - a1) <= gap
        grown = True
        while grown and cands:
            grown = False
            tree_a = STRtree(anchors) if anchors else None
            rest = []
            for e in cands:
                ax = e.axis
                near = [int(k) for k in tree_a.query(ax.buffer(max(d, gap)))] if tree_a is not None else []
                hit = any(anchors[k].distance(ax) <= d or continues(e, aw[k]) for k in near)
                if not hit:
                    rest.append(e)
                    continue
                e.id = st.ctx.ids.new("wall")
                e.detection_methods = set(e.detection_methods) | {"approach1_wall"}
                e.notes = list(getattr(e, "notes", [])) + ["combined: wall from approach 1 where SymPointV2 built none"]
                st.walls.append(e)
                anchors.append(ax)
                aw.append(e)
                added += 1
                grown = True
            cands = rest
        st._combined_detached_walls = len(cands)
        return added

    def _add_columns(self, st, a1) -> int:
        added = 0
        have = [(e.polygon.centroid.x, e.polygon.centroid.y, max(e.size)) for e in st.columns
                if e.status != Status.REJECTED and e.polygon is not None]
        for col in a1.columns:
            if col.status != Status.ACCEPTED or col.polygon is None:
                continue
            p = (col.polygon.centroid.x, col.polygon.centroid.y)
            if any(math.dist(p, (x, y)) <= max(s, max(col.size), 0.3) for x, y, s in have):
                continue
            e = copy.deepcopy(col)
            e.id = st.ctx.ids.new("column")
            e.detection_methods = set(e.detection_methods) | {"approach1_column"}
            e.notes = list(getattr(e, "notes", [])) + ["combined: column from approach 1 (SymPointV2 has no column class)"]
            st.columns.append(e)
            have.append((p[0], p[1], max(col.size)))
            added += 1
        return added


__all__ = ["CombinedPipeline", "approach1_elements", "fill_unlabelled", "merge_duplicate_openings", "votes_of"]
