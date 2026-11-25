"""Label consensus: the drawing's own structure votes on the model's line-by-line answer.

Four independent votes, strongest first (a later vote never changes what an earlier one set, and
none of them overrides a confident answer of the model about a building element):

0. **Entities.** A DXF entity (a polyline, a spline) is one object: when the model labels only a
   small part of its length, the stray label does not make the whole entity that class.
1. **Blocks.** A block is one object drawn once and inserted many times. The same line of a block
   definition gets the same class in every insert (majority of the inserts the model labelled),
   and a symbol-sized block whose labelled lines are mostly one kind (a door, a window, a piece of
   furniture) is that kind as a whole.
2. **Layers.** A layer the model labels consistently (most of its labelled length is one kind) is a
   layer of that kind: its unlabelled lines get the layer's class, and "wall" lines in a layer of
   symbols (door leaves, furniture) follow the layer (walls are filled in only when the model
   already labels half of the layer). A big layer in which the model labels (almost) nothing is
   not a layer of building elements: stray "wall" lines in it are dropped.
3. **Layer names.** Where the model is silent, the name of the layer (approach 1's configurable
   keyword table, e.g. A-WALL, A-GLAZ, Fenster) gives a weak label; "wall" lines on a layer named
   as annotation, dimensions, grid, hatch or furniture are dropped, however many of them the
   model calls wall (kitchen counters and wardrobes look like walls).

Every changed line records where its label came from (``origin``), the model's own answer is kept
in ``model_label``; the returned log counts the changes per vote."""
from __future__ import annotations

import copy
import math
from collections import defaultdict

from ..classes import CLASSES, GROUP_OF

# kinds of thing a vote is about (the model's classes that mean the same for a vote)
FAMILY = {"wall": "wall", "curtain_wall": "glazing", "window": "glazing", "door": "door", "railing": "railing",
          "stair": "stair", "furniture": "furniture", "opening": "opening", "other": "furniture"}
SYMBOLS = ("door", "glazing", "furniture", "stair")
# layer-name category (approach 1's layer semantics) -> model class given to silent lines
NAME_CLASS = {"wall": "wall", "column": "wall", "door": "single door", "window": "window", "stair": "stairs",
              "balcony": "railing"}
NAME_FAMILY = {"wall": "wall", "column": "wall", "door": "door", "window": "glazing", "stair": "stair",
               "balcony": "railing", "furniture": "furniture"}
NOT_BUILDING = ("annotation", "dimension", "grid", "hatch", "area", "ignore", "furniture")


def family_of(label):
    return FAMILY.get(GROUP_OF.get(label, "other"), "furniture") if label else None


class _Votes:
    """Length-weighted class votes."""

    def __init__(self):
        self.by_class: dict[str, float] = defaultdict(float)
        self.labelled = 0.0
        self.total = 0.0
        self.n = 0
        self.n_labelled = 0

    def add(self, label, score, length):
        self.total += length
        self.n += 1
        if label:
            self.by_class[label] += length * max(score, 1e-3)
            self.labelled += length
            self.n_labelled += 1

    def families(self):
        fam: dict[str, float] = defaultdict(float)
        for c, v in self.by_class.items():
            fam[family_of(c)] += v
        return fam

    def top(self):
        """(family, purity among the labelled, class) of the strongest family, or None."""
        fam = self.families()
        if not fam:
            return None
        f = max(fam, key=fam.get)
        cls = max((c for c in self.by_class if family_of(c) == f), key=self.by_class.get)
        return f, fam[f] / max(sum(fam.values()), 1e-12), cls

    @property
    def coverage(self):
        return self.labelled / self.total if self.total > 0 else 0.0


def refine_labels(drawing, layer_profiles, prims, labels, cfg) -> tuple[dict, dict]:
    """(labels with the votes applied, log). ``labels`` is not modified."""
    o = (cfg.get("reasoning") or {}).get("label_consensus", {})
    log: dict = {"model_labelled_entities": len(labels.get("entities", {}))}
    if not o.get("enabled", True):
        return labels, {"skipped": "disabled"}
    ents = copy.deepcopy(labels.get("entities", {}))
    length: dict[str, float] = defaultdict(float)
    layer_of: dict[str, str] = {}
    pts: dict[str, list] = defaultdict(list)
    for rid, layer, _cmd, p4, L, _disp in prims:
        length[rid] += float(L)
        layer_of[rid] = layer
        pts[rid].append(p4)
    raw = drawing.raw_entities
    changed: dict[str, str] = {}

    def lab(rid):
        e = ents.get(rid)
        return (e["label"], float(e["score"])) if e else (None, 0.0)

    def assign(rid, cls, score, origin):
        old = ents.get(rid)
        if old is not None and old["label"] == cls:
            return False
        e = {"label": cls, "score": round(float(score), 4), "length": round(length[rid], 4), "origin": origin}
        if old is not None:
            e["model_label"] = old.get("model_label", old["label"])
            e["model_score"] = old.get("model_score", old["score"])
        ents[rid] = e
        changed[rid] = origin
        return True

    def drop(rid, origin):
        if rid in ents:
            del ents[rid]
            changed[rid] = origin
            return True
        return False

    strong = float(o.get("keep_model_score", 0.8))

    # ------------------------------------------------------------------ 0. entities
    min_part = float(o.get("min_entity_labelled_share", 0.3))
    n_part = 0
    for rid, e in list(ents.items()):
        if length[rid] > 0 and float(e.get("length", length[rid])) < min_part * length[rid]:
            n_part += drop(rid, "entity_share")
    log["entities"] = {"dropped_small_labelled_part": n_part}

    # ------------------------------------------------------------------ 1. blocks
    bo = o.get("blocks", {})
    if bo.get("enabled", True):
        n_line = n_whole = 0
        by_def: dict[tuple, list] = defaultdict(list)          # (block, line of the definition) -> inserts
        by_insert: dict[tuple, list] = defaultdict(list)       # (block, insert) -> lines
        for rid in length:
            r = raw.get(rid)
            if r is None or not r.block_name or not r.block_path:
                continue
            if r.source_block_handle:
                by_def[(r.block_name, r.source_block_handle)].append(rid)
            by_insert[(r.block_name, r.block_path[-1])].append(rid)
        min_share, min_purity = float(bo.get("min_labelled_share", 0.5)), float(bo.get("min_purity", 0.7))
        for key, rids in by_def.items():
            if len(rids) < 2:
                continue
            v = _Votes()
            for rid in rids:
                v.add(*lab(rid), 1.0)                          # one vote per insert
            t = v.top()
            if t is None or v.n_labelled < 2 or v.n_labelled / v.n < min_share or t[1] < min_purity:
                continue
            fam, purity, cls = t
            sc = purity * sum(lab(r)[1] for r in rids if family_of(lab(r)[0]) == fam) / max(
                1, sum(1 for r in rids if family_of(lab(r)[0]) == fam))
            for rid in rids:
                if family_of(lab(rid)[0]) != fam:
                    n_line += assign(rid, cls, sc, "block_vote")
        # a symbol-sized block that is mostly one kind of symbol is that symbol as a whole
        max_size = float(bo.get("symbol_max_size_m", 6.0))
        whole_cov, whole_purity = float(bo.get("whole_min_coverage", 0.3)), float(bo.get("whole_min_purity", 0.7))
        per_block: dict[str, list] = defaultdict(list)
        for (name, ins), rids in by_insert.items():
            per_block[name].append(rids)
        for name, inserts in per_block.items():
            v = _Votes()
            size = 0.0
            for rids in inserts:
                xs = [float(p[0]) for r in rids for p4 in pts[r] for p in p4]
                ys = [float(p[1]) for r in rids for p4 in pts[r] for p in p4]
                size = max(size, math.hypot(max(xs) - min(xs), max(ys) - min(ys)))
                for rid in rids:
                    v.add(*lab(rid), length[rid])
            t = v.top()
            if t is None or size > max_size or v.coverage < whole_cov or t[1] < whole_purity or t[0] not in SYMBOLS:
                continue
            fam, purity, cls = t
            for rids in inserts:
                for rid in rids:
                    if family_of(lab(rid)[0]) != fam:
                        n_whole += assign(rid, cls, purity * 0.8, "block_symbol")
        log["blocks"] = {"lines_from_other_inserts": n_line, "lines_of_symbol_blocks": n_whole}

    # ------------------------------------------------------------------ 2. layers (the model's own vote)
    lo = o.get("layers", {})
    layer_votes: dict[str, _Votes] = defaultdict(_Votes)
    for rid in length:
        layer_votes[layer_of[rid]].add(*lab(rid), length[rid])
    kinds = {}
    for ly in layer_votes:
        p = layer_profiles.get(ly) if layer_profiles else None
        k = p.primary if p is not None else None
        kinds[ly] = k if (k and p.category_scores.get(k, 0.0) >= float(o.get("names", {}).get("min_name_score", 0.6))) else None
    names_on = bool(o.get("names", {}).get("enabled", True))
    consistent: dict[str, tuple] = {}
    background: set[str] = set()
    if lo.get("enabled", True):
        min_n, min_len = int(lo.get("min_entities", 10)), float(lo.get("min_length_m", 5.0))
        for ly, v in layer_votes.items():
            if v.n < min_n or v.total < min_len:
                continue
            t = v.top()
            need = float(lo.get("fill_min_coverage_wall", 0.5) if t is not None and t[0] == "wall"
                         else lo.get("fill_min_coverage", 0.25))
            named_other = names_on and kinds[ly] in NOT_BUILDING and t is not None and t[0] == "wall"
            if named_other:
                continue                       # the name says it is not structure: decided by the names vote
            if t is not None and v.coverage >= need and t[1] >= float(lo.get("fill_min_purity", 0.85)):
                consistent[ly] = t
            elif v.coverage <= float(lo.get("background_max_coverage", 0.03)) or (
                    kinds[ly] in NOT_BUILDING and v.coverage <= float(lo.get("background_max_coverage_named", 0.15))):
                background.add(ly)
        n_fill = n_follow = n_drop = 0
        fill_score = float(lo.get("fill_score", 0.6))
        for rid in list(length):
            ly = layer_of[rid]
            l, s = lab(rid)
            if ly in consistent:
                fam, purity, cls = consistent[ly]
                if l is None:
                    n_fill += assign(rid, cls, fill_score * purity, "layer_vote")
                elif family_of(l) == "wall" and fam in SYMBOLS and rid not in changed:
                    n_follow += assign(rid, cls, fill_score * purity, "layer_vote")
            elif ly in background and family_of(l) == "wall" and rid not in changed:
                n_drop += drop(rid, "layer_background")
        log["layers"] = {"consistent": {ly: [t[0], round(t[1], 3), round(layer_votes[ly].coverage, 3)]
                                        for ly, t in sorted(consistent.items())},
                         "background": sorted(background), "lines_filled": n_fill,
                         "wall_lines_following_symbol_layer": n_follow, "stray_wall_lines_dropped": n_drop}

    # ------------------------------------------------------------------ 3. layer names
    no = o.get("names", {})
    if no.get("enabled", True):
        n_name = n_drop = 0
        name_score = float(no.get("score", 0.5))
        used = {}
        for rid in list(length):
            ly = layer_of[rid]
            k = kinds.get(ly)
            if k is None or ly in consistent:
                continue
            l, s = lab(rid)
            if k in NAME_CLASS and l is None:
                if assign(rid, NAME_CLASS[k], name_score, "layer_name"):
                    n_name += 1
                    used[ly] = k
            elif k in NOT_BUILDING and family_of(l) == "wall" and rid not in changed:
                if drop(rid, "layer_name"):
                    n_drop += 1
                    used[ly] = k
            elif k in ("door", "window", "stair", "furniture") and family_of(l) == "wall" and s < strong \
                    and rid not in changed and k in NAME_CLASS:
                if assign(rid, NAME_CLASS[k], name_score, "layer_name"):
                    n_name += 1
                    used[ly] = k
        log["names"] = {"layers": dict(sorted(used.items())), "lines_labelled": n_name, "stray_wall_lines_dropped": n_drop}

    # ------------------------------------------------------------------ result
    out = dict(labels)
    out["entities"] = ents
    prim = dict(labels.get("primitives", {}))
    if changed:
        for k, pr in enumerate(prims):
            rid = pr[0]
            if rid in changed:
                e = ents.get(rid)
                if e is None:
                    prim.pop(k, None)
                    prim.pop(str(k), None)
                else:
                    prim.pop(str(k), None)
                    prim[k] = [CLASSES.index(e["label"]), e["score"]]
        dropped = {rid for rid, why in changed.items() if rid not in ents}
        if dropped and labels.get("instances"):
            out["instances"] = [i for i in labels["instances"] if not set(i["members"]) <= dropped]
    out["primitives"] = prim
    by = defaultdict(int)
    for why in changed.values():
        by[why] += 1
    log["changed"] = dict(sorted(by.items()))
    log["labelled_entities"] = len(ents)
    return out, log
