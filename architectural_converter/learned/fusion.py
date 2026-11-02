"""Turn SymPointV2 labels into evidence on the engine's candidates.

For each candidate the labels of its OWN source DXF entities are aggregated by length:
``share[group]`` = labelled length in the group / labelled length, ``coverage`` = labelled length
/ total length of its sources. Then:

* every candidate gets ``learned_label`` evidence = share of the group that matches its type;
* walls, columns and stairs whose lines are confidently furniture are vetoed;
* window candidates whose lines are a door (or a wardrobe front) are re-typed as doors, ones that
  are furniture are rejected; door candidates that are furniture are rejected;
* generic openings (gaps without symbol) become doors / windows when door / window labelled
  lines lie in the gap.

Decisions carry the model's class and score in notes and status reasons (traceability), and
candidates with a manual / AI decision are never touched."""
from __future__ import annotations

import numpy as np

from ..models.base_entity import Status

TYPE_GROUP = {"wall": ("wall",), "column": ("wall",), "stair": ("stair",), "door": ("door", "opening"),
              "window": ("window",), "opening": ("door", "opening")}


class Labels:
    def __init__(self, data, cfg):
        self.cfg = cfg["learned"]
        self.sp = self.cfg["sympoint"]
        self.ent = data.get("entities", {}) if data else {}
        self.group_of = {c: g for g, cs in self.cfg["groups"].items() for c in cs}
        self.model = (data or {}).get("model", "SymPointV2")

    def __bool__(self):
        return bool(self.ent)

    def summary(self, rids, lengths=None):
        """(share per group, coverage, best class, mean score of best group) for a set of raw ids."""
        tot, lab = 0.0, 0.0
        g_len: dict[str, float] = {}
        c_len: dict[str, float] = {}
        g_sc: dict[str, list] = {}
        for rid in rids:
            L = (lengths or {}).get(rid, 1.0)
            tot += L
            e = self.ent.get(rid)
            if e is None or e["score"] < self.sp["min_label_score"]:
                continue
            g = self.group_of.get(e["label"], "other")
            lab += L
            g_len[g] = g_len.get(g, 0.0) + L
            c_len[e["label"]] = c_len.get(e["label"], 0.0) + L
            g_sc.setdefault(g, []).append(e["score"])
        if lab <= 0:
            return {}, 0.0, None, 0.0
        share = {g: v / lab for g, v in g_len.items()}
        best = max(c_len, key=c_len.get)
        bg = self.group_of.get(best, "other")
        return share, lab / max(tot, 1e-12), best, float(np.mean(g_sc[bg]))


def _lengths(drawing):
    out: dict[str, float] = {}
    for g in drawing.entities:
        if g.valid and g.points is not None and len(g.points) >= 2 and g.kind in ("line", "arc", "curve", "circle"):
            P = np.asarray(g.points, float)
            out[g.raw_id] = out.get(g.raw_id, 0.0) + float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    return out


def _note(e, labels, best, share, score, what):
    e.notes.append(f"{labels.model}: {what} ({best}, {share:.0%} of its lines, score {score:.2f})")


def fuse_resolved(walls, columns, stairs, labels: Labels, drawing) -> dict:
    """Walls, columns, stairs (after candidate resolution)."""
    log = {"walls_vetoed": 0, "columns_vetoed": 0, "stairs_vetoed": 0, "with_evidence": 0}
    if not labels:
        return log
    sp = labels.sp
    lengths = _lengths(drawing)
    for kind, coll in (("wall", walls), ("column", columns), ("stair", stairs)):
        for e in coll:
            if e.manual_status is not None or not e.source_entities:
                continue
            share, cov, best, sc = labels.summary(e.source_entities, lengths)
            if cov < sp["min_coverage"]:
                continue
            match = sum(share.get(g, 0.0) for g in TYPE_GROUP[kind])
            e.add_evidence("learned_label", match)
            log["with_evidence"] += 1
            furn = share.get("furniture", 0.0) + share.get("wardrobe", 0.0)
            if e.status != Status.REJECTED and furn >= sp["veto_share"] and sc >= sp["veto_score"]:
                e.veto(f"{labels.model}: lines are furniture ({best})")
                _note(e, labels, best, furn, sc, "furniture")
                log[f"{kind}s_vetoed"] += 1
    return log


def fuse_openings(b, labels: Labels, drawing, locked=frozenset()) -> dict:
    """Doors / windows / generic openings (classify stage, before the plausibility rules)."""
    log = {"windows_to_doors": 0, "rejected": 0, "openings_typed": 0, "with_evidence": 0}
    if not labels:
        return log
    sp = labels.sp
    lengths = _lengths(drawing)
    for o in b.openings.values():
        if o.id in locked or o.manual_status is not None or o.status == Status.REJECTED:
            continue
        if o.type == "opening" or not o.source_entities:
            _type_gap(o, b, labels, drawing, lengths, log)
            continue
        share, cov, best, sc = labels.summary(o.source_entities, lengths)
        if cov < sp["min_coverage"]:
            continue
        log["with_evidence"] += 1
        door, win = share.get("door", 0.0) + share.get("opening", 0.0), share.get("window", 0.0)
        ward, furn = share.get("wardrobe", 0.0), share.get("furniture", 0.0)
        o.add_evidence("learned_label", sum(share.get(g, 0.0) for g in TYPE_GROUP[o.type]))
        if furn >= sp["veto_share"] and sc >= sp["veto_score"]:
            o.veto(f"{labels.model}: lines are furniture ({best})")
            _note(o, labels, best, furn, sc, "furniture")
            log["rejected"] += 1
        elif o.type == "window" and (door >= sp["retype_share"] or ward >= sp["retype_share"]):
            o.type, o.sill_height = "door", None
            o.subtype = "sliding_or_closet" if ward >= sp["retype_share"] or best in ("sliding door", "folding door") \
                else best.replace(" ", "_")
            o.status_reason = f"{labels.model}: window re-typed as {o.subtype}"
            _note(o, labels, best, max(door, ward), sc, "door, not a window")
            o.set_evidence("learned_label", max(door, ward))
            log["windows_to_doors"] += 1
    return log


def _type_gap(o, b, labels, drawing, lengths, log):
    """Generic opening: look at labelled lines lying in the gap."""
    from shapely.geometry import LineString
    span = LineString([o.p0, o.p1]).buffer(labels.sp["opening_search_m"])
    rids = set()
    for g in drawing.entities:
        if g.valid and g.points is not None and g.raw_id in labels.ent and len(g.points) >= 2:
            c = np.asarray(g.points, float).mean(axis=0)
            if span.bounds[0] <= c[0] <= span.bounds[2] and span.bounds[1] <= c[1] <= span.bounds[3] \
                    and span.contains(LineString(g.points[:2]).centroid):
                rids.add(g.raw_id)
    if not rids:
        return
    share, cov, best, sc = labels.summary(rids, lengths)
    if not share:
        return
    door, win = share.get("door", 0.0) + share.get("opening", 0.0), share.get("window", 0.0)
    if door >= labels.sp["retype_share"] and sc >= labels.sp["veto_score"]:
        o.type, o.subtype = "door", best.replace(" ", "_")
    elif win >= labels.sp["retype_share"] and sc >= labels.sp["veto_score"]:
        o.type, o.subtype = "window", best.replace(" ", "_")
    else:
        return
    o.add_evidence("learned_label", max(door, win))
    o.status_reason = f"{labels.model}: gap typed as {o.type}"
    _note(o, labels, best, max(door, win), sc, f"{o.type} symbol in the gap")
    log["openings_typed"] += 1
