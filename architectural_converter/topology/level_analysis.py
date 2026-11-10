"""Multi-storey analysis: which part of the drawing is which floor, and how floors stack.

Permit sets and many office drawings place every floor plan side by side in model space (often
with the same site context repeated around each one), plus elevations and sections. This module:

1. splits the extracted walls into separate *views* (spatial clusters of wall bodies);
2. reads each view's identity from its own text: finished-floor-level annotations
   ("708.75 F.F.L", "FFL +3.20", "NIV. 0.00") and level names ("GROUND FLOOR", "BASEMENT 2",
   room tags such as "Lobby GF", "Entrance F1-2", "B3 PARKING");
3. classifies views: floor plan, elevation/section (several levels annotated in one view or
   explicit titles), duplicate (a second view of an already identified level, e.g. a
   structural or sanitary plan), or unidentified;
4. aligns every floor plan onto a reference floor by voting on the displacement between their
   wall/linework end points (repeated cores, columns and site outlines dominate the vote);
5. derives floor-to-floor heights from consecutive levels.

Stacking is only applied when at least two different levels are identified; otherwise the
drawing stays a single storey (nothing is invented). Every decision is recorded on the level.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import Point, box
from shapely.strtree import STRtree

from ..models.base_entity import Status

FFL_RE = re.compile(r"([+-]?\d{1,4}(?:[.,]\d{1,3})?)\s*(?:m\b)?\s*(?:F\.?\s*F\.?\s*L\.?|FFL|F\.?L\.?\b|NGF|N\.?G\.?F\.?)", re.I)
FFL_PREFIX_RE = re.compile(r"(?:F\.?\s*F\.?\s*L\.?|FFL|NIV(?:EAU)?\.?|LEVEL|EL\.?)\s*[:=]?\s*([+-]?\d{1,4}(?:[.,]\d{1,3})?)", re.I)
NUMBER_RE = re.compile(r"^\s*([+-]?\d{1,4}[.,]\d{1,3})\s*$")
FFL_WORD_RE = re.compile(r"^\s*(?:F\.?\s*F\.?\s*L\.?|FFL)\s*$", re.I)
NONPLAN_RE = re.compile(r"\b(ELEVATION|ELEVATIONS|SECTION|SECTIONS|COUPE|FACADE|FAÇADE|ANSICHT|SCHNITT|ALZADO|SECCION)\b", re.I)

ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
            "fifth": 5, "5th": 5, "sixth": 6, "6th": 6, "seventh": 7, "7th": 7, "eighth": 8, "8th": 8,
            "ninth": 9, "9th": 9, "tenth": 10, "10th": 10}


def level_rank_from_text(text: str) -> tuple[int, str] | None:
    """(rank, canonical name) from a title or room tag; ground = 0, basements negative."""
    t = " " + re.sub(r"[_\-]", " ", text.upper()) + " "
    m = re.search(r"\bBASEMENT\s*(\d+)\b|\bSOUS[\s-]?SOL\s*(\d+)?\b|\bSS\s*(\d)\b", t)
    if m:
        n = int(next(g for g in m.groups() if g) if any(m.groups()) else 1)
        return -n, f"Basement {n}"
    if re.search(r"\bGROUND\s+FLOOR\b|\bREZ[\s-]DE[\s-]CHAUSS|\bRDC\b|\bERDGESCHOSS\b|\bPLANTA\s+BAJA\b", t):
        return 0, "Ground floor"
    m = re.search(r"\b(FIRST|1ST|SECOND|2ND|THIRD|3RD|FOURTH|4TH|FIFTH|5TH|SIXTH|6TH|SEVENTH|7TH|EIGHTH|8TH|NINTH|9TH|TENTH|10TH)\s+FLOOR\b", t)
    if m:
        n = ORDINALS[m.group(1).lower()]
        return n, f"Floor {n}"
    m = re.search(r"\b(?:FLOOR|LEVEL|ETAGE|ÉTAGE|NIVEAU|OG)\s*(\d{1,2})\b", t)
    if m:
        return int(m.group(1)), f"Floor {int(m.group(1))}"
    if re.search(r"\bATTIC\b|\bCOMBLES?\b|\bPENTHOUSE\b", t):
        return 90, "Attic"
    if re.search(r"\bROOF\b|\bTOITURE\b|\bDACH\b", t):
        return 99, "Roof"
    return None


def level_tag(text: str) -> tuple[int, str] | None:
    """Short level tags used inside room labels: B3, GF, F1, ... (weaker evidence)."""
    t = text.upper()
    m = re.search(r"(?<![A-Z0-9])B(\d)(?![0-9])", t)
    if m:
        return -int(m.group(1)), f"Basement {m.group(1)}"
    if re.search(r"(?<![A-Z0-9])GF(?![A-Z0-9])", t):
        return 0, "Ground floor"
    m = re.search(r"(?<![A-Z0-9])F(\d)(?![0-9])", t)
    if m:
        return int(m.group(1)), f"Floor {m.group(1)}"
    return None


ATTIC_RANK, ROOF_RANK = 90, 99


def rank_name(rank: int) -> str:
    if rank < 0:
        return f"Basement {-rank}"
    if rank == 0:
        return "Ground floor"
    if rank == ATTIC_RANK:
        return "Attic"
    if rank == ROOF_RANK:
        return "Roof"
    return f"Floor {rank}"


def _reconcile(plans, log):
    """Make level names agree with the FFL values.

    FFL values are measured; names often come from texts that merely mention another floor
    ("stair to ground floor"). The FFL-annotated views are put in elevation order and the offset
    between that order and the named ranks is voted; with a clear majority every FFL view gets
    the rank of its position (names that disagree are corrected and say so)."""
    ffl = [lv for lv in plans if lv.elevation is not None]
    if len(ffl) < 2:
        return
    elevs = []
    for e in sorted(lv.elevation for lv in ffl):
        if not elevs or e - elevs[-1] > 0.3:
            elevs.append(e)
    pos = {lv.id: min(range(len(elevs)), key=lambda i: abs(elevs[i] - lv.elevation)) for lv in ffl}
    votes: dict[int, int] = {}
    for lv in ffl:
        if lv.rank is not None and lv.rank < ATTIC_RANK:
            k = lv.rank - pos[lv.id]
            votes[k] = votes.get(k, 0) + 1
    if not votes:
        return
    k, n = max(votes.items(), key=lambda kv: (kv[1], -abs(kv[0])))
    total = sum(votes.values())
    if n < 2 or n <= total / 2:
        return
    log["rank_offset"] = {"offset": k, "votes": n, "of": total}
    top = len(elevs) - 1
    for lv in ffl:
        r = pos[lv.id] + k
        if lv.rank is not None and lv.rank >= ATTIC_RANK and pos[lv.id] == top:
            continue                                   # attic / roof named on the top level: keep
        if lv.rank != r:
            old = lv.name or "unnamed"
            lv.rank, lv.name = r, rank_name(r)
            lv.sources.append(f"name corrected from '{old}' by FFL order")


def _fill_elevations(plans, cfg):
    """Views with a level name but no FFL get an elevation from the named levels around them."""
    known = sorted((lv.rank, lv.elevation) for lv in plans if lv.elevation is not None and lv.rank is not None
                   and lv.rank < ATTIC_RANK)
    if not known:
        return
    hs = [(e2 - e1) / (r2 - r1) for (r1, e1), (r2, e2) in zip(known, known[1:]) if r2 != r1]
    typical = float(np.median(hs)) if hs else float(cfg["phase2"]["wall_height_m"])
    top_e = max(lv.elevation for lv in plans if lv.elevation is not None)
    for lv in plans:
        if lv.elevation is not None or lv.rank is None:
            continue
        if lv.rank >= ATTIC_RANK:
            lv.elevation = top_e + typical
        else:
            r0, e0 = min(known, key=lambda re_: abs(re_[0] - lv.rank))
            lv.elevation = e0 + (lv.rank - r0) * typical
        lv.sources.append(f"elevation {lv.elevation:.2f} inferred from level name")
        lv.confidence = min(lv.confidence, 0.6)


def parse_ffl(texts: list[tuple[str, tuple[float, float], float]]) -> list[float]:
    """Finished-floor-level values from texts (value with FFL marker, or number next to an FFL word)."""
    vals = []
    words = [(p, h) for s, p, h in texts if FFL_WORD_RE.match(s)]
    for s, p, h in texts:
        for rx in (FFL_RE, FFL_PREFIX_RE):
            for m in rx.finditer(s):
                try:
                    vals.append(float(m.group(1).replace(",", ".")))
                except ValueError:
                    pass
        m = NUMBER_RE.match(s)
        if m and words:
            # level marker blocks often keep the number and the "F.F.L" word as separate texts
            r = max(h, 0.05) * 8
            if any(math.dist(p, q) <= r + 3 * max(hq, 0.05) for q, hq in words):
                vals.append(float(m.group(1).replace(",", ".")))
    return vals


@dataclass
class Level:
    id: str
    name: str
    rank: int | None
    elevation: float | None
    kind: str                       # floor_plan | elevation_or_section | duplicate | unidentified
    bbox: tuple[float, float, float, float]
    translation: tuple[float, float] = (0.0, 0.0)
    height: float | None = None
    z: float = 0.0
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.5
    alignment: dict[str, Any] = field(default_factory=dict)
    wall_ids: list[str] = field(default_factory=list)
    outline: list = field(default_factory=list)         # closed outline(s) of the storey (level_slabs.py)
    outline_source: str | None = None                   # plan | wall_vertices

    def to_dict(self) -> dict[str, Any]:
        return {"outline": [[[round(float(x), 4), round(float(y), 4)] for x, y in p.exterior.coords] for p in self.outline],
                "outline_source": self.outline_source, "id": self.id, "name": self.name, "rank": self.rank,
                "elevation": round(self.elevation, 3) if self.elevation is not None else None,
                "kind": self.kind, "bbox": [round(v, 3) for v in self.bbox],
                "translation": [round(self.translation[0], 4), round(self.translation[1], 4)],
                "height": round(self.height, 3) if self.height else None, "z": round(self.z, 3),
                "sources": self.sources, "confidence": round(self.confidence, 3), "alignment": self.alignment,
                "walls": len(self.wall_ids)}


def _cluster_views(walls, gap: float):
    """Union-find of wall bodies whose bounding boxes come within ``gap`` of each other."""
    boxes = []
    for w in walls:
        p = w.polygon(0.05)
        if p is None:
            continue
        boxes.append((w, p.bounds))
    n = len(boxes)
    if n == 0:
        return []
    tree = STRtree([box(*b) for _, b in boxes])
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, (_, b) in enumerate(boxes):
        for j in tree.query(box(b[0] - gap, b[1] - gap, b[2] + gap, b[3] + gap)):
            j = int(j)
            if j > i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        ws = [boxes[i][0] for i in idx]
        bs = np.array([boxes[i][1] for i in idx])
        out.append((ws, (float(bs[:, 0].min()), float(bs[:, 1].min()), float(bs[:, 2].max()), float(bs[:, 3].max()))))
    out.sort(key=lambda g: (g[1][0], -g[1][3]))
    return out


def _align(points_ref: np.ndarray, points: np.ndarray, grid: float, max_pts: int = 400):
    """Translation d maximising matches of ``points + d`` onto ``points_ref`` (vote on differences)."""
    if len(points_ref) == 0 or len(points) == 0:
        return None, 0, 0
    rng = np.random.default_rng(0)
    a = points if len(points) <= max_pts else points[rng.choice(len(points), max_pts, replace=False)]
    b = points_ref if len(points_ref) <= 6000 else points_ref[rng.choice(len(points_ref), 6000, replace=False)]
    diff = (b[None, :, :] - a[:, None, :]).reshape(-1, 2)
    q = np.round(diff / grid).astype(np.int64)
    keys, counts = np.unique(q, axis=0, return_counts=True)
    k = int(np.argmax(counts))
    best = keys[k]
    sel = np.all(q == best, axis=1)
    d = diff[sel].mean(axis=0)
    return (float(d[0]), float(d[1])), int(counts[k]), len(a)


def analyze_levels(walls, labels, strokes, tol, cfg, ids=None):
    """Returns (levels, wall_level: dict wall_id -> level_id | None, log)."""
    lc = cfg.get("levels", {}) if hasattr(cfg, "get") else {}
    gap = float(lc.get("view_gap_m", 4.0))
    live = [w for w in walls if w.status != Status.REJECTED and not w.is_hypothesis]
    views = _cluster_views(live, gap)
    views = [v for v in views if len(v[0]) >= int(lc.get("min_walls_per_view", 4))]
    log = {"views": len(views)}
    if len(views) < 2:
        return [], {}, dict(log, stacked=False, reason="single view")
    label_pts = [Point(lb.position) for lb in labels]
    ltree = STRtree(label_pts) if label_pts else None
    levels: list[Level] = []
    fmax = int(lc.get("sheet_frame_max_walls", 8))

    def encloses(a, b):
        return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]
    for k, (ws, bb) in enumerate(views):
        if len(ws) <= fmax and any(len(ws2) > len(ws) and encloses(bb, bb2) for ws2, bb2 in views if bb2 is not bb):
            levels.append(Level(id=f"level_{k + 1:02d}", name="Sheet frame", rank=None, elevation=None,
                                kind="sheet_frame", bbox=bb, wall_ids=[w.id for w in ws],
                                sources=["few lines enclosing a drawing: sheet border / title block"]))
            continue
        w_, h_ = bb[2] - bb[0], bb[3] - bb[1]
        margin = max(2.0, 0.15 * max(w_, h_))
        region = box(bb[0] - margin, bb[1] - margin, bb[2] + margin, bb[3] + margin)
        texts = []
        if ltree is not None:
            for i in ltree.query(region, predicate="contains"):
                lb = labels[int(i)]
                texts.append((lb.text, lb.position, lb.height))
        ffl = parse_ffl(texts)
        names = [r for r in (level_rank_from_text(t) for t, _, _ in texts) if r]
        tags = [r for r in (level_tag(t) for t, _, _ in texts) if r]
        nonplan = any(NONPLAN_RE.search(t) for t, _, _ in texts)
        lv = Level(id=f"level_{k + 1:02d}", name="", rank=None, elevation=None, kind="unidentified", bbox=bb,
                   wall_ids=[w.id for w in ws])
        spread = (max(ffl) - min(ffl)) if ffl else 0.0
        if nonplan or (len(ffl) >= 3 and spread > float(lc.get("max_ffl_spread_in_plan_m", 2.0))):
            lv.kind = "elevation_or_section"
            lv.sources.append("titles: elevation/section" if nonplan else f"several levels annotated ({spread:.2f} m apart)")
            lv.name = "Elevation / section"
            levels.append(lv)
            continue
        if ffl:
            lv.elevation = float(np.median(ffl))
            lv.sources.append(f"FFL text x{len(ffl)}")
        src = names or tags
        if src:
            ranks = {}
            for r, nm in src:
                ranks[(r, nm)] = ranks.get((r, nm), 0) + 1
            (rank, nm), cnt = max(ranks.items(), key=lambda kv: kv[1])
            lv.rank, lv.name = rank, nm
            lv.sources.append(("title" if names else "room tags") + f" '{nm}' x{cnt}")
        if lv.elevation is not None or lv.rank is not None:
            lv.kind = "floor_plan"
            lv.confidence = 0.9 if (lv.elevation is not None and lv.rank is not None) else 0.7
        levels.append(lv)
    plans = [lv for lv in levels if lv.kind == "floor_plan"]
    _reconcile(plans, log)
    # roof plans show parapets / roof outlines: not a storey to stack
    for lv in plans:
        if lv.rank == ROOF_RANK:
            lv.kind = "roof_plan"
            lv.sources.append("roof plan: not stacked as a storey")
    plans = [lv for lv in levels if lv.kind == "floor_plan"]
    # one view per level: a second view of the same level (structural, sanitary, furniture plan...) is a duplicate
    def key(lv):
        return ("rank", lv.rank) if lv.rank is not None else ("elev", round(lv.elevation, 1))
    by_key: dict = {}
    for lv in plans:
        by_key.setdefault(key(lv), []).append(lv)
    for lst in by_key.values():
        lst.sort(key=lambda lv: (lv.elevation is None, -len(lv.wall_ids)))      # FFL-annotated, then largest
        for dup in lst[1:]:
            dup.kind = "duplicate"
            dup.sources.append(f"same level as {lst[0].id}")
    plans = [lv for lv in levels if lv.kind == "floor_plan"]
    if len(plans) < 2:
        return levels, {}, dict(log, stacked=False, reason="fewer than two identified floor levels")
    _fill_elevations(plans, cfg)
    plans.sort(key=lambda lv: (lv.elevation if lv.elevation is not None else 1e9, lv.rank or 0))
    for lv in plans:
        if not lv.name:
            lv.name = f"Level {lv.elevation:+.2f}" if lv.elevation is not None else lv.id
    # alignment onto the reference (largest) floor by voting on linework end points
    typical = [lv for lv in plans if lv.sources and any(x.startswith("FFL") for x in lv.sources)
               and lv.rank is not None and -50 < lv.rank < 50] or plans
    ref = max(typical, key=lambda lv: len(lv.wall_ids))
    def pts_in(bb):
        P = []
        reg = box(*bb)
        for st in strokes:
            ln = st.line()
            if ln.intersects(reg):
                P.append(st.p0)
                P.append(st.p1)
        return np.array(P, dtype=float) if P else np.zeros((0, 2))
    ref_pts = pts_in(ref.bbox)
    grid = max(tol.snap * 4, 0.02)
    for lv in plans:
        if lv is ref:
            lv.alignment = {"reference": True}
            continue
        d, votes, n = _align(ref_pts, pts_in(lv.bbox), grid)
        fallback = (ref.bbox[0] - lv.bbox[0], ref.bbox[1] - lv.bbox[1])
        if d is None or votes < max(8, 0.1 * n):
            lv.translation = fallback
            lv.alignment = {"method": "bounding box (weak geometric match)", "votes": votes, "points": n}
            lv.confidence *= 0.6
        else:
            lv.translation = d
            lv.alignment = {"method": "linework vote", "votes": votes, "points": n, "reference": ref.id}
    # elevations / heights
    p2 = cfg["phase2"]
    known = [lv for lv in plans if lv.elevation is not None]
    base = min(lv.elevation for lv in known) if known else 0.0
    default_h = float(p2["wall_height_m"])
    z = 0.0
    for i, lv in enumerate(plans):
        if lv.elevation is not None:
            lv.z = lv.elevation - base
        else:
            lv.z = z
        nxt = plans[i + 1] if i + 1 < len(plans) else None
        if nxt is not None and nxt.elevation is not None and lv.elevation is not None:
            lv.height = max(2.0, min(6.0, nxt.elevation - lv.elevation))
            lv.sources.append("height from next FFL")
        else:
            lv.height = None
        z = lv.z + (lv.height or default_h)
    measured = sorted(lv.height for lv in plans if lv.height)
    if measured:                                   # top floor: typical measured storey height
        typical = measured[len(measured) // 2]
        for lv in plans:
            if lv.height is None:
                lv.height = typical
                lv.sources.append("height = typical storey height (no FFL above)")
    wall_level = {}
    for lv in levels:
        for wid in lv.wall_ids:
            wall_level[wid] = lv.id
    log.update({"stacked": True, "floor_plans": len(plans), "reference": ref.id,
                "kinds": {k: sum(1 for lv in levels if lv.kind == k) for k in
                          ("floor_plan", "elevation_or_section", "duplicate", "roof_plan", "sheet_frame",
                           "unidentified")}})
    return levels, wall_level, log
