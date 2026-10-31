"""Layer semantic analysis (Stage 3).

Layer names give *hints*. Each layer gets category scores from configurable keyword
tables, plus statistics (entity distribution, colour, linetype, lengths) and a geometric
profile (share of linework that has a parallel partner at wall-like spacing). Detectors
read ``wall_evidence`` as one evidence feature among many."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np


def tokenize(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    toks = [t.lower() for t in re.split(r"[^A-Za-z0-9]+", s) if t]
    out = []
    for t in toks:
        m = re.match(r"^([a-z]+)(\d+)$", t)  # WALL200 -> wall, 200
        if m:
            out.extend([m.group(1), m.group(2)])
        else:
            out.append(t)
    return out


def category_scores(name: str, categories: dict) -> dict[str, float]:
    toks = tokenize(name)
    scores: dict[str, float] = {}
    for cat, kw in categories.items():
        best = 0.0
        for tok in toks:
            for strength, value in (("strong", 1.0), ("medium", 0.6)):
                for k in kw.get(strength, []):
                    if tok == k or (len(k) >= 4 and tok.startswith(k)) or (len(tok) >= 5 and len(k) >= 5 and k.startswith(tok)):
                        best = max(best, value)
        if best > 0:
            scores[cat] = best
    return scores


@dataclass
class LayerProfile:
    name: str
    category_scores: dict[str, float] = field(default_factory=dict)
    primary: str | None = None
    wall_evidence: float = 0.5
    entity_counts: dict[str, int] = field(default_factory=dict)
    total_length: float = 0.0
    median_length: float = 0.0
    colors: dict[str, int] = field(default_factory=dict)
    linetypes: dict[str, int] = field(default_factory=dict)
    is_off: bool = False
    parallel_share: float = 0.0
    block_names: dict[str, int] = field(default_factory=dict)

    def to_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def analyze_layers(drawing, segments, cfg, tol) -> dict[str, LayerProfile]:
    sem = cfg["layer_semantics"]
    cats = sem["categories"]
    wev = sem["wall_evidence"]
    profiles: dict[str, LayerProfile] = {}

    def prof(name):
        if name not in profiles:
            li = drawing.layers.get(name)
            sc = category_scores(name, cats)
            p = LayerProfile(name=name, category_scores=sc, is_off=bool(li and (li.is_off or li.is_frozen)))
            if sc:
                p.primary = max(sc, key=sc.get)
                # strongest category drives the wall evidence; ties use the max evidence
                top = max(sc.values())
                vals = [wev.get(c, wev["_unknown"]) for c, v in sc.items() if v == top]
                p.wall_evidence = float(max(vals)) if p.primary == "wall" else float(min(vals))
                if "wall" in sc and p.primary != "wall":
                    p.wall_evidence = max(p.wall_evidence, 0.5 * sc["wall"])
            else:
                p.wall_evidence = wev["_unknown"]
            profiles[name] = p
        return profiles[name]

    for name in drawing.layers:
        prof(name)
    for raw in drawing.raw_entities.values():
        p = prof(raw.layer)
        p.entity_counts[raw.dxftype] = p.entity_counts.get(raw.dxftype, 0) + 1
        col = str(raw.attributes.get("color", "BYLAYER"))
        p.colors[col] = p.colors.get(col, 0) + 1
        lt = str(raw.attributes.get("linetype", "BYLAYER"))
        p.linetypes[lt] = p.linetypes.get(lt, 0) + 1
        if raw.dxftype == "INSERT":
            nm = raw.attributes.get("name", "?")
            p.block_names[nm] = p.block_names.get(nm, 0) + 1
    by_layer: dict[str, list[float]] = {}
    for s in segments:
        if s.dup_of is None:
            by_layer.setdefault(s.layer, []).append(s.length)
    for name, lens in by_layer.items():
        p = prof(name)
        p.total_length = float(np.sum(lens))
        p.median_length = float(np.median(lens))
    return profiles


def layer_parallel_shares(strokes, pairs) -> dict[str, float]:
    """Share of each layer's stroke length that participates in a wall-like parallel pair."""
    tot: dict[str, float] = {}
    used: dict[str, float] = {}
    paired = set()
    for pr in pairs:
        paired.add(pr.i)
        paired.add(pr.j)
    for k, st in enumerate(strokes):
        tot[st.layer] = tot.get(st.layer, 0.0) + st.length
        if k in paired:
            used[st.layer] = used.get(st.layer, 0.0) + st.length
    return {k: used.get(k, 0.0) / v for k, v in tot.items() if v > 0}
