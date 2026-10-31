"""Stage 11: stable intermediate architectural model + JSON / SQLite export.

The model (schema ``archmodel`` 1.x) is the *only* contract between Phase 1 and Phase 2.
Phase 2 does not need to know how anything was detected: it reads validated walls, openings,
rooms, columns and relationships. Unknown values (heights, single-line thickness) are
``null``; Phase 2 resolves them from its own configurable defaults.

To target an existing Phase 2 database, implement ``Phase2Adapter.write``."""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from .. import __version__
from ..geometry.geometric_utils import coords_list
from ..models.base_entity import Status

SCHEMA = "archmodel"
SCHEMA_VERSION = "1.0.0"


def _src(drawing, ids):
    return drawing.raw_refs(ids)


def build_model(st, cfg) -> dict[str, Any]:
    b = st.building
    d = st.drawing
    walls = []
    for w in b.walls.values():
        x = w.to_dict()
        x["source_entities"] = _src(d, w.source_entities)
        walls.append(x)
    openings = []
    for o in b.openings.values():
        x = o.to_dict()
        x["source_entities"] = _src(d, o.source_entities)
        openings.append(x)
    rooms = []
    for r in b.rooms.values():
        x = r.to_dict()
        x["source_entities"] = _src(d, r.source_entities)
        rooms.append(x)
    columns = []
    for c in b.columns.values():
        x = c.to_dict()
        x["source_entities"] = _src(d, c.source_entities)
        columns.append(x)
    hyps = [h.to_dict() for h in b.hypotheses.values()]
    hrooms = [r.to_dict() for r in b.room_hypotheses.values()]
    rel = []
    for w in b.walls.values():
        if w.status == Status.REJECTED:
            continue
        for oid in w.openings:
            rel.append({"type": "hosts", "from": w.id, "to": oid})
        for rid in sorted(w.rooms):
            rel.append({"type": "bounds", "from": w.id, "to": rid})
        for vid in sorted(w.connections):
            if w.id < vid:
                rel.append({"type": "connects", "from": w.id, "to": vid})
    G = b.graphs.get("rooms")
    if G is not None:
        for a, c, data in G.edges(data=True):
            rel.append({"type": "adjacent", "from": a, "to": c, "walls": data.get("walls", []),
                        "openings": data.get("openings", [])})
    for r in b.rooms.values():
        for o in sorted(r.openings):
            rel.append({"type": "opening_of", "from": o, "to": r.id})
    rep = st.report.to_dict() if st.report else {"summary": {}, "issues": []}
    counts = {}
    for name, coll in (("walls", walls), ("openings", openings), ("rooms", rooms), ("columns", columns)):
        c = {}
        for x in coll:
            c[x["status"]] = c.get(x["status"], 0) + 1
        counts[name] = c
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generator": {"name": "architectural_converter", "version": __version__},
        "units": "m",
        "source": {
            "file": d.source_file,
            "dxf_units": d.units.unit if d.units else None,
            "unit_confidence": d.units.confidence if d.units else None,
            "insunits_header": d.units.insunits if d.units else None,
            "transform": d.transform.to_dict(),
            "metadata": {k: v for k, v in d.metadata.items() if k != "paperspace_entity_counts"},
            "summary": d.summary(),
        },
        "export_statuses": cfg["runtime"]["export_statuses"],
        "building": {
            "footprint": coords_list(b.footprint) if b.footprint is not None and b.footprint.geom_type == "Polygon" else
            ([coords_list(g) for g in b.footprint.geoms] if b.footprint is not None and hasattr(b.footprint, "geoms") else None),
            "levels": [{"id": "L0", "name": "Level 0", "elevation": 0.0, "height": None}],
        },
        "walls": walls,
        "openings": openings,
        "rooms": rooms,
        "columns": columns,
        "hypotheses": {"walls": hyps, "rooms": hrooms},
        "relationships": rel,
        "validation": rep,
        "statistics": {"counts": counts, "timings_s": st.timings, "logs": _jsonable(st.logs),
                       "thickness_clusters": st.ctx.thickness_clusters},
        "corrections": st.corrections,
    }


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (int, float, str)) or x is None:
        return x
    if hasattr(x, "item"):
        return x.item()
    return str(x)


def phase2_view(model: dict) -> dict:
    """Filter the model to what Phase 2 should build (configured export statuses)."""
    ok = set(model.get("export_statuses", ["ACCEPTED", "NEEDS_REVIEW"]))
    m = dict(model)
    m["walls"] = [w for w in model["walls"] if w["status"] in ok and not w.get("is_hypothesis")]
    wall_ids = {w["id"] for w in m["walls"]}
    m["openings"] = [o for o in model["openings"] if o["status"] in ok and o.get("host_wall") in wall_ids]
    m["rooms"] = [r for r in model["rooms"] if r["status"] in ok]
    m["columns"] = [c for c in model["columns"] if c["status"] in ok]
    return m


def export_json(model: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=1, default=_jsonable)
    return path


SQL = """
CREATE TABLE IF NOT EXISTS model_info (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS walls (id TEXT PRIMARY KEY, classification TEXT, status TEXT, confidence REAL,
  kind TEXT, start_x REAL, start_y REAL, end_x REAL, end_y REAL, thickness REAL, thickness_source TEXT,
  height REAL, length REAL, polygon_json TEXT, geometry_json TEXT, detection_methods TEXT,
  evidence_json TEXT, reconstruction_json TEXT, is_hypothesis INTEGER);
CREATE TABLE IF NOT EXISTS openings (id TEXT PRIMARY KEY, type TEXT, subtype TEXT, classification TEXT,
  status TEXT, confidence REAL, host_wall TEXT, center_x REAL, center_y REAL, start_x REAL, start_y REAL,
  end_x REAL, end_y REAL, width REAL, height REAL, sill_height REAL, host_offset_start REAL,
  host_offset_end REAL, evidence_json TEXT);
CREATE TABLE IF NOT EXISTS rooms (id TEXT PRIMARY KEY, name TEXT, classification TEXT, status TEXT,
  confidence REAL, area REAL, perimeter REAL, polygon_json TEXT, label_texts TEXT, evidence_json TEXT,
  is_hypothesis INTEGER);
CREATE TABLE IF NOT EXISTS columns (id TEXT PRIMARY KEY, classification TEXT, status TEXT, confidence REAL,
  shape TEXT, size_x REAL, size_y REAL, center_x REAL, center_y REAL, height REAL, polygon_json TEXT);
CREATE TABLE IF NOT EXISTS relationships (type TEXT, from_id TEXT, to_id TEXT, data_json TEXT);
CREATE TABLE IF NOT EXISTS source_entities (element_id TEXT, raw_id TEXT, handle TEXT, dxftype TEXT,
  layer TEXT, block TEXT);
CREATE TABLE IF NOT EXISTS validation_issues (id TEXT PRIMARY KEY, category TEXT, code TEXT, severity TEXT,
  entity_id TEXT, x REAL, y REAL, description TEXT, suggestion TEXT, confidence REAL, sources_json TEXT);
"""


def export_sqlite(model: dict, path: str) -> str:
    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        for t in ("model_info", "walls", "openings", "rooms", "columns", "relationships", "source_entities",
                  "validation_issues"):
            cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.executescript(SQL)
        for k in ("schema", "schema_version", "units"):
            cur.execute("INSERT INTO model_info VALUES (?, ?)", (k, str(model[k])))
        cur.execute("INSERT INTO model_info VALUES (?, ?)", ("source", json.dumps(model["source"], default=_jsonable)))
        cur.execute("INSERT INTO model_info VALUES (?, ?)", ("building", json.dumps(model["building"])))
        cur.execute("INSERT INTO model_info VALUES (?, ?)", ("corrections", json.dumps(model.get("corrections", {}))))
        all_walls = model["walls"] + model["hypotheses"]["walls"]
        for w in all_walls:
            g = w["geometry"]
            s = g.get("start") or (g.get("axis") or [[None, None]])[0]
            e = g.get("end") or (g.get("axis") or [[None, None]])[-1]
            cur.execute("INSERT INTO walls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (w["id"], w["classification"], w["status"], w["confidence"], g["kind"], s[0], s[1], e[0], e[1],
                         w["thickness"], w["thickness_source"], w["height"], g.get("length"), json.dumps(g.get("polygon")),
                         json.dumps(g), json.dumps(w["detection_methods"]), json.dumps(w["evidence"]),
                         json.dumps(w["reconstruction"], default=_jsonable), int(bool(w.get("is_hypothesis")))))
            for s_ in w.get("source_entities", []):
                if isinstance(s_, dict):
                    cur.execute("INSERT INTO source_entities VALUES (?,?,?,?,?,?)",
                                (w["id"], s_["id"], s_["handle"], s_["type"], s_["layer"], s_["block"]))
        for o in model["openings"]:
            p = o["position"]
            hi = o.get("host_interval") or [None, None]
            cur.execute("INSERT INTO openings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (o["id"], o["type"], o["subtype"], o["classification"], o["status"], o["confidence"],
                         o["host_wall"], p["center"][0], p["center"][1], p["start"][0], p["start"][1], p["end"][0],
                         p["end"][1], o["width"], o["height"], o["sill_height"], hi[0], hi[1], json.dumps(o["evidence"])))
            for s_ in o.get("source_entities", []):
                cur.execute("INSERT INTO source_entities VALUES (?,?,?,?,?,?)",
                            (o["id"], s_["id"], s_["handle"], s_["type"], s_["layer"], s_["block"]))
        for r in model["rooms"] + model["hypotheses"]["rooms"]:
            cur.execute("INSERT INTO rooms VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (r["id"], r["name"], r["classification"], r["status"], r["confidence"], r["area"], r["perimeter"],
                         json.dumps(r["polygon"]), json.dumps(r["label_texts"]), json.dumps(r["evidence"]),
                         int(bool(r.get("is_hypothesis")))))
        for c in model["columns"]:
            cur.execute("INSERT INTO columns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (c["id"], c["classification"], c["status"], c["confidence"], c["shape"], c["size"][0],
                         c["size"][1], c["center"][0] if c["center"] else None, c["center"][1] if c["center"] else None,
                         c["height"], json.dumps(c["polygon"])))
        for rl in model["relationships"]:
            data = {k: v for k, v in rl.items() if k not in ("type", "from", "to")}
            cur.execute("INSERT INTO relationships VALUES (?,?,?,?)", (rl["type"], rl["from"], rl["to"], json.dumps(data)))
        for i in model["validation"]["issues"]:
            loc = i.get("location") or [None, None]
            cur.execute("INSERT INTO validation_issues VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (i["id"], i["category"], i["code"], i["severity"], i["entity_id"], loc[0], loc[1],
                         i["description"], i["suggestion"], i["confidence"], json.dumps(i["source_entities"])))
        con.commit()
    finally:
        con.close()
    return path


class Phase2Adapter:
    """Base class for writing the model into an existing Phase 2 database schema."""

    def write(self, model: dict) -> None:  # pragma: no cover - integration point
        raise NotImplementedError


class SQLiteAdapter(Phase2Adapter):
    def __init__(self, path: str):
        self.path = path

    def write(self, model: dict) -> None:
        export_sqlite(model, self.path)
