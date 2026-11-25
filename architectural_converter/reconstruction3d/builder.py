"""Phase 2 (reference implementation): architectural model -> 3D meshes (Shapely + Trimesh).

Reads ONLY the exported architectural model (``archmodel``). Unknown heights/thicknesses are
resolved here from ``config/phase2.json``; every default that is used is listed in
``metadata.defaults_used`` so it is never confused with measured data."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..pipeline.database_exporter import phase2_view
from .mesh_utils import box, extrude

COLORS = {"exterior_wall": [0.80, 0.78, 0.74, 1.0], "interior_wall": [0.92, 0.90, 0.86, 1.0],
          "partition_wall": [0.95, 0.95, 0.93, 1.0], "wall": [0.88, 0.86, 0.82, 1.0],
          "railing": [0.55, 0.60, 0.66, 1.0], "balcony": [0.66, 0.70, 0.72, 1.0], "ground": [0.70, 0.68, 0.62, 1.0],
          "column": [0.62, 0.60, 0.58, 1.0], "floor": [0.76, 0.66, 0.52, 1.0],
          "roof_tiles": [0.72, 0.36, 0.25, 1.0], "stair": [0.70, 0.55, 0.42, 1.0], "review": [0.98, 0.75, 0.45, 1.0], "glass": [0.55, 0.75, 0.95, 0.45]}


class Model3D:
    def __init__(self):
        self.meshes: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {"defaults_used": {}}

    def add(self, mid, mtype, cls, V, F, color):
        if len(V) == 0 or len(F) == 0:
            return
        self.meshes.append({"id": mid, "type": mtype, "classification": cls, "vertices": np.asarray(V, float),
                            "faces": np.asarray(F, int), "color": color})

    def default(self, key, value, who):
        self.metadata["defaults_used"].setdefault(key, {"value": value, "elements": []})["elements"].append(who)
        return value

    def to_json(self) -> dict:
        return {"metadata": self.metadata,
                "meshes": [{"id": m["id"], "type": m["type"], "classification": m["classification"],
                            "color": m["color"], "level": m.get("level"), "positions": np.round(m["vertices"], 4).ravel().tolist(),
                            "indices": m["faces"].ravel().tolist()} for m in self.meshes]}

    def to_trimesh_scene(self):
        import trimesh
        scene = trimesh.Scene()
        for m in self.meshes:
            from trimesh.visual.material import PBRMaterial
            from trimesh.visual.texture import TextureVisuals
            rgba = [float(x) for x in m["color"]]
            mat = PBRMaterial(name=f"mat_{m['type']}", baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.9,
                              alphaMode="BLEND" if rgba[3] < 1 else "OPAQUE", doubleSided=True)
            tm = trimesh.Trimesh(vertices=m["vertices"], faces=m["faces"], process=False,
                                 visual=TextureVisuals(material=mat))
            scene.add_geometry(tm, node_name=m["id"], geom_name=m["id"])
        return scene

    def export_glb(self, path: str | None = None) -> bytes:
        data = self.to_trimesh_scene().export(file_type="glb")
        if path:
            with open(path, "wb") as fh:
                fh.write(data)
        return data


def build_3d(model: dict, cfg) -> Model3D:
    p2 = cfg["phase2"]
    view = phase2_view(model)
    out = Model3D()
    out.metadata["source_schema"] = f"{model['schema']} {model['schema_version']}"
    ops = {}
    for o in view["openings"]:
        ops.setdefault(o["host_wall"], []).append(o)
    # multi-storey placement: each floor plan is moved onto the reference floor and lifted to its level
    levels = {lv["id"]: lv for lv in model.get("building", {}).get("levels", []) if lv.get("id")}
    stacked = bool(model.get("building", {}).get("stacked"))
    out.metadata["levels"] = [{"id": lv["id"], "name": lv["name"], "z": lv["z"], "height": lv["height"]}
                              for lv in levels.values() if lv.get("kind") == "floor_plan"] if stacked else []

    def level_height(e):
        lv = levels.get(e.get("level")) if stacked else None
        return lv.get("height") if lv else None
    for w in view["walls"]:
        g = w["geometry"]
        if w.get("classification") == "railing" and not w.get("height") and g["kind"] in ("straight", "centerline"):
            u = np.array(g["direction"], float)
            n = np.array([-u[1], u[0]])
            a, bb = np.array(g["start"], float), np.array(g["end"], float)
            t = w.get("thickness") or out.default("unknown_wall_thickness_m", p2["unknown_wall_thickness_m"], w["id"])
            _railing(out, w, u, float(a @ n), float(a @ u), float(bb @ u), t, p2)
            continue
        if w.get("classification") == "railing" and not w.get("height"):
            # balcony railing / parapet: an outdoor edge, not a storey-high wall
            H = out.default("railing_height_m", p2.get("railing_height_m", 1.1), w["id"])
        else:
            H = w.get("height") or level_height(w) or out.default("wall_height_m", p2["wall_height_m"], w["id"])
        t = w.get("thickness") or out.default("unknown_wall_thickness_m", p2["unknown_wall_thickness_m"], w["id"])
        color = COLORS.get(w.get("classification") or "wall", COLORS["wall"])
        if w["status"] == "NEEDS_REVIEW":
            color = COLORS["review"]
        if g["kind"] in ("straight", "centerline"):
            u = np.array(g["direction"], float)
            n = np.array([-u[1], u[0]])
            a = np.array(g["start"], float)
            b = np.array(g["end"], float)
            s0, s1 = float(a @ u), float(b @ u)
            c = float(a @ n)
            voids = []
            for o in ops.get(w["id"], []):
                hi = o.get("host_interval")
                if not hi:
                    continue
                x0, x1 = max(min(hi), s0), min(max(hi), s1)
                if x1 - x0 <= 1e-6:
                    continue
                if o["type"] == "window":
                    z0 = o.get("sill_height") or out.default("window_sill_m", p2["window_sill_m"], o["id"])
                    z1 = o.get("height") and z0 + o["height"] or out.default("window_head_m", p2["window_head_m"], o["id"])
                elif o["type"] == "door":
                    z0, z1 = 0.0, (o.get("height") or out.default("door_height_m", p2["door_height_m"], o["id"]))
                else:
                    z0, z1 = 0.0, (o.get("height") or out.default("opening_height_m", p2["opening_height_m"], o["id"]))
                voids.append((x0, x1, z0, min(z1, H), o))
            voids.sort(key=lambda v: v[0])
            Vs, Fs = [], []
            cur = s0
            for x0, x1, z0, z1, o in voids:
                if x0 > cur:
                    Vs.append(box(u, c, cur, x0, t, 0.0, H))
                lo = max(cur, x0)
                if z0 > 0:
                    Vs.append(box(u, c, lo, x1, t, 0.0, z0))
                if z1 < H:
                    Vs.append(box(u, c, lo, x1, t, z1, H))
                if o["type"] == "window":
                    gv, gf = box(u, c, lo, x1, max(0.01, t * 0.1), z0, z1)
                    out.add(o["id"], "window", o.get("classification"), gv, gf, COLORS["glass"])
                cur = max(cur, x1)
            if cur < s1:
                Vs.append(box(u, c, cur, s1, t, 0.0, H))
            V, F = _merge(Vs)
            out.add(w["id"], "wall", w.get("classification"), V, F, color)
        else:
            poly = g.get("polygon")
            if poly:
                V, F = extrude(poly["exterior"], 0.0, H)
                out.add(w["id"], "wall", w.get("classification"), V, F, color)
    for col in view["columns"]:
        H = col.get("height") or level_height(col) or out.default("column_height_m", p2["column_height_m"], col["id"])
        if col.get("polygon"):
            V, F = extrude(col["polygon"]["exterior"], 0.0, H)
            out.add(col["id"], "column", col.get("classification"), V, F, COLORS["column"])
    for sf in view.get("stairs", []):
        _stair_mesh(out, sf)
    slab = p2["slab_thickness_m"]
    for r in view["rooms"]:
        if r.get("polygon"):
            if r.get("floor_parts") is not None:         # a manual floor cut went through this room
                V, F = _merge([extrude(q, -slab, 0.0) for q in r["floor_parts"]]) if r["floor_parts"] else (np.zeros((0, 3)), np.zeros((0, 3), int))
            else:
                V, F = extrude(r["polygon"]["exterior"], -slab, 0.0)
            out.add(r["id"], "floor", r.get("classification"), V, F,
                    COLORS["balcony"] if r.get("classification") == "balcony" else COLORS["floor"])
    ground_levels = _ground(out, model, view, p2, levels, stacked)
    if stacked and p2.get("stack_levels", True):
        ground_levels.update(_level_slabs(out, p2, levels))
        _pitched_roofs(out, p2, levels, view, ground_levels)
    # stack_levels false: every floor stays where it is drawn, side by side on the ground
    out.metadata["stacked_in_3d"] = bool(stacked and p2.get("stack_levels", True))
    if stacked and p2.get("stack_levels", True):
        level_of = {e["id"]: e.get("level") for k in ("walls", "openings", "rooms", "columns", "stairs")
                    for e in view.get(k, [])}
        level_of.update(ground_levels)
        for m in out.meshes:
            lv = levels.get(level_of.get(m["id"]) or level_of.get(m["id"].removesuffix("_glass")))
            if lv is None:
                continue
            tx, ty = lv.get("translation") or (0.0, 0.0)
            m["vertices"] = m["vertices"] + np.array([tx, ty, lv.get("z") or 0.0])
            m["level"] = lv["id"]
    allv = np.vstack([m["vertices"] for m in out.meshes]) if out.meshes else np.zeros((1, 3))
    out.metadata["bounds"] = [allv.min(axis=0).round(3).tolist(), allv.max(axis=0).round(3).tolist()]
    out.metadata["counts"] = {k: sum(1 for m in out.meshes if m["type"] == k) for k in ("wall", "window", "column", "floor", "stair", "roof")}
    return out


def _ground(out, model, view, p2, levels, stacked) -> dict:
    """Ground slabs for the areas the model lists in building.ground (computed by Phase 1, see
    topology/ground.py): no geometry operations here, only extrusion."""
    from shapely.geometry import Polygon
    from shapely.ops import triangulate
    if not p2.get("ground", {}).get("enabled", True):
        return {}
    slab = p2["slab_thickness_m"]
    level_of, area = {}, 0.0
    for k, gp in enumerate(model.get("building", {}).get("ground") or []):
        pg = gp.get("polygon")
        if not pg or (stacked and gp.get("level") not in levels):
            continue
        try:
            if pg.get("holes"):
                p = Polygon(pg["exterior"], pg["holes"])
                parts = [list(t.exterior.coords) for t in triangulate(p) if p.contains(t.representative_point())]
            else:
                parts = [pg["exterior"]]
            V, F = _merge([extrude(r, -slab, 0.0) for r in parts])
        except Exception:  # noqa: BLE001  (an optional slab never stops the 3D build)
            continue
        gid = f"ground_{k + 1:03d}"
        out.add(gid, "floor", "ground", V, F, COLORS["ground"])
        level_of[gid] = gp.get("level")
        area += gp.get("area") or 0.0
    if level_of:
        out.metadata["ground_slabs"] = {"count": len(level_of), "area_m2": round(area, 2)}
    return level_of


def _level_slabs(out, p2, levels) -> dict:
    """Stacked storeys: a base slab under and a roof slab over each floor-plan level, from the
    level's closed outline (building.levels[].outline, computed by Phase 1).

    The base slab lies a few millimetres inside the room floors, and the roof slab inside the
    floor of the storey above (or on top of the walls of the highest storey), so the slabs show
    exactly where nothing else covers. Levels named as attic / roof keep their top open."""
    lc = p2.get("level_slabs", {})
    if not lc.get("enabled", True):
        return {}
    slab, eps = p2["slab_thickness_m"], 0.005
    plans = sorted((lv for lv in levels.values() if lv.get("kind") == "floor_plan"), key=lambda lv: lv.get("z") or 0.0)
    no_roof = [n.lower() for n in lc.get("no_roof_names", [])]
    level_of, info = {}, []
    for k, lv in enumerate(plans):
        rings = lv.get("outline") or []
        H = lv.get("height") or p2["wall_height_m"]
        name = (lv.get("name") or "").lower()
        open_top = any(n in name for n in no_roof)
        pitched = bool(open_top and (lv.get("roof") or {}).get("kind") == "pitched"
                       and p2.get("pitched_roof", {}).get("enabled", True))
        top = k == len(plans) - 1
        made = {"level": lv["id"], "name": lv.get("name"), "outline": lv.get("outline_source"), "base": False,
                "roof": False if not open_top else ("pitched" if pitched else "open (attic / roof storey)")}
        try:
            base = _merge([extrude(r, -slab + eps, -eps) for r in rings]) if rings else None
            if top:
                z0, z1 = H, H + slab
            else:                                  # = the floor zone of the storey above (levels can be
                up = (plans[k + 1].get("z") or 0.0) - (lv.get("z") or 0.0)     # further apart than the wall height)
                z0, z1 = up - slab + eps, up - eps
            roof = _merge([extrude(r, z0, z1) for r in rings]) if rings and not open_top else None
        except Exception:  # noqa: BLE001  (an optional slab never stops the 3D build)
            base = roof = None
        if base is not None and len(base[0]):
            out.add(f"{lv['id']}_base", "floor", "level_slab", base[0], base[1], COLORS["ground"])
            level_of[f"{lv['id']}_base"] = lv["id"]
            made["base"] = True
        if roof is not None and len(roof[0]):
            out.add(f"{lv['id']}_roof", "roof", "level_roof", roof[0], roof[1], COLORS["ground"])
            level_of[f"{lv['id']}_roof"] = lv["id"]
            made["roof"] = True
        info.append(made)
    out.metadata["level_slabs"] = info
    return level_of


def _pitched_roofs(out, p2, levels, view, level_of) -> None:
    """Attic / roof storeys: a tiled hip roof over the footprint Phase 1 gives (building.levels[].roof),
    and everything of that storey trimmed under its slope (the attic room is inside the roof).
    Runs before the storeys are moved to their levels: z = 0 is the attic floor.

    * Cores rising through the roof (footprint abutments) get a flat top at the roof's top height
      and closed sides; what stands in them keeps its height up to that top.
    * Dormers (footprint dormers) are gabled boxes on the eave line.
    * Walls of the storey that stand outside the roof (terrace edges) are parapets: capped at
      railing_height_m (``walls_outside_roof``: "parapet", or "keep")."""
    from .roof import as_polygon, dormer_meshes, hip_roof
    pr = p2.get("pitched_roof", {})
    if not pr.get("enabled", True):
        return
    tile = pr.get("tile", {})
    cols = tile.get("colors") or [COLORS["roof_tiles"]]
    dcfg = pr.get("dormers", {})
    of = {e["id"]: e.get("level") for k in ("walls", "openings", "columns", "stairs") for e in view.get(k, [])}
    for lv in levels.values():
        roof = lv.get("roof") or {}
        if lv.get("kind") != "floor_plan" or roof.get("kind") != "pitched" or not roof.get("footprint"):
            continue
        pitch = float(pr.get("pitch_deg", 30.0))
        over, knee = float(pr.get("overhang_m", 0.0)), float(pr.get("knee_wall_m", 1.0))
        total = pr.get("max_height_m") or lv.get("height") or p2["wall_height_m"]
        eave = knee - over * np.tan(np.radians(pitch))         # an overhanging eave hangs below the knee-wall top
        max_rise = max(0.3, float(total) - eave)
        rows, decks, sides, dw, dr, surfaces, rise = ([], []), [], [], [], [], [], 0.0
        for fp in roof["footprint"]:
            try:
                r = hip_roof(fp, pitch, max_rise, eave, float(tile.get("course_m", 0.35)),
                             float(tile.get("thickness_m", 0.05)), int(tile.get("max_courses", 60)))
            except Exception:  # noqa: BLE001  (an optional roof never stops the 3D build)
                continue
            for k in (0, 1):
                if len(r["courses"][k][1]):
                    rows[k].append(r["courses"][k])
            if len(r["deck"][1]):
                decks.append(r["deck"])
            if len(r["abutment_sides"][1]):
                sides.append(r["abutment_sides"])
            if r["surface"] is not None:
                surfaces.append(r["surface"])
                if dcfg.get("enabled", True) and isinstance(fp, dict):
                    for d in fp.get("dormers") or []:
                        try:
                            m = dormer_meshes(d, r["surface"], eave, float(dcfg.get("pitch_deg", 45.0)),
                                              float(dcfg.get("min_rise_m", 0.4)))
                        except Exception:  # noqa: BLE001
                            m = None
                        if m:
                            dw.append(m[0])
                            dr.append(m[1])
            rise = max(rise, r["rise"])
        if not rows[0] and not rows[1]:
            continue
        for k, sfx in ((0, "_roof"), (1, "_roof_b")):          # alternate rows of tiles, two tones
            if rows[k]:
                V, F = _merge(rows[k])
                out.add(f"{lv['id']}{sfx}", "roof", "pitched_roof", V, F, cols[k % len(cols)])
                level_of[f"{lv['id']}{sfx}"] = lv["id"]
        for mid, cls, parts, color in ((f"{lv['id']}_roof_deck", "roof_deck", decks, tile.get("deck_color") or COLORS["ground"]),
                                       (f"{lv['id']}_roof_core", "roof_abutment", sides, COLORS["exterior_wall"]),
                                       (f"{lv['id']}_dormers", "dormer", dw, COLORS["exterior_wall"]),
                                       (f"{lv['id']}_dormer_roofs", "dormer_roof", dr, cols[0])):
            if parts:
                V, F = _merge(parts)
                out.add(mid, "roof", cls, V, F, color)
                level_of[mid] = lv["id"]
        # trim the storey's own elements under the slope (a chord under a roof plane stays under it);
        # what stands in a core keeps its height up to the core's top; what stands outside the roof
        # (on a terrace) is a parapet: nothing up there is carried by it
        near_m = float(pr.get("outside_tolerance_m", 0.3))
        parapet = float(p2.get("railing_height_m", 1.1)) if pr.get("walls_outside_roof", "parapet") == "parapet" else None
        cap_types = set(pr.get("outside_roof_types", ["wall", "column"]))
        trimmed = outside = 0
        for m in out.meshes:
            base = m["id"].removesuffix("_glass")
            if m["type"] in ("floor", "roof") or (of.get(m["id"]) or of.get(base)) != lv["id"]:
                continue
            Vm = m["vertices"]
            capped = False
            for i in np.nonzero(Vm[:, 2] > 1e-6)[0]:
                z = None
                for s in surfaces:
                    h = s.height(float(Vm[i, 0]), float(Vm[i, 1]), near_m)
                    if h is not None:
                        z = h if z is None else max(z, h)
                if z is not None:
                    if Vm[i, 2] > z - 0.02:
                        Vm[i, 2] = max(0.0, z - 0.02)
                        trimmed += 1
                elif parapet is not None and (m["type"] in cap_types or base != m["id"]) and Vm[i, 2] > parapet + 1e-6:
                    Vm[i, 2] = parapet
                    capped = True
            outside += capped
        out.metadata.setdefault("pitched_roofs", []).append(
            {"level": lv["id"], "name": lv.get("name"), "pitch_deg": pitch, "eave_m": round(float(eave), 3),
             "ridge_m": round(float(eave + rise), 3), "footprint": roof.get("source"),
             "open_middle": any(isinstance(fp, dict) and fp.get("holes") for fp in roof["footprint"]),
             "abutments": sum(len(fp.get("abutments") or []) for fp in roof["footprint"] if isinstance(fp, dict)),
             "dormers": len(dw), "capped_outside_roof": outside, "vertices_trimmed": trimmed, "regularization": roof.get("regularization"),
             "tile": {"course_m": tile.get("course_m", 0.35), "thickness_m": tile.get("thickness_m", 0.05)}})


def _railing(out, w, u, c, s0, s1, t, p2):
    """Balcony wall: a low solid parapet with glass above it up to the railing height."""
    top = out.default("railing_height_m", p2.get("railing_height_m", 1.1), w["id"])
    par = min(top, out.default("balcony_parapet_height_m", p2.get("balcony_parapet_height_m", 0.5), w["id"]))
    V, F = box(u, c, s0, s1, t, 0.0, par)
    out.add(w["id"], "wall", w.get("classification"), V, F, COLORS["railing"])
    if top > par:
        gv, gf = box(u, c, s0, s1, max(0.02, t * 0.25), par, top)
        out.add(w["id"] + "_glass", "window", "balcony_glass", gv, gf, COLORS["glass"])


def _stair_mesh(out, sf):
    """Solid stepped block: step k fills the strip between tread lines k-1 and k at (k+1) risers."""
    o = np.array(sf["origin"], float)
    u = np.array(sf["u"], float)
    r = np.array(sf["run"], float)
    w0, w1 = sf["tread_span"]
    rise = sf.get("rise") or 0.175
    z0 = sf.get("base_height") or 0.0              # next flights of a stair start at their landing
    offs = sf["tread_offsets"]
    prev = -sf["going"]
    parts = []
    for k, y in enumerate(offs):
        ring = [tuple(o + u * x + r * yy) for x, yy in ((w0, prev), (w1, prev), (w1, y), (w0, y))]
        parts.append(extrude(ring, max(0.0, z0 - 0.15), z0 + (k + 1) * rise))
        prev = y
    land = sf.get("landing")
    if land and land.get("exterior"):
        zl = z0 + (len(offs) + 1) * rise           # top of this flight = the landing
        parts.append(extrude(land["exterior"], zl - 0.15, zl))
    V, F = _merge(parts)
    out.add(sf["id"], "stair", "stair", V, F, COLORS["stair"])


def _merge(parts):
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.vstack(Vs), np.vstack(Fs)
