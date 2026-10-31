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
          "column": [0.62, 0.60, 0.58, 1.0], "floor": [0.76, 0.66, 0.52, 1.0],
          "review": [0.98, 0.75, 0.45, 1.0], "glass": [0.55, 0.75, 0.95, 0.45]}


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
                            "color": m["color"], "positions": np.round(m["vertices"], 4).ravel().tolist(),
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
    for w in view["walls"]:
        g = w["geometry"]
        H = w.get("height") or out.default("wall_height_m", p2["wall_height_m"], w["id"])
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
        H = col.get("height") or out.default("column_height_m", p2["column_height_m"], col["id"])
        if col.get("polygon"):
            V, F = extrude(col["polygon"]["exterior"], 0.0, H)
            out.add(col["id"], "column", col.get("classification"), V, F, COLORS["column"])
    slab = p2["slab_thickness_m"]
    for r in view["rooms"]:
        if r.get("polygon"):
            V, F = extrude(r["polygon"]["exterior"], -slab, 0.0)
            out.add(r["id"], "floor", r.get("classification"), V, F, COLORS["floor"])
    allv = np.vstack([m["vertices"] for m in out.meshes]) if out.meshes else np.zeros((1, 3))
    out.metadata["bounds"] = [allv.min(axis=0).round(3).tolist(), allv.max(axis=0).round(3).tolist()]
    out.metadata["counts"] = {k: sum(1 for m in out.meshes if m["type"] == k) for k in ("wall", "window", "column", "floor")}
    return out


def _merge(parts):
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.vstack(Vs), np.vstack(Fs)
