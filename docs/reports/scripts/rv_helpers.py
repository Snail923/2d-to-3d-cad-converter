import sys
sys.path.insert(0, '/home/user/latest')
from sympoint_converter import CLASSES, load_config
from sympoint_converter.sympoint_io import export_tiles, load_drawing
EMPTY = [CLASSES.index("wall"), 0.0, 0.0]

def check(req, resp):
    rt = {t["id"]: t for t in resp["tiles"]}
    bad = [t["id"] for t in req["tiles"] if t["id"] not in rt or len(rt[t["id"]]["semantic"]) != len(t["ids"])]
    return bad

def by_id(path, resp, cfg):
    req = export_tiles(load_drawing(path), cfg); bad = check(req, resp)
    if bad: raise RuntimeError(f"response does not match the request tiling: {bad[:5]}")
    rt = {t["id"]: t for t in resp["tiles"]}
    return {rid: s for t in req["tiles"] for rid, s in zip(t["ids"], rt[t["id"]]["semantic"])}

def remap(path, ids, cfg, votes=None):
    req = export_tiles(load_drawing(path), cfg); tiles = []; filled = 0
    for t in req["tiles"]:
        sem = []
        for rid in t["ids"]:
            s = ids.get(rid, EMPTY)
            if votes and rid in votes and (s[1] < cfg["min_label_score"] or s[2] < cfg["min_label_mass"]):
                s = [CLASSES.index(votes[rid]), 0.6, 0.6]; filled += 1
            sem.append(s)
        tiles.append({"id": t["id"], "semantic": sem, "instances": []})
    return {"model": "SymPointV2 (remapped)", "tiles": tiles}, filled

def a1_votes(model):
    v = {}
    for o in model["openings"]:
        if o["status"] == "REJECTED": continue
        c = {"door": "single door", "window": "window"}.get(o["type"])
        if c:
            for s in o.get("source_entities", []): v[s["id"]] = c
    for w in model["walls"]:
        if w["status"] == "REJECTED" or w.get("is_hypothesis"): continue
        for s in w.get("source_entities", []): v.setdefault(s["id"], "wall")
    return v

