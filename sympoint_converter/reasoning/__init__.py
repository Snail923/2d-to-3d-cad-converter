"""Reasoning on top of SymPointV2 (approach 2).

The model answers line by line and tile by tile, so its answer is locally noisy: lines of one
object get different classes, some lines get none, stray lines are called wall. A drawing is not
noisy in that way: a block is one object, a layer holds one kind of thing, walls form a connected
structure that encloses rooms, a room has one name, and the inside of a building has a floor.
Each module here states one such fact and uses it to correct the model's answer or the result
built from it. The modules are independent, are switched on and off in ``config.json``
(``reasoning.<module>.enabled``), never read drawing-specific names or sizes, and each returns a
log of what it changed (shown in the web app and stored in the model's extraction log).

    label_consensus   before extraction  blocks, layers and layer names vote on the labels
    wall_structure    after topology     wall groups that enclose no room and host no opening
    room_names        in topology        one room, one name: a missing wall between two names
                      (shared code: architectural_converter/topology/label_rules.py)
    enclosed floors   in validation      areas mostly surrounded by walls get a floor
                      (shared code: architectural_converter/topology/ground.py)"""
from __future__ import annotations

MODULES = ("label_consensus", "wall_structure")


def enabled(cfg: dict, name: str) -> bool:
    return bool((cfg.get("reasoning") or {}).get(name, {}).get("enabled", True))


def options(cfg: dict, name: str) -> dict:
    return (cfg.get("reasoning") or {}).get(name, {})


def summary(st) -> list[list[str]]:
    """[[module, what it did]] of a finished run, in plain words (web app, reports)."""
    out = []
    d = st.logs.get("detect", {}).get("label_consensus") or {}
    if d.get("skipped"):
        out.append(["label consensus", "off"])
    elif "error" in d:
        out.append(["label consensus", "skipped: " + d["error"]])
    elif d:
        ch = d.get("changed", {})
        b = ch.get("block_vote", 0) + ch.get("block_symbol", 0)
        out.append(["label consensus",
                    f"{d.get('model_labelled_entities', 0):,} entities labelled by the model, {d.get('labelled_entities', 0):,} "
                    f"after the votes: {b:,} lines set by their block, {ch.get('layer_vote', 0):,} by their layer's "
                    f"majority, {(d.get('names') or {}).get('lines_labelled', 0):,} by the layer name; "
                    f"{ch.get('layer_background', 0) + (d.get('names') or {}).get('stray_wall_lines_dropped', 0):,} "
                    "stray wall lines dropped"])
    t = st.logs.get("topology", {})
    n = t.get("room_names") or {}
    if n.get("skipped"):
        out.append(["room names", "off"])
    else:
        out.append(["room names", f"{len(n.get('split_rooms', []))} missing wall(s) added between different room names, "
                                  f"{len(n.get('named_rooms', []))} named space(s) closed into a room"])
    w = t.get("wall_structure") or {}
    if w.get("skipped"):
        out.append(["wall structure", "off" if w["skipped"] == "disabled" else "not applied (no rooms found)"])
    elif "error" in w:
        out.append(["wall structure", "skipped: " + w["error"]])
    elif w:
        out.append(["wall structure", f"{w.get('rejected_walls', 0)} wall(s) ({w.get('rejected_length_m', 0):g} m) rejected: "
                                      f"their group encloses no room and has no door or window; {w.get('review_walls', 0)} "
                                      "kept for review"])
    g = st.logs.get("ground") or {}
    if g:
        out.append(["floors", f"{g.get('unassigned_floor_pieces', 0)} area(s), {g.get('area_m2', 0):g} m², inside the building "
                              "without a room got a ground slab"])
    return out
