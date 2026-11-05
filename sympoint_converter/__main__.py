"""Command line for approach 2 (SymPointV2).

    python -m sympoint_converter health --url http://localhost:8790
    python -m sympoint_converter label drawing.dxf --url http://localhost:8790 -o out/
    python -m sympoint_converter label drawing.dxf --url ... --tile-m 15 20 30     # compare tile sizes
    python -m sympoint_converter label drawing.dxf -o out/                          # offline: request only
    python -m sympoint_converter label drawing.dxf --response resp.json -o out/     # offline: view answer
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import load_config
from .client import ServiceError, health, predict
from .sympoint_io import export_tiles, labels_from_response, load_drawing, primitives
from .viewer import render_html, summarize


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def cmd_health(a):
    try:
        print(json.dumps(health(a.url), indent=1))
    except ServiceError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


def cmd_label(a):
    t0 = time.time()
    print(f"Reading {a.input} ...")
    d = load_drawing(a.input, a.units)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, os.path.splitext(os.path.basename(a.input))[0])
    sizes = a.tile_m or [None]
    if a.response and len(sizes) > 1:
        print("--response holds the answer for one request: give a single --tile-m", file=sys.stderr)
        return 2
    rc = 0
    for tm in sizes:
        cfg = load_config({"tile_m": tm, "overlap_m": a.overlap_m, "min_label_score": a.min_score,
                           "include_hatch_boundaries": True if a.hatches else None})
        sfx = f"_t{cfg['tile_m']:g}" if len(sizes) > 1 else ""
        prims = primitives(d, cfg)
        req = export_tiles(d, cfg, prims)
        print(f"{len(prims):,} primitives in {len(req['tiles'])} tile(s) of {cfg['tile_m']:g} m "
              f"(overlap {cfg['overlap_m']:g} m), units {d.units.unit}")
        resp = None
        if a.url:
            try:
                resp = predict(a.url, req, cfg["request_tiles_per_call"], cfg["timeout_s"])
            except ServiceError as e:
                print(e, file=sys.stderr)
                return 1
            _dump(resp, f"{base}{sfx}.sympoint.response.json")
        elif a.response:
            with open(a.response, encoding="utf-8") as fh:
                resp = json.load(fh)
        if resp is None or a.request_out:
            p = a.request_out or f"{base}{sfx}.sympoint.request.json"
            _dump(req, p)
            print(f"request -> {p}")
            if resp is None:
                print("No --url / --response: run on the GPU machine\n"
                      f"  python sympoint_server.py --config <yaml> --checkpoint <pth> --in {p} --out resp.json\n"
                      f"then: python -m sympoint_converter label {a.input} --response resp.json -o {a.out}")
                continue
        labels = labels_from_response(req, resp, cfg["min_label_score"], cfg["min_label_mass"])
        summ = summarize(prims, labels, d)
        raw = d.raw_entities
        ents = {rid: {**v, "handle": raw[rid].handle if rid in raw else None,
                      "type": raw[rid].dxftype if rid in raw else None,
                      "layer": raw[rid].layer if rid in raw else None}
                for rid, v in labels["entities"].items()}
        meta = {"file": os.path.basename(a.input), "units": d.units.unit, "tiles": len(req["tiles"]),
                "tile_m": cfg["tile_m"], "overlap_m": cfg["overlap_m"], "min_label_score": cfg["min_label_score"],
                "seconds": resp.get("seconds")}
        _dump({"meta": meta, "summary": summ, "entities": ents, "instances": labels["instances"]},
              f"{base}{sfx}.sympoint.labels.json")
        with open(f"{base}{sfx}.sympoint.html", "w", encoding="utf-8") as fh:
            fh.write(render_html(prims, labels, d, f"SymPointV2 labels: {os.path.basename(a.input)}"
                                 + (f" (tiles {cfg['tile_m']:g} m)" if sfx else ""), meta))
        g = ", ".join(f"{k} {100 * v:.0f}%" for k, v in summ["groups"].items())
        print(f"labelled {100 * summ['labelled_length_share']:.0f}% of the line length: {g or 'nothing'}")
        print(f"viewer -> {base}{sfx}.sympoint.html")
    print(f"done in {time.time() - t0:.1f} s")
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sympoint_converter", description="Approach 2: SymPointV2 labels for DXF drawings")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("health", help="check the SymPointV2 service")
    h.add_argument("--url", default="http://localhost:8790")
    lb = sub.add_parser("label", help="label every line of a drawing and write the viewer")
    lb.add_argument("input", help="DXF (or DWG with ODA File Converter)")
    lb.add_argument("-o", "--out", default="out")
    lb.add_argument("--url", help="SymPointV2 service, e.g. http://localhost:8790")
    lb.add_argument("--response", help="SymPointV2 answer from offline inference (same drawing and tile size)")
    lb.add_argument("--request-out", help="also write the request JSON here")
    lb.add_argument("--tile-m", type=float, nargs="+", help="tile size(s) in metres (default from config.json)")
    lb.add_argument("--overlap-m", type=float)
    lb.add_argument("--min-score", type=float, help="ignore labels below this score")
    lb.add_argument("--units", choices=["mm", "cm", "dm", "m", "in", "ft"], help="force drawing units")
    lb.add_argument("--hatches", action="store_true", help="also send hatch boundary lines to the model")
    a = ap.parse_args(argv)
    return {"health": cmd_health, "label": cmd_label}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
