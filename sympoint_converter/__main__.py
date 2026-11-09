"""Command line for approach 2 (SymPointV2).

The local SymPointV2 service is started automatically in the background when a command needs it
(and keeps running for the next drawings):

    python -m sympoint_converter build drawing.dxf -o out/           # walls/doors/windows -> JSON, SQLite, GLB, report
    python -m sympoint_converter compare drawing.dxf -o out/         # approach 1 vs approach 2, side by side
    python -m sympoint_converter compare --benchmark -o out/         # ... on the 13 benchmark drawings (with scores)
    python -m sympoint_converter label drawing.dxf -o out/           # raw labels viewer (step 1)
    python -m sympoint_converter server start | stop | status | log
    python -m sympoint_converter web                                  # web app with both methods (opens the browser)

``--url`` uses a service you started yourself (no autostart); ``--response`` an offline answer.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

from . import load_config
from .client import ServiceError
from .service_manager import ServiceSetupError, log_file, start, status, stop, tail
from .sympoint_io import load_drawing


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1 if path.endswith(".model.json") else None)


def _cfg(a, **extra):
    over = {"tile_m": getattr(a, "tile_m", None), "overlap_m": getattr(a, "overlap_m", None),
            "min_label_score": getattr(a, "min_score", None),
            "include_hatch_boundaries": True if getattr(a, "hatches", False) else None,
            "cache_labels": False if getattr(a, "no_cache", False) else None, **extra}
    if getattr(a, "no_autostart", False):
        over["service"] = {"autostart": False}
    return load_config(over)


def _response(a):
    if getattr(a, "response", None):
        with open(a.response, encoding="utf-8") as fh:
            return json.load(fh)
    return None


# ------------------------------------------------------------------ server
def cmd_server(a):
    cfg = load_config()
    if a.action == "start":
        st = start(cfg)
        print(json.dumps({k: v for k, v in st.items() if k != "error"}, indent=1))
    elif a.action == "stop":
        print("stopped" if stop(cfg) else "no service started by sympoint_converter is running")
    elif a.action == "status":
        print(json.dumps(status(a.url or cfg["service"]["url"]), indent=1))
    else:
        print(f"{log_file()}:\n{tail(60)}")
    return 0


def cmd_health(a):
    st = status(a.url or load_config()["service"]["url"])
    print(json.dumps(st, indent=1))
    return 0 if st["running"] else 1


# ------------------------------------------------------------------ label (step 1)
def cmd_label(a):
    from .labels import get_labels
    from .viewer import render_html, summarize
    t0 = time.time()
    print(f"Reading {a.input} ...")
    d = load_drawing(a.input, a.units)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, os.path.splitext(os.path.basename(a.input))[0])
    sizes = a.tile_m or [None]
    if a.response and len(sizes) > 1:
        print("--response holds the answer for one request: give a single --tile-m", file=sys.stderr)
        return 2
    for tm in sizes:
        cfg = _cfg(a, tile_m=tm)
        sfx = f"_t{cfg['tile_m']:g}" if len(sizes) > 1 else ""
        if a.request_out or (not a.url and not a.response and a.offline):
            from .sympoint_io import export_tiles
            p = a.request_out or f"{base}{sfx}.sympoint.request.json"
            _dump(export_tiles(d, cfg), p)
            print(f"request -> {p}")
            if a.offline:
                print("Offline: on the GPU machine run\n"
                      f"  python sympoint_server.py --config <yaml> --checkpoint <pth> --in {p} --out resp.json\n"
                      f"then: python -m sympoint_converter label {a.input} --response resp.json -o {a.out}")
                continue
        labels, prims, req, resp, source = get_labels(d, a.input, cfg, a.url, _response(a))
        if source == "service":
            _dump(resp, f"{base}{sfx}.sympoint.response.json")
        summ = summarize(prims, labels, d)
        raw = d.raw_entities
        ents = {rid: {**v, "handle": raw[rid].handle if rid in raw else None,
                      "type": raw[rid].dxftype if rid in raw else None,
                      "layer": raw[rid].layer if rid in raw else None}
                for rid, v in labels["entities"].items()}
        meta = {"file": os.path.basename(a.input), "units": d.units.unit, "tiles": len(req["tiles"]),
                "tile_m": cfg["tile_m"], "overlap_m": cfg["overlap_m"], "min_label_score": cfg["min_label_score"],
                "seconds": resp.get("seconds"), "labels_source": source}
        _dump({"meta": meta, "summary": summ, "entities": ents, "instances": labels["instances"]},
              f"{base}{sfx}.sympoint.labels.json")
        with open(f"{base}{sfx}.sympoint.html", "w", encoding="utf-8") as fh:
            fh.write(render_html(prims, labels, d, f"SymPointV2 labels: {os.path.basename(a.input)}"
                                 + (f" (tiles {cfg['tile_m']:g} m)" if sfx else ""), meta))
        g = ", ".join(f"{k} {100 * v:.0f}%" for k, v in summ["groups"].items())
        print(f"labelled {100 * summ['labelled_length_share']:.0f}% of the line length: {g or 'nothing'}")
        print(f"viewer -> {base}{sfx}.sympoint.html")
    print(f"done in {time.time() - t0:.1f} s")
    return 0


# ------------------------------------------------------------------ build (step 2)
def cmd_build(a):
    from architectural_converter.pipeline.database_exporter import export_json, export_sqlite
    from architectural_converter.visualization.feature_overlay import render_overlay
    from architectural_converter.visualization.validation_viewer import html_report
    from .pipeline import SymPointPipeline
    from .viewer import render_html
    cfg = _cfg(a)
    corr = None
    if a.corrections:
        with open(a.corrections, encoding="utf-8") as fh:
            corr = json.load(fh)
    ov = {"runtime": {"forced_units": a.units}} if a.units else None
    t0 = time.time()
    pipe = SymPointPipeline(cfg, engine_overrides=ov, url=a.url, response=_response(a))
    st = pipe.process(a.input, corrections=corr)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, os.path.splitext(os.path.basename(a.input))[0] + ".sympoint")
    export_json(st.model, base + ".model.json")
    export_sqlite(st.model, base + ".model.sqlite")
    svg = render_overlay(st)
    with open(base + ".overlay.svg", "w", encoding="utf-8") as fh:
        fh.write(svg)
    with open(base + ".report.html", "w", encoding="utf-8") as fh:
        fh.write(html_report(st, svg))
    with open(base + ".html", "w", encoding="utf-8") as fh:
        fh.write(render_html(st.sp_prims, st.sp_labels, st.drawing, f"SymPointV2 labels: {os.path.basename(a.input)}",
                             {"file": os.path.basename(a.input), "units": st.drawing.units.unit,
                              "tiles": len(st.sp_request["tiles"]), "tile_m": cfg["tile_m"]}))
    if not a.no_3d:
        from architectural_converter.reconstruction3d import build_3d
        m3 = build_3d(st.model, pipe.cfg)
        m3.export_glb(base + ".glb")
    from .compare import counts
    c = counts(st.model)
    lg = st.logs["detect"]
    print("approach 2 (SymPointV2): " + " ".join(f"{k}={v}" for k, v in c.items())
          + f"  ({time.time() - t0:.1f} s, labels: {lg['labels_source']})")
    print(f"outputs: {base}.*  (model.json, model.sqlite, overlay.svg, report.html, labels viewer .html"
          f"{', glb' if not a.no_3d else ''})")
    return 0


# ------------------------------------------------------------------ compare
def cmd_compare(a):
    from .compare import run, strip_states, to_html, to_markdown
    cfg = _cfg(a)
    paths = list(a.inputs)
    if a.benchmark:
        from architectural_converter.benchmark.generate import DEFAULT_DIR
        paths += sorted(glob.glob(os.path.join(DEFAULT_DIR, "*.dxf")))
    if not paths:
        print("give DXF files or --benchmark", file=sys.stderr)
        return 2
    res = run(paths, cfg, a.url, _response(a))
    os.makedirs(a.out, exist_ok=True)
    stem = "benchmark" if a.benchmark and not a.inputs else os.path.splitext(os.path.basename(paths[0]))[0]
    base = os.path.join(a.out, f"{stem}.compare")
    with open(base + ".html", "w", encoding="utf-8") as fh:
        fh.write(to_html(res))
    _dump(strip_states(res), base + ".json")
    md = to_markdown(res)
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    print(f"report -> {base}.html  (also .md, .json)")
    return 0


# ------------------------------------------------------------------ web app (both methods)
def _site_outdated(root, site) -> bool:
    man = os.path.join(site, "build_manifest.json")
    if not os.path.exists(man):
        return True
    try:
        with open(man, encoding="utf-8") as fh:
            if not json.load(fh).get("sympoint"):
                return True
    except (OSError, ValueError):
        return True
    built = os.path.getmtime(man)
    for pkg in ("web", "architectural_converter", "sympoint_converter"):
        for dp, _, fs in os.walk(os.path.join(root, pkg)):
            if "__pycache__" in dp:
                continue
            for f in fs:
                if f.endswith((".py", ".json", ".js", ".html", ".css")) and os.path.getmtime(os.path.join(dp, f)) > built:
                    return True
    return False


def cmd_web(a):
    import functools
    import shutil
    import subprocess
    import webbrowser
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from .service_manager import ensure_running, state_dir
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    builder = os.path.join(root, "scripts", "build_site.py")
    site = os.path.abspath(a.site or os.path.join(root, "site"))
    if not os.path.exists(builder):
        print(f"scripts/build_site.py not found next to the package ({root}): run this from a clone of the repository.",
              file=sys.stderr)
        return 2
    if a.rebuild or _site_outdated(root, site):
        print("building the web app (both methods) ...")
        cmd = [sys.executable, builder, "--out", site]
        if not a.with_benchmark:
            cmd.append("--skip-benchmark")
        wheels = os.path.join(site, "wheels")
        if os.path.isdir(wheels) and glob.glob(os.path.join(wheels, "*.whl")):     # reuse, no download
            keep = os.path.join(state_dir(), "wheels")
            shutil.rmtree(keep, ignore_errors=True)
            shutil.copytree(wheels, keep)
            cmd += ["--wheels-from", keep]
        subprocess.check_call(cmd, cwd=root)
    cfg = load_config()
    if not a.no_gpu:
        try:
            ensure_running(cfg)
        except (ServiceError, ServiceSetupError) as e:
            print(f"Approach 2 is not available until the SymPointV2 service runs:\n{e}\n"
                  "(approach 1 works anyway)", file=sys.stderr)
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    handler = functools.partial(Quiet, directory=site)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"web app: {url}  (Ctrl+C stops the web server; the GPU service keeps running: "
          "python -m sympoint_converter server stop)")
    if not a.no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sympoint_converter", description="Approach 2: SymPointV2 learned extraction")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, tiles=True):
        p.add_argument("-o", "--out", default="out")
        p.add_argument("--url", help="use this running service (no autostart), e.g. http://localhost:8790")
        p.add_argument("--response", help="SymPointV2 answer from offline inference (same drawing and tile size)")
        p.add_argument("--no-autostart", action="store_true", help="do not start the local service automatically")
        p.add_argument("--no-cache", action="store_true", help="ask the model again even if the labels are cached")
        if tiles:
            p.add_argument("--overlap-m", type=float)
            p.add_argument("--min-score", type=float, help="ignore labels below this score (0-1)")
            p.add_argument("--hatches", action="store_true", help="also send hatch boundary lines to the model")
        p.add_argument("--units", choices=["mm", "cm", "dm", "m", "in", "ft"], help="force drawing units")

    s = sub.add_parser("server", help="start / stop / status / log of the local SymPointV2 service")
    s.add_argument("action", choices=["start", "stop", "status", "log"])
    s.add_argument("--url")
    h = sub.add_parser("health", help="check the SymPointV2 service")
    h.add_argument("--url")

    lb = sub.add_parser("label", help="label every line of a drawing and write the viewer (step 1)")
    lb.add_argument("input")
    common(lb)
    lb.add_argument("--tile-m", type=float, nargs="+", help="tile size(s) in metres (default from config.json)")
    lb.add_argument("--request-out", help="also write the request JSON here")
    lb.add_argument("--offline", action="store_true", help="only write the request JSON (GPU elsewhere)")

    b = sub.add_parser("build", help="build the building model with approach 2 (JSON, SQLite, report, GLB)")
    b.add_argument("input")
    common(b)
    b.add_argument("--tile-m", type=float)
    b.add_argument("--corrections", help="manual corrections JSON (same format as approach 1)")
    b.add_argument("--no-3d", action="store_true")

    c = sub.add_parser("compare", help="approach 1 vs approach 2 on the same drawings")
    c.add_argument("inputs", nargs="*")
    common(c)
    c.add_argument("--tile-m", type=float)
    c.add_argument("--benchmark", action="store_true", help="add the 13 benchmark drawings (scored against references)")

    w = sub.add_parser("web", help="serve the web app with both methods (starts the GPU service, opens the browser)")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--site", help="site directory (default: <repo>/site, built when missing or outdated)")
    w.add_argument("--rebuild", action="store_true", help="rebuild the site")
    w.add_argument("--with-benchmark", action="store_true", help="also run the benchmark for the Benchmark tab when building")
    w.add_argument("--no-gpu", action="store_true", help="do not start the SymPointV2 service")
    w.add_argument("--no-browser", action="store_true")

    a = ap.parse_args(argv)
    try:
        return {"server": cmd_server, "health": cmd_health, "label": cmd_label, "build": cmd_build,
                "compare": cmd_compare, "web": cmd_web}[a.cmd](a)
    except (ServiceError, ServiceSetupError) as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
