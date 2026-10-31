"""Assemble the static GitHub Pages site into ``site/``.

    python scripts/build_site.py [--skip-benchmark] [--wheels-from DIR]

* copies ``web/`` (HTML/JS/CSS, vendored three.js)
* zips the ``architectural_converter`` package (loaded into Pyodide in the browser)
* downloads the pure-Python wheels not shipped with Pyodide (pinned versions)
* copies the benchmark drawings as samples (+ reference geometry)
* runs the benchmark and writes ``benchmark.json`` for the Benchmark tab
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Pure-python wheels installed with micropip (deps=False). numpy / shapely / sqlite3 come from Pyodide itself.
WHEELS = ["ezdxf==1.4.5", "pyparsing==3.2.1", "typing_extensions==4.14.1", "fonttools==4.56.0",
          "networkx==3.4.2", "trimesh==5.2.0"]


def build(out: str, skip_benchmark: bool, wheels_from: str | None):
    from architectural_converter import __version__
    from architectural_converter.benchmark.generate import DEFAULT_DIR
    from architectural_converter.benchmark.plans import PLANS

    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(os.path.join(ROOT, "web"), out)
    # engine package
    pkg_zip = os.path.join(out, "engine.zip")
    with zipfile.ZipFile(pkg_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for path in glob.glob(os.path.join(ROOT, "architectural_converter", "**", "*"), recursive=True):
            rel = os.path.relpath(path, ROOT)
            if "__pycache__" in rel or "/drawings/" in rel or rel.endswith(".pyc") or os.path.isdir(path):
                continue
            z.write(path, rel)
    # wheels
    wdir = os.path.join(out, "wheels")
    os.makedirs(wdir, exist_ok=True)
    if wheels_from:
        for w in glob.glob(os.path.join(wheels_from, "*.whl")):
            shutil.copy(w, wdir)
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                               "--platform", "any", "--python-version", "3.13", "-d", wdir, *WHEELS])
    wheels = sorted(os.path.basename(p) for p in glob.glob(os.path.join(wdir, "*.whl")))
    # samples
    sdir = os.path.join(out, "samples")
    os.makedirs(sdir, exist_ok=True)
    samples = []
    for fn in PLANS:
        spec, _ = fn()
        name = spec.name
        src = os.path.join(DEFAULT_DIR, name + ".dxf")
        if not os.path.exists(src):
            continue
        shutil.copy(src, sdir)
        shutil.copy(src.replace(".dxf", ".reference.json"), sdir)
        samples.append({"file": name + ".dxf", "reference": name + ".reference.json",
                        "title": f"{name.replace('_', ' ')} — {spec.description}"})
    # benchmark
    bench = {}
    if not skip_benchmark:
        from architectural_converter.benchmark.run_benchmark import run
        bench = run()
        for v in bench.values():
            v.pop("trace", None)
    with open(os.path.join(out, "benchmark.json"), "w", encoding="utf-8") as fh:
        json.dump(bench, fh, indent=1)
    manifest = {"version": __version__, "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "package": "engine.zip", "wheels": wheels, "samples": samples}
    with open(os.path.join(out, "build_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    open(os.path.join(out, ".nojekyll"), "w").close()
    print(f"site built in {out}: {len(wheels)} wheels, {len(samples)} samples, benchmark={'yes' if bench else 'skipped'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "site"))
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--wheels-from")
    a = ap.parse_args()
    build(a.out, a.skip_benchmark, a.wheels_from)
