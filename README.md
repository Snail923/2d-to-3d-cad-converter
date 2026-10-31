# 2D → 3D CAD Converter: deterministic architectural extraction engine

Turns 2D architectural **DXF/DWG** floor plans into a structured building model (walls, rooms, doors,
windows, columns and their relationships) and reconstructs it in **3D**. It uses no machine learning: every
decision comes from computational geometry, graph topology and configurable architectural rules, and each
decision records the evidence it was based on.

**Web app:** `https://snail923.github.io/2d-to-3d-cad-converter/` (needs GitHub Pages enabled, see below).
The page runs the real Python engine in your browser (Pyodide), so your drawings never leave your computer.

| | |
|---|---|
| Phase 1 | ingestion → preprocessing → semantic context → 5 wall detectors + doors/windows/columns → evidence resolver → reconstruction → topology/rooms → classification → validation → export (JSON + SQLite) |
| Phase 2 | reference 3D reconstruction (Shapely + Trimesh) from the exported model only → GLB |
| Review UI | 2D overlay with layer toggles, evidence and source-entity inspection, accept/reject, add wall, connect endpoints, draw room, re-run, 3D viewer, exports |

## Try it

1. Open the web app (or run it locally, see below).
2. Pick a sample drawing and press **Load sample**, or use **Open DXF…** with your own file.
3. In **2D review**, click any element to see its confidence, evidence, reconstruction steps and source DXF entities
   (they are highlighted in pink). Use **Accept / Reject**, **Add wall**, **Connect endpoints** or **Draw room**, then
   **Re-run with corrections**.
4. **3D model** shows the Phase 2 reconstruction. **Validation** lists every problem with its location, source
   entities and a suggested correction. **Export** downloads the model (`.json`), the database (`.sqlite`), the 3D model
   (`.glb`) or an HTML report.

DWG files must first be converted to DXF. ODA File Converter is a desktop program, so it can't run in a browser.
The command line handles DWG automatically when ODA is installed.

### Enable the GitHub Pages site (one-time)

The workflow `.github/workflows/pages.yml` runs the tests, builds the site and deploys it on every push.
Deployment needs two settings that only the repository owner can change:

1. **Settings → Pages → Build and deployment → Source: "GitHub Actions".**
2. GitHub Pages on a **private** repository requires a paid plan (Pro/Team/Enterprise). On a free account,
   either make the repository public (**Settings → General → Danger zone → Change visibility**) or run the app locally.

Then re-run the workflow (**Actions → Test and deploy the web app → Run workflow**).
If deployment is refused with a branch-protection message, allow the branch under
**Settings → Environments → github-pages → Deployment branches**, or merge to `main`.

### Run the web app locally

```bash
pip install -r requirements.txt
python scripts/build_site.py          # builds ./site (downloads 6 pure-Python wheels)
python -m http.server -d site 8000    # open http://localhost:8000
```

## Command line and Python API

```bash
pip install -r requirements.txt
python -m architectural_converter plan.dxf -o out/            # JSON, SQLite, overlay SVG, HTML report, GLB
python -m architectural_converter plan.dwg -o out/            # needs ODA File Converter (PATH or ODA_FILE_CONVERTER)
python -m architectural_converter plan.dxf --corrections plan.corrections.json
python -m architectural_converter plan.dxf --until detect     # stop after a stage (debugging)
python -m architectural_converter --benchmark                 # score all benchmark drawings
python -m pytest -q                                           # unit, integration and benchmark regression tests
```

```python
from architectural_converter import PipelineConfig
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.reconstruction3d import build_3d

pipeline = ExtractionPipeline(PipelineConfig({"runtime": {"forced_units": None}}))
result = pipeline.process("building.dxf")                 # or .dwg
model = result.model                                      # archmodel 1.0 (see docs/DATA_MODEL.md)
result = pipeline.rerun(result, {"status_overrides": {"wall_0012": "REJECTED"}})  # detection is cached
build_3d(result.model, pipeline.cfg).export_glb("building.glb")
```

## Benchmark (synthetic drawings with exact reference geometry)

There are 13 generated drawings covering the categories in the brief: residential, multi-unit, commercial with a
column grid, office, hotel with nested and mirrored room-module blocks, irregular plan at survey coordinates, diagonal
and curved walls, fragmented and duplicated linework, everything on layer 0, hatches and overlaps, single-line walls,
MLINE walls and wide polylines. Units are mm, cm, m, inches or unitless.

All 13 reach ≥ 0.93 wall precision and 1.00 wall recall. Rooms, doors and windows are at 1.00 precision and recall
on all of them. Full table: [docs/BENCHMARK.md](docs/BENCHMARK.md).

> These drawings are synthetic. They prove the algorithms and guard against regressions, but they are cleaner than
> typical production drawings. Add your own DXF files with verified reference geometry to measure real-world accuracy;
> the metrics and regression tests pick them up automatically.

## Documentation

* [docs/AUDIT_AND_PLAN.md](docs/AUDIT_AND_PLAN.md): audit of the repository (it was empty) and the implementation plan
* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): stages, algorithms, configuration, how to add a detector
* [docs/DATA_MODEL.md](docs/DATA_MODEL.md): the Phase 1 → Phase 2 contract (`archmodel` JSON / SQLite)
* [docs/LIMITATIONS.md](docs/LIMITATIONS.md): known limitations
* [docs/BENCHMARK.md](docs/BENCHMARK.md): benchmark results and metric definitions

## Repository layout

```
architectural_converter/
  ingestion/        DWG→DXF (ODA), DXF loading, entity + block extraction, metadata
  preprocessing/    units, coordinate normalisation, cleaning, duplicates, adaptive tolerances, layer semantics
  geometry/         primitives/strokes, parallel pairing, spatial index, curves, endpoints, utils
  detectors/        BaseDetector + parallel / polyline / hatch / layer / single-line walls, doors, windows, columns, text, symbols
  classification/   candidate resolver, wall / room classification, rules
  reconstruction/   collinear merging, junctions (L/T/curved), opening-aware gaps, face association, snapping
  topology/         wall graphs, rooms, adjacency, missing-room hypotheses, boundary support
  validation/       wall / room / opening / geometry checks, report
  models/           Wall, Opening, Room, Column, Building, Drawing
  pipeline/         ExtractionPipeline, config, corrections, exporter (JSON / SQLite / Phase 2 adapter)
  reconstruction3d/ Phase 2 reference (Shapely + Trimesh → GLB)
  visualization/    SVG overlay, HTML report, web view payload
  benchmark/        plan specs, DXF writer, drawings + references, metrics, runner, regression baseline
  config/           all thresholds and weights (JSON)
web/                browser UI (HTML/JS, three.js, Pyodide worker)
scripts/            build_site.py
tests/              ingestion, preprocessing, detectors, reconstruction, topology, integration
```
