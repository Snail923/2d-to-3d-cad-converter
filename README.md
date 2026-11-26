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
| Multi-storey | permit sets with all floors side by side: views are split, floor levels read from "F.F.L" annotations and titles, elevations / sections / duplicate (structural, sanitary) plans excluded, floors aligned and stacked at their real heights |
| Stairs & rules | straight stair flights (treads are never read as walls; stepped 3D); plausibility rules: windows must sit in a wall, window-like lines in interior partitions become sliding / closet doors to review, rooms without access are flagged |
| Phase 2 | reference 3D reconstruction (Shapely + Trimesh) from the exported model only → GLB |
| Review UI | 2D overlay with layer toggles, evidence and source-entity inspection, accept/reject, add wall, connect endpoints, draw room, re-run; Blender-style 3D editor (instant edits, customisable keys); exports |

## Two approaches (kept separate for comparison)

| | package | extraction | status |
|---|---|---|---|
| **Approach 1** | `architectural_converter/` | geometry, graph topology and rules only (this README) | complete |
| **Approach 2** | `sympoint_converter/` | the learned model [SymPointV2](https://github.com/nicehuster/SymPointV2) labels every DXF line (wall, door types, window, curtain wall, stairs, furniture, ...); walls, doors, windows, columns and stairs are built from the labels, then approach 1's reconstruction, rooms, classification, validation, export and 3D run unchanged | complete; `python -m sympoint_converter compare <dxf>` runs both side by side (the GPU service starts by itself) |

Approach 2 reuses approach 1's DXF front end and build stages; approach 1 never imports approach 2.
In the web app, choose **Method** in the header: approach 1, approach 2 (needs the SymPointV2 service on your
computer) or **Compare both**. `python -m sympoint_converter web` starts the service and opens the app locally.
Setup (local NVIDIA GPU, no change to the system CUDA) and usage: [sympoint_converter/README.md](sympoint_converter/README.md).

## Try it

1. Open the web app (or run it locally, see below).
2. Pick a sample drawing and press **Load sample**, or use **Open DXF…** with your own file.
3. In **2D review**, click any element to see its confidence, evidence, reconstruction steps and source DXF entities
   (they are highlighted in pink). Use **Accept / Reject**, **Add wall**, **Connect endpoints** or **Draw room**, then
   **Re-run with corrections**.
4. **3D model** shows the Phase 2 reconstruction (for stacked buildings, **floors up to** shows the building up to one floor). **Validation** lists every problem with its location, source
   entities and a suggested correction. **Export** downloads the model (`.json`), the database (`.sqlite`), the 3D model
   (`.glb`) or an HTML report.

DWG files must first be converted to DXF. ODA File Converter is a desktop program, so it can't run in a browser.
The command line handles DWG automatically when ODA is installed.

### 3D editor (the *3D model* tab)

The 3D view is an editor that works like Blender: changes are rebuilt in the browser in milliseconds
(0.2 ms to move a wall on the 7-storey permit set), the pipeline is not re-run. Every key can be changed
(**Keys** / Ctrl+,), together with the navigation style, snapping and the default sizes; the settings are
kept in the browser and can be exported / imported as JSON. **?** (F1) lists all keys.

| | default key | |
|---|---|---|
| navigate | middle drag / Shift+middle / wheel (or Alt+left) | orbit / pan / zoom; *classic* navigation: left orbits, right pans |
| select | click, Shift+click, A, Alt+A | a door or window is selected by clicking in its opening |
| move / rotate / scale | G / R / S | X, Y, Z constrain (Z: height); type a number for an exact value; Ctrl toggles snapping; click or Enter confirms, right-click or Esc cancels. A door / window slides along its wall, Z moves its sill; S on a wall with Z changes its height |
| delete, duplicate | X or Delete, Shift+D | deleting a door / window fills the gap |
| draw walls | W | click corners (snaps to wall ends and the grid, Shift: 45° steps), Enter / right-click ends |
| cut a door / window | D / V | point at a wall, a preview follows the mouse, click cuts it |
| fill a gap | F | a door / window: closes it; two walls in line: one wall; two walls at an angle: both run to their corner; one wall: its end runs to the next wall |
| floor / roof | P / O | click the outline, Enter finishes; roofs are flat, shed, gable or hip (Tab while drawing) and sit on the walls under them |
| properties | N | exact numbers of the selection (lengths, thickness, heights, widths, sill, roof pitch...) |
| undo / redo | Ctrl+Z / Ctrl+Shift+Z | |

**Export GLB** writes the edited scene; **Save scene** / **Open scene** keep the edits as a JSON file to continue
later. The 3D edits are separate from the 2D corrections: loading or re-running a drawing asks before it
replaces an edited scene.

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
sympoint_converter/ approach 2: SymPointV2 request / labels, raw-label viewer, CLI, GPU service (service/)
web/                browser UI (HTML/JS, three.js, Pyodide worker)
scripts/            build_site.py
tests/              ingestion, preprocessing, detectors, reconstruction, topology, integration
```
