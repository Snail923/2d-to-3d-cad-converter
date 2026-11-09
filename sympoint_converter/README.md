# Approach 2: SymPointV2 (learned extraction)

This package is the second, **separate** approach of the converter. Approach 1 (`architectural_converter/`) finds
walls, doors and windows with geometry and rules only. In approach 2 a trained model,
[SymPointV2](https://github.com/nicehuster/SymPointV2), decides what every line, arc and circle of the drawing is
(wall, single / double / sliding door, window, curtain wall, stairs, furniture, ...; 35 FloorPlanCAD classes), and
the building model is built from those labels with approach 1's **build** stages, unchanged:

```
approach 1:  DXF -> front end -> 5 rule-based detectors + resolver -> reconstruct -> rooms -> classify -> validate -> export / 3D
approach 2:  DXF -> front end -> SymPointV2 labels -> extract.py   -> reconstruct -> rooms -> classify -> validate -> export / 3D
```

Shared: the front end (DXF reading, units, coordinate normalisation, strokes, texts) and everything after the
extraction. Not shared: no detector, resolver or veto of approach 1 runs in approach 2, and approach 1 never
imports approach 2 (a test checks this). Approach 2's evidence weights are merged into its own copy of the
configuration (`engine_overrides` in `config.json`); approach 1's config files are not changed.

The model runs in a small local service on a machine with an NVIDIA GPU. The SymPointV2 repository has no license
file, so it is **not** copied into this project: the setup clones it, and the service runs inside that clone.

## 1. One-time setup (Linux / WSL2, NVIDIA GPU)

Your system CUDA is **not** changed. Everything goes into one conda environment called `sympoint`: a private
CUDA 12.8 compiler, GCC 13 and PyTorch built for CUDA 12.8 (needed for RTX 50xx cards).

Prerequisites: git, the NVIDIA driver (`nvidia-smi` must show **CUDA Version 12.8 or higher**) and conda.
If you have no conda, install Miniforge (home folder only, no sudo):

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/Miniforge3.sh
```

```bash
bash ~/Miniforge3.sh -b -p ~/miniforge3 && ~/miniforge3/bin/conda init bash
```

Open a new terminal, then run the setup from this repository (safe to re-run; finished steps are skipped):

```bash
bash sympoint_converter/service/setup.sh ~/sympointv2
```

It clones SymPointV2, creates the `sympoint` env, installs the private CUDA 12.8 compiler and GCC 13, PyTorch
(cu128), patches and compiles the point operations (`pointops`). detectron2 is **not** needed: SymPointV2 imports
it only for training, and the server stands in for it.

**Weights.** Download them from the Google Drive link in the SymPointV2 README. The archive (`spv2-rep.zip`) holds
`best.pth` and the `svg_pointT.yaml` it was trained with. Use the two together:

```bash
cp spv2-rep/best.pth ~/sympointv2/weights.pth && cp spv2-rep/svg_pointT.yaml ~/sympointv2/svg_pointT_release.yaml
```

The converter itself runs in a normal Python 3.11+ environment (**not** the `sympoint` env):

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

## 2. Use it (the service starts by itself)

You do not start the service by hand. The first command that needs the model finds `~/sympointv2` and the
`sympoint` env, copies the current `service/server.py` there, starts the service in the background, waits until
it is ready and keeps it running for the next drawings. Labels are cached per drawing, so a second run of the
same drawing does not need the GPU at all.

```bash
python -m sympoint_converter compare samples/real/apartment-building-plan.dxf -o out/
```

| command | what it does | main outputs in `out/` |
|---|---|---|
| `compare <dxf> ...` | runs approach 1 and approach 2 on the same drawings | `<name>.compare.html` (tables + both 2D overlays), `.md`, `.json` |
| `compare --benchmark` | the same on the 13 benchmark drawings, **scored** against their exact references | `benchmark.compare.html` |
| `build <dxf>` | approach 2 only: the full model | `<name>.sympoint.model.json`, `.model.sqlite`, `.glb`, `.report.html`, `.overlay.svg`, labels viewer `.html` |
| `label <dxf>` | raw labels only (step 1), optionally `--tile-m 15 20 30` | `<name>.sympoint.html` viewer, `.labels.json` |
| `server status` / `stop` / `start` / `log` | the background service | log in `~/.cache/sympoint_converter/server.log` |
| `web` | the web app with both methods (see below) | `http://127.0.0.1:8000` |

Options: `--url http://host:8790` uses a service you started yourself (on this or another machine, no
autostart); `--no-cache` asks the model again; `--response resp.json` uses an offline answer (see below);
`--corrections` (build) applies manual corrections in approach 1's format.

If a CUDA error stops the service, the next request restarts it once automatically. `server stop` frees the GPU.
Settings for the service are under `service` in `config.json` (`SYMPOINT_HOME`, `SYMPOINT_PYTHON` and
`SYMPOINT_URL` override them).

**Offline** (GPU on another machine): `label drawing.dxf -o out/ --offline` writes the request JSON; on the GPU
machine run `python sympoint_server.py --config ... --checkpoint ... --in request.json --out resp.json`; then
`build drawing.dxf --response resp.json -o out/` (or `label` / `compare` with `--response`).

## Web app: choose the method in the browser

```bash
python -m sympoint_converter web
```

builds the web app when needed, starts the SymPointV2 service, serves the page on `http://127.0.0.1:8000` and
opens it. In the header choose **Method**:

* **Approach 1 · rules**: runs entirely in the browser (as before, no GPU needed),
* **Approach 2 · SymPointV2**: the page cuts the drawing into tiles in the browser, sends them to the service on
  your computer (address and tile size in the *SymPointV2 service* panel; nothing leaves your machine) and builds
  the model from the labels with the same engine,
* **Compare both**: runs both. A **Compare** tab shows counts, agreement and, for the benchmark samples, the
  scores of both against the reference; the **Approach 1 / Approach 2** switch in the 2D toolbar chooses which
  result you review, correct, re-run and export (exports of approach 2 are named `*.sympoint.*`).

*SymPointV2 classes* (2D toolbar) colours the drawing's lines by the class the model gave them. Changing the method
after a drawing is loaded runs only what is missing; labels are kept for the drawing while the page is open.
The published GitHub Pages site has the same selector: it works with the service running on your computer
(`python -m sympoint_converter server start`), as browsers allow pages to call `127.0.0.1`.

## 3. How labels become walls, doors and windows (`extract.py`)

| element | from the labels |
|---|---|
| walls | parallel wall-labelled strokes 4-90 cm apart form a wall band (faces -> thickness, centre line). Bands side by side (a line inside a wall) are merged. Concentric wall-labelled arcs give curved walls. Wall-labelled lines without a second face become single-line walls in single-line drawings, and wide polylines use their width as thickness |
| short pieces | a pair shorter than its own thickness is a **column** when wall lines close it at both ends, a **wall nib / pier** when closed at one end or when longer walls continue in the same line, and is **dropped** when open at both ends (the end caps of two walls facing each other across a door / window gap) |
| columns | SymPointV2 has no column class (FloorPlanCAD draws columns as walls): small compact closed wall-labelled outlines, plus the closed short pieces above |
| doors | model instances of a door class, else clusters of door-labelled lines split by their swing arcs. Hinge = arc centre; the chord runs from the hinge along the wall the closed leaf lies in (not the wall the open leaf rests against); two arcs leaf to leaf = double door. Without arc (sliding / folding): the extent of the symbol along the nearest wall |
| windows | window, bay window, blind window **and curtain wall** labels (on real drawings window glazing often comes out as curtain wall): the extent along the nearest wall; pieces of one window on the same wall are merged |
| stairs | clusters of stair-labelled lines: the treads are the parallel lines of the dominant direction, their spacing the going |
| railings | railing-labelled lines (pairs a few cm apart, else single lines) become railing walls with `railing_label` evidence; the shared balcony detection makes the space they bound a balcony, and 3D builds them as a low parapet with glass above |

Each element keeps its source DXF entities and the model's class and score (`sympoint_label` evidence, notes).
Hosting in walls, gaps, junctions, rooms, classification, validation, export and 3D are approach 1's, including
the shared rules that a door needs a wall (missing walls re-checked around doors), that every area inside the
building gets a floor, and that balcony walls are a low parapet with glass above (see `docs/ARCHITECTURE.md`).

**Measured with perfect labels** (an oracle answering from the benchmark's layer names, `tests/sympoint`): the
build reaches wall, room, door, window and column precision / recall of 0.98-1.00 on 12 of the 13 benchmark
drawings, including closed-polyline walls, loose door / window linework, column grids, curved walls,
single-line walls, MLINEs and wide polylines. (The 13th draws everything on layer `0`, which an oracle reading
layer names cannot label.) So the result on a real drawing mostly depends on the quality of SymPointV2's labels,
which is exactly what `compare` measures.

## Step 1 checks (`label` viewer)

* **Walls:** are the wall face lines brown and continuous? Are door leaves, glazing or furniture wrongly brown?
* **Doors vs windows:** pink shades are doors, blue windows, violet *curtain wall*. On
  `apartment-building-plan.dxf` almost all window lines come out as curtain wall (approach 2 treats them as windows).
* **Scores** are the share of the chosen class in the model's answer for that line (0-1). SymPointV2's raw scores
  are sums over its queries and can exceed 1; lines the model barely covers (`min_label_mass`) count as "no label".
* **Layers × groups table:** a wall layer should be almost all wall, a door layer almost all door.
* **Tile size:** FloorPlanCAD drawings are crops of apartment / office plans. 20 m (default) and 15 m worked better
  than 30 m on the apartment plan.

## Troubleshooting

* **`CUDA error: device-side assert triggered`**: SymPointV2's neighbour search
  (`modules/pointops/src/knnquery/knnquery_cuda_kernel.cu`) keeps its candidates in arrays of 100, but its decoder
  asks for 256 neighbours. `setup.sh` patches the arrays to 512 and rebuilds; on start the server checks the
  neighbour search against PyTorch and refuses to run with an unpatched build.
* **`checkpoint does not match this config`**: use the `.yaml` that came with the weights.
* **`GPU out of memory`**: lower `max_primitives_per_tile` or `tile_m` in `config.json`.
* **The service does not start**: `python -m sympoint_converter server log` shows its output.

## Configuration (`sympoint_converter/config.json`)

| key | meaning |
|---|---|
| `tile_m`, `overlap_m` | tile size and context overlap in metres; a line takes its label from the tile whose core holds it |
| `max_primitives_per_tile` | tiles with more lines are split into quarters (GPU memory) |
| `min_label_score`, `min_label_mass` | below these a line counts as "no label" |
| `cache_labels` | reuse the answer for the same DXF + tiling (`~/.cache/sympoint_converter/labels`) |
| `service.*` | where SymPointV2 and its env are, weights / config names, autostart, start timeout |
| `build.*` | thickness range, parallel tolerance, column sizes, door / window widths, clustering, stair going |
| `engine_overrides` | evidence weights of approach 2 for the shared build stages |

## How the drawing is sent

`sympoint_io.py` writes the same fields as SymPointV2's `parse_svg_v5.py` (`commands`, `args` = 4 points per
primitive, `lengths`, `layerIds`, `widths`, ...), in metres with y pointing down like SVG. Polylines and MLINEs
are sent as their exploded lines and arcs; text, dimensions and hatches are not sent (hatches optionally).
Every primitive keeps its DXF entity id, so each label maps back to a DXF handle. The server feeds every tile
through SymPointV2's own loader (`SVGDataset.load`), exactly like its `tools/inference.py`.
