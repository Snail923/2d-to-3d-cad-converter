# Approach 2: SymPointV2 (learned extraction)

This package is the second, **separate** approach of the converter. Approach 1 (`architectural_converter/`) finds
walls, doors and windows with geometry and rules only. Approach 2 lets a trained model,
[SymPointV2](https://github.com/nicehuster/SymPointV2), label every line, arc and circle of the drawing (wall,
single / double / sliding door, window, stairs, furniture, ...; 35 FloorPlanCAD classes).

The two approaches share only approach 1's front end (DXF reading, unit detection, coordinate normalisation).
Nothing in `architectural_converter/` is changed or called by the model, so results can be compared cleanly.

**Status. Step 1 (now):** run the model and inspect its raw labels on our drawings. **Step 2 (next):** build
walls / doors / windows from those labels with the existing reconstruction, topology, validation, export and 3D,
then compare with approach 1 on the same drawings.

The model runs in a small local service on a machine with an NVIDIA GPU. The SymPointV2 repository has no
license file, so it is **not** copied into this project: you clone it yourself and run the service inside it.

## 1. Install the service (Linux / WSL2, NVIDIA GPU)

Your system CUDA is **not** changed. Everything goes into one conda environment called `sympoint`:
a private CUDA 12.8 compiler, GCC 13 and PyTorch built for CUDA 12.8 (needed for RTX 50xx cards).

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
(cu128), compiles the point operations (`pointops`) and copies `service/server.py` to `~/sympointv2/sympoint_server.py`.
detectron2 is **not** needed: SymPointV2 imports it only for training, and the server stands in for it.

**Weights.** Download them from the Google Drive link in the SymPointV2 README. The archive (`spv2-rep.zip`) holds
`best.pth` and the `svg_pointT.yaml` it was trained with. Use the two together:

```bash
cp spv2-rep/best.pth ~/sympointv2/weights.pth && cp spv2-rep/svg_pointT.yaml ~/sympointv2/svg_pointT_release.yaml
```

## 2. Start the service

```bash
cd ~/sympointv2 && conda activate sympoint && python sympoint_server.py --config svg_pointT_release.yaml --checkpoint weights.pth
```

On start it prints how much of the model the checkpoint filled; it stops if the config and the weights don't
match (that would give random labels). Check it from another terminal:

```bash
python -m sympoint_converter health --url http://localhost:8790
```

## 3. Label a drawing and look at the result

In a normal Python 3.11+ environment with this repository's requirements (`pip install -r requirements.txt`),
**not** the `sympoint` env:

```bash
python -m sympoint_converter label samples/real/apartment-building-plan.dxf --url http://localhost:8790 -o out/
```

Outputs in `out/`:

| file | content |
|---|---|
| `<name>.sympoint.html` | viewer: every line in its class colour (grey = no label), zoom / pan, hover for class, score, layer, DXF handle; class, instance and layer tables; score slider |
| `<name>.sympoint.labels.json` | label per DXF entity (class, score, handle, layer), instances, summary |
| `<name>.sympoint.response.json` | the raw model answer |

Try several tile sizes in one run (one viewer per size):

```bash
python -m sympoint_converter label samples/real/apartment-building-plan.dxf --url http://localhost:8790 -o out/ --tile-m 15 20 30
```

**Offline** (GPU on another machine): `label drawing.dxf -o out/` writes the request JSON; run
`python sympoint_server.py --config ... --checkpoint ... --in request.json --out resp.json` on the GPU machine, then
`label drawing.dxf --response resp.json -o out/`.

## What to look at (step 1)

* **Walls:** are the wall face lines brown (wall) and continuous? Are door leaves, window glazing or furniture
  wrongly brown?
* **Doors vs windows:** pink shades are doors, blue are windows, violet is *curtain wall* (glazed facade). On
  `apartment-building-plan.dxf` almost all window lines come out as curtain wall, so curtain wall is reported as
  its own group. Check sliding doors and wardrobe fronts too.
* **Scores** are the share of the chosen class in the model's answer for that line (0-1). SymPointV2's raw
  scores are sums over its queries and can exceed 1; lines the model barely covers (`min_label_mass`) count as
  "no label".
* **Layers × groups table:** a wall layer should be almost all wall; a door layer almost all door. This is the
  quickest way to spot a systematic error.
* **Scores:** move the min-score slider. Good labels usually stay when the threshold is raised.
* **Tile size:** FloorPlanCAD drawings are crops of apartment / office plans. If labels look poor, compare
  `--tile-m 15 20 30`.

## Troubleshooting

* **`CUDA error: device-side assert triggered`** (and the server stops): SymPointV2's neighbour search
  (`modules/pointops/src/knnquery/knnquery_cuda_kernel.cu`) keeps its candidates in arrays of 100, but its decoder
  asks for 256 neighbours. `setup.sh` patches the arrays to 512 and rebuilds; on start the server checks the
  neighbour search against PyTorch and refuses to run with an unpatched build.
* **`checkpoint does not match this config`**: use the `.yaml` that came with the weights.
* **`GPU out of memory`**: lower `max_primitives_per_tile` or `tile_m` in `config.json`.

## Configuration (`sympoint_converter/config.json`)

| key | meaning |
|---|---|
| `tile_m`, `overlap_m` | tile size and context overlap in metres; a line takes its label from the tile whose core holds it |
| `max_primitives_per_tile` | tiles with more lines are split into quarters (GPU memory) |
| `include_hatch_boundaries` | also send hatch outlines (`--hatches`) |
| `min_label_score` | labels below this score count as "no label" |
| `request_tiles_per_call`, `timeout_s` | batching of the requests to the service |

## How the drawing is sent

`sympoint_io.py` writes the same fields as SymPointV2's `parse_svg_v5.py` (`commands`, `args` = 4 points per
primitive, `lengths`, `layerIds`, `widths`, ...), in metres with y pointing down like SVG. Polylines and MLINEs
are sent as their exploded lines and arcs; text, dimensions and hatches are not sent (hatches optionally).
Every primitive keeps its DXF entity id, so each label maps back to a DXF handle. The server feeds every tile
through SymPointV2's own loader (`SVGDataset.load`), exactly like its `tools/inference.py`.
