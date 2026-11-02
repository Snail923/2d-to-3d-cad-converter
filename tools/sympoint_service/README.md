# SymPointV2 symbol detector (local GPU service)

[SymPointV2](https://github.com/nicehuster/SymPointV2) labels every line, arc and circle of a floor plan with one of
35 classes. It was trained on FloorPlanCAD. The classes include single / double / sliding / folding doors,
windows, wardrobe, bed, sofa, toilet, stairs, wall, curtain wall and railing. The converter uses these labels as
**evidence**, not as decisions:

* walls, columns and stair flights whose lines are confidently furniture are rejected,
* a "window" whose lines are a door or a wardrobe front becomes a (sliding / closet) door,
* window / door candidates that are furniture are rejected,
* gaps without a symbol become doors or windows when labelled door / window lines lie in them,
* every candidate gets a `learned_label` evidence value next to the rule-based evidence.

The model never creates or moves geometry. Every decision names the class and score in the element's notes.

The SymPointV2 repository has no license file. It is **not** copied into this project. You clone it yourself and
run this service inside your clone.

## Setup (Linux or WSL2 with an NVIDIA GPU; tiles are processed one at a time)

```bash
bash tools/sympoint_service/setup.sh ~/sympointv2
```

The script:
1. clones SymPointV2 into `~/sympointv2` and creates the conda environment `sympoint` (Python 3.10),
2. installs a current PyTorch that supports your GPU. The repository pins torch 1.10 + CUDA 11.1, which does not
   run on RTX 40xx / 50xx cards. RTX 50xx gets the CUDA 12.8 build, older cards torch 2.5.1 + CUDA 12.4,
3. removes the obsolete `#include <THC/THC.h>` lines from the point ops (nothing else from THC is used) and
   compiles them. This needs the CUDA toolkit (`nvcc`) matching the PyTorch CUDA version,
4. copies `server.py` into the clone as `sympoint_server.py`.

Then download the released weights (Google Drive link in the SymPointV2 README) to `~/sympointv2/weights.pth`.

## Run

```bash
cd ~/sympointv2 && conda activate sympoint
python sympoint_server.py --config configs/svg/svg_pointT.yaml --checkpoint weights.pth      # http://127.0.0.1:8790
```

* **Web app:** open the **SymPointV2 symbol detector** panel and keep the address `http://localhost:8790`.
  Press **Test connection**, then tick **run after every drawing is loaded** or press **Run SymPointV2 now**. If the
  local model verification is switched on too, it runs after SymPointV2.
* **Command line (same machine or over the network):**
  `python -m architectural_converter drawing.dxf --sympoint-url http://localhost:8790`
* **Offline:** `python -m architectural_converter drawing.dxf --sympoint-request req.json`, then on the GPU machine
  `python sympoint_server.py --config ... --checkpoint ... --in req.json --out resp.json`, then
  `python -m architectural_converter drawing.dxf --sympoint-response resp.json`.

## How the drawing is sent

`architectural_converter/learned/sympoint_io.py` writes the same fields as SymPointV2's own `parse_svg_v5.py`
(`commands`, `args` = 4 points per primitive, `lengths`, `layerIds`, `widths`, …). Coordinates are in metres with y
pointing down like SVG. Every primitive keeps its DXF entity id, so labels map back to DXF handles. Large drawings
are cut into overlapping square tiles (`config/learned.json`: `tile_m` 30 m, `overlap_m` 4 m). A primitive takes its
label from the tile whose core contains it. The tests check that this JSON passes through SymPointV2's own data
loader (`svgnet/data/svg3.py`) unchanged.

## Things to check on your drawings

* **Tile size.** FloorPlanCAD drawings are crops of apartment / office plans. If labels look poor, try
  `tile_m` 20–40.
* **Domain.** The model was trained on one drafting style. Symbols from other offices may get lower scores, and
  low-score labels are ignored (`min_label_score`). For your own symbol library, fine-tune SymPointV2 on a few of
  your drawings with the repository's training script.
* **Thresholds.** `veto_share` / `veto_score` / `retype_share` in `config/learned.json` decide how sure the model
  must be before it can overrule the rules.
