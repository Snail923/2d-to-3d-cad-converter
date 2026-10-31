# Audit and Technical Implementation Plan

## 1. Audit of the existing project (2026-10-31)

| Item requested by the brief | Finding |
|---|---|
| Existing DWG conversion | **None.** The repository `Snail923/2d-to-3d-cad-converter` had no commits and no branches. |
| Existing DXF parser / entity extraction | None |
| Coordinate & unit handling | None |
| Wall / room detection | None |
| Database schema | None |
| Phase 2 (Shapely + Trimesh) code and its data requirements | None in this repository |

**Consequence.** The brief's rules "do not rewrite before understanding" and
"keep Phase 2 compatible" cannot point at concrete code, because there is no
code. The Phase 1 engine is therefore built from scratch in the module layout
the brief suggests. Phase 2 compatibility is handled like this:

* Phase 1 writes one **stable, versioned intermediate model** (`archmodel` 1.x, JSON, plus
  an equivalent SQLite database). See `docs/DATA_MODEL.md`.
* All Phase 2 access goes through `pipeline/database_exporter.py`. Supporting a different,
  existing Phase 2 schema means adding one adapter class (`Phase2Adapter`). No detector
  needs to change.
* A reference Phase 2 implementation (`reconstruction3d/`, Shapely + Trimesh) consumes
  only the intermediate model. That proves the contract works end to end, and the web
  demo uses it to show the 3D result.

> If your existing Phase 1/Phase 2 code lives somewhere else, push it to this repository.
> The adapter layer is the place to connect it. The engine stays as it is.

## 2. Design principles applied

1. **Deterministic only.** No ML, no learned weights, no external inference. All scoring is
   a transparent weighted evidence model with weights in JSON config.
2. **Units first.** Every drawing is normalised to metres (scale and offset are stored), so all
   architectural thresholds are expressed in real-world units. Drawing-specific tolerances
   (snap, duplicates, angle) are then *estimated from the drawing* (`preprocessing/tolerance.py`).
3. **Nothing is invented silently.** Reconstructed geometry (junction extensions, bridged gaps,
   unidentified openings, missing-wall hypotheses) is always stored in a separate
   field with a reason and a confidence. Hypotheses are never exported as walls unless
   accepted in the review UI.
4. **Traceability.** Raw DXF entities (`RawEntity`) are never modified. Every normalised
   primitive, segment, candidate and exported object carries the ids of the raw entities
   it came from (handle, layer, type, block path).
5. **Independent stages.** Ingestion → preprocessing → context → detectors → resolver →
   reconstruction → topology → classification → validation → export. Each stage has
   its own package and can be run alone (`ExtractionPipeline.run_stage`).
6. **Replaceable detectors.** Every detector subclasses `BaseDetector`, declares its
   dependencies, and returns standard `Candidate` objects with evidence. Detectors are
   registered in configuration, so they can be swapped or disabled.

## 3. Migration / build plan (incremental, test after every stage)

| Stage | Module(s) | Exit criterion |
|---|---|---|
| 1 Ingestion | `ingestion/*` | all listed entity types extracted, nested INSERTs, handles preserved (unit tests) |
| 2 Preprocessing | `preprocessing/*`, `geometry/*` | unit inference correct on mm/cm/m/inch/unitless drawings; duplicates & overlaps identified |
| 3 Semantic context | `preprocessing/layer_analyzer.py`, `detectors/symbol_analysis.py`, `detectors/text_detector.py` | layer hints, block roles, room labels |
| 4 Wall detection | `detectors/*_wall_detector.py`, `classification/candidate_resolver.py` | wall precision/recall on benchmark |
| 5 Reconstruction | `reconstruction/*` | junctions closed, fragments merged, openings not closed |
| 6 Openings | `detectors/door_detector.py`, `detectors/window_detector.py`, `reconstruction/opening_reconstruction.py` | door/window P/R |
| 7 Topology & rooms | `topology/*` | room P/R + IoU |
| 8 Classification | `classification/*` | exterior/interior/partition, room types |
| 9 Validation | `validation/*` | issue report with locations & sources |
| 10 Review UI | `web/` (browser, Pyodide) | toggle layers, inspect evidence, accept/reject, add wall, connect endpoints, room override, re-run |
| 11 Export | `pipeline/database_exporter.py` | JSON + SQLite, unknown values kept as `null` |
| 12 Benchmark | `benchmark/*`, `tests/*` | 12 synthetic drawings with reference geometry; regression thresholds |

## 4. Why a browser (Pyodide) review UI instead of PySide6

The request is a web page that can be tested from GitHub. A PySide6 desktop app cannot be
hosted on GitHub Pages. The *same* Python package runs unchanged in the browser through
Pyodide (numpy, shapely and networkx are official Pyodide packages, and ezdxf and trimesh
are pure-Python wheels). So the hosted page runs the real engine. It is not a JavaScript
re-implementation. The UI layer is plain HTML, SVG and three.js.

## 5. Dependencies

`ezdxf`, `shapely`, `networkx`, `numpy`, `trimesh`. Spatial indexing uses Shapely's
`STRtree`, so `rtree` is not needed. SciPy is **not** used. The only clustering needed
(1-D thickness clusters, endpoint gap statistics) is a few lines of NumPy, and SciPy
would add about 20 MB to the browser download. ODA File Converter is optional and only used for
`.dwg` input (CLI only; it cannot run in a browser).

## 6. Known limits of this version (also see `docs/LIMITATIONS.md`)

* DWG input needs ODA File Converter installed locally. The web page accepts DXF.
* Benchmark drawings are synthetic (generated with known ground truth). Accuracy on your
  real drawings must be measured by adding them to `benchmark/` with reference geometry.
* Curved walls are detected from concentric ARC pairs. Spline-drawn curved walls are not.
