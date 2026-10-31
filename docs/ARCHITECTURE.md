# Architecture

```
DWG ─(ODA)→ DXF ─→ ingest ─→ preprocess ─→ context ─→ detectors ─→ resolve ─→ [corrections]
                                                                               │
          export ←─ validate ←─ classify ←─ topology ←─ reconstruct ←──────────┘
            │
            └─→ archmodel JSON / SQLite ─→ Phase 2 (reconstruction3d) ─→ GLB
```

Every stage lives in its own package, reads the configuration (`architectural_converter/config/*.json`), and can be
run on its own: `ExtractionPipeline.process(path, until="detect")`, then `run_stage(state, "resolve")`, and so on.
`rerun(state, corrections)` replays everything from `resolve` using a cached deep copy of the detector output.

## Stage 1: ingestion (`ingestion/`)
* `dxf_loader` uses ezdxf *recover* mode; audit errors and fixes become issues.
* `entity_extractor` walks model space (paper space optional) and expands every `INSERT` recursively with its full
  transform (nested, rotated, mirrored, non-uniformly scaled, MINSERT arrays). It applies the layer-0 rule (entities on
  layer `0` inherit the INSERT layer) and records the INSERT chain (`block_path`) and the handle of the entity inside
  the BLOCK definition (`source_block_handle`).
* Supported entity types: LINE, LWPOLYLINE/POLYLINE (bulges become exact ARC parts linked to the parent polyline;
  constant width is kept), ARC (normalised to counter-clockwise in world space, so mirrored blocks are handled),
  CIRCLE, SPLINE/ELLIPSE (flattened), HATCH (all boundary loops), SOLID/TRACE/3DFACE, MLINE (+ exploded lines), TEXT,
  MTEXT, ATTRIB, DIMENSION (measurement + definition points), POINT. Other types are kept as raw records and counted.
* `RawEntity` is never modified. `GeoEntity` is the normalised copy that the later stages work on.

## Stage 2: preprocessing (`preprocessing/`, `geometry/`)
* **Units** (`unit_detection`): `$INSUNITS` is only a prior. Each candidate unit (mm, cm, dm, m, in, ft) is scored on
  how plausible the drawing becomes in metres: robust extent, dominant spacing of parallel lines (wall thickness),
  quarter-arc radii (door swings) and text height. If the header disagrees with the geometry, the geometry wins and a
  warning is recorded.
* **Normalisation**: `model = (dxf − offset) · scale`, where the offset is the robust lower-left corner of the drawing.
  This avoids precision loss on drawings placed at survey coordinates. The transform is exported, so the mapping is
  reversible.
* **Adaptive tolerances** (`tolerance.py`): separate values for duplicates (from numeric precision), endpoint snapping
  (from a natural break in the near-miss endpoint-gap distribution, capped below the thinnest plausible wall),
  intersections, parallelism (length-aware angle tolerance) and thickness search range.
* **Cleaning**: degenerate entities are flagged, not deleted. Exact and contained duplicates are marked
  (`dup_of`), and partial overlaps are reported. Collinear fragments on the same layer and role are joined into
  *strokes* that keep their member segments and the position of each along the stroke.
* **Robust geometry**: all set operations on derived geometry use snap-rounding overlay on a 1 µm grid
  (`robust_union/difference/intersection`). Inscribed width uses the pole of inaccessibility (no buffering). In
  Pyodide a GEOS exception aborts the interpreter, so this hardening is required.

## Stage 3: semantic context (`detectors/context.py`)
* **Layer analysis**: names are tokenised (separators, camelCase, accents, numbers) and matched against
  configurable multilingual keyword tables (`config/layer_semantics.json`) to give category scores and a
  `wall_evidence` value. Per-layer statistics: entity counts, colours, linetypes, lengths, share of parallel-paired
  linework.
* **Symbols** (`symbol_analysis`): every INSERT gets a role (door, window, column, furniture, assembly, unknown).
  The role combines block-name and layer hints with a geometric signature: a swing arc plus a leaf line means a door;
  equal parallel thin lines mean a window; a compact closed or filled shape means a column. Large wall-bearing blocks
  (*assemblies*, for example hotel room modules) are transparent, so their lines are ordinary linework.
* **Texts**: room names (multilingual keyword table), room numbers, area annotations, tags. **Dimensions**:
  measurement plus definition points. **Fills**: hatch and solid regions classified as wall fill (thin everywhere),
  column fill, area fill or other.

## Stage 4: detectors (`detectors/`)
All detectors subclass `BaseDetector(detect(drawing, context) → candidates)`, declare `requires`, are registered with
`@register` and are enabled in `config/detection.json`. A failing detector is skipped and reported; it does not stop
the pipeline.

| detector | idea |
|---|---|
| `parallel_wall_detector` | Pairs strokes that are parallel (length-aware tolerance), separated by a distance in the thickness search range, nearly constant in offset, and overlapping lengthwise. **Emptiness**: a parallel line inside the band cuts the pair and is reported as a *band obstruction* (glazing). Lines running straight through the whole band, such as grid or axis lines, are ignored. **Side consistency**: a boundary line carries material on one side only, so a greedy pass rejects e.g. the cavity between two adjacent walls. Thickness classes are clustered from the drawing (weighted 1-D clustering). Concentric arc pairs give curved walls. |
| `polyline_wall_detector` | Closed thin outlines (erosion thinness, aspect, t ≈ 2A/P), constant-width polylines (width = thickness), MLINE (style width × scale). |
| `hatch_wall_detector` | Only *wall-fill* regions. Their boundaries are paired with the extra condition that the material lies inside the fill. Unpaired thin parts become polygon walls. |
| `layer_wall_detector` | Produces layer-evidence *linework* candidates only. They are attached to geometric walls that share the same boundary segments, or discarded. |
| `single_line_wall_detector` | Unpaired lines with support from layer, network connectivity (gap-tolerant along the wall direction), and enclosed room-sized faces. Small isolated networks (furniture) are excluded. A drawing-level convention check applies: if most wall-like linework is already double-line, a single line also needs explicit wall-layer evidence. Thickness stays `None`. |
| `door_detector` | Swing leaves come from blocks and loose geometry (arc with plausible radius and sweep, plus leaf line). The opening chord runs from the hinge to the closed-leaf point. Double doors are paired across sources. Sliding doors are door blocks without arcs. |
| `window_detector` | Window symbols, glazing line groups (in a wall band or on a window layer, not on furniture layers), and band obstructions inside real wall bands (a thickness class of the drawing, with the wall continuing beside it). |
| `column_detector` | Compact closed or filled shapes, symbols, and loops drawn by their own short lines. Evidence includes fill, layer, repetition and grid alignment. |

## Unified resolver (`classification/candidate_resolver.py`)
* Merges same-band candidates. Evidence is merged per feature with `max`, so the same geometric evidence is never
  counted twice, even when several detectors used it.
* Attaches layer evidence, hatch-fill coverage and dimension support (a DIMENSION whose definition points lie on both
  faces and whose value equals the thickness).
* Absorbs polygon walls explained by straight walls. Vetoes walls that are actually glazing (thinner than the window
  depth), door leaves, or column outlines. Resolves conflicting overlaps.
* **Confidence** is the weighted mean of the evidence features present (weights in config). A configurable **gate**
  reduces confidence when no strong source agrees (a thickness class of the drawing, hatch, MLINE or polyline width,
  a dimension, or a manual correction). Thresholds: ACCEPTED ≥ 0.68, NEEDS_REVIEW ≥ 0.45, otherwise REJECTED.

## Stage 5: reconstruction (`reconstruction/`)
1. `line_merging`: merges collinear same-band fragments when the gap is ≤ snap, or when another wall fills the gap
   (the wall runs through an X or T junction).
2. `intersection_resolver`: L, T and curved junctions. An end is extended to the axis intersection plus half the other
   wall's thickness divided by sin(angle), but only for non-parallel walls and small extensions (≤ 1.5 × thickness).
   Every extension is recorded, and the detected interval is kept.
3. Merge again: T-junctions are now filled, so walls run through them.
4. `opening_reconstruction`: a ray is cast from each free wall end along the wall axis:
   * the gap is opening-sized and a door/window chord lies in it → the opening is hosted and the wall made continuous;
   * the gap is opening-sized with no symbol → an *unidentified opening* (NEEDS_REVIEW, hypothesis);
   * the gap is smaller than an opening and continues the same wall → closed as a recorded low-confidence
     reconstruction;
   * a gap crossed by another wall is never bridged (e.g. partitions facing each other across a corridor);
   * wider gaps stay open (validation and room recovery report them).
   Symbols drawn over continuous walls are hosted by projection.
5. `wall_pairing`: links face strokes to each final wall, measures face support, and re-measures thickness from both
   faces.

## Stage 7: topology (`topology/`)
* Graphs (NetworkX): the planar axis graph (noded centre lines) and the wall-contact graph. Topology consistency
  (attached ends, membership of the main network; parallel neighbours do not count as junctions) is fed back into
  wall confidence.
* **Rooms**: the solid is the union of wall bodies (openings are bridged by their host walls), centre-line walls
  with a thin topological width and accepted columns, closed with a snap-sized morphological closing. Each hole is a
  room candidate. Tiny regions, narrow cavities (inscribed width < 0.7 m) and very large regions are rejected or
  flagged. Rooms are scored on boundary support (accepted vs uncertain walls vs openings), labels, area-label match
  and shape.
* Adjacency covers rooms sharing walls and rooms connected through doors or openings.
* **Missing-room recovery**: from free ends, collinear extensions and links between ends are proposed. A proposal is
  kept only if it encloses a new plausible room, and gains confidence when it captures an orphan label. It is output
  as a hypothesis wall and room (≤ 0.5 confidence, NEEDS_REVIEW) and is never added silently.

## Stages 8–9: classification and validation
* Walls are exterior when one face runs along the building footprint. Partitions are interior walls at or below a
  thickness limit derived from the drawing's own thickness classes. Openings are exterior or interior. Columns are
  engaged or free-standing. Rooms are typed from labels first, then geometry (corridor, service room).
* Validation produces issues with entity id, source DXF entities (handle, layer, type, block), location, problem,
  suggested correction and confidence.

## Stage 10: review and manual corrections
The web UI (`web/`) or a corrections JSON (`pipeline/corrections.py`) supplies status overrides, added walls
(including endpoint connections and accepted hypotheses), room boundary overrides and opening type overrides.
Corrections are separate from the source data and are re-applied on every re-run.

## Adding a detector
```python
from architectural_converter.detectors.base_detector import BaseDetector, register

@register
class MyWallDetector(BaseDetector):
    name = "my_wall_detector"
    produces = "wall"
    requires = ("parallel_wall_detector",)       # optional: read ctx.candidates[...] / ctx.thickness_clusters

    def detect(self, drawing, ctx):
        cands = [...]                            # models.wall.Wall with evidence + source_entities
        return self.finalize(cands)              # scores with config weights + gate
```
Add its name to `enabled_detectors` and the weights of any new evidence features to `config/detection.json`. Then
add it to the resolver's wall list if it produces walls.
