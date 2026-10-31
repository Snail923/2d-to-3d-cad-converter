# Known limitations (v1.0)

**Validation scope**
* The benchmark uses **synthetic** drawings with exact ground truth. Real production drawings (xrefs, messy layering,
  annotation clutter, partial plans, multiple levels on one sheet) have not been measured yet. Add real drawings with
  reference JSON to `architectural_converter/benchmark/drawings/` to measure them; the regression tests pick them up.
* No existing Phase 1/Phase 2 code or database schema was present in the repository, so compatibility is provided
  through the documented `archmodel` contract and a `Phase2Adapter` hook, not against a specific existing schema.

**Input**
* DWG needs the ODA File Converter (CLI only). The browser accepts DXF.
* Model space only by default (`runtime.include_paperspace` enables layouts, but viewport transforms are not applied).
* XREFs are not resolved (bind them before export). Proxy, ACIS and 3D solid entities are kept as raw records only.
* One storey per drawing. Multiple plans on one sheet are treated as one building.

**Detection**
* Curved walls are detected from concentric ARC pairs (including polyline bulges). Spline- or ellipse-drawn curved
  walls are not. Openings in curved walls are not hosted.
* Windows drawn with only two face-aligned lines and no glazing line or symbol are indistinguishable from a
  continuous wall and are not detected.
* A wall whose two faces are on unrelated layers *and* fragmented differently may come out as several collinear
  pieces. They are merged when gaps are small or filled by junctions.
* Thickness-cluster evidence assumes several walls share each thickness class. A drawing with one unique thickness
  per wall gets lower (NEEDS_REVIEW) confidence.
* Furniture placed parallel and very close to a wall (e.g. a bed 25 cm from the wall, all on layer `0`) can form a
  wall-like pair. These candidates are typically NEEDS_REVIEW (see benchmark drawing 09).
* Single-line walls get `thickness = null`. Phase 2 applies `unknown_wall_thickness_m`.

**Reconstruction / topology**
* Gaps of opening size without a symbol become *unidentified openings* (flagged). Gaps wider than the configured
  maximum opening (2.8 m, or the symbol width) are not bridged; missing-room recovery proposes hypotheses instead.
* Room polygons are net areas (inside wall faces). For single-line walls they extend to the centre lines.
* Heights, sill heights and levels are not read from the plan (they rarely exist in 2D). DXF `thickness` (extrusion)
  attributes are preserved as raw attributes but not used.

**Performance**
* Pure Python. Typical floor plans (≤ 10 000 segments) take seconds, also in the browser. Very large site plans can
  take minutes, and parallel pairing is the dominant cost.
