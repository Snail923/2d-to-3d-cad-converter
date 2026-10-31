# Data model: `archmodel` 1.0 (Phase 1 → Phase 2 contract)

All coordinates are **metres** in the model frame. `source.transform` maps model coordinates back to the DXF:
`dxf_xy = model_xy / scale_to_m + offset_drawing_units`. Unknown values are `null` and are never guessed.

```jsonc
{
  "schema": "archmodel", "schema_version": "1.0.0", "units": "m",
  "source": { "file": "...", "dxf_units": "mm", "unit_confidence": 1.0, "insunits_header": 4,
              "transform": { "scale_to_m": 0.001, "offset_drawing_units": [x, y] }, "metadata": {...}, "summary": {...} },
  "export_statuses": ["ACCEPTED", "NEEDS_REVIEW"],
  "building": { "footprint": {...}, "levels": [{ "id": "L0", "elevation": 0.0, "height": null }] },
  "walls": [{
      "id": "wall_0012", "type": "wall", "classification": "exterior_wall|interior_wall|partition_wall",
      "status": "ACCEPTED|NEEDS_REVIEW|REJECTED", "status_reason": null, "confidence": 0.93,
      "geometry": { "kind": "straight|centerline|arc|polygon", "start": [x, y], "end": [x, y], "direction": [ux, uy],
                    "length": 5.2, "axis": [[x, y], ...] /* arc/polygon */, "arc": {...}, "polygon": { "exterior": [...], "holes": [...] } },
      "thickness": 0.2,               // null for single-line walls
      "thickness_source": "measured|polyline_width|mline|manual|unknown",
      "height": null,                  // unknown -> Phase 2 default
      "openings": ["door_0003"], "connections": ["wall_0013"], "rooms": ["room_0002"],
      "detection_methods": ["parallel_wall_detector", "layer_wall_detector"],
      "evidence": { "parallelism": 1.0, "thickness_consistency": 0.91, "layer_match": 1.0, "topology_consistency": 1.0 },
      "source_entities": [{ "id": "e_000123", "handle": "1F4", "type": "LINE", "layer": "A-WALL", "block": null, "block_path": [] }],
      "reconstruction": { "extensions": [...], "merged": [...], "face_support": {...}, "hypothesis_gaps": [...] },
      "detected_interval": [s0, s1], "is_hypothesis": false
  }],
  "openings": [{
      "id": "door_0003", "type": "door|window|opening", "subtype": "single_swing|double_swing|sliding|fixed|unidentified",
      "classification": "interior_door|exterior_door|exterior_window|passage|...",
      "position": { "center": [x, y], "start": [x, y], "end": [x, y] }, "width": 0.9, "depth": null,
      "height": null, "sill_height": null,          // unknown -> Phase 2 defaults
      "host_wall": "wall_0012", "host_interval": [s0, s1],   // along the host wall direction (s = dot(p, direction))
      "hinge": [x, y], "swing": {...}, "rooms": ["room_0001", "room_0002"], "is_hypothesis": false, ...
  }],
  "rooms": [{ "id": "room_0002", "name": "KITCHEN", "classification": "kitchen", "polygon": {...}, "area": 15.6,
              "perimeter": 16.0, "centroid": [x, y], "label_texts": [...], "area_text": 15.6,
              "bounding_walls": [...], "openings": [...], "adjacent_rooms": [...], "connected_rooms": [...],
              "boundary_support": { "accepted_walls": 0.92, "review_walls": 0.0, "openings": 0.08, "inscribed_width": 3.1 } }],
  "columns": [{ "id": "column_0001", "shape": "rectangular", "size": [0.5, 0.5], "center": [x, y], "polygon": {...}, "height": null }],
  "hypotheses": { "walls": [ /* proposed missing walls (NEEDS_REVIEW, confidence <= 0.5) */ ], "rooms": [ /* rooms they would enclose */ ] },
  "relationships": [ { "type": "hosts|bounds|connects|adjacent|opening_of", "from": "...", "to": "..." } ],
  "validation": { "summary": {...}, "issues": [{ "id", "category", "code", "severity", "entity_id", "source_entities",
                  "location", "description", "suggestion", "confidence" }] },
  "statistics": { "counts": {...}, "timings_s": {...}, "logs": {...}, "thickness_clusters": [...] },
  "corrections": { /* the manual corrections applied */ }
}
```

**What Phase 2 should build:** `pipeline.database_exporter.phase2_view(model)` keeps the elements whose status is in
`export_statuses`, removes hypotheses, and keeps only openings whose host wall is kept. The reference Phase 2
(`reconstruction3d`) builds:
* straight walls as boxes along the axis, with voids for openings (doors 0 → door height; windows sill → head;
  unidentified openings 0 → opening height);
* arc and polygon walls as extrusions; columns as extrusions; rooms as floor slabs.

Every default it uses (`config/phase2.json`) is listed in `metadata.defaults_used`, with the ids of the elements it
was applied to.

**SQLite** (`export_sqlite`) has the tables `model_info`, `walls`, `openings`, `rooms`, `columns`, `relationships`,
`source_entities` and `validation_issues`. Complex fields are stored as JSON text. To write into an existing Phase 2
database, implement `Phase2Adapter.write(model)`.
