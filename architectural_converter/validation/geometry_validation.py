"""Drawing-level checks: units, ingestion problems, degenerate geometry, duplicates."""
from __future__ import annotations


def validate_geometry(report, drawing, ctx, dup_stats):
    u = drawing.units
    if u is not None:
        if u.confidence < 0.4:
            report.add("geometry", "uncertain_units", "warning", None, None,
                       f"Drawing units inferred as {u.unit} with low confidence ({u.confidence:.2f}).",
                       "Set the units explicitly (runtime.forced_units) if the scale is wrong.", u.confidence, set())
        if "warning" in u.evidence:
            report.add("geometry", "unit_header_mismatch", "warning", None, None, u.evidence["warning"],
                       "Check $INSUNITS; geometry-based units were used.", 0.7, set())
    for iss in drawing.issues:
        if iss.get("severity") == "warning":
            report.add("geometry", "ingestion_warning", "info", None, None, iss["message"],
                       "Inspect the source entity in CAD.", 0.5, set())
    if dup_stats.get("exact", 0) + dup_stats.get("contained", 0) > 0:
        report.add("geometry", "duplicate_entities", "info", None, None,
                   f"{dup_stats['exact']} exact and {dup_stats['contained']} contained duplicate segments were ignored "
                   f"({dup_stats['partial_overlaps']} partial overlaps).",
                   "Clean the drawing with OVERKILL if needed.", 0.9, set())
    if ctx.stats.get("fragmented_strokes"):
        report.add("geometry", "fragmented_linework", "info", None, None,
                   f"{ctx.stats['segments_in_fragmented_strokes']} segments were joined into "
                   f"{ctx.stats['fragmented_strokes']} continuous strokes.", "No action needed.", 0.9, set())
