# Benchmark report (25 Nov 2026)

`benchmark_report.pdf`: how approach 1 (rules) and approach 2 (SymPointV2) work, the simulated benchmark,
the real GPU results on the 13 benchmark drawings and the two real drawings, and what is missing.

`scripts/` holds the analysis scripts that produced it. They were run against branch
`claude/practical-planck-ugw5wn` and the GPU answers (`gpu_out/*.sympoint.response.json`); paths inside them
point to the analysis machine and need adjusting before a re-run.

| script | what it does |
|---|---|
| `scratch_bench.py`, `scratch_bench_strip.py`, `scratch_bench_hybrid.py` | simulated-label benchmark (with layers, layers renamed to 0, approach-1 fallback) |
| `label_accuracy.py` | line-level accuracy of the raw model answer per class |
| `real_variants.py`, `rv_helpers.py` | rebuild approach 2 from the real answers: rules on/off, no layer names, fallback |
| `real_drawings.py`, `apt_truth.py` | real drawings: build variants, score apartment doors / windows |
| `plot_level.py` | plan images of one storey |
| `make_pdf_report.py`, `report_template.html` | the PDF |
