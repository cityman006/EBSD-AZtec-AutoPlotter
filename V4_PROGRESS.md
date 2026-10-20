# V4 progress and verification

## Dependency and preservation

Remote main 30d590c; V3 Draft PR #1 still open/unmerged at bc81e41.
Development is an independent clone. Five reviewed local V3 fixes (ROI source
initialization, measured split lattice parameters, Matplotlib font filtering,
render/Schmid progress) carried into M1. Original uncommitted files untouched.

## M1 modern GUI and batch foundation

Implemented: runnable optional PySide6 workbench, collapsible docks, light/dark
themes and saved bilingual navigation, bounded HDF5 preview, original-grid
hover values, Ctrl-drag ROI, map switching, process isolation and logging;
immutable per-scan batches, montage-aware grouping, memory budget, serial jobs,
failure continuation, cancellation, verified-cache repeat/resume, CSV manifests;
native single-map exports and selectable maps/layout; Tk fallback remains.

Fresh evidence: 4 M1 batch/Qt tests passed; 12 M1+MAD tests passed; pip check
passed. Existing full regression: 67 passed, 2 upstream warnings (53.25 s),
run outside sandbox after sandbox Tk invalid-handle crash. No tests removed.

Still partial: disk-backed multilevel tiled viewer, linked simultaneous viewers,
complete bilingual control text, live GUI progress percentages, configurable
parallel analyses, real 4K multi-monitor validation, and large real-data benchmark.
Current preview is bounded; it must not be described as a full tiled viewer.

## Subsequent milestones

M2 quantitative, M3 patches/EDS/graphs, M4 mechanics/diffusion modules in
development, not yet integrated or certified. Scientific external validation
is pending. Actual real-data labels and an independently validated surrogate
are required for mechanics generalization and mechanical-target generation.

No raw data, caches, virtualenvs or model weights are committed or uploaded.
