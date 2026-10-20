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

## M2 quantitative development

Added configurable RX proxy and per-phase denominators/QC, anisotropic physical
grain descriptors, Euler scalars/RGB, true inverse-pole directions and numeric
ODF metrics, archived cache actions and a v4_quantitative batch mode with
Summary.csv. Unknown values are NaN with a reason, not example values.
Corrected V3 MRD normalization with exact Haar cell weights and an algorithm ID;
old caches are preserved and rejected for new validated metrics. Euler units
come from attributes or the documented H5OINA contract, never magnitude.

Fresh evidence: 18 M2/V3/pipeline checks passed; 13 root M2+Qt+batch integration
checks passed. Real 1009x756 full V1/V2/quantitative/render workflow returned 0
in 29.913 s; original raw SHA256 unchanged. Read-only 4317x4102 (17,708,334 pixels,
1,905,189,929 byte input) streamed all rows in 6.267 s, source stat unchanged.
Whole-process peak working set across both runs: 539,811,840 bytes. These are
local observations, not extrapolated timings or scientific reference agreement.

## Subsequent milestones

M3 implemented and integrated: 3x5/uneven/ROI/pixel/physical patches with exact
seven-channel crops, masks/units/provenance, per-tile selection, CSV/JSON index;
H5OINA flattened EDS window/timing maps with all-point coordinate verification;
native cps preserved, valid counts-to-cps conversion, phase intensity summaries;
physical real-grain graphs, GroupSplit and optional Torch Dataset adapters.
Qt preview and isolated child actions for patches, EDS and graphs are runnable.

Fresh M3 evidence: 19 module tests passed, 24 module+GUI+cached-action tests
passed. Actual 1009x756 EDS import verified all 762,804 coordinate pairs and
seven source cps arrays exactly; actual 3x5 export with EDS and grain graph
completed in new private run directories. Source data remains read-only.
New V2 source hashes prevent wrong-cache pairing; legacy pair identity remains
unverified and is reported. EDS stitching across montage shards and interactive
landmark editing remain pending. No chemical wt.% or at.% is inferred.

M2 quantitative, M3 patches/EDS/graphs, M4 mechanics/diffusion modules in
development, not yet integrated or certified. Scientific external validation
is pending. Actual real-data labels and an independently validated surrogate
are required for mechanics generalization and mechanical-target generation.

No raw data, caches, virtualenvs or model weights are committed or uploaded.
