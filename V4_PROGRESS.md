# V4 progress and verification — 4.0.0.dev1

This development increment has runnable implementations in M1–M4, with stacked
Draft PRs. **The complete attachment acceptance list is not yet fulfilled.**
Research baselines and software tests are distinct from external scientific
validation. Missing work is recorded below rather than represented as usable.

## Dependency and preservation

Remote main: `30d590ca8344e481380a5fbac38c1e1402b4d471`. V3 Draft
[PR #1](https://github.com/cityman006/EBSD-AZtec-AutoPlotter/pull/1) remains
unmerged at `bc81e4170a7c7f6c5b117624bd74551903b230d7`.
The independent V4 clone retains reviewed V3 operational fixes; original V3
uncommitted files and raw experimental sources are untouched.

| Stage | Branch / target | Draft PR | Stage head before M4 |
|---|---|---|---|
| M1 | feature/v4-modern-gui-batch → feature/advanced-crystallography-v3 | [#2](https://github.com/cityman006/EBSD-AZtec-AutoPlotter/pull/2) | 44a3d00d90e727fe64293da54a07e896cdb498a0 |
| M2 | feature/v4-ebsd-quantitative → feature/v4-modern-gui-batch | [#3](https://github.com/cityman006/EBSD-AZtec-AutoPlotter/pull/3) | 58d5e3a8b4e43da83fd011c8c70a8213993ac8d7 |
| M3 | feature/v4-patches-eds-gnn → feature/v4-ebsd-quantitative | [#4](https://github.com/cityman006/EBSD-AZtec-AutoPlotter/pull/4) | b71ef45e7fe00a31f31ddbdc8105874edcaf522e |
| M4 | feature/v4-mechanics-diffusion → feature/v4-patches-eds-gnn | pending creation after final checks | final SHA in Git/PR |

No main commits, automatic merge, force-push, or public/private setting changes.

## Implemented paths

- M1: optional PySide6 Studio, collapsible docks/themes, persisted bilingual
  navigation, import worker, serial batch child processes, frozen settings,
  montage grouping, failure continuation, cancel/verified resume, queue/logs,
  cache history, font/format/map/layout controls, native publication exports.
  The integrated viewer adds lazy power-of-two disk tiles (≤512² per read),
  a bounded base preview, stale-result guards, original coordinates/ROI and
  measured dy/dx. Tk and V1/V2/V3 remain available.
- M2: phase-specific RX **proxy**, GOS/KAM/GROD QC and threshold sensitivity,
  physical anisotropic grain descriptors/CDF, Euler scalars/RGB, inverse-pole
  direction statistics, numeric per-phase and per-sample tables. Correct Haar
  MRD normalization and exact PF disk-cell intersections are versioned;
  legacy texture caches are preserved and rejected for new metrics. Euler
  units use metadata/format evidence, never numerical magnitude.
- M3: exact native seven-channel grid/pixel/µm/ROI crops, masks, overlap/padding
  rules, enabled-tile selection, CSV/JSON indexes; H5OINA EDS windows/timing,
  coordinate verification and explicit registration, native cps preservation;
  real 2D grain graphs and physical boundary lengths, source/hash/phase schema,
  grouped Dataset and optional Windows Torch DataLoader.
- M4: independent mechanics-label import with units/frame/source mapping;
  grouped learned CPU GNN/ridge, train-only scalers, held-out metrics,
  research prediction/uncertainty exports. Real inference requires matching
  material, loading, frame, load step, time and relevant phase schema.
  Numerical conditional DDPM learns epsilon and samples a reverse chain;
  real HCP/BCC fields use actual phase-aware crystallographic representatives.
  GUI label gates, four AI tabs, isolated processes, pause/cancel and durable
  incomplete checkpoints/compatible weight resume are integrated. Mechanical
  target generation stays disabled without an independently validated surrogate.

## Fresh evidence

Focused M4: **36 passed**, including phase ID/context gates, real learning,
symmetry invariance, cancelled checkpoint recovery and four-format readback.
Independent review reran six relevant cases: **6 passed, 1.60 s**. It found and
fixed a sub-unit boundary-weight normalization error and GUI CrashExit status.
Qt/preview/process tests cover real child launch, pause/cancel, stale map results,
physical aspect, masks and bounded reads from an instrumented 100000×200000 grid.
Final full Windows regression: **167 passed, 2 upstream deprecation warnings,
57.72 s** with `.venv\Scripts\python.exe -m pytest tests -q
--basetemp=.local/test-v4-final-verified`. Existing Tk tests ran with normal
desktop process permissions after a sandbox handle restriction; no tests were
removed or validation disabled. `compileall` and `git diff --check` pass.

Local dependency check: `pip check` passes. Python 3.12, PySide6 6.12; optional
Torch 2.14.1+cpu was actually installed/tested, including Windows worker spawn.
CI does not install Torch and skips that optional test; no GPU result is claimed.
Offscreen Qt screenshots at 100% and 200% scale were inspected; these are not
physical 4K/multi-monitor measurements. No font binaries are redistributed.

Real 1009×756 V1/V2/quantitative/render workflow returned 0 in **29.913 s**;
raw SHA256 unchanged. Read-only 4317×4102 (17,708,334 pixels, 1,905,189,929 bytes)
streamed all rows in **6.267 s**, source stat unchanged. Whole-process peak
working set across those runs: **539,811,840 bytes**. This is a reader benchmark,
not a full large-map grain/ODF analysis benchmark or extrapolated performance.

Real 3×5 export with EDS and a 30,202-node/41,715-undirected-edge grain graph
completed. All 762,804 coordinate pairs and seven cps arrays were checked;
source cps values were preserved exactly. A legacy graph pair lacks the new
source SHA256 and is explicitly marked unverified. No chemical concentration
or external segmentation accuracy is inferred.

Latest actual Render Only used the existing cache, created a NEW render run,
and returned 0: **18 PNG, 10 TIFF, 10 PDF, 10 SVG** were reopened/parsed;
native maps retain 1009×756, manifest success. Poppler rendering of the paper
PDF was visually checked. Private outputs/evidence are under `.local/`:

- `real-validation/.../renders/render_20261020_164237_407205_e0c5ac74/`
- `real-validation/performance_report.json`, `final_export_verification.json`
- `m4-smoke/`, `m4-adapter-smoke/` (clearly synthetic; no material validation).

## Remaining acceptance gaps

- Full bilingual control translation, linked simultaneous viewers, progress
  percentages, configurable parallel analyses, actual 4K/multi-monitor QA.
- Full large-map grain/texture analysis, EDS montage stitching and GUI landmark
  registration editor; complete publication-template/interactive styling suite.
- Independent MTEX/AZtec agreement, RX metallurgy calibration, instrument frame
  transform verification, real segmentation/reconstruction reference accuracy.
- Independent real mechanics labels/training/blind test, cross-material
  generalization, calibrated uncertainty and strong OOD detection.
- U-Net/FNO/full deep PyG architectures and GPU validation; current ridge and
  one-hop GNN are explicitly named research baselines.
- Large latent/convolutional diffusion, grain-size target conditioning,
  generated texture/grain-distribution validation, physical realism and any
  mechanically conditioned inverse design. DDPM is limited to 4096 encoded
  variables/tile; requested phase fractions are measured, not enforced.
- Exact optimizer/RNG recovery; current resume is compatible weight training.

All real data, models, caches, local paths/settings and virtualenvs are excluded
from the public change set. No project-wide license was invented; third-party
and optional MTEX notices remain. See the module documents and Chinese guide.
