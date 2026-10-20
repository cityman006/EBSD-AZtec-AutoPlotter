# Changelog

## V4 research development M2

- Phase-specific RX proxy, physical grain statistics, Euler maps and per-phase
  quantitative summaries; independent cached-action runs and batch Summary.csv.
- Corrected MRD normalization using Haar probability weights; legacy ODF grids
  retained and rejected for newly versioned metrics.
- Removed Euler unit guessing; explicit attributes and H5OINA format evidence.
- Synthetic output validation and real small/large-file performance recorded.

## V4 research development M1

- Added optional PySide6 Studio and portable per-folder Windows launch/install.
- Kept Tk fallback and V1/V2/V3 algorithms and archive interfaces.
- Added frozen serial batch manifests, montage grouping, cache-verified resume,
  sample failure isolation, source-change protection and memory budget checks.
- Added selected maps/layout, per-map timing, one PNG encoding, and native
  embedded rasters with vector text in single-map PDF/SVG.
- Carried forward reviewed local V3 operational fixes without altering original
  uncommitted checkout. See V4_PROGRESS.md for remaining acceptance gaps.
