# EDS row montage and landmark registration

These additions operate on an existing EBSD archive. They never rewrite raw
H5/H5OINA files, the feature cache or an old analysis/render result. Each Import
EDS or patch export creates a new `studio_runs/...` directory; scientific files
publish through the existing transaction receipt mechanism.

## Authoritative row montage

The supported montage is the project's existing **row-partitioned scan**,
identified by its analysis manifest, scan identity and explicit row ranges.
This is not an image-based 2D tile alignment algorithm.

Both preserved-global Header dimensions and the local Header dimensions emitted
by the Compact_v1 extractor are supported. Local Header dimensions require
explicit `global_nx`, `global_ny`, `start_row_in_original` and
`end_row_in_original_exclusive` provenance. Manifest dimensions/ranges, root
provenance and actual EBSD arrays must agree; fractional counts, booleans,
contradictions, gaps and overlaps fail validation. Source coordinates are read
unchanged, and numeric EBSD caches from equivalent local/global Header fixtures
are identical.

EDS windows and Live/Real Time can use native part-row vectors or a preserved
full-grid vector in each shard. Only the authoritative row range contributes to
the assembled map. All measured EDS/EBSD coordinates, steps, units, window names,
dtypes and timing schemas are verified. Local Y coordinate resets are converted
only using the explicit part offset and measured step, with that evidence saved.
Arbitrary offsets or mismatched grids are not silently aligned. A small relative
step difference is insufficient evidence if accumulated coordinate drift exceeds
the precision tolerance.

Native complete maps keep their dtype and exact values, including integer
counts. Source cps is never divided by live time again. Explicit counts can
produce an additional counts/s channel only with measured live time and known
time units; invalid/nonpositive time remains masked. No concentration, wt.% or
at.% is inferred. Native timing is retained separately from registered maps.

If any authoritative part lacks EDS, the whole EDS modality is explicitly
unavailable, with missing/available row intervals recorded. No missing shard is
filled with zero-valued intensities. Malformed present data fails rather than
being relabelled as absent.

Reads are in row blocks. Owned scientific buffers use at most 64 MiB heap per
EDSResult, then temporary file-backed memmaps; consumers must call `close()` or
use the result as a context manager. This is not an OS hard process-memory cap.
Temporary coordinate arrays during affine sampling are also bounded by rows.
Affine interpolation still needs a full-source finite mask and safe-value
scratch copy for one channel at a time; these are outside the owned-buffer
budget. Large native images therefore require additional process memory.

```python
from ebsd_autoplot.io import H5Source
from ebsd_autoplot.eds import load_source_eds, save_eds

source = H5Source(selected_montage_scan_spec)
with load_source_eds(source, block_rows=32, count_rate=False) as eds:
    save_eds(eds, new_output_directory)
```

## GUI workflow

1. Open the desired archived `features.h5`. In **EDS**, Import EDS + phase
   statistics automatically resolves that archive's original scan and montage.
   Existing output paths are not reused. Include verified EDS in Slicing uses
   the same assembly path.
2. For explicit registration, choose **Edit EDS landmarks**. Select a native
   H5OINA source, its EBSD sibling scan path and an EDS window, then Read preview.
   This can be a different native EDS grid from the target cache. The editor
   reads bounded previews in background threads and performs no raw analysis.
3. Click the same feature in the source EDS and target BC panels to add a pair.
   Drag to pan, wheel to zoom, or edit the four numeric table columns. Coordinates
   are original **x,y pixel centres**, not preview pixels. Physical aspect ratios
   use measured steps; unknown source steps are explicitly labelled.
4. Supply at least three noncollinear pairs; use additional well-distributed
   features and independent check features. Fit affine displays the source-to-
   target matrix, each residual, RMS/max in target pixels and conditioning QC.
   Three pairs can fit exactly even when they identify the wrong features.
5. Apply freezes this project for the selected target. Import EDS then reads
   all native channels and applies the explicit linear inverse map to the target
   cache grid. Outside/invalid support is NaN plus a false mask. Timing remains
   on its original grid. The same project can be used by Include EDS patch export.
6. Save JSON writes a **new** file atomically and refuses existing paths. Load
   JSON validates source/target identities and refits its landmarks; it does not
   trust a stored success or external-validation claim. Editing a point, channel
   or source invalidates Apply. Changed/replaced source or target files require
   fresh previews and a fit. Switching the active cache clears an incompatible
   applied project. Clear manual EDS registration restores automatic coordinate
   verification.

Fit residuals use the same landmarks used to estimate the matrix. They are
**not independent registration accuracy, crystallographic validation, chemical
quantification or a calibration certificate**. The editor supports an explicit
affine and does not model nonlinear distortion, automate contrast matching or
silently assemble arbitrary overlapping stage tiles.

## Verification boundaries

Regression checks use uneven row shards, exporter-faithful local Headers,
known affine geometry, exact source values/masks/units, source mutation,
cancellation, private temporary buffers, actual Qt clicks and JSON round trips.
Archive actions explicitly prohibit calling raw EBSD analysis and verify raw,
cache and previous-result bytes remain unchanged.

Fresh integrated M4 Windows regression: **306 passed, 2 upstream orix/diffpy
deprecation warnings, 124.93 s**, using `python -m pytest tests -q
--basetemp=.local/test-v4-eds-registration-full`. Focused backend checks:
**53 passed, 6.38 s**; local/global Header ingestion and preservation checks:
**49 passed, 9.34 s**; actual Qt editor checks including interpreter shutdown:
**20 passed, 86.29 s**. These sets overlap and are not additive.

An intact 756×1009 experimental H5OINA was also imported read-only through the
actual cached action. All seven exported window arrays and dtypes matched their
source exactly; units stayed cps, the transaction validated, and raw/cache/old
paper-PDF hashes stayed unchanged. No real row-shard acquisition was available;
its layout checks use faithful fixtures from the actual extractor contract.
Other vendor EDS layouts, arbitrary stage-tile mosaics and independent measured
landmark-registration accuracy remain unverified.
