# V4 dev2: bilingual, linked-view and parallel-analysis verification

This increment implements the three prioritized software paths. It does not
certify all V4 scientific research functions. The original V1/V2/V3 calculation
paths, archive layout, formats, fonts and Render Only remain in the regression
suite. All four PRs remain stacked drafts; main is unchanged.

## Acceptance evidence

| Requirement | Implementation | Fresh checks |
|---|---|---|
| English/Chinese presentation | `i18n.py`, Studio bindings, bundled Qt standard-dialog translator | Catalog inventory with a positive missing-key control; actual Qt round trips preserve navigation, mode, canonical combo data, paths, font, JSON, formats and values; dynamic research/preview states |
| Source-isolated linked maps | `linked_views.py`, MapView navigation hooks | Real 1/2/4 panels; native camera, cursor and ROI; link-off and different-source independence; stale-source invalidation; absent maps clear old images; same-source map/layout changes retain overlays |
| Correct first display | Active-panel initial fit and resize propagation guards | A failing real Qt first-layout test reproduced an undersized map; regression now compares scale against actual viewport geometry |
| Bounded parallel analysis | `batch.py` single coordinator, independent subprocesses | Real process barriers prove overlap and concurrency cap; summed active-memory admission serializes when needed; actual synthetic EBSD children finish in independent archives |
| Cancellation and recovery | All-child control, bounded cleanup, PID/creation-token checks | Cancel two active children with a third queued; queued work stays resumable; failures continue; callback/interrupt cleanup; do not duplicate a still-running prior child |
| Concurrent GUI edits | Frozen batch manifest and isolated child processes | While two scientific children run, change language/theme/view layout/maps and future analysis/output settings; running options and output roots remain unchanged |
| Correct stream text | Incremental UTF-8 Qt decoder and explicit UTF-8 child environments | Split Chinese UTF-8 into individual bytes; actual GBK-parent child tests preserve Chinese logs, JOB_EVENT paths, manifest and CSV without replacement characters |
| Source-safe patch selection | Preview path/size/mtime snapshot and export validation | Preview A, switch to B, reject old selection; re-preview B succeeds; replace B and reject again; source switch clears preview table/export enablement |
| Atomic controls on Windows | Private fsynced temporary JSON, bounded sharing-lock retry | WinError 5/32/33 retry; old target remains readable until replace; permanent lock has bounded timeout; serialization/nonretryable failure removes only owned temporary |
| Safe tile shutdown | Generation guards and destroyed-signal handling | Stale field result rejected, bounded hyperslabs, deleted Qt signal target does not cause shutdown exception |

Human-facing controls, help, tabs, headers, dialogs, known validation and dynamic
states are bilingual. Original technical/backend logs, external error details,
scientific IDs, source phase names, user strings and font family names remain
verbatim. Processing item data are never translated.

Parallelism defaults to one and accepts 1–8 children. Memory admission sums
active reservations, including a 256 MiB process baseline, row buffers and the
largest estimated render/grain ROI stage. This is an estimate, not an operating
system hard limit. Native child thread pools are limited to one.

## Reproduced defects and focused verification

- Windows control replace: three original cancellation checks first failed
  with a sharing error. Retry/parallel regression: **29 passed, 12.45 s**.
- Final M3 language/linked/patch source regression: **32 passed, 11.62 s**.
- GBK-parent batch event test first failed with a corrupted Chinese result path
  despite child exit 0. Batch/parallel/recovery after the fix: **22 passed,
  12.49 s**. Actual M1 Qt GBK-parent regression also failed before the fix;
  after the fix **2 passed, 1.83 s**.
- M2 Unicode integration: **16 passed, 13.79 s**.
- Integrated M4 Qt/parallel interaction and encoding: **9 passed, 16.15 s**.
- Final full integrated suite after all fixes: **212 passed, 2 upstream
  orix/diffpy deprecation warnings, 74.21 s**, using
  `python -m pytest tests -q --basetemp=.local/test-v4-ui-parallel-final-utf8`.
  Existing Tk tests require normal desktop process permissions in this session;
  no tests were removed or scientific checks disabled. Independent final
  encoding review reran **2 passed, 3.20 s**. `pip check`, `compileall` and
  `git diff --check` also pass.

Private read-only experimental-cache screenshots were inspected in Chinese at
100% and English at 200% scale, with BC/Phase/KAM/IPF-Z, linked native ROI and
cursor. These offscreen checks are not a physical 4K/multi-monitor test. The
screenshots and cache are excluded from Git. No raw experimental analysis was
rerun for this UI increment.

## Verification boundaries

CI installs the GUI requirements and runs Windows Python 3.12 tests offscreen.
Optional Torch tests can be skipped there; the local environment has actually
tested CPU Torch and Windows DataLoader spawn. Neither implies GPU validation.
Latest-head GitHub checks are available on Draft PRs
[#2](https://github.com/cityman006/EBSD-AZtec-AutoPlotter/pull/2),
[#3](https://github.com/cityman006/EBSD-AZtec-AutoPlotter/pull/3),
[#4](https://github.com/cityman006/EBSD-AZtec-AutoPlotter/pull/4) and
[#5](https://github.com/cityman006/EBSD-AZtec-AutoPlotter/pull/5).

Dev3 subsequently adds [authoritative EDS row montage and landmark editing](V4_EDS_MONTAGE_REGISTRATION.md).
Fine-grained overall percentage progress, physical multi-monitor QA, external crystallographic comparison,
metallurgical RX calibration, real mechanics blind tests and generated-structure
physical validation remain outstanding. See the complete acceptance-gap list in
[V4_PROGRESS.md](../V4_PROGRESS.md); no software test is represented as an
independent material-science certification.
