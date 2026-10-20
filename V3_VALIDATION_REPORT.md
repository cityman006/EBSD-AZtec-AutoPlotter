# V3 validation report

## Implementation status

- Implemented: explicit phase/spec classification; source lattice extraction; Bunge ZXZ/radian and quaternion convention metadata; HCP/BCC PF families; numerical SO(3) kernel ODF with Haar normalization; area/grain weighting; HCP α slip-family Schmid/tau analysis; twelve ideal Burgers variants; graph/anchor-supported alpha-to-parent candidates; unknown/ambiguous retention; prior-beta coverage, residual and confidence report; archive checkpoints and GUI/CLI modes.
- Tested locally: focused synthetic unit/integration tests listed below; Python compile check; V3 texture archive stage with transactional artifacts.
- Pending external validation: MTEX comparison, real experimental EBSD, large-map performance, instrument reference-frame transforms, robust multi-parent/noisy data performance, and metallurgical review of thresholds and confidence calibration.

## Verification evidence

The focused V3 tests passed: **9 passed**. The broader local regression run excluding `tests/test_gui_jobs.py` passed: **61 passed, 2 upstream orix/diffpy deprecation warnings**. `compileall`, `pip check` and `git diff --check` passed. A live GUI startup/event-loop smoke run reported a visible Tk window, a working file-selection command, and all four V3 analysis modes. GitHub Actions on Windows Server 2025 / Python 3.12 ran the unfiltered suite and passed: **67 passed, 2 upstream orix/diffpy deprecation warnings**.

The unfiltered full test process in the local sandbox did not complete: Windows raised a fatal `0xc0000008` invalid-handle exception while `tests/test_gui_jobs.py` launched a child process from a Tk event-loop test. The separate clean GitHub-hosted Windows run completed successfully, so this remains a local sandbox limitation rather than a project CI failure. The V3 archive integration test also passed locally. The source fixtures are synthetic and include lattice fields solely for deterministic tests; no experiment data was used.

MTEX/MATLAB was not found in the environment and was not installed. The optional script and CTF sidecar therefore remain unexecuted; no numeric agreement with MTEX is asserted. The repository CI, if enabled, checks Python behavior only.

## Interpretation limits

Schmid values describe resolved geometry under the user-specified load direction/stress tensor; they do not infer local stress, strain, CRSS, or activation. BOR-derived prior-beta maps are inferred candidate partitions. Coverage is a fraction of α pixels assigned by the current graph/anchor rule, not a prior-beta volume fraction. Single α grains and weakly supported graph components can be non-unique and remain unknown/ambiguous.
