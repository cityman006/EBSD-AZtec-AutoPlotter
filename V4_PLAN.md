# V4 Materials Intelligence Studio implementation plan

V3 dependency confirmed: Draft PR #1, feature/advanced-crystallography-v3, bc81e41.
Original V3 checkout and its uncommitted edits remain intact. V4 development uses
an independent clone; reviewed V3 fixes are carried forward only in M1.

Stacked Draft PRs (no merge, no direct main changes):

1. feature/v4-modern-gui-batch → feature/advanced-crystallography-v3
2. feature/v4-ebsd-quantitative → feature/v4-modern-gui-batch
3. feature/v4-patches-eds-gnn → feature/v4-ebsd-quantitative
4. feature/v4-mechanics-diffusion → feature/v4-patches-eds-gnn

Each milestone requires focused checks, full regression, status and limitations,
then an isolated commit and Draft PR. Synthetic fixtures establish software
behavior only. MTEX/AZtec agreement, instrument transforms, RX metallurgy and
ML generalization require independent experimental validation.

Owners: root integrates M1/GUI/batch/CI; quantitative adds isolated M2 modules;
patches adds isolated M3 modules; research adds isolated M4 modules.
