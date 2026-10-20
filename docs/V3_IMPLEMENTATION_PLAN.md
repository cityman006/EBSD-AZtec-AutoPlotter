# V3 implementation and validation plan

Work branch: feature/advanced-crystallography-v3, based on origin/main at 30d590ca8344e481380a5fbac38c1e1402b4d471.

## Work slices

1. **Orientation contract and validation fixtures:** lock phase/spec lookup, radians, quaternion convention, sample/crystal frames, proper hexagonal and cubic symmetries; add deterministic synthetic tests.
2. **Quantitative texture:** family pole figures with crystallographic reciprocal metrics and a symmetry-aware SO(3) kernel ODF, explicit MRD normalization, sampling/weighting metadata, reproducible numeric HDF5/CSV/JSON outputs and vector-annotated plots. Compare definitions with an MTEX script where available.
3. **Schmid fields:** define α-Ti slip families from four-index directions/planes, remove symmetry duplicates, compute signed m and |m| for uniaxial load, and keep stress-derived τ in MPa separate.
4. **α variants and BOR:** generate the 12 symmetry-distinct Burgers child orientations, compute disorientation fits and ambiguity-aware assignments; add synthetic single-parent/multi-parent/noise cases.
5. **Prior-β research reconstruction:** infer candidate parents from alpha variants and adjacency, preserve unknowns, report residual/coverage/confidence QC, and support β anchors without treating inferred volume as measured. Validate the Python baseline synthetically; inspect installed MATLAB/MTEX without installing it.
6. **Integration and compatibility:** add opt-in CLI/GUI entry points and immutable V3 archive/cache results using the existing analysis/render run model; preserve Render Only and style-only rerender behavior. Update docs, validation report, and optional Python 3.12 CI.
7. **Final verification and PR:** run focused tests after each slice, full regression suite, diff/security/data checks, commit by module, push only the feature branch, create a Draft PR targeting main, inspect CI, and leave it unmerged.

## Evidence policy

Feature claims will be split into implemented/tested, awaiting external reference validation, and not implemented/blocked. No MATLAB/MTEX scientific agreement or full-size-data performance will be claimed unless directly executed and measured.
