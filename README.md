# EBSD-AZtec AutoPlotter V1+V2+V3

A Windows Python/Tkinter research application for processing Oxford Instruments AZtec H5OINA/HDF5 EBSD maps. The repository does not declare a semantic version or ship a compiled installer; V3 is an opt-in research feature set on the `feature/advanced-crystallography-v3` branch.

## Features

- **V1 plotting:** row-blocked scan reads, IPF X/Y/Z, phase, KAM, band contrast, MAD, pattern quality, and overview layouts. Input Euler units follow the H5OINA metadata contract; the cache records the input unit.
- **V2 analysis:** optional ROI grain segmentation, grain summaries, boundary/GOS/GROD-style fields and grain adjacency graph. V2 results remain exploratory.
- **V3 texture:** pole figures for HCP `{0001}`, `{10-10}`, `{11-20}` and BCC `{100}`, `{110}`, `{111}` families; PF intensity is MUD (mean valid-bin density = 1). The numeric ODF is a symmetry-summed Gaussian kernel on a Bunge ZXZ SO(3) grid, normalized with the Haar measure and stored in HDF5. Pixel-area and equal-grain weighting are explicit choices. The plots are not substitutes for numeric ODF data.
- **V3 α-Ti Schmid analysis:** basal `<a>`, prismatic `<a>`, pyramidal `<a>`, and pyramidal `<c+a>` families, derived from measured HCP lattice parameters and sample-frame loading direction. Uniaxial Schmid factor `m` is dimensionless; stress-resolved shear `τ=sᵀσn` is separately labeled in MPa. No local stress, CRSS-based activation, or “first slip” claim is inferred.
- **V3 BOR / prior β:** constructs twelve ideal Burgers variants, fits α-grain candidates, retains unknown/ambiguous cases, and reports inferred parent coverage, OR residuals and confidence. A single α grain does not uniquely identify its parent. Inferred prior-β coverage is not a measured volume fraction. `variant_boundary_type_map` distinguishes same-parent variant boundaries, inferred parent boundaries, and unknown edges.
- **Archive and export:** per-scan/per-run outputs, resumable checkpoints, analysis cache, independent Render Only, installed-font selection and PNG/TIFF/PDF/SVG exports are retained. V3 results are transactional additions inside the selected analysis run.

## Inputs and coordinate conventions

Supported inputs are `.h5oina` and compatible AZtec `.h5` files, including montage scan manifests. V3 requires explicit source phase names and supported space groups (named HCP Ti, SG 194; named BCC Ti, SG 229); it does not infer phase identity from numeric raster IDs. V3 also requires measured HCP lattice dimensions for `c/a`; it does not insert nominal Ti constants.

The V3 orientation contract is Bunge ZXZ Euler angles in radians at input, SciPy quaternion order `xyzw`, and active crystal-Cartesian to sample-Cartesian rotations. Source specimen/scanning metadata is recorded, but those transforms are not applied until their AZtec frame/sign conventions are independently verified. The Schmid loading vector is specified in sample coordinates. ODF sample rotation is an explicit left action, defaulting to identity.

## Windows setup and use

1. Install 64-bit Python 3.12 with Tcl/Tk.
2. Clone the repository and run `INSTALL_WINDOWS.cmd`.
3. Start with `START_WINDOWS.cmd`.

Alternatively, from PowerShell in the repository:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe gui.py
```

In the GUI choose `v3_texture`, `v3_schmid`, `v3_variants`, or `v3_prior_beta`. The grain analyses use the ROI field; Schmid uses the sample-coordinate loading axis. Equivalent CLI examples:

```powershell
python -m ebsd_autoplot scan.h5oina --output-root output --analysis v3_texture --v3-weighting area
python -m ebsd_autoplot scan.h5oina --output-root output --analysis v3_schmid --roi 0 0 800 800 --v3-load-axis 0 0 1
python -m ebsd_autoplot scan.h5oina --output-root output --analysis v3_variants --roi 0 0 800 800
python -m ebsd_autoplot scan.h5oina --output-root output --analysis v3_prior_beta --roi 0 0 800 800
```

Large files are read in row blocks for V1; V3 ODF currently samples a seeded maximum number of orientations and uses a fixed numeric SO(3) grid. Schmid and grain reconstruction require an ROI and enforce a pixel safety limit. Large full-map V3 performance has not been benchmarked.

## Scientific status and verification limits

V3 outputs are research analyses. Synthetic checks cover phase mapping, HCP/BCC symmetry utilities, PF families, ODF Haar normalization, Schmid system geometry, the twelve BOR variants, isolated-grain ambiguity, archive integration and inferred-parent QC. These checks do not establish agreement with an independent scientific package or experimental reference.

`mtex/reconstruct_prior_beta.m` is an optional MTEX comparison path using the CTF and orientation JSON exported by the application. MATLAB/MTEX is not a runtime requirement for normal Python processing. No MATLAB/MTEX installation or actual comparison run was available during this implementation; no MTEX agreement is claimed. No real EBSD experiment is included in the public source tree. Full-size performance, instrument-frame correction, multi-parent reconstruction accuracy, noise robustness and independent MTEX comparison remain unverified; see [`V3_VALIDATION_REPORT.md`](V3_VALIDATION_REPORT.md).

The GUI, pipeline and all standard analysis paths are Python; Windows supplies Tkinter/Tcl. MATLAB/MTEX is an optional external scientific reference, not a dependency installed by the application.

## License

No project-wide open-source license has been selected. The MTEX example carries its separate GPL-2.0 notice. Third-party Python packages retain their upstream licenses.
