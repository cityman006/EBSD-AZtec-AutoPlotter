# EBSD-AZtec AutoPlotter V1+V2

A Windows desktop research tool for batch inspection, plotting, and exploratory analysis of Oxford Instruments AZtec EBSD HDF5 data. It provides a Python/Tkinter GUI and command-line pipeline. The project has no declared semantic release number or installer executable.

## Features

- **V1 plotting:** streams scan data from '.h5oina' and compatible '.h5' files, including files containing multiple scans; produces IPF X/Y/Z, phase, KAM, band contrast, MAD, pattern quality, and a 2x4 overview panel.
- **V2 research analysis:** optional ROI grain segmentation and summaries, GOS/GROD-style maps, neighbor graph and BOR candidate screening. These are exploratory outputs and need validation against domain references before scientific conclusions.
- **Large inputs:** reads data in row blocks; '--block-rows' (GUI: Processing block rows) trades memory for throughput. Large montage files can be split into scan-specific HDF5 shards with a JSON manifest, retaining original scan names and separating scans.
- **Run safety:** outputs are isolated by run and scan; interrupted jobs record state for recovery. Results are written through temporary files and committed atomically where supported.
- **Export:** PNG, TIFF, PDF, and SVG. Raster maps are embedded in PDF/SVG; labels remain vector text. Font selection is limited to fonts installed on Windows.
- **GUI:** inspect, V1, V1+V2, and V2 modes; ROI and block-row settings; format picker with individual checkboxes and select-all/clear-all.

## Input and outputs

Supported inputs are Oxford AZtec '.h5oina' and compatible '.h5' EBSD maps. Use only data files you are authorized to process. The public source repository does not contain experimental inputs or generated result folders.

Each run creates a separate output directory so later runs do not replace earlier results. HDF5 feature caches such as 'features.h5' contain derived values used for independent re-rendering; they are not raw input files.

## Windows setup

1. Install 64-bit Python 3.12 from [python.org](https://www.python.org/downloads/windows/), with Tcl/Tk support.
2. Clone this repository and open its folder.
3. Double-click 'INSTALL_WINDOWS.cmd' (requires internet for Python packages), then 'START_WINDOWS.cmd'.
4. To repair/check an existing environment, use 'REPAIR_WINDOWS.cmd'.

Alternatively, from PowerShell in the project directory:

    py -3.12 -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    & .\.venv\Scripts\python.exe -m pytest tests -q
    & .\.venv\Scripts\python.exe gui.py

The application is implemented in Python; Windows supplies Tkinter and Tcl/Tk. No separate analysis application is required for normal V1/V2 processing. MATLAB with MTEX is only needed to run the optional external reconstruction example.

## Scientific calculations and limits

- **IPF:** uses 'orix''s TSL color key and the phase/crystal symmetry metadata. The AZtec color-key output has not been calibrated as identical.
- **KAM:** four-neighbor, same-phase local misorientation; neighbors above 5 degrees are excluded.
- **MAD:** the source quantity is read in radians and converted to degrees for display; its fixed 0-2 degree scale was checked against the source unit metadata and sample values.
- **V2:** grain/texture summaries and Burgers orientation relationship candidate screening are research aids. Candidate screening does not certify prior-beta parent reconstruction. Texture preview is not a full ODF.
- **MTEX:** 'mtex/reconstruct_prior_beta.m' is an optional reference example and has not been executed or validated in this delivery environment. See the adjacent MTEX GPL-2.0 notice.

## Verification status

The local Windows environment passed the project test suite (58 passed, 2 upstream deprecation warnings), a GUI startup/event-loop smoke check, and a V1+V2 small real-sample acceptance run. The source input hash was unchanged by the run. This evidence does not validate AZtec color matching, external scientific accuracy, full-size performance, or MATLAB/MTEX reconstruction. The large 17.7-million-pixel case and independent scientific reference comparisons remain unverified.

Run tests with 'python -m pytest tests -q'. Runtime requirements are listed in 'requirements.txt'; the Windows Python 3.12 lock file records a tested package set.

## License

No project-wide open-source license has been selected. The standalone MTEX example is derived from the official MTEX TiBetaReconstruction example and is accompanied by its GPL-2.0 license notice. Third-party Python packages are installed from their upstream distributions; see their respective project pages and license files.
