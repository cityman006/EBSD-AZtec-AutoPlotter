"""EDS numeric maps and window integrals with explicit units and registration.

No peak fitting, background subtraction or elemental quantification is claimed.
Known H5OINA Window Integral layouts are discovered with verified coordinates;
other layouts require an explicit map/spectral dataset mapping.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import h5py
import numpy as np
from .registration import resample_map, validate_affine
from .storage import check_cancelled, transactional_output


@dataclass
class EDSResult:
    maps: dict = field(default_factory=dict)
    masks: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    available: bool = False
    timing_maps: dict = field(default_factory=dict)
    timing_masks: dict = field(default_factory=dict)
    timing_units: dict = field(default_factory=dict)
    _owned_arrays: list = field(default_factory=list, repr=False)
    _temporary_files: list = field(default_factory=list, repr=False)
    _memory_bytes: int = field(default=0, repr=False)

    def allocate(self, shape, dtype):
        """Allocate owned bounded-memory storage; close() releases only this storage."""
        return allocate_eds_array(self, shape, dtype)

    def close(self):
        """Release only buffers allocated by this result, never caller arrays."""
        for array in self._owned_arrays:
            if isinstance(array, np.memmap) and not array._mmap.closed:
                array.flush();array._mmap.close()
        for handle in self._temporary_files:handle.close()
        self._owned_arrays.clear();self._temporary_files.clear();self._memory_bytes=0

    def __enter__(self):return self
    def __exit__(self, *_):self.close()


def allocate_eds_array(result, shape, dtype):
    from .eds_montage import allocate_eds_array as allocate
    return allocate(result, shape, dtype)


def load_source_eds(source, *, count_rate=False, block_rows=32, registration=None):
    """Read native EDS using authoritative H5Source row-part provenance."""
    from .eds_montage import load_source_eds as load
    return load(source, count_rate=count_rate, block_rows=block_rows, registration=registration)


def load_native_h5oina_eds(path, scan_path="/1/EBSD", *, count_rate=False, block_rows=32):
    """Read native EDS without asserting registration to any EBSD cache."""
    from .eds_montage import load_native_h5oina_eds as load
    return load(path, scan_path, count_rate=count_rate, block_rows=block_rows)


def read_eds_preview(path, scan_path="/1/EBSD", channel=None, max_side=1024):
    """Read a bounded decimated native EDS preview with an immutable source stamp."""
    from .eds_montage import read_eds_preview as read
    return read(path, scan_path, channel=channel, max_side=max_side)


def _unit(dataset):
    declarations=[]
    aliases={"cps":"cps","counts/s":"cps","count/s":"cps","cts/s":"cps","countss-1":"cps",
             "countspersecond":"cps","countpersecond":"cps","counts":"counts","count":"counts","cts":"counts","ct":"counts",
             "s":"s","sec":"s","second":"s","seconds":"s","ms":"ms","msec":"ms","millisecond":"ms","milliseconds":"ms",
             "us":"us","usec":"us","microsecond":"us","microseconds":"us","um":"um","micrometer":"um","micrometre":"um",
             "deg":"deg","degree":"deg","degrees":"deg","°":"deg","rad":"rad","radian":"rad","radians":"rad",
             "px":"px","pixel":"px","pixels":"px"}
    for key in dataset.attrs:
        if str(key).casefold() not in ('unit','units'):continue
        values=np.asarray(dataset.attrs[key]).reshape(-1)
        if not len(values):raise ValueError('Empty unit declaration: '+dataset.name)
        for value in values:
            if not isinstance(value,(str,bytes,np.str_,np.bytes_)):raise ValueError('Unit declaration must be text: '+dataset.name)
            text=value.decode('utf-8') if isinstance(value,(bytes,np.bytes_)) else str(value)
            if not text.strip():raise ValueError('Empty unit declaration: '+dataset.name)
            normalized=text.strip().lower().replace('µ','u').replace('μ','u').replace(' ','')
            declarations.append((text,aliases.get(normalized,normalized)))
    if not declarations:return "unknown (not provided by source)"
    if len({value[1] for value in declarations})!=1:raise ValueError('Conflicting unit declarations: '+dataset.name)
    return declarations[0][0]


def load_eds(path, dataset_paths=None, *, expected_shape=None, energy_windows=None,
             energy_axis_path=None, spectrum_path=None, spatial_metadata=None,
             registration=None, block_rows=32):
    """Load explicit named maps or spectral window integrals.

    dataset_paths={element: HDF5 dataset path}; energy_windows={label:(lo,hi)}
    in source energy-axis units, inclusive. Spectra must be H,W,energy. Counts
    are summed in row blocks. registration is an explicit source->EBSD pixel
    affine; otherwise caller must supply spatial_metadata['aligned_to_ebsd']=True
    to assert co-registration even for equal shapes. Missing EDS returns an
    unavailable result, never zero-valued element channels.
    """
    if path is None:
        return EDSResult(metadata={"status": "unavailable", "reason": "No EDS source supplied"})
    if not dataset_paths and not energy_windows:
        return EDSResult(metadata={"status": "unavailable", "reason": "No explicit EDS dataset mapping supplied"})
    if int(block_rows) < 1:
        raise ValueError("EDS block_rows must be positive")
    result = EDSResult(metadata={"source": str(Path(path).resolve()), "status": "available",
                                "spatial": spatial_metadata or {},
                                "quantification": "source maps/window counts only; no inferred concentrations"})
    with h5py.File(path, "r") as src:
        for name, dataset_path in (dataset_paths or {}).items():
            if dataset_path not in src:
                raise KeyError(f"EDS dataset not found: {dataset_path}")
            ds = src[dataset_path]
            if ds.ndim != 2:
                raise ValueError(f"EDS map must be H,W: {dataset_path}")
            arr = ds[:]
            result.maps[name] = arr; result.masks[name] = np.isfinite(arr)
            result.units[name] = _unit(ds)
        if energy_windows:
            if not spectrum_path or not energy_axis_path:
                raise ValueError("EDS windows require explicit spectrum and energy-axis paths")
            cube, eds_energy = src[spectrum_path], src[energy_axis_path]
            energies = np.asarray(eds_energy[:], float)
            if cube.ndim != 3 or energies.ndim != 1 or cube.shape[-1] != len(energies):
                raise ValueError("EDS spectrum shape must be H,W,energy with matching 1D axis")
            if not np.isfinite(energies).all() or np.any(np.diff(energies) <= 0):
                raise ValueError("EDS energy axis must be finite and strictly increasing")
            result.metadata["energy_unit"] = _unit(eds_energy)
            result.metadata["energy_windows"] = {}
            for name, limits in energy_windows.items():
                if name in result.maps:
                    raise ValueError(f"Repeated EDS channel: {name}")
                lo, hi = map(float, limits)
                if not np.isfinite([lo, hi]).all() or lo > hi:
                    raise ValueError("EDS energy window must be finite lo <= hi")
                selected = np.flatnonzero((energies >= lo) & (energies <= hi))
                if len(selected) == 0:
                    raise ValueError(f"EDS window contains no energy channels: {name}")
                arr = np.empty(cube.shape[:2], float); valid = np.empty(arr.shape, bool)
                for first in range(0, cube.shape[0], int(block_rows)):
                    check_cancelled()
                    block = np.asarray(cube[first:first + int(block_rows), :, selected], float)
                    valid[first:first + len(block)] = np.isfinite(block).all(axis=-1)
                    arr[first:first + len(block)] = block.sum(axis=-1)
                arr[~valid] = np.nan
                result.maps[name] = arr; result.masks[name] = valid; result.units[name] = _unit(cube)
                result.metadata["energy_windows"][name] = {"limits": [lo, hi], "channels": selected.tolist(),
                                                               "interpretation": "inclusive raw window sum"}
    if expected_shape is not None:
        shape = tuple(map(int, expected_shape))
        if registration is None:
            if any(arr.shape != shape for arr in result.maps.values()):
                raise ValueError("EDS/EBSD grid mismatch; explicit affine registration is required")
            if not (spatial_metadata or {}).get("aligned_to_ebsd", False):
                raise ValueError("Equal EDS/EBSD dimensions do not establish alignment; provide explicit registration evidence")
            result.metadata["registration"] = {"method": "caller-asserted aligned grids", "resampled": False}
        else:
            affine = validate_affine(registration)
            for name in result.maps:
                arr, valid = resample_map(result.maps[name], shape, affine, result.masks[name])
                result.maps[name], result.masks[name] = arr, valid
            result.metadata["registration"] = {"method": "explicit affine", "source_to_ebsd_pixels": affine.tolist(),
                                                  "resampled": True, "interpolation": "linear", "outside": "NaN + false mask"}
    result.available = bool(result.maps)
    return result


def _unit_key(value):
    return str(value).strip().lower().replace("µ", "u").replace("μ", "u").replace(" ", "")


def _length_factor_um(unit):
    values = {"um": 1., "micrometer": 1., "micrometre": 1., "nm": .001, "mm": 1000., "m": 1e6}
    if _unit_key(unit) not in values:
        raise ValueError(f"Unknown or missing spatial length unit: {unit}")
    return values[_unit_key(unit)]


def _header_geometry(group):
    def scalar(name):
        values = np.asarray(group[name][()]).reshape(-1)
        if len(values) != 1 or values.dtype.kind not in 'iuf' or not np.isfinite(values[0]):
            raise ValueError(f"Expected one finite H5OINA header value: {name}")
        return float(values[0])
    nx, ny = scalar("X Cells"), scalar("Y Cells")
    if nx != int(nx) or ny != int(ny) or min(nx, ny) < 1:
        raise ValueError("H5OINA X/Y Cells must be positive integer pixel counts")
    evidence = {name: _unit(group[name]) for name in ("X Cells", "Y Cells", "X Step", "Y Step")}
    # Named 'Cells' fields are the documented H5OINA raster-count contract.
    for name in ("X Cells", "Y Cells"):
        if _unit_key(evidence[name]) not in ("px", "pixel", "pixels", "unknown(notprovidedbysource)"):
            raise ValueError(f"H5OINA raster count {name} has incompatible unit: {evidence[name]}")
    steps = []
    for name in ("X Step", "Y Step"):
        step = scalar(name)
        if step <= 0:
            raise ValueError("H5OINA spatial steps must be positive")
        steps.append(step)
    return (int(ny), int(nx)), steps, evidence


def _flat_rows(dataset, first, stop, shape):
    h, w = shape
    if dataset.shape == (h * w,):
        return np.asarray(dataset[first * w:stop * w]).reshape(stop - first, w)
    if dataset.shape == (h * w, 1):
        return np.asarray(dataset[first * w:stop * w, 0]).reshape(stop - first, w)
    if dataset.shape == (h, w):
        return np.asarray(dataset[first:stop, :])
    raise ValueError(f"H5OINA dataset {dataset.name} has inconsistent flattened raster shape {dataset.shape}; expected {shape}")


def _read_flat_map(dataset, shape, block_rows):
    if not np.issubdtype(dataset.dtype, np.number):
        raise ValueError(f"H5OINA scientific channel is not numeric: {dataset.name}")
    result = np.empty(shape, dtype=dataset.dtype)
    for first in range(0, shape[0], block_rows):
        check_cancelled()
        stop = min(shape[0], first + block_rows)
        result[first:stop] = _flat_rows(dataset, first, stop, shape)
    return result


def load_h5oina_eds(path, scan_path="/1/EBSD", *, count_rate=False, block_rows=32, registration=None):
    """Discover sibling H5OINA EDS Window Integral channels with unit evidence.

    Flattened maps are read in row blocks. Native window values and timing
    values are retained. Source 'cps' stays cps and is never divided by live time
    again. With count_rate=True, explicit counts generate separate '<name>_cps'
    channels, masked where live time is missing/nonfinite/nonpositive.

    Implicit alignment requires equal header grids/physical steps AND all EBSD
    and EDS X/Y coordinates with known length units to match in row blocks.
    Otherwise an explicit source-to-EBSD pixel affine is required.
    """
    if path is None:
        return EDSResult(metadata={"status": "unavailable", "reason": "No H5OINA source supplied"})
    if isinstance(block_rows, bool) or int(block_rows) != block_rows or int(block_rows) < 1:
        raise ValueError("EDS block_rows must be a positive integer")
    block_rows = int(block_rows)
    scan_path = "/" + str(scan_path).strip("/")
    if scan_path.rsplit("/", 1)[-1] != "EBSD":
        raise ValueError("scan_path must identify an H5OINA EBSD group")
    eds_path = scan_path.rsplit("/", 1)[0] + "/EDS"
    p = Path(path).resolve()
    with h5py.File(p, "r") as src:
        windows_path = eds_path + "/Data/Window Integral"
        if windows_path not in src or not isinstance(src[windows_path], h5py.Group):
            return EDSResult(metadata={"status": "unavailable", "source": str(p), "scan_path": scan_path,
                                      "reason": "No sibling H5OINA EDS Window Integral maps"})
        windows = src[windows_path]
        names = sorted(name for name, ds in windows.items() if isinstance(ds, h5py.Dataset))
        if not names:
            return EDSResult(metadata={"status": "unavailable", "source": str(p), "scan_path": scan_path,
                                      "reason": "H5OINA Window Integral group has no numeric datasets"})
        eds_shape, eds_steps, eds_units = _header_geometry(src[eds_path + "/Header"])
        ebsd_shape, ebsd_steps, ebsd_units = _header_geometry(src[scan_path + "/Header"])
        verification = {"method": "header plus every X/Y position in row blocks", "source_shape_hw": list(eds_shape),
                        "target_shape_hw": list(ebsd_shape), "eds_header_units": eds_units,
                        "ebsd_header_units": ebsd_units, "compared_pixels": 0, "aligned": False}
        problems = []
        if eds_shape != ebsd_shape:
            problems.append("EDS/EBSD raster dimensions differ")
        try:
            ds = np.asarray([v * _length_factor_um(eds_units[name]) for v, name in zip(eds_steps, ("X Step", "Y Step"))])
            es = np.asarray([v * _length_factor_um(ebsd_units[name]) for v, name in zip(ebsd_steps, ("X Step", "Y Step"))])
            verification.update(eds_step_um_xy=ds.tolist(), ebsd_step_um_xy=es.tolist())
            if not np.allclose(ds, es, rtol=1e-5, atol=1e-9):
                problems.append("EDS/EBSD physical step sizes differ")
        except ValueError as exc:
            problems.append(str(exc))
        eds_data, ebsd_data = src[eds_path + "/Data"], src[scan_path + "/Data"]
        max_error = {}; coordinate_units = {}
        if eds_shape == ebsd_shape:
            for axis in ("X", "Y"):
                try:
                    if axis not in eds_data or axis not in ebsd_data:
                        raise ValueError(f"Missing EDS/EBSD {axis} coordinates")
                    edu, ebu = _unit(eds_data[axis]), _unit(ebsd_data[axis])
                    coordinate_units[axis] = {"eds": edu, "ebsd": ebu}
                    edf, ebf = _length_factor_um(edu), _length_factor_um(ebu)
                    error = 0.; all_match = True; max_tolerance = 0.
                    for first in range(0, eds_shape[0], block_rows):
                        check_cancelled()
                        stop = min(eds_shape[0], first + block_rows)
                        ev = _flat_rows(eds_data[axis], first, stop, eds_shape).astype(float) * edf
                        bv = _flat_rows(ebsd_data[axis], first, stop, ebsd_shape).astype(float) * ebf
                        finite = np.isfinite(ev) & np.isfinite(bv)
                        if not finite.all():all_match = False
                        if np.any(finite):error = max(error, float(np.max(np.abs(ev[finite] - bv[finite]))))
                        # Float32 coordinate generators may differ by one ULP.
                        # Bound numerical precision explicitly, while accepting
                        # no more than 0.001 of the measured pixel spacing.
                        spacing = min(verification.get("ebsd_step_um_xy", [1e-3, 1e-3]))
                        edeps = np.finfo(eds_data[axis].dtype).eps if np.issubdtype(eds_data[axis].dtype, np.floating) else 0.
                        ebeps = np.finfo(ebsd_data[axis].dtype).eps if np.issubdtype(ebsd_data[axis].dtype, np.floating) else 0.
                        magnitude = max(float(np.max(np.abs(ev[finite]))) if np.any(finite) else 0.,
                                        float(np.max(np.abs(bv[finite]))) if np.any(finite) else 0., spacing)
                        rounding = 2 * max(edeps, ebeps) * magnitude
                        tolerance = min(spacing * 1e-3, max(spacing * 1e-5 + 1e-10, rounding))
                        max_tolerance = max(max_tolerance, tolerance)
                        if not np.allclose(ev, bv, rtol=0, atol=tolerance, equal_nan=False):all_match = False
                    max_error[axis] = error
                    verification.setdefault("coordinate_tolerance_um", {})[axis] = float(max_tolerance)
                    verification["coordinate_tolerance_policy"] = "max(1e-5 pixel step, two source-dtype precision units), capped at 0.001 pixel step"
                    if not all_match:problems.append(f"EDS/EBSD {axis} pixel positions differ")
                except ValueError as exc:
                    problems.append(str(exc))
            verification["compared_pixels"] = int(np.prod(eds_shape)) if len(max_error) == 2 else 0
        verification.update(coordinate_units=coordinate_units, max_coordinate_error_um=max_error,
                            mismatch_reasons=problems, aligned=not problems)
        if problems and registration is None:
            raise ValueError("EDS/EBSD co-registration not verified; explicit affine registration required: " + "; ".join(problems))
        stat = p.stat()
        result = EDSResult(available=True, metadata={"status": "available", "source": str(p), "scan_path": scan_path,
            "eds_path": eds_path, "source_stamp": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "format_contract": "H5OINA Header X/Y Cells define row-major raster; EDS Data/Window Integral names identify vendor X-ray windows",
            "grid_verification": verification, "window_channels": {}, "rate_channels": {},
            "quantification": "X-ray window intensities only; not elemental wt% or at%; no concentration inferred"})
        live_seconds = None
        for time_name in ("Live Time", "Real Time"):
            if time_name in eds_data:
                arr = _read_flat_map(eds_data[time_name], eds_shape, block_rows)
                unit = _unit(eds_data[time_name]); result.timing_maps[time_name] = arr
                result.timing_masks[time_name] = np.isfinite(arr) & (arr >= 0)
                result.timing_units[time_name] = unit
                if time_name == "Live Time":
                    factor = {"s": 1., "sec": 1., "second": 1., "seconds": 1., "ms": .001, "us": 1e-6}.get(_unit_key(unit))
                    if factor is not None:live_seconds = arr.astype(float) * factor
        for name in names:
            check_cancelled()
            ds = windows[name]; arr = _read_flat_map(ds, eds_shape, block_rows); unit = _unit(ds)
            result.maps[name] = arr; result.masks[name] = np.isfinite(arr); result.units[name] = unit
            evidence = {"dataset": ds.name, "unit": unit, "unit_source": "dataset attribute" if not unit.startswith("unknown") else "not available",
                        "representation": "unchanged native source values"}
            for attr in ("Atomic Number", "X-ray Line"):
                if attr in ds.attrs:
                    value = ds.attrs[attr]
                    evidence[attr] = value.decode() if isinstance(value, bytes) else value.tolist() if isinstance(value, np.ndarray) else value
            result.metadata["window_channels"][name] = evidence
            if count_rate:
                if _unit_key(unit) in ("cps", "counts/s", "count/s", "cts/s", "countss-1"):
                    result.metadata["rate_channels"][name] = {"method": "already a source count rate; no second division by live time", "unit": unit}
                elif _unit_key(unit) in ("counts", "count", "cts", "ct"):
                    if live_seconds is None:
                        raise ValueError(f"Count-rate conversion requires Live Time with explicit second-compatible unit: {name}")
                    rate_name = name + "_cps"
                    if rate_name in names:
                        raise ValueError(f"Derived count-rate name conflicts with a source window: {rate_name}")
                    valid = np.isfinite(arr) & np.isfinite(live_seconds) & (live_seconds > 0)
                    rate = np.full(eds_shape, np.nan)
                    np.divide(arr, live_seconds, out=rate, where=valid)
                    result.maps[rate_name] = rate; result.masks[rate_name] = valid; result.units[rate_name] = "counts/s"
                    result.metadata["rate_channels"][rate_name] = {"method": "source raw counts / source live time converted to seconds",
                         "source_window": name, "timing_dataset": eds_data["Live Time"].name, "timing_unit": result.timing_units["Live Time"]}
                else:
                    raise ValueError(f"Cannot infer counts or rates from source unit {unit!r}: {name}")
        if registration is None:
            result.metadata["registration"] = {"method": "verified H5OINA header and all X/Y positions", "resampled": False,
                                                   "aligned_to_ebsd": True}
        else:
            affine = validate_affine(registration)
            for name in result.maps:
                result.maps[name], result.masks[name] = resample_map(result.maps[name], ebsd_shape, affine, result.masks[name])
            result.metadata["registration"] = {"method": "explicit affine", "source_to_ebsd_pixels": affine.tolist(),
                "resampled": True, "interpolation": "linear; rates calculated on native EDS grid before resampling", "outside": "NaN + false mask"}
        result.metadata["timing_grid"] = "native EDS grid retained; timing arrays are not resampled"
        return result


@transactional_output()
def save_eds(result, outdir, *, phase_map=None, indexed=None, phase_names=None):
    """Save EDS scientific arrays/masks/provenance and optional phase intensities.

    phase_map/indexed must use the aligned EBSD grid. Phase means/sums describe
    source window intensity or counts/s; they are never chemical concentrations.
    Unavailable EDS produces only an explicit unavailable manifest.
    """
    import json
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    if not isinstance(result, EDSResult):raise TypeError("Expected EDSResult")
    manifest = {"schema_version": 1, "available": result.available, "metadata": result.metadata,
                "units": result.units, "channels": list(result.maps),
                "scientific_status": "window intensity and registration evidence; not wt% or at%", "phase_intensity_statistics": []}
    if result.available:
        if not result.maps:raise ValueError("Available EDS result has no maps")
        shape = next(iter(result.maps.values())).shape
        if any(arr.shape != shape for arr in result.maps.values()):raise ValueError("EDS map grids differ")
        if phase_map is not None:
            phases = np.asarray(phase_map)
            if phases.shape != shape:raise ValueError("Phase/EDS grids differ")
            valid_indexed = np.asarray(indexed, bool) if indexed is not None else np.isfinite(phases) & (phases > 0)
            if valid_indexed.shape != shape:raise ValueError("Indexed/EDS grids differ")
            for name, arr in result.maps.items():
                for pid in np.unique(phases[valid_indexed]):
                    use = valid_indexed & (phases == pid) & result.masks[name] & np.isfinite(arr)
                    vals = arr[use]
                    manifest["phase_intensity_statistics"].append({"channel": name, "phase_id": int(pid),
                        "phase_name": (phase_names or {}).get(int(pid), (phase_names or {}).get(str(int(pid)), f"phase {int(pid)}")),
                        "valid_pixels": int(len(vals)), "unit": result.units[name],
                        "mean_intensity": float(vals.mean()) if len(vals) else None,
                        "std_intensity": float(vals.std()) if len(vals) else None,
                        "sum_intensity": float(vals.sum()) if len(vals) else None,
                        "interpretation": "window intensity summary; not concentration"})
        with h5py.File(outdir / "eds_fields.h5", "x") as dst:
            dst.attrs["metadata_json"] = json.dumps(result.metadata, ensure_ascii=False)
            maps, masks = dst.create_group("channels"), dst.create_group("masks")
            for name, arr in result.maps.items():
                check_cancelled()
                if "/" in name or "\\" in name:raise ValueError("EDS channel labels must not contain path separators")
                if result.masks[name].shape != shape:raise ValueError(f"EDS mask shape differs: {name}")
                ds = maps.create_dataset(name, data=arr, compression="gzip"); ds.attrs["unit"] = result.units[name]
                masks.create_dataset(name, data=result.masks[name], compression="gzip")
            timing, tmask = dst.create_group("native_timing"), dst.create_group("native_timing_masks")
            for name, arr in result.timing_maps.items():
                ds = timing.create_dataset(name, data=arr, compression="gzip"); ds.attrs["unit"] = result.timing_units[name]
                ds.attrs["grid"] = "native EDS, before registration"
                tmask.create_dataset(name, data=result.timing_masks[name], compression="gzip")
        manifest["fields"] = "eds_fields.h5"
    (outdir / "eds_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
