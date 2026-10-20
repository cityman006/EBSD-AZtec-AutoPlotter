"""Explicit pixel-centre affine registration; never assumes equal grid = registered.

Coordinates are (x, y), integer pixel centres. Affines map source -> target.
These routines do not estimate image registration from unrelated contrasts.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates


def file_stamp(path):
    """Read-only source identity used by landmark projects; not a content hash."""
    from pathlib import Path
    path = Path(path).resolve()
    stat = path.stat()
    if not path.is_file():
        raise ValueError("Registration source must be a file")
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _shape_hw(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Registration shape must be positive integer H,W")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1 for v in value):
        raise ValueError("Registration shape must be positive integer H,W")
    return tuple(map(int, value))


def validate_landmark_config(config, cache_path=None, source_shape=None):
    """Validate a frozen source-bound project and recompute landmark residuals.

    A fit residual is never independent validation. The consumer must compare
    source_shape with the actual native EDS grid it loaded before resampling.
    Target cache dimensions and both file stamps are checked read-only here.
    """
    import copy
    import os
    import h5py
    from .storage import canonical_path
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported landmark registration schema")
    result = copy.deepcopy(config)
    def same_path(left, right):
        return os.path.normcase(str(canonical_path(left))) == os.path.normcase(str(canonical_path(right)))
    def checked_stamp(record, role):
        if not isinstance(record, dict) or not {"path", "size", "mtime_ns"} <= record.keys():
            raise ValueError(f"Missing {role} registration file stamp")
        try:
            actual = file_stamp(record["path"])
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"Registration {role} file changed or is unavailable") from exc
        if not same_path(actual["path"], record["path"]) or any(actual[k] != record[k] for k in ("size", "mtime_ns")):
            raise ValueError(f"Registration {role} file changed; fit fresh landmarks")
        return actual
    source = checked_stamp(result.get("source_stamp"), "source")
    target = checked_stamp(result.get("target_cache_stamp"), "target cache")
    if not result.get("source_path") or not same_path(result["source_path"], source["path"]):
        raise ValueError("Registration source path differs from its file stamp")
    if cache_path is not None and not same_path(cache_path, target["path"]):
        raise ValueError("Registration belongs to a different target cache")
    if not isinstance(result.get("scan_path"), str) or not result["scan_path"].startswith("/"):
        raise ValueError("Registration requires an explicit source scan path")
    source_hw = _shape_hw(result.get("source_shape_hw"))
    target_hw = _shape_hw(result.get("target_shape_hw"))
    if source_shape is not None and tuple(source_shape) != source_hw:
        raise ValueError("Native EDS shape differs from landmark project shape")
    with h5py.File(target["path"], "r") as cache:
        if "indexed" not in cache or cache["indexed"].shape != target_hw:
            raise ValueError("Target cache shape differs from landmark project shape")
    try:
        source_xy = np.asarray(result.get("source_xy"), float)
        target_xy = np.asarray(result.get("target_xy"), float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Landmark coordinates must be finite paired x,y points") from exc
    for points, shape in ((source_xy, source_hw), (target_xy, target_hw)):
        if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
            raise ValueError("Landmark coordinates must be finite paired x,y points")
        if np.any(points < 0) or np.any(points[:, 0] > shape[1]-1) or np.any(points[:, 1] > shape[0]-1):
            raise ValueError("Landmark coordinates are outside their native image bounds")
    fitted, qc = fit_affine(source_xy, target_xy)
    supplied = validate_affine(result.get("matrix"))
    if not np.allclose(supplied, fitted, rtol=1e-8, atol=1e-7):
        raise ValueError("Registration matrix differs from the fitted landmarks")
    result.update(source_path=source["path"], source_stamp=source, target_cache_stamp=target,
                  source_shape_hw=list(source_hw), target_shape_hw=list(target_hw),
                  source_xy=source_xy.tolist(), target_xy=target_xy.tolist(), matrix=fitted.tolist(),
                  fit_qc=qc, coordinate_convention="source EDS to target EBSD; x,y pixel centres")
    return result


def validate_affine(affine):
    matrix = np.asarray(affine, dtype=float)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, [0., 0., 1.]))
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Affine must be a finite 3x3 (or 2x3) source-to-target matrix")
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("Perspective transforms are not supported")
    if abs(np.linalg.det(matrix[:2, :2])) < 1e-12:
        raise ValueError("Affine is singular")
    return matrix


def transform_points(xy, affine):
    points = np.asarray(xy, dtype=float)
    if points.shape[-1:] != (2,) or not np.isfinite(points).all():
        raise ValueError("Coordinates must be finite (...,2) x,y points")
    matrix = validate_affine(affine)
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def fit_affine(source_xy, target_xy):
    """Fit explicit landmarks, returning matrix and residual QC (target pixels).

    Three noncollinear pairs determine an affine but provide no independent
    residual validation. Four or more pairs are strongly recommended.
    """
    src, dst = np.asarray(source_xy, float), np.asarray(target_xy, float)
    if src.ndim != 2 or src.shape[1] != 2 or src.shape != dst.shape or len(src) < 3:
        raise ValueError("At least three paired x,y landmarks are required")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Landmarks must be finite")
    design = np.column_stack((src, np.ones(len(src))))
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("Source landmarks are collinear")
    coef, _, _, _ = np.linalg.lstsq(design, dst, rcond=None)
    affine = validate_affine(np.vstack((coef.T, [0., 0., 1.])))
    residual = np.linalg.norm(transform_points(src, affine) - dst, axis=1)
    return affine, {"landmarks": len(src), "residuals_target_px": residual.tolist(),
                    "rms_target_px": float(np.sqrt(np.mean(residual**2))),
                    "max_target_px": float(residual.max()),
                    "independently_validated": False,
                    "warning": "Training-landmark residual; external registration validation required"}


def resample_map(data, target_shape, affine, valid_mask=None, order=1):
    """Explicit inverse-mapped scalar/RGB sampling; NaN outside valid support.

    Returns (values, validity_mask). Linear output is valid only when all
    contributing source pixels are valid, preventing missing data dilution.
    Use order=0 for categorical maps. No extrapolation or image resizing.
    """
    arr = np.asarray(data)
    if arr.ndim not in (2, 3) or len(target_shape) != 2 or min(target_shape) < 1:
        raise ValueError("Expected H,W[,C] data and positive target H,W")
    if order not in (0, 1):
        raise ValueError("Only nearest (0) and linear (1) interpolation are supported")
    mask = np.isfinite(arr).all(axis=-1) if arr.ndim == 3 else np.isfinite(arr)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, bool)
        if valid_mask.shape != arr.shape[:2]:
            raise ValueError("Validity mask shape differs from source")
        mask &= valid_mask
    safe = np.where(mask[..., None], arr, 0) if arr.ndim == 3 else np.where(mask, arr, 0)
    return _sample_prepared(safe, mask, target_shape, affine, order)


def _sample_prepared(safe, mask, target_shape, affine, order):
    # The full source validity/safe arrays are prepared once per map, never once
    # per target block. scipy writes floating interpolants from native dtypes.
    yy, xx = np.indices(tuple(map(int, target_shape)), dtype=float)
    coords = transform_points(np.stack((xx, yy), -1), np.linalg.inv(validate_affine(affine)))
    near = np.abs(coords - np.rint(coords)) < 1e-10
    coords[near] = np.rint(coords[near])
    ij = np.stack((coords[..., 1], coords[..., 0]))
    support = map_coordinates(mask, ij, output=np.float64, order=order, mode="constant", cval=0, prefilter=False)
    valid = support >= 1 - 1e-10
    if safe.ndim == 2:
        values = map_coordinates(safe, ij, output=np.float64, order=order, mode="constant", cval=0, prefilter=False)
    else:
        values = np.stack([map_coordinates(safe[..., c], ij, output=np.float64, order=order,
                                          mode="constant", cval=0, prefilter=False)
                           for c in range(safe.shape[-1])], -1)
    values[~valid] = np.nan
    return values, valid


def resample_map_blocked(data, target_shape, affine, valid_mask=None, order=1,
                         block_rows=32, out=None, valid_out=None):
    """Same pixel-centre inverse map, with bounded target coordinate buffers.

    Optional output arrays may be caller-owned disk memmaps. Linear interpolation
    is explicit; outside/invalid support remains NaN plus a false mask.
    """
    from .storage import check_cancelled
    shape = _shape_hw(target_shape)
    if isinstance(block_rows, bool) or int(block_rows) != block_rows or block_rows < 1:
        raise ValueError("Registration block_rows must be a positive integer")
    matrix = validate_affine(affine)
    arr = np.asarray(data)
    if arr.ndim not in (2, 3):
        raise ValueError("Expected H,W[,C] source data")
    if order not in (0, 1):
        raise ValueError("Only nearest (0) and linear (1) interpolation are supported")
    mask = np.isfinite(arr).all(axis=-1) if arr.ndim == 3 else np.isfinite(arr)
    if valid_mask is not None:
        if np.shape(valid_mask) != arr.shape[:2]:
            raise ValueError("Validity mask shape differs from source")
        mask &= np.asarray(valid_mask, bool)
    check_cancelled()
    safe = np.where(mask[..., None], arr, 0) if arr.ndim == 3 else np.where(mask, arr, 0)
    output_shape = (*shape, arr.shape[2]) if arr.ndim == 3 else shape
    if out is None: out = np.empty(output_shape, dtype=float)
    if valid_out is None: valid_out = np.empty(shape, dtype=bool)
    if out.shape != output_shape or not np.issubdtype(out.dtype, np.floating) or valid_out.shape != shape or valid_out.dtype != bool:
        raise ValueError("Registration output shape or dtype differs from target")
    for first in range(0, shape[0], int(block_rows)):
        check_cancelled()
        stop = min(shape[0], first + int(block_rows))
        local = matrix.copy(); local[1, 2] -= first
        out[first:stop], valid_out[first:stop] = _sample_prepared(safe, mask, (stop-first, shape[1]), local, order)
    check_cancelled()
    return out, valid_out
