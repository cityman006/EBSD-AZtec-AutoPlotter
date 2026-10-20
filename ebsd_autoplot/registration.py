"""Explicit pixel-centre affine registration; never assumes equal grid = registered.

Coordinates are (x, y), integer pixel centres. Affines map source -> target.
These routines do not estimate image registration from unrelated contrasts.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates


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
    yy, xx = np.indices(tuple(map(int, target_shape)), dtype=float)
    coords = transform_points(np.stack((xx, yy), -1), np.linalg.inv(validate_affine(affine)))
    # Round tiny inverse-matrix noise at exact boundaries, not real offsets.
    coords[np.abs(coords - np.rint(coords)) < 1e-10] = np.rint(coords[np.abs(coords - np.rint(coords)) < 1e-10])
    ij = np.stack((coords[..., 1], coords[..., 0]))
    support = map_coordinates(mask.astype(float), ij, order=order, mode="constant", cval=0, prefilter=False)
    valid = support >= 1 - 1e-10
    safe = np.where(mask[..., None], arr, 0) if arr.ndim == 3 else np.where(mask, arr, 0)
    if arr.ndim == 2:
        values = map_coordinates(safe.astype(float), ij, order=order, mode="constant", cval=0, prefilter=False)
    else:
        values = np.stack([map_coordinates(safe[..., c].astype(float), ij, order=order,
                                          mode="constant", cval=0, prefilter=False)
                           for c in range(arr.shape[-1])], -1)
    values[~valid] = np.nan
    return values, valid
