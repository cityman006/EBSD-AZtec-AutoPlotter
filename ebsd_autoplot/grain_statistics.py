"""Physical, phase-resolved statistics of an existing V2 label raster.

The input segmentation is never changed. Pixel-edge perimeter and second-moment
ellipse axes are raster descriptors, not a calibrated intercept grain size.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from .storage import check_cancelled


def weighted_quantile(values, quantile, weights=None):
    """Inverse empirical CDF (including the lower median at an exact boundary)."""
    values = np.asarray(values, float)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not good.any():
        return float("nan")
    order = np.argsort(values[good], kind="stable")
    v, w = values[good][order], weights[good][order]
    index = np.searchsorted(np.cumsum(w), float(quantile) * w.sum(), side="left")
    return float(v[min(index, len(v) - 1)])


def grain_statistics(labels, phase, indexed, dx_um, dy_um, *, kam=None,
                     kam_valid=None, grod=None, min_pixels=8, exclude_edge=False):
    """Return one row per positive label, including explicit exclusion reasons."""
    labels = np.asarray(labels)
    phase = np.asarray(phase)
    indexed = np.asarray(indexed, bool)
    if labels.ndim != 2 or labels.shape != phase.shape or phase.shape != indexed.shape:
        raise ValueError("Grain, phase and indexing maps must share a 2D ROI")
    dx, dy = float(dx_um), float(dy_um)
    if not np.isfinite([dx, dy]).all() or min(dx, dy) <= 0:
        raise ValueError("Verified positive x/y steps in micrometres are required")
    if int(min_pixels) < 1:
        raise ValueError("Minimum grain size must be positive")
    for array in (kam, kam_valid, grod):
        if array is not None and np.shape(array) != labels.shape:
            raise ValueError("Quality maps must have the same shape as grain labels")
    # Count the physical length of each exposed pixel face, including the ROI
    # boundary. Label differences are evaluated separately from area filtering.
    perimeter = np.zeros_like(labels, float)
    perimeter[:, 0] += dy; perimeter[:, -1] += dy
    perimeter[0, :] += dx; perimeter[-1, :] += dx
    different = labels[:, 1:] != labels[:, :-1]
    perimeter[:, 1:] += different * dy; perimeter[:, :-1] += different * dy
    different = labels[1:, :] != labels[:-1, :]
    perimeter[1:, :] += different * dx; perimeter[:-1, :] += different * dx
    order = np.argsort(labels.ravel(), kind="stable")
    sorted_ids = labels.ravel()[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1, len(order)]
    rows = []
    for start, stop in zip(starts[:-1], starts[1:]):
        gid = int(sorted_ids[start])
        if gid <= 0:
            continue
        check_cancelled()
        loc = order[start:stop]
        yy, xx = np.unravel_index(loc, labels.shape)
        ph, counts = np.unique(phase.ravel()[loc], return_counts=True)
        pid = int(ph[np.argmax(counts)])
        mixed = len(ph) != 1 or pid <= 0
        edge = bool(np.any((xx == 0) | (xx == labels.shape[1] - 1) |
                           (yy == 0) | (yy == labels.shape[0] - 1)))
        count = len(loc)
        indexed_count = int(np.count_nonzero(indexed.ravel()[loc]))
        reasons = []
        if mixed: reasons.append("mixed_or_unindexed_phase_label")
        if count < int(min_pixels): reasons.append("below_minimum_pixels")
        if indexed_count != count: reasons.append("contains_unindexed_pixels")
        if exclude_edge and edge: reasons.append("edge_truncated")
        # Match source scan positions and graph/slicing coordinates: index 0
        # is the centre of the first pixel, not the outer pixel-cell edge.
        physical = np.column_stack((xx * dx, yy * dy))
        centered = physical - physical.mean(axis=0)
        covariance = centered.T @ centered / count + np.diag([dx * dx, dy * dy]) / 12
        eigen = np.linalg.eigvalsh(covariance).clip(min=0)
        minor, major = 4 * np.sqrt(eigen)
        area = count * dx * dy
        length = float(perimeter.ravel()[loc].sum())
        ka = np.array([], float)
        if kam is not None:
            ka = np.asarray(kam).ravel()[loc]
            kv = np.isfinite(ka)
            if kam_valid is not None: kv &= np.asarray(kam_valid, bool).ravel()[loc]
            ka = ka[kv]
        ga = np.array([], float) if grod is None else np.asarray(grod).ravel()[loc]
        ga = ga[np.isfinite(ga)]
        rows.append({"grain_id": gid, "phase_id": pid, "pixels": count,
                     "indexed_pixels": indexed_count, "indexed_fraction": indexed_count / count,
                     "area_um2": area, "ecd_um": 2 * np.sqrt(area / np.pi),
                     "perimeter_um": length, "major_axis_um": float(major),
                     "minor_axis_um": float(minor), "aspect_ratio": float(major / minor),
                     "circularity": float(4 * np.pi * area / length ** 2),
                     "centroid_x_um": float(physical[:, 0].mean()),
                     "centroid_y_um": float(physical[:, 1].mean()),
                     "edge_truncated": edge, "qc_included": not reasons,
                     "qc_reason": ";".join(reasons) or "included",
                     "KAM_mean_deg": float(ka.mean()) if len(ka) else float("nan"),
                     "KAM_valid_count": len(ka),
                     "GOS_deg": float(ga.mean()) if len(ga) == count else float("nan"),
                     "GROD_p95_deg": float(np.percentile(ga, 95)) if len(ga) == count else float("nan"),
                     "GROD_valid_count": len(ga)})
    return rows


def summarize_grain_sizes(rows):
    included = [r for r in rows if r["qc_included"]]
    ecd = np.array([r["ecd_um"] for r in included], float)
    area = np.array([r["area_um2"] for r in included], float)
    return {"grain_count": len(included), "excluded_grain_count": len(rows) - len(included),
            "edge_truncated_grain_count": sum(bool(r["edge_truncated"]) for r in rows),
            "ECD_number_mean_um": float(ecd.mean()) if len(ecd) else float("nan"),
            "ECD_number_median_um": float(np.median(ecd)) if len(ecd) else float("nan"),
            "ECD_number_P10_um": float(np.percentile(ecd, 10)) if len(ecd) else float("nan"),
            "ECD_number_P90_um": float(np.percentile(ecd, 90)) if len(ecd) else float("nan"),
            "ECD_number_variance_um2": float(np.var(ecd)) if len(ecd) else float("nan"),
            "ECD_area_weighted_mean_um": float(np.average(ecd, weights=area)) if len(ecd) else float("nan"),
            "ECD_area_weighted_median_um": weighted_quantile(ecd, .5, area),
            "ECD_area_weighted_P10_um": weighted_quantile(ecd, .1, area),
            "ECD_area_weighted_P90_um": weighted_quantile(ecd, .9, area),
            "ECD_area_weighted_variance_um2": float(np.average((ecd-np.average(ecd, weights=area))**2, weights=area)) if len(ecd) else float("nan"),
            "grain_size_status": "available_raster_descriptor" if len(ecd) else "unavailable_no_QC_grains"}


def grain_size_cdf(rows):
    """Empirical number/area CDF, grouping tied ECD values at the same step."""
    selected = [r for r in rows if r["qc_included"]]
    if not selected: return []
    ecd = np.array([r["ecd_um"] for r in selected])
    area = np.array([r["area_um2"] for r in selected])
    values, inverse, counts = np.unique(ecd, return_inverse=True, return_counts=True)
    area_at_value = np.bincount(inverse, weights=area)
    return [{"ecd_um": float(v), "grain_count_at_ECD": int(n), "area_um2_at_ECD": float(a),
             "number_CDF": float(cn), "area_CDF": float(ca)}
            for v, n, a, cn, ca in zip(values, counts, area_at_value,
                np.cumsum(counts)/len(ecd), np.cumsum(area_at_value)/area.sum())]


def attach_v2_grain_metadata(rows, v2_dir):
    """Preserve V2's symmetry-aware mean orientation and graph neighbour count.

    Missing mean/graph data remain unavailable; a missing graph is never
    represented as a measured zero-neighbour grain.
    """
    v2_dir = Path(v2_dir)
    means = {}
    if (v2_dir / "grains.csv").is_file():
        with (v2_dir / "grains.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream): means[int(row["grain_id"])] = row
    neighbours = {r["grain_id"]: set() for r in rows}
    graph_available = (v2_dir / "grain_neighbour_graph.csv").is_file()
    if graph_available:
        with (v2_dir / "grain_neighbour_graph.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                a, b = int(row["grain_a"]), int(row["grain_b"])
                neighbours.setdefault(a, set()).add(b); neighbours.setdefault(b, set()).add(a)
    for row in rows:
        previous = means.get(row["grain_id"], {})
        same_phase = int(previous.get("phase_id", -1)) == row["phase_id"]
        values = np.array([float(previous.get("mean_q"+axis, "nan")) for axis in ("x", "y", "z", "w")])
        valid = same_phase and np.isfinite(values).all() and np.isclose(np.linalg.norm(values), 1, atol=1e-4)
        for axis, value in zip(("x", "y", "z", "w"), values): row["mean_q"+axis] = float(value) if valid else float("nan")
        row["mean_orientation_status"] = "available_V2_symmetry_aligned_xyzw" if valid else "unavailable_no_matching_V2_mean"
        row["neighbour_count"] = len(neighbours[row["grain_id"]]) if graph_available else float("nan")
        row["neighbour_count_status"] = "available_V2_graph" if graph_available else "unavailable_no_V2_graph"
    return rows
