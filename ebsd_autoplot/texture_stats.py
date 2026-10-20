"""Auditable statistics of V3 numeric texture caches and true inverse poles."""
from __future__ import annotations

import json
import os
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .v3_crystal import normalize_quaternions

ODF_ALGORITHM_ID = "so3-gaussian-haar-mrd-v2"
PF_ALGORITHM_ID = "lambert-disk-cell-area-mud-v2"


def feature_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": os.path.normcase(str(path)), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def odf_haar_weights(phi1_rad, Phi_rad, phi2_rad):
    """Exact cell masses for a Bunge Euler grid covering SO(3).

    Periodic phi axes exclude their repeated 2*pi endpoint. Phi includes both
    endpoints; each node owns the interval to its neighbours' midpoints.
    Integrating sin(Phi) analytically gives positive endpoint cell weights and
    a discrete Haar probability measure summing to one even on coarse grids.
    """
    p1, p, p2 = (np.asarray(a, float) for a in (phi1_rad, Phi_rad, phi2_rad))
    if any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0) for a in (p1, p, p2)):
        raise ValueError("ODF coordinates must be finite and strictly increasing 1D axes")
    if any(a.min() < 0 or a.max() >= 2*np.pi for a in (p1, p2)):
        raise ValueError("Periodic Euler axes must lie in [0,2*pi)")
    if not np.isclose(p[0], 0) or not np.isclose(p[-1], np.pi):
        raise ValueError("Bunge Phi grid must include 0 and pi")
    widths = []
    for angle in (p1, p2):
        forward = np.diff(np.r_[angle, angle[0] + 2*np.pi])
        widths.append((forward + np.roll(forward, 1)) / (4*np.pi))
    edges = np.r_[0., (p[1:] + p[:-1]) / 2, np.pi]
    beta_mass = (np.cos(edges[:-1]) - np.cos(edges[1:])) / 2
    return widths[0][:, None, None] * beta_mass[None, :, None] * widths[1][None, None, :]


def odf_metrics(phi1_rad, Phi_rad, phi2_rad, mrd):
    """Haar-weighted texture index and entropy of a nonnegative SO(3) grid.

    Grid nodes own analytic Haar cell masses. The grid is explicitly normalized
    for standalone metrics; source normalization is recorded independently.
    """
    p1, p, p2 = (np.asarray(a, float) for a in (phi1_rad, Phi_rad, phi2_rad))
    mrd = np.asarray(mrd, float)
    if mrd.shape != (len(p1), len(p), len(p2)) or min(len(p1), len(p), len(p2)) < 2:
        raise ValueError("ODF dimensions must match the three Euler coordinate axes")
    if not np.isfinite(mrd).all() or (mrd < 0).any():
        raise ValueError("ODF density must be finite and nonnegative")
    if any(not np.isfinite(a).all() or np.any(np.diff(a) <= 0) for a in (p1, p, p2)):
        raise ValueError("ODF coordinates must be finite and strictly increasing")
    if p.min() < 0 or p.max() > np.pi + 1e-7:
        raise ValueError("Bunge Phi axis must be within [0,pi]")
    haar = odf_haar_weights(p1, p, p2)
    mean = float(np.sum(mrd * haar))
    if mean <= 0: raise ValueError("ODF has no positive mass")
    relative = mrd / mean
    positive = relative > 0
    return {"texture_index_J": float(np.sum(relative ** 2 * haar)),
            "ODF_entropy_relative_uniform": float(-np.sum(haar[positive] * relative[positive] * np.log(relative[positive]))),
            "ODF_peak_MRD": float(relative.max()), "source_Haar_mean_MRD": mean,
            "source_normalization_status": "unit_MRD" if np.isclose(mean, 1, rtol=.01) else "explicitly_renormalized_for_metrics",
            "metric_normalization": "ODF divided by its discrete Haar-weighted mean; uniform J=1"}


def read_texture_statistics(odf_file, phase_id, *, features_file=None):
    """Return unavailable with a reason if no compatible phase grid is present."""
    missing = {"texture_status": "unavailable_no_matching_numeric_texture_cache",
               "texture_index_J": float("nan"), "ODF_peak_MRD": float("nan"),
               "PF_peak_MUD": float("nan")}
    if odf_file is None or not Path(odf_file).is_file(): return missing
    with h5py.File(odf_file, "r") as f:
        if int(f.attrs.get("schema_version", 0)) != 1:
            return dict(missing, texture_status="unavailable_unsupported_texture_schema")
        if f.attrs.get("odf_algorithm_id") != ODF_ALGORITHM_ID:
            return dict(missing, texture_status="unavailable_legacy_ODF_normalization_regenerate_from_cache")
        if features_file is not None:
            if f.attrs.get("features_signature_json") != json.dumps(feature_signature(features_file), sort_keys=True):
                return dict(missing, texture_status="unavailable_texture_source_mismatch")
        if not f.attrs.get("configuration_hash"):
            return dict(missing, texture_status="unavailable_texture_configuration_unknown")
        groups = [g for g in f.values() if isinstance(g, h5py.Group) and int(g.attrs.get("phase_id", -1)) == int(phase_id)]
        if len(groups) != 1: return missing
        if "MRD" not in groups[0]:
            return dict(missing, texture_status=groups[0].attrs.get("status", missing["texture_status"]))
        group = groups[0]
        try:
            metrics = odf_metrics(*(group[k][...] for k in ("phi1_rad", "Phi_rad", "phi2_rad", "MRD")))
        except (ValueError, KeyError) as exc:
            return dict(missing, texture_status="unavailable_invalid_texture_grid", texture_reason=str(exc))
        if metrics["source_normalization_status"] != "unit_MRD":
            return dict(missing, texture_status="unavailable_invalid_MRD_normalization",
                        source_Haar_mean_MRD=metrics["source_Haar_mean_MRD"])
        peaks = [float(np.nanmax(g["MUD"][...])) for name, g in group.items()
                 if name.startswith("PF_") and isinstance(g, h5py.Group) and "MUD" in g
                 and g.attrs.get("pf_algorithm_id") == PF_ALGORITHM_ID and "bin_disk_area" in g
                 and np.isfinite(g["MUD"][...]).any()]
        return {**metrics, "PF_peak_MUD": max(peaks) if peaks else float("nan"),
                "PF_status": "available_disk_area_normalized" if peaks else "unavailable_legacy_or_missing_disk_area_PF",
                "texture_status": "available_exploratory_V3_numeric_grid",
                "texture_algorithm_id": ODF_ALGORITHM_ID,
                "texture_configuration_hash": f.attrs["configuration_hash"],
                "texture_source_file": str(Path(odf_file).resolve()),
                "texture_parameters_json": group.attrs.get("ODF_parameters_json", "{}")}


def inverse_pole_vectors(quaternions, space_group, sample_direction=(0, 0, 1)):
    """Use q^-1 d and orix's Laue fundamental sector; this is an IPF, not PF."""
    from orix.vector import Vector3d
    from orix.quaternion import symmetry
    laue = {194: symmetry.D6h, 229: symmetry.Oh}.get(int(space_group))
    if laue is None: raise ValueError("IPF statistics require validated HCP(194) or BCC(229) symmetry")
    direction = np.asarray(sample_direction, float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-12:
        raise ValueError("Sample direction must be a finite nonzero 3-vector")
    q = normalize_quaternions(quaternions)
    crystal = Rotation.from_quat(q).inv().apply(direction / np.linalg.norm(direction))
    return Vector3d(crystal).in_fundamental_sector(laue).data


def inverse_pole_statistics(quaternions, space_group, sample_direction=(0, 0, 1)):
    vectors = inverse_pole_vectors(quaternions, space_group, sample_direction)
    # Second moment is sign-invariant; sector projection is recorded and the
    # vectors can be exported for independent MTEX verification.
    moment = vectors.T @ vectors / len(vectors)
    return vectors, {"IPF_orientation_count": len(vectors),
        "IPF_type": "sample direction projected into crystal Laue fundamental sector using orix",
        "IPF_sample_direction": json.dumps(list(map(float, sample_direction))),
        "IPF_second_moment_eigenvalues": json.dumps(np.linalg.eigvalsh(moment).tolist()),
        "IPF_status": "available_exploratory_requires_MTex_comparison"}
