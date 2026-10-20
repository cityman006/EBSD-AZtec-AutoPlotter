"""Quantitative PF and symmetry-aware SO(3) kernel ODF from immutable V1 caches.

ODF values are MRD (orientation density divided by the Haar-uniform density).
The numeric grid uses Bunge ZXZ Euler coordinates and Haar weights
sin(Phi) d(phi1) d(Phi) d(phi2) / (8*pi**2). Figures are renderings of these
values, not substitutes for the numeric density.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.spatial.transform import Rotation

from .storage import check_cancelled, transactional_output
from .v3_crystal import ORIENTATION_CONVENTION, classify_phase_specs, point_group, normalize_quaternions
from .texture_stats import ODF_ALGORITHM_ID, PF_ALGORITHM_ID, feature_signature, odf_haar_weights


HCP_FAMILIES = {
    "0001": ((0, 0, 0, 1),),
    "10-10": ((1, 0, -1, 0),),
    "11-20": ((1, 1, -2, 0),),
}
BCC_FAMILIES = {
    "100": ((1, 0, 0),),
    "110": ((1, 1, 0),),
    "111": ((1, 1, 1),),
}


def _phase_metadata(meta):
    specs = {int(k): v for k, v in meta.get("phase_specs", {}).items()}
    kinds = classify_phase_specs(specs)
    return specs, kinds


def _reciprocal_hcp_normal(plane, c_over_a):
    h, k, i, l = plane
    if h + k + i != 0:
        raise ValueError("HCP plane must satisfy h+k+i=0")
    n = np.array([h, (h + 2 * k) / np.sqrt(3), l / c_over_a], float)
    return n / np.linalg.norm(n)


def _pole_family(space_group, family, c_over_a):
    if int(space_group) == 194:
        if family not in HCP_FAMILIES:
            raise ValueError(f"Unsupported HCP pole family {family!r}")
        base = _reciprocal_hcp_normal(HCP_FAMILIES[family][0], c_over_a)
    elif int(space_group) == 229:
        if family not in BCC_FAMILIES:
            raise ValueError(f"Unsupported BCC pole family {family!r}")
        base = np.asarray(BCC_FAMILIES[family][0], float)
        base /= np.linalg.norm(base)
    else:
        raise ValueError(f"PF symmetry not supported for space group {space_group}")
    vectors = point_group(space_group).apply(np.broadcast_to(base, (len(point_group(space_group)), 3)))
    unique = []
    for v in vectors:
        v = v / np.linalg.norm(v)
        if not any(abs(float(v @ old)) > 1 - 1e-7 for old in unique):
            unique.append(v)
    return np.asarray(unique)


def _sample_pixels(indexed, feature_ds, phase_id, max_samples, seed):
    phase_ds = None
    h, w = indexed.shape
    total = h * w
    count = min(total, max(1, int(max_samples)))
    rng = np.random.default_rng(seed)
    # One uniformly selected location per equal-area raster stratum. Random
    # probes are row-grouped so no full-resolution map is materialized.
    positions = np.floor((np.arange(count) + rng.random(count)) * total / count).astype(np.int64)
    positions = np.clip(positions, 0, total - 1)
    order = np.argsort(positions // w, kind="stable")
    positions = positions[order]
    rows, cols = positions // w, positions % w
    selected = []
    at = 0
    for row in np.unique(rows):
        check_cancelled()
        take = np.flatnonzero(rows == row)
        phase_row = feature_ds[4, int(row), :]
        valid_row = indexed[int(row), :]
        good = (phase_row[cols[take]] == phase_id) & (valid_row[cols[take]] > 0)
        selected.extend(positions[take[good]].tolist())
        at += len(take)
    if not selected:
        return np.empty((0, 4), dtype=np.float64), np.empty(0, dtype=np.float64)
    selected = np.asarray(selected, dtype=np.int64)
    rows, cols = selected // w, selected % w
    qs = np.empty((len(selected), 4), dtype=np.float64)
    for row in np.unique(rows):
        ix = np.flatnonzero(rows == row)
        q_row = feature_ds[0:4, int(row), :]
        qs[ix] = np.moveaxis(q_row[:, cols[ix]], 0, -1)
    return normalize_quaternions(qs), np.ones(len(qs), dtype=np.float64)


def _sample_grains(grains_csv, phase_id, max_samples, seed):
    if grains_csv is None:
        raise ValueError("grain-weighted ODF needs the V2 grains.csv cache")
    rows = []
    with Path(grains_csv).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["phase_id"]) == phase_id:
                rows.append(row)
    if len(rows) > max_samples:
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in np.sort(rng.choice(len(rows), int(max_samples), replace=False))]
    if not rows:
        return np.empty((0, 4)), np.empty(0)
    q = np.array([[float(r[f"mean_q{x}"]) for x in ("x", "y", "z", "w")] for r in rows])
    return normalize_quaternions(q), np.ones(len(q), dtype=np.float64)


def _odf_grid(samples, weights, space_group, bandwidth_deg, grid_step_deg, seed):
    if bandwidth_deg <= 0 or grid_step_deg <= 0:
        raise ValueError("ODF bandwidth and grid step must be positive degrees")
    if len(samples) < 50:
        raise ValueError(f"Only {len(samples)} orientations; at least 50 are required for quantitative ODF")
    n = max(4, int(np.ceil(360 / grid_step_deg)))
    n_beta = max(3, int(np.ceil(180 / grid_step_deg)) + 1)
    phi1 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    Phi = np.linspace(0, np.pi, n_beta)
    phi2 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    grid_euler = np.stack(np.meshgrid(phi1, Phi, phi2, indexing="ij"), axis=-1).reshape(-1, 3)
    grid_q = Rotation.from_euler("ZXZ", grid_euler, degrees=False).as_quat()
    sym = point_group(space_group)
    left_q = np.repeat(samples[:, None, :], len(sym), axis=1).reshape(-1, 4)
    right_q = np.tile(sym.as_quat(), (len(samples), 1))
    equiv_q = (Rotation.from_quat(left_q) * Rotation.from_quat(right_q)).as_quat()
    expanded_weights = np.repeat(weights / len(sym), len(sym))
    sigma = np.deg2rad(float(bandwidth_deg))
    density = np.empty(len(grid_q), dtype=np.float64)
    # A Gaussian in the SO(3) geodesic angle is normalized numerically against
    # Haar measure below, so reported MRD integrates to one random multiple.
    for start in range(0, len(grid_q), 128):
        check_cancelled()
        dot = np.clip(np.abs(grid_q[start:start + 128] @ equiv_q.T), 0.0, 1.0)
        angle = 2 * np.arccos(dot)
        density[start:start + 128] = np.exp(-0.5 * (angle / sigma) ** 2) @ expanded_weights
    density = density.reshape(len(phi1), len(Phi), len(phi2))
    haar = odf_haar_weights(phi1, Phi, phi2)
    norm = float(np.sum(density * haar))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("ODF normalization failed")
    # haar is already a probability measure: uniform MRD=1. Multiplying by
    # 8*pi**2 again incorrectly scaled all old MRD values by that factor.
    mrd = density / norm
    integral = float(np.sum(mrd * haar))
    return phi1, Phi, phi2, mrd.astype(np.float32), {
        "density_measure": "SO(3) Haar measure",
        "algorithm_id": ODF_ALGORITHM_ID,
        "normalization": "integral(MRD dmu_Haar_probability)=1; uniform MRD=1",
        "quadrature": "periodic phi cells; exact cos(Phi) interval masses",
        "integral_check": integral,
        "kernel": "symmetry-summed Gaussian of SO(3) geodesic angle; numerically Haar-normalized",
        "bandwidth_deg": float(bandwidth_deg),
        "grid_step_deg": float(grid_step_deg),
        "seed": int(seed),
    }


def _pf_bin_areas(edges, radius=np.sqrt(2)):
    """Analytic rectangle/disk intersection areas, including partial rim bins."""
    edges = np.asarray(edges, float)
    r = float(radius)
    if r <= 0 or edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("PF disk bins require positive radius and increasing edges")
    xx, yy = np.meshgrid(edges, edges, indexing="ij")
    x, y = np.clip(np.abs(xx), 0, r), np.clip(np.abs(yy), 0, r)
    a = np.sqrt(np.maximum(r*r-y*y, 0))
    def integral(t):
        return .5*(t*np.sqrt(np.maximum(r*r-t*t, 0)) + r*r*np.arcsin(np.clip(t/r, 0, 1)))
    primitive = np.where(x <= a, x*y, y*a+integral(x)-integral(a))
    primitive *= np.sign(xx)*np.sign(yy)
    areas = primitive[1:,1:] - primitive[:-1,1:] - primitive[1:,:-1] + primitive[:-1,:-1]
    return np.maximum(areas, 0)


def _plot_pf(path, poles, *, title, bins=180, formats=("png",)):
    poles = np.asarray(poles, float).copy()
    if poles.ndim != 2 or poles.shape[1] != 3 or not len(poles) or not np.isfinite(poles).all():
        raise ValueError("PF requires finite nonempty unit pole vectors")
    norms = np.linalg.norm(poles, axis=1)
    if np.any(norms <= 0): raise ValueError("Zero pole vector is invalid")
    poles /= norms[:, None]
    poles[poles[:, 2] < 0] *= -1  # antipodal upper-hemisphere convention
    denom = np.sqrt(np.maximum(1 + poles[:, 2], 1e-15))
    # Equal-area Lambert azimuthal projection, disk radius sqrt(2).
    x = np.sqrt(2) * poles[:, 0] / denom
    y = np.sqrt(2) * poles[:, 1] / denom
    edges = np.linspace(-np.sqrt(2), np.sqrt(2), bins + 1)
    counts, _, _ = np.histogram2d(np.clip(x, edges[0], edges[-1]),
                                 np.clip(y, edges[0], edges[-1]), bins=(edges, edges))
    areas = _pf_bin_areas(edges)
    inside = areas > np.finfo(float).eps
    mud = np.full_like(counts, np.nan, dtype=np.float64)
    total = counts.sum()
    if total != len(poles) or np.any(counts[~inside]):
        raise ValueError("PF disk histogram failed pole-count conservation")
    mud[inside] = counts[inside]/total * (2*np.pi)/areas[inside]
    vmin = max(0.0, float(np.nanpercentile(mud, 2)))
    vmax = max(vmin + 1e-6, float(np.nanpercentile(mud, 98)))
    fig, ax = plt.subplots(figsize=(5.2, 4.8), layout="constrained")
    im = ax.pcolormesh(edges, edges, mud.T, shading="auto", cmap="viridis", norm=Normalize(vmin, vmax))
    ax.add_patch(plt.Circle((0, 0), np.sqrt(2), fill=False, color="black", lw=0.8))
    ax.set_aspect("equal"); ax.set_xlim(-1.48, 1.48); ax.set_ylim(-1.48, 1.48)
    ax.set_xlabel("Sample X (Lambert equal-area)"); ax.set_ylabel("Sample Y")
    ax.set_title(title + "\nAntipodal upper hemisphere; PF, not IPF")
    fig.colorbar(im, ax=ax, label="MUD (uniform hemisphere density = 1)")
    for fmt in formats: fig.savefig(path.with_suffix("." + fmt), dpi=220)
    plt.close(fig)
    return mud


@transactional_output()
def analyze_texture(features_file, outdir, *, families=None, weighting="area",
                    bandwidth_deg=10.0, grid_step_deg=15.0, max_samples=2000,
                    seed=11, sample_rotation_euler_deg=(0.0, 0.0, 0.0),
                    formats=("png",), min_phase_samples=50):
    """Write PF MUD, normalized SO(3) ODF grid, plots, and auditable parameters."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    formats = tuple(dict.fromkeys(str(x).lower().lstrip(".") for x in formats))
    if not formats or set(formats) - {"png", "pdf", "svg", "tiff"}:
        raise ValueError("V3 texture formats must be PNG, PDF, SVG, or TIFF")
    if int(min_phase_samples) < 50:
        raise ValueError("Quantitative texture requires at least 50 samples per phase")
    summaries = []
    sample_rotation = Rotation.from_euler("ZXZ", np.deg2rad(sample_rotation_euler_deg))
    with h5py.File(features_file, "r") as f:
        meta = json.loads(f.attrs["metadata_json"])
        specs, kinds = _phase_metadata(meta)
        if "features" not in f or "indexed" not in f:
            raise ValueError("A compatible V1 feature cache is required")
        if meta.get("angular_units_input") not in ("radian", "degree"):
            raise ValueError("V1 cache does not declare verified Euler input units")
        h, w = f["indexed"].shape
        with h5py.File(outdir / "odf_grid.h5", "w") as odf:
            odf.attrs["schema_version"] = 1
            odf.attrs["odf_algorithm_id"] = ODF_ALGORITHM_ID
            odf.attrs["pf_algorithm_id"] = PF_ALGORITHM_ID
            odf.attrs["features_signature_json"] = json.dumps(feature_signature(features_file), sort_keys=True)
            texture_configuration = {"families": families, "weighting": weighting,
                "bandwidth_deg": bandwidth_deg, "grid_step_deg": grid_step_deg,
                "max_samples": max_samples, "seed": seed,
                "min_phase_samples": int(min_phase_samples),
                "sample_rotation_euler_deg": list(sample_rotation_euler_deg),
                "algorithm_id": ODF_ALGORITHM_ID}
            texture_configuration["pf_algorithm_id"] = PF_ALGORITHM_ID
            odf.attrs["configuration_json"] = json.dumps(texture_configuration, sort_keys=True)
            odf.attrs["configuration_hash"] = hashlib.sha256(odf.attrs["configuration_json"].encode()).hexdigest()
            odf.attrs["orientation_convention"] = ORIENTATION_CONVENTION
            odf.attrs["plot_type"] = "PF is a pole projection; ODF is a normalized SO(3) density"
            for pid, kind in kinds.items():
                check_cancelled()
                spec = specs[pid]
                lattice = spec.get("lattice_parameters") or {}
                ca_value = spec.get("c_over_a", lattice.get("c_over_a"))
                ca = float(ca_value or 1.0)
                if weighting == "grain":
                    samples, weights = _sample_grains(Path(features_file).parent / "v2" / "grains.csv", pid, max_samples, seed)
                elif weighting == "area":
                    samples, weights = _sample_pixels(f["indexed"], f["features"], pid, max_samples, seed)
                else:
                    raise ValueError("weighting must be 'area' or 'grain'")
                if len(samples) < int(min_phase_samples):
                    group = odf.create_group(f"phase_{pid}_{kind}")
                    group.attrs["phase_id"] = int(pid); group.attrs["phase_name"] = str(spec["name"])
                    group.attrs["space_group"] = int(spec["space_group"])
                    group.attrs["status"] = "unavailable_insufficient_phase_samples"
                    group.attrs["orientation_samples"] = len(samples)
                    group.attrs["minimum_required_samples"] = int(min_phase_samples)
                    summaries.append({"phase_id": pid, "phase": spec["name"], "kind": "PF_ODF",
                        "status": "unavailable_insufficient_phase_samples", "orientation_samples": len(samples),
                        "minimum_required_samples": int(min_phase_samples), "PF_peak_MUD": float("nan"),
                        "ODF_peak_MRD": float("nan"), "texture_index_J": float("nan")})
                    continue
                if kind == "alpha" and ca_value is None:
                    raise ValueError(f"Measured HCP lattice parameters missing for phase {pid}; no nominal Ti constant is substituted")
                # Explicit sample-frame rotation is a left action. The source
                # scanning/specimen rotations are recorded but not guessed.
                oriented = (sample_rotation * Rotation.from_quat(samples)).as_quat()
                if kind == "alpha":
                    phase_families = families or tuple(HCP_FAMILIES)
                else:
                    phase_families = families or tuple(BCC_FAMILIES)
                group = odf.create_group(f"phase_{pid}_{kind}")
                group.attrs["phase_id"] = int(pid); group.attrs["phase_name"] = str(spec["name"])
                group.attrs["space_group"] = int(spec["space_group"])
                for family in phase_families:
                    crystal_poles = _pole_family(spec["space_group"], family, ca)
                    all_poles = np.concatenate([Rotation.from_quat(q).apply(crystal_poles) for q in oriented], axis=0)
                    pf_path = outdir / f"PF_{kind}_{family}"
                    mud = _plot_pf(pf_path, all_poles, title=f"{spec['name']} {{family {family}}}", formats=formats)
                    pgrp = group.create_group(f"PF_{family}")
                    pgrp.create_dataset("MUD", data=mud.astype(np.float32), compression="gzip")
                    pf_edges = np.linspace(-np.sqrt(2), np.sqrt(2), mud.shape[0]+1)
                    areas = _pf_bin_areas(pf_edges)
                    pgrp.create_dataset("bin_disk_area", data=areas)
                    pgrp.create_dataset("x_edges", data=pf_edges); pgrp.create_dataset("y_edges", data=pf_edges)
                    pgrp.attrs["pf_algorithm_id"] = PF_ALGORITHM_ID
                    pgrp.attrs["area_weighted_mean_MUD"] = float(np.nansum(mud*areas)/(2*np.pi))
                    pgrp.attrs["family"] = family; pgrp.attrs["projection"] = "Lambert equal-area"
                    pgrp.attrs["antipodal"] = True; pgrp.attrs["valid_pole_count"] = len(all_poles)
                    pgrp.attrs["mean_valid_bin_MUD"] = float(np.nanmean(mud))
                    summaries.append({"phase_id": pid, "phase": spec["name"], "family": family,
                                      "kind": "PF_MUD", "orientation_samples": len(samples),
                                      "pole_count": len(all_poles), "mean_valid_bin_MUD": float(np.nanmean(mud)),
                                      "plot_type": "pole figure, not IPF"})
                phi1, Phi, phi2, mrd, odf_params = _odf_grid(
                    oriented, weights, int(spec["space_group"]), bandwidth_deg, grid_step_deg, seed)
                group.create_dataset("phi1_rad", data=phi1)
                group.create_dataset("Phi_rad", data=Phi)
                group.create_dataset("phi2_rad", data=phi2)
                group.create_dataset("MRD", data=mrd, compression="gzip", shuffle=True)
                group.attrs["ODF_parameters_json"] = json.dumps(odf_params)
                section_dir = outdir / f"ODF_{kind}_sections"; section_dir.mkdir(exist_ok=True)
                for i in range(0, len(phi2), max(1, int(round(30 / grid_step_deg)))):
                    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
                    mesh = ax.pcolormesh(np.rad2deg(phi1), np.rad2deg(Phi), mrd[:, :, i].T,
                                         shading="auto", cmap="magma")
                    ax.set_xlabel("phi1 (degrees)"); ax.set_ylabel("Phi (degrees)")
                    angle = np.rad2deg(phi2[i])
                    ax.set_title(f"{spec['name']} ODF section phi2={angle:g}° | MRD")
                    fig.colorbar(mesh, ax=ax, label="MRD, Haar-normalized")
                    for fmt in formats: fig.savefig(section_dir / f"phi2_{angle:03.0f}.{fmt}", dpi=220)
                    plt.close(fig)
                summaries.append({"phase_id": pid, "phase": spec["name"], "kind": "ODF_MRD",
                                  "orientation_samples": len(samples), **odf_params})
        summary = {
            "schema_version": 1, "method": "symmetry-summed Gaussian SO(3) kernel; numerical Haar normalization",
            "odf_algorithm_id": ODF_ALGORITHM_ID,
            "pf_algorithm_id": PF_ALGORITHM_ID,
            "weighting": weighting, "sampling": "seeded equal-area raster strata" if weighting == "area" else "one equally weighted V2 grain mean",
            "seed": int(seed), "max_samples": int(max_samples), "orientation_convention": ORIENTATION_CONVENTION,
            "min_phase_samples": int(min_phase_samples),
            "source_euler_unit": meta["angular_units_input"],
            "source_scanning_rotation_rad": meta.get("scanning_rotation_rad"),
            "source_specimen_orientation_euler": meta.get("specimen_orientation_euler"),
            "sample_rotation_euler_deg_applied": list(map(float, sample_rotation_euler_deg)),
            "source_rotation_policy": "recorded, not applied because sign/reference semantics require instrument validation",
            "phase_mapping": {str(pid): {"kind": kinds[pid], "name": specs[pid]["name"],
                                           "space_group": specs[pid]["space_group"]} for pid in kinds},
            "results": summaries,
            "scientific_status": "exploratory; compare numeric grid with MTEX under matching settings before publication",
        }
        (outdir / "texture_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with (outdir / "texture_summary.csv").open("w", encoding="utf-8", newline="") as stream:
            cols = sorted({k for row in summaries for k in row})
            writer = csv.DictWriter(stream, fieldnames=cols); writer.writeheader(); writer.writerows(summaries)
        (outdir / "parameters.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
