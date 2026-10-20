"""α-Ti crystallographic slip-system and Schmid/tau field analysis."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from .storage import check_cancelled, transactional_output
from .v3_crystal import (
    ORIENTATION_CONVENTION, canonical_axis, hcp_direction_cartesian,
    hcp_plane_normal_cartesian, point_group, normalize_quaternions,
)

SLIP_SEEDS = {
    "basal_a": ((0, 0, 0, 1), (1, 1, -2, 0)),
    "prismatic_a": ((1, 0, -1, 0), (1, -2, 1, 0)),
    "pyramidal_a": ((1, 0, -1, 1), (1, -2, 1, 0)),
    "pyramidal_c+a": ((1, 0, -1, 1), (1, 1, -2, -3)),
}


def slip_systems_alpha(c_over_a: float):
    """Generate unique symmetry-related α slip systems from valid 4-index seeds."""
    symmetry = point_group(194)
    result = {}
    for family, (plane4, direction4) in SLIP_SEEDS.items():
        n0 = hcp_plane_normal_cartesian(plane4, c_over_a)
        s0 = hcp_direction_cartesian(direction4, c_over_a)
        s0 /= np.linalg.norm(s0)
        if abs(float(n0 @ s0)) > 1e-8:
            raise ValueError(f"Invalid non-orthogonal seed for {family}: s·n={n0 @ s0}")
        systems = []
        for op in symmetry:
            n = canonical_axis(op.apply(n0))
            s = canonical_axis(op.apply(s0))
            if not any(abs(n @ old["normal"]) > 1 - 1e-7 and abs(s @ old["direction"]) > 1 - 1e-7
                       for old in systems):
                systems.append({"plane_4": list(plane4), "direction_4": list(direction4),
                                "normal": n, "direction": s})
        systems.sort(key=lambda row: tuple(np.round(np.r_[row["normal"], row["direction"]], 7)))
        for i, row in enumerate(systems, 1):
            row["system_id"] = i
            row["s_dot_n"] = float(row["normal"] @ row["direction"])
        result[family] = systems
    return result


def _load_axis(axis):
    v = np.asarray(axis, dtype=np.float64)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-12:
        raise ValueError("Loading direction must be a finite nonzero Sample-X/Y/Z vector")
    return v / np.linalg.norm(v)


@transactional_output()
def analyze_schmid(features_file, outdir, *, load_axis=(0, 0, 1), roi=None,
                   block_rows=32, stress_tensor_mpa=None, crss_mpa=None,
                   formats=("png", "pdf", "svg"), max_pixels=2_200_000,
                   progress=print):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    load_axis = _load_axis(load_axis)
    formats = tuple(dict.fromkeys(str(x).lower().lstrip(".") for x in formats))
    if not formats or set(formats) - {"png", "tiff", "pdf", "svg"}:
        raise ValueError("Schmid formats must be PNG, TIFF, PDF, or SVG")
    sigma = None
    if stress_tensor_mpa is not None:
        sigma = np.asarray(stress_tensor_mpa, dtype=np.float64)
        if sigma.shape != (3, 3) or not np.all(np.isfinite(sigma)):
            raise ValueError("Stress tensor must be a finite 3x3 matrix in MPa")
        if not np.allclose(sigma, sigma.T, atol=1e-9, rtol=0):
            raise ValueError("Stress tensor must be symmetric")
    if crss_mpa is not None:
        if any(k not in SLIP_SEEDS or not np.isfinite(float(v)) or float(v) <= 0 for k, v in crss_mpa.items()):
            raise ValueError("CRSS weights must be positive MPa values keyed by supported slip family")
    with h5py.File(features_file, "r") as f:
        progress("Schmid: reading phase metadata")
        meta = json.loads(f.attrs["metadata_json"])
        phase_specs = {int(k): v for k, v in meta.get("phase_specs", {}).items()}
        alpha_ids = [pid for pid, spec in phase_specs.items()
                     if int(spec.get("space_group", 0)) == 194 and
                     any(x in str(spec.get("name", "")).casefold() for x in ("alpha", "α", "hcp", "titanium-hex"))]
        if len(alpha_ids) != 1:
            raise ValueError("Schmid analysis requires exactly one explicitly named Ti HCP (space group 194) phase")
        pid = alpha_ids[0]; spec = phase_specs[pid]
        lattice = spec.get("lattice_parameters") or {}
        c_over_a = spec.get("c_over_a", lattice.get("c_over_a"))
        if c_over_a is None:
            raise ValueError("Measured α-Ti lattice parameters are absent; no nominal c/a value will be substituted")
        systems = slip_systems_alpha(float(c_over_a))
        h, w = f["indexed"].shape
        if roi is None:
            if h * w > max_pixels:
                raise ValueError(f"Full map {w}x{h} exceeds Schmid safety limit; provide an explicit ROI")
            x = y = 0; rw = w; rh = h
        else:
            x, y, rw, rh = map(int, roi)
            if min(x, y) < 0 or min(rw, rh) <= 0 or x + rw > w or y + rh > h:
                raise ValueError("Schmid ROI is outside the feature cache")
            if rw * rh > max_pixels:
                raise ValueError(f"Schmid ROI exceeds {max_pixels} pixels")
        fields_path = outdir / "schmid_fields.h5"
        with h5py.File(fields_path, "w") as dst:
            dst.attrs["schema_version"] = 1
            dst.attrs["orientation_convention"] = ORIENTATION_CONVENTION
            dst.attrs["phase_id"] = pid; dst.attrs["phase_name"] = spec["name"]
            dst.attrs["roi_xywh"] = (x, y, rw, rh)
            dst.attrs["load_axis_sample_unit"] = load_axis
            dst.attrs["stress_tensor_sample_mpa"] = sigma if sigma is not None else np.full((3, 3), np.nan)
            dst.attrs["signed_m_convention"] = "canonicalized slip direction; tension-positive; signed value depends on chosen slip direction representative"
            dst.attrs["stress_unit"] = "MPa" if sigma is not None else "not applicable"
            fg = dst.create_group("families")
            for family in systems:
                grp = fg.create_group(family)
                grp.create_dataset("signed_m" if sigma is None else "tau_mpa", shape=(rh, rw), dtype="f4", chunks=True, compression="gzip")
                grp.create_dataset("abs_m" if sigma is None else "abs_tau_mpa", shape=(rh, rw), dtype="f4", chunks=True, compression="gzip")
                grp.create_dataset("system_id", shape=(rh, rw), dtype="u2", chunks=True, compression="gzip")
            max_name = "Schmid_Factor_Max" if sigma is None else "Tau_Absolute_Max_MPa"
            dst.create_dataset(max_name, shape=(rh, rw), dtype="f4", chunks=True, compression="gzip")
            dst.create_dataset("Active_Slip_System_Candidate", shape=(rh, rw), dtype="u2", chunks=True, compression="gzip")
            for row0 in range(0, rh, max(1, int(block_rows))):
                check_cancelled()
                row1 = min(row0 + max(1, int(block_rows)), rh)
                progress(f"Schmid: computing rows {row0}:{row1} of {rh}")
                q = np.moveaxis(f["features"][0:4, y + row0:y + row1, x:x + rw], 0, -1).reshape(-1, 4)
                ph = np.rint(f["features"][4, y + row0:y + row1, x:x + rw]).astype(np.uint8).ravel()
                indexed = f["indexed"][y + row0:y + row1, x:x + rw].astype(bool).ravel()
                valid = (ph == pid) & indexed
                best_global = np.zeros(len(q), dtype=np.uint16)
                best_value = np.full(len(q), np.nan, dtype=np.float32)
                max_abs = np.zeros(len(q), dtype=np.float32)
                rots = Rotation.from_quat(normalize_quaternions(q[valid])) if np.any(valid) else None
                family_number = 0
                for family, candidates in systems.items():
                    family_number += 1
                    signed_best = np.full(len(q), np.nan, dtype=np.float32)
                    abs_best = np.full(len(q), np.nan, dtype=np.float32)
                    sys_best = np.zeros(len(q), dtype=np.uint16)
                    local_best = np.full(int(valid.sum()), -1.0, dtype=np.float64)
                    local_signed = np.zeros(int(valid.sum()), dtype=np.float64)
                    local_id = np.zeros(int(valid.sum()), dtype=np.uint16)
                    for item in candidates:
                        s = rots.apply(item["direction"])
                        n = rots.apply(item["normal"])
                        if sigma is None:
                            val = (s @ load_axis) * (n @ load_axis)
                            metric = np.abs(val)
                        else:
                            val = np.einsum("ni,ij,nj->n", s, sigma, n)
                            metric = np.abs(val)
                        take = metric > local_best
                        local_best[take] = metric[take]; local_signed[take] = val[take]
                        local_id[take] = item["system_id"]
                    signed_best[valid] = local_signed
                    abs_best[valid] = local_best
                    sys_best[valid] = local_id
                    base = "families/" + family
                    dst[base]["signed_m" if sigma is None else "tau_mpa"][row0:row1] = signed_best.reshape(row1-row0, rw)
                    dst[base]["abs_m" if sigma is None else "abs_tau_mpa"][row0:row1] = abs_best.reshape(row1-row0, rw)
                    dst[base]["system_id"][row0:row1] = sys_best.reshape(row1-row0, rw)
                    take_global = (abs_best > max_abs) & valid
                    max_abs[take_global] = abs_best[take_global]
                    best_global[take_global] = family_number * 100 + sys_best[take_global]
                    best_value[take_global] = signed_best[take_global]
                dst[max_name][row0:row1] = np.where(valid, max_abs, np.nan).reshape(row1-row0, rw)
                dst["Active_Slip_System_Candidate"][row0:row1] = best_global.reshape(row1-row0, rw)
    grain_map_path = Path(features_file).parent / "v2" / "v2_grains.h5"
    grain_rows = []
    if grain_map_path.is_file():
        with h5py.File(grain_map_path, "r") as gf, h5py.File(fields_path, "r") as sf:
            grain_ids = gf["grain_id"][:]
            v2_roi = tuple(map(int, gf.attrs.get("roi_xywh", (0, 0, w, h))))
            if v2_roi != (x, y, rw, rh) or grain_ids.shape != (rh, rw):
                raise ValueError("V2 grain map ROI differs from Schmid ROI")
            with h5py.File(features_file, "r") as source:
                phase_roi = np.rint(source["features"][4, y:y+rh, x:x+rw]).astype(np.uint8)
            for gid in np.unique(grain_ids):
                if gid == 0: continue
                mask = (grain_ids == gid) & (phase_roi == pid)
                if not np.any(mask): continue
                row = {"grain_id": int(gid), "phase_id": pid,
                       "pixel_count": int(mask.sum()), "roi_xywh": json.dumps([x, y, rw, rh])}
                for family in systems:
                    arr = sf[f"families/{family}/abs_m"][:] if sigma is None else sf[f"families/{family}/abs_tau_mpa"][:]
                    vals = arr[mask]
                    row[f"{family}_mean_abs"] = float(np.nanmean(vals))
                    row[f"{family}_max_abs"] = float(np.nanmax(vals))
                grain_rows.append(row)
    with (outdir / "grain_schmid.csv").open("w", encoding="utf-8", newline="") as stream:
        cols = ["grain_id", "phase_id", "pixel_count", "roi_xywh",
                *[f"{family}_{stat}" for family in systems for stat in ("mean_abs", "max_abs")]]
        writer = csv.DictWriter(stream, fieldnames=cols); writer.writeheader(); writer.writerows(grain_rows)
    params = {
        "schema_version": 1, "method": "resolved single-crystal slip-system loading",
        "load_axis_sample_unit": load_axis.tolist(), "stress_tensor_sample_MPa": sigma.tolist() if sigma is not None else None,
        "quantity": "tau=s^T sigma n (MPa)" if sigma is not None else "m=(s·l)(n·l), dimensionless",
        "max_uniaxial_abs_m_theoretical": 0.5 if sigma is None else None,
        "phase_id": pid, "phase_name": spec["name"], "space_group": spec["space_group"],
        "c_over_a": float(c_over_a), "roi_xywh": [x, y, rw, rh],
        "slip_systems": {family: [{k: v for k, v in row.items() if k not in ("normal", "direction")} |
                                  {"normal": row["normal"].tolist(), "direction": row["direction"].tolist()}
                                  for row in rows] for family, rows in systems.items()},
        "crss_mpa": crss_mpa, "crss_interpretation": "user-supplied relative activation weighting only; no activation claim" if crss_mpa else None,
        "orientation_convention": ORIENTATION_CONVENTION,
        "scientific_status": "exploratory; no local stress/strain or first-activated system is inferred from Schmid factor",
    }
    (outdir / "parameters.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("Schmid: exporting maps")
    _render_schmid(fields_path, outdir, formats, progress=progress)
    progress("Schmid: complete")
    return params


def _render_schmid(fields_path, outdir, formats, progress=print):
    with h5py.File(fields_path, "r") as f:
        names = list(f["families"])
        max_name = "Schmid_Factor_Max" if "Schmid_Factor_Max" in f else "Tau_Absolute_Max_MPa"
        for name in [*names, max_name]:
            check_cancelled()
            progress(f"Schmid: rendering {name}")
            ds = f["families"][name]["abs_m"] if name in f["families"] and "abs_m" in f["families"][name] else (
                f["families"][name]["abs_tau_mpa"] if name in f["families"] else f[max_name])
            arr = ds[:]
            factor = max(1, int(np.ceil(max(arr.shape) / 1500)))
            shown = arr[::factor, ::factor]
            vmax = 0.5 if max_name == "Schmid_Factor_Max" else float(np.nanpercentile(shown, 99))
            vmax = max(vmax, 1e-8)
            fig, ax = plt.subplots(figsize=(6, 4.8), layout="constrained")
            im = ax.imshow(shown, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
            axis = np.asarray(f.attrs["load_axis_sample_unit"], dtype=float)
            axis_label = "σ tensor" if np.isfinite(f.attrs["stress_tensor_sample_mpa"]).any() else f"Sample axis {axis.tolist()}"
            ax.set_title(f"{'|m|' if max_name == 'Schmid_Factor_Max' else '|τ| (MPa)'} — {name}\n"
                         f"{axis_label}; exploratory")
            ax.set_xlabel("ROI pixel X"); ax.set_ylabel("ROI pixel Y")
            fig.colorbar(im, ax=ax, label="dimensionless |m|" if max_name == "Schmid_Factor_Max" else "MPa")
            safe = name.replace("+", "_plus")
            for fmt in formats:
                progress(f"Schmid: exporting {name} {fmt.upper()}")
                fig.savefig(outdir / f"{safe}.{fmt}", dpi=220)
            plt.close(fig)
