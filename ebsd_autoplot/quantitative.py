"""V4 M2 quantitative analysis from immutable V1/V2 caches, with atomic outputs."""
from __future__ import annotations

import csv
import hashlib
import json
import uuid
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from PIL import Image

from .euler_maps import EULER_DISPLAY_CONVENTION, quaternion_euler_maps
from .grain_statistics import (attach_v2_grain_metadata, grain_size_cdf,
                               grain_statistics, summarize_grain_sizes)
from .rx_state import RxThresholds, STATE_NAMES, classify_rx, threshold_sensitivity
from .storage import check_cancelled, transactional_output
from .texture_stats import inverse_pole_statistics, read_texture_statistics


def _csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        names = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader(); writer.writerows(rows)


def _json_default(obj):
    from dataclasses import asdict, is_dataclass
    if is_dataclass(obj): return asdict(obj)
    if isinstance(obj, np.generic): return obj.item()
    raise TypeError(type(obj).__name__)


def _save_map(outdir, name, array, *, formats, title, font, extent, cmap=None,
              vmin=None, vmax=None, unit=None):
    check_cancelled()
    with plt.rc_context({"font.family": font, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(6.6, 4.9), layout="constrained")
        kwargs = {"extent": extent, "origin": "upper", "interpolation": "nearest"}
        if array.ndim == 2: kwargs.update(cmap=cmap or "viridis", vmin=vmin, vmax=vmax)
        image = ax.imshow(array, **kwargs)
        ax.set_xlabel("X (um)"); ax.set_ylabel("Y (um)"); ax.set_title(title)
        if array.ndim == 2: fig.colorbar(image, ax=ax, label=unit or "")
        for fmt in formats:
            check_cancelled()
            image.set_interpolation("none" if fmt in ("pdf", "svg") else "nearest")
            fig.savefig(outdir / f"{name}.{fmt}", dpi=180)
        plt.close(fig)


def _native_maps(outdir, rx, euler, rgb, formats, roi, dx, dy):
    """One source pixel per exported pixel; scientific scalars retain TIFF float."""
    native = outdir / "native_maps"; native.mkdir()
    raster_formats = set(formats) & {"png", "tiff"}
    for fmt in raster_formats:
        check_cancelled()
        image = Image.fromarray(rx.astype(np.uint8))
        if fmt == "png":
            image = image.convert("P")
            palette = [112,112,112, 60,171,105, 240,187,69, 196,78,82]
            image.putpalette(palette + [0] * (768-len(palette)))
        image.save(native / f"RX_state.{fmt}")
        Image.fromarray(np.rint(rgb.clip(0,1)*255).astype(np.uint8)).save(native / f"Euler_RGB.{fmt}")
        for i, name in enumerate(("phi1", "Phi", "phi2")):
            scalar = euler[..., i]
            if fmt == "tiff": image = Image.fromarray(scalar.astype(np.float32))
            else:
                maximum = 180 if i == 1 else 360
                encoded = np.full(scalar.shape, 65535, np.uint16)
                valid = np.isfinite(scalar)
                encoded[valid] = np.rint(scalar[valid]/maximum*65534).astype(np.uint16)
                image = Image.fromarray(encoded)
            image.save(native / f"Euler_{name}.{fmt}")
    metadata = {"shape_HW": list(rx.shape), "roi_xywh": list(roi), "dx_um": dx, "dy_um": dy,
        "resampling": "none; one original ROI pixel per image pixel", "RX_scalar_codes": STATE_NAMES,
        "Euler_scalar_TIFF": "float32 degrees, invalid=NaN",
        "Euler_scalar_PNG": "uint16; invalid=65535; degree=value*max_angle/65534; phi1/phi2 max_angle=360, Phi=180",
        "Euler_RGB": "uint8 RGB from explicit Bunge Euler normalization; invalid black; not IPF",
        "annotated_PDF_SVG": "embedded full-resolution raster with interpolation=none and vector text/axes"}
    (native / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _ipf_scatter(outdir, phase_id, vectors, formats, font):
    """A true inverse-pole orientation scatter; never labelled as density."""
    denominator = 1 + vectors[:, 2]
    with plt.rc_context({"font.family": font, "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(5.8, 5.0), layout="constrained")
        ax.scatter(vectors[:, 0]/denominator, vectors[:, 1]/denominator, s=9, alpha=.5)
        ax.set_aspect("equal"); ax.set_xlabel("Crystal x / (1 + z)"); ax.set_ylabel("Crystal y / (1 + z)")
        ax.set_title(f"Phase {phase_id}: IPF sample-Z orientation scatter\nLaue fundamental sector; stereographic; not density")
        for fmt in formats: fig.savefig(outdir / f"IPF_Z_scatter_phase_{phase_id}.{fmt}", dpi=180)
        plt.close(fig)


@transactional_output()
def analyze_quantitative(features_file, v2_dir, outdir, *, formats=("png",),
                         font="DejaVu Sans", thresholds=None, min_pixels=8,
                         exclude_edge=False, min_indexed_fraction=0.0,
                         min_phase_grains=1, min_beta_grains=5, min_beta_pixels=50,
                         sensitivity_scales=(.8, 1.0, 1.2), odf_file=None,
                         ipf_statistics=True, ipf_max_samples=2000,
                         max_pixels=2200000, progress=print):
    """Write RX proxy, grains, Euler and phase summaries for the V2 ROI.

    Does not read/reanalyse raw EBSD data and never opens caches for writing.
    RX thresholds are exploratory defaults that require material calibration.
    Texture metrics are reported only from a supplied numeric V3 grid.
    """
    outdir, v2_dir = Path(outdir), Path(v2_dir)
    formats = tuple(dict.fromkeys(str(f).lower().lstrip(".") for f in formats))
    if not formats or set(formats) - {"png", "tiff", "pdf", "svg"}:
        raise ValueError("Formats must be PNG, TIFF, PDF or SVG")
    if int(ipf_max_samples) < 1 or int(max_pixels) < 1:
        raise ValueError("Sample and ROI limits must be positive")
    outdir.mkdir(parents=True, exist_ok=True)
    with h5py.File(features_file, "r") as f, h5py.File(v2_dir / "v2_grains.h5", "r") as g:
        from .storage import digest
        pair_hash=g.attrs.get('source_features_sha256')
        if pair_hash and pair_hash!=digest(features_file):
            raise ValueError('V2 grain cache belongs to a different features file')
        metadata = json.loads(f.attrs["metadata_json"])
        if metadata.get("angular_units_input") not in ("radian", "degree"):
            raise ValueError("Cache must declare verified input Euler units")
        full_h, full_w = f["indexed"].shape
        roi = tuple(map(int, g.attrs.get("roi_xywh", (0, 0, full_w, full_h))))
        x, y, w, h = roi
        if min(x, y) < 0 or min(w, h) < 1 or x+w > full_w or y+h > full_h or w*h > max_pixels:
            raise ValueError("Invalid or oversized quantitative ROI; use the V2 ROI safety limit")
        sl = (slice(y, y+h), slice(x, x+w))
        labels = g["grain_id"][...].astype(np.int64)
        if labels.shape != (h, w): raise ValueError("V2 grain map and declared ROI disagree")
        phase = np.rint(f["features"][4, sl[0], sl[1]]).astype(np.int32)
        indexed = f["indexed"][sl] > 0
        kam = f["features"][5, sl[0], sl[1]]
        kam_valid = f["kam_valid"][sl] > 0
        q = np.moveaxis(f["features"][0:4, sl[0], sl[1]], 0, -1)
        grod = g["GROD_deg"][...]
    dx, dy = float(metadata["x_step_um"]), float(metadata["y_step_um"])
    specs = metadata["phase_specs"]
    if not specs: raise ValueError("Phase metadata is required for quantitative analysis")
    unknown_phases = set(np.unique(phase[indexed]).tolist()) - {int(pid) for pid in specs}
    if unknown_phases:
        raise ValueError(f"Indexed phase IDs lack metadata: {sorted(unknown_phases)}")
    euler, rgb, euler_valid = quaternion_euler_maps(q, indexed)
    if np.any(indexed & ~euler_valid):
        raise ValueError("Indexed cache contains invalid quaternions; quantitative analysis cannot validate orientations")
    progress("Quantitative: physical grain statistics and QC")
    grains = grain_statistics(labels, phase, indexed, dx, dy, kam=kam,
        kam_valid=kam_valid, grod=grod, min_pixels=min_pixels, exclude_edge=exclude_edge)
    attach_v2_grain_metadata(grains, v2_dir)
    # V2 GROD is relative to its symmetry-aligned mean; its all-pixel mean is
    # GOS. Missing GROD is unavailable instead of substituting KAM or zero.
    rx_kwargs = {"min_indexed_fraction": min_indexed_fraction, "min_phase_grains": min_phase_grains,
                 "min_beta_grains": min_beta_grains, "min_beta_pixels": min_beta_pixels}
    rx, rx_rows, assignments = classify_rx(grains, labels, phase, indexed, specs,
                                          thresholds=thresholds, **rx_kwargs)
    sensitivity = threshold_sensitivity(grains, labels, phase, indexed, specs,
        thresholds=thresholds, scales=sensitivity_scales, **rx_kwargs)
    for row in grains:
        row["rx_state_code"] = assignments.get(row["grain_id"], 0)
        row["rx_state"] = STATE_NAMES[row["rx_state_code"]]
        row["RX_QC_grade"] = "passes_configured_proxy_gates" if row["rx_state_code"] else "unclassified"
        row["RX_confidence_status"] = "uncalibrated_QC_grade_only; no probability/confidence estimate"
        row["centroid_x_um"] += x * dx; row["centroid_y_um"] += y * dy
    config = {"thresholds": thresholds or {str(pid): RxThresholds() for pid in specs},
              "min_pixels": min_pixels, "exclude_edge": exclude_edge,
              "sensitivity_scales": list(sensitivity_scales), "roi_xywh": roi,
              "odf_file": str(odf_file) if odf_file else None,
              "ipf_statistics": ipf_statistics, "ipf_max_samples": ipf_max_samples, **rx_kwargs}
    config_json = json.dumps(config, sort_keys=True, default=_json_default)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    run_id = "quantitative_" + uuid.uuid4().hex[:12]
    phase_rows = []
    ipf_vectors = {}
    for raw_pid, spec in specs.items():
        pid = int(raw_pid)
        mask = (phase == pid) & indexed
        valid_kam = mask & kam_valid & np.isfinite(kam)
        ka = kam[valid_kam]
        group = [row for row in grains if row["phase_id"] == pid]
        gos = np.array([row["GOS_deg"] for row in group if row["qc_included"] and np.isfinite(row["GOS_deg"])])
        texture = read_texture_statistics(odf_file, pid, features_file=features_file)
        ipf = {"IPF_status": "unavailable_disabled" if not ipf_statistics else "unavailable_no_valid_orientations"}
        if ipf_statistics and np.any(mask & euler_valid):
            samples = q[mask & euler_valid]
            eligible_orientations = len(samples)
            if len(samples) > ipf_max_samples:
                samples = samples[np.linspace(0, len(samples)-1, ipf_max_samples, dtype=int)]
            try:
                vectors, ipf = inverse_pole_statistics(samples, int(spec["space_group"]))
                ipf["IPF_total_eligible_orientations"] = eligible_orientations
                ipf["IPF_sampling"] = "deterministic equally spaced eligible raster indices; bounded by ipf_max_samples"
                ipf_vectors[pid] = vectors
            except (ImportError, ValueError) as exc:
                ipf = {"IPF_status": "unavailable_" + type(exc).__name__, "IPF_reason": str(exc)}
        row = {"run_id": run_id, "configuration_hash": config_hash,
            "software_version": "4.0-research", "scan_id": metadata.get("scan_path", "cache_ROI"),
            "phase_id": pid, "phase_name": spec.get("name", str(pid)),
            "Nx": full_w, "Ny": full_h, "ROI_Nx": w, "ROI_Ny": h, "dx_um": dx, "dy_um": dy,
            "ROI_x_min_um": x*dx, "ROI_x_max_um": (x+w)*dx, "ROI_y_min_um": y*dy, "ROI_y_max_um": (y+h)*dy,
            "all_ROI_pixels": w*h, "indexed_pixels_all_phases": int(indexed.sum()),
            "unindexed_pixels": int((~indexed).sum()), "indexed_fraction_all_phases": float(indexed.mean()),
            "phase_indexed_pixels": int(mask.sum()), "phase_area_fraction_all_ROI": float(mask.mean()),
            "KAM_valid_count": len(ka), "KAM_mean_deg": float(ka.mean()) if len(ka) else float("nan"),
            "KAM_median_deg": float(np.median(ka)) if len(ka) else float("nan"),
            "KAM_p95_deg": float(np.percentile(ka, 95)) if len(ka) else float("nan"),
            "KAM_status": "available" if len(ka) else "unavailable_no_valid_neighbours",
            "GOS_mean_deg": float(gos.mean()) if len(gos) else float("nan"),
            "GOS_median_deg": float(np.median(gos)) if len(gos) else float("nan"),
            "RX_confidence_status": "uncalibrated_QC_grade_only; not a confidence probability",
            **summarize_grain_sizes(group), **texture, **ipf}
        for rxrow in rx_rows:
            if rxrow["phase_id"] == pid:
                row[rxrow["state"] + "_area_fraction_classified"] = rxrow["area_fraction_classified"]
                row[rxrow["state"] + "_number_fraction_classified"] = rxrow["number_fraction_classified"]
                row[rxrow["state"] + "_area_fraction_all_ROI"] = rxrow["area_fraction_all_ROI"]
                row[rxrow["state"] + "_area_fraction_indexed_phase"] = rxrow["area_fraction_indexed_phase"]
                row["RX_status"] = rxrow["status"]
        phase_rows.append(row)
    _csv(outdir / "grain_statistics.csv", grains)
    cdf_rows = []
    for pid in specs:
        cdf_rows.extend(dict(row, phase_id=int(pid)) for row in grain_size_cdf(
            [g for g in grains if g["phase_id"] == int(pid)]))
    _csv(outdir / "grain_size_cdf.csv", cdf_rows)
    _csv(outdir / "per_phase_rx_state.csv", rx_rows)
    _csv(outdir / "threshold_sensitivity.csv", sensitivity)
    _csv(outdir / "sample_phase_summary.csv", phase_rows)
    with h5py.File(outdir / "RX_state.h5", "w") as dst:
        dst.attrs["schema_version"] = 1; dst.attrs["roi_xywh"] = roi
        dst.attrs["x_step_um"] = dx; dst.attrs["y_step_um"] = dy
        dst.attrs["scientific_status"] = "research proxy; thresholds not calibrated against RX ground truth"
        dst.attrs["confidence_status"] = "uncalibrated QC grade only; no probability/confidence estimated"
        dst.attrs["configuration_json"] = config_json
        dst.attrs["configuration_hash"] = config_hash
        dst.attrs["state_codes_json"] = json.dumps(STATE_NAMES)
        dst.create_dataset("RX_state", data=rx, compression="lzf")
        dst.create_dataset("grain_id", data=labels, compression="lzf")
        dst.create_dataset("indexed", data=indexed, compression="lzf")
        dst.create_dataset("phase_id", data=phase, compression="lzf")
    with h5py.File(outdir / "Euler_maps.h5", "w") as dst:
        dst.attrs["schema_version"] = 1; dst.attrs["roi_xywh"] = roi
        dst.attrs["x_step_um"] = dx; dst.attrs["y_step_um"] = dy
        dst.attrs["convention"] = EULER_DISPLAY_CONVENTION
        dst.attrs["source_Euler_units"] = metadata["angular_units_input"]
        dst.attrs["unit"] = "degree"
        for i, name in enumerate(("phi1_deg", "Phi_deg", "phi2_deg")):
            dst.create_dataset(name, data=euler[..., i], compression="lzf")
        dst.create_dataset("Euler_RGB", data=rgb, compression="lzf")
        dst.create_dataset("valid", data=euler_valid, compression="lzf")
    with h5py.File(outdir / "IPF_statistics.h5", "w") as dst:
        dst.attrs["schema_version"] = 1
        dst.attrs["convention"] = "q inverse sample-Z; orix Laue fundamental sector; vectors, not PF"
        for pid, vectors in ipf_vectors.items():
            dst.create_dataset(f"phase_{pid}_crystal_directions", data=vectors, compression="lzf")
    extent = [x*dx, (x+w)*dx, (y+h)*dy, y*dy]
    progress("Quantitative: exporting RX, grain and Euler figures")
    _native_maps(outdir, rx, euler, rgb, formats, roi, dx, dy)
    for pid, vectors in ipf_vectors.items(): _ipf_scatter(outdir, pid, vectors, formats, font)
    _save_map(outdir, "RX_state", rx, formats=formats, font=font, extent=extent,
        title="Recrystallization proxy (requires material calibration)\n0 unclassified | 1 RX | 2 substructured | 3 deformed",
        cmap=ListedColormap(["#707070", "#3cab69", "#f0bb45", "#c44e52"]), vmin=-.5, vmax=3.5,
        unit="Proxy class code")
    unique, inverse = np.unique(labels, return_inverse=True)
    ecd_by_id = {row["grain_id"]: row["ecd_um"] for row in grains if row["qc_included"]}
    values = np.array([ecd_by_id.get(int(gid), np.nan) for gid in unique], np.float32)
    ecd_map = values[inverse].reshape(labels.shape)
    _save_map(outdir, "Grain_ECD", ecd_map, formats=formats, font=font, extent=extent,
        title="Grain ECD | existing V2 segmentation", unit="ECD (um)")
    _save_map(outdir, "Euler_RGB", rgb, formats=formats, font=font, extent=extent,
        title="Euler RGB | Bunge ZXZ (not IPF)")
    for i, name in enumerate(("phi1", "Phi", "phi2")):
        _save_map(outdir, "Euler_" + name, euler[..., i], formats=formats, font=font, extent=extent,
            title=f"Euler {name} | Bunge ZXZ", vmin=0, vmax=180 if i == 1 else 360, unit="degrees")
    with plt.rc_context({"font.family": font, "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(11,4.5), layout="constrained")
        histogram_rows = []
        for raw_pid, spec in specs.items():
            selected = [row for row in grains if row["phase_id"] == int(raw_pid) and row["qc_included"]]
            sizes = np.array([row["ecd_um"] for row in selected])
            areas = np.array([row["area_um2"] for row in selected])
            if len(sizes):
                edges = np.histogram_bin_edges(sizes, bins="auto")
                number, _ = np.histogram(sizes, bins=edges)
                area, _ = np.histogram(sizes, bins=edges, weights=areas)
                axes[0].stairs(number/len(sizes), edges, label=str(spec.get("name", raw_pid)))
                axes[1].stairs(area/areas.sum(), edges, label=str(spec.get("name", raw_pid)))
                histogram_rows.extend({"phase_id": int(raw_pid), "ECD_bin_left_um": float(a),
                    "ECD_bin_right_um": float(b), "number_count": int(n), "area_um2": float(ar),
                    "number_fraction": float(n/len(sizes)), "area_fraction": float(ar/areas.sum())}
                    for a,b,n,ar in zip(edges[:-1], edges[1:], number, area))
        for ax, kind in zip(axes, ("Number", "Area")):
            ax.set_xlabel("Grain ECD (um)"); ax.set_ylabel(kind + " fraction per bin")
            ax.set_title(kind + " weighted grain size")
            if ax.get_legend_handles_labels()[0]: ax.legend()
        fig.suptitle("QC-included raster grains | edge policy in configuration")
        for fmt in formats: fig.savefig(outdir / f"Grain_size_distribution.{fmt}", dpi=180)
        plt.close(fig)
    _csv(outdir / "grain_size_histogram.csv", histogram_rows)
    summary = {"schema_version": 1, "run_id": run_id, "configuration_hash": config_hash,
        "cache_pair_identity": 'verified_sha256' if pair_hash else 'unverified_legacy_no_source_hash',
        "software_version": "4.0-research", "scientific_status": "exploratory; not scientifically validated for publication",
        "limitations": ["RX is an uncalibrated GOS/KAM/GROD proxy, not measured recrystallization",
                        "Grain shape and size inherit V2 segmentation, cleaning and ROI truncation",
                        "Euler RGB is not an IPF color map; gimbal-lock triplets are nonunique",
                        "Texture metrics require a separately generated numeric V3 cache; compare with MTEX"],
        "roi_xywh": roi, "source_features_file": str(Path(features_file).resolve()),
        "configuration": json.loads(config_json), "phases": phase_rows}
    (outdir / "quantitative_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    progress("Quantitative: complete (research status)")
    return summary
