"""Ambiguity-aware inferred prior-beta maps and reconstruction quality audit."""
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

from .crystal import ipf_colors
from .storage import check_cancelled, transactional_output
from .v3_crystal import ORIENTATION_CONVENTION, classify_phase_specs, normalize_quaternions


@transactional_output()
def reconstruct_prior_beta(features_file, v2_dir, variant_dir, outdir, *, formats=("png", "pdf", "svg")):
    """Render graph-supported beta parent candidates; unknowns remain unassigned."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    formats = tuple(dict.fromkeys(str(x).lower().lstrip(".") for x in formats))
    if not formats or set(formats) - {"png", "tiff", "pdf", "svg"}:
        raise ValueError("Prior-beta formats must be PNG, TIFF, PDF, or SVG")
    v2_dir = Path(v2_dir); variant_dir = Path(variant_dir)
    with h5py.File(features_file, "r") as f, h5py.File(v2_dir / "v2_grains.h5", "r") as gf, \
            h5py.File(variant_dir / "alpha_variant_fields.h5", "r") as vf:
        meta = json.loads(f.attrs["metadata_json"])
        specs = {int(k): v for k, v in meta["phase_specs"].items()}
        kinds = classify_phase_specs(specs)
        beta_ids = [pid for pid, kind in kinds.items() if kind == "beta"]
        alpha_ids = [pid for pid, kind in kinds.items() if kind == "alpha"]
        if len(alpha_ids) != 1 or len(beta_ids) > 1:
            raise ValueError("Prior-beta reconstruction needs one HCP alpha phase and at most one measured beta phase")
        beta_id = beta_ids[0] if beta_ids else 0
        roi = tuple(map(int, gf.attrs["roi_xywh"]))
        if tuple(map(int, vf.attrs["roi_xywh"])) != roi:
            raise ValueError("V2 and variant-cache ROI coordinates do not match")
        x, y, rw, rh = roi
        grain_id = gf["grain_id"][:]
        variant = vf["variant_id_map"][:]
        fit = vf["variant_fit_map_deg"][:]
        confidence = vf["confidence_map"][:]
        parent = vf["parent_id_map"][:]
        if any(a.shape != (rh, rw) for a in (grain_id, variant, fit, confidence, parent)):
            raise ValueError("V3 ROI caches have incompatible shapes")
        phase = np.rint(f["features"][4, y:y+rh, x:x+rw]).astype(np.uint8)
        alpha_mask = phase == alpha_ids[0]
        assigned = alpha_mask & (parent > 0)
        unassigned = alpha_mask & (parent == 0)
        parent_q = {}
        grain_info = {}
        with (variant_dir / "alpha_variant_grains.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                gid = int(row["grain_id"]); grain_info[gid] = row
                if int(row["parent_id"]) and row.get("parent_q"):
                    parent_q[gid] = normalize_quaternions(np.asarray(json.loads(row["parent_q"])).reshape(1, 4))[0]
        # Store one parent orientation per inferred component. Components are
        # assembled from alpha grains; a β orientation is labeled inferred.
        q_for_parent = {}
        for gid, q in parent_q.items():
            pid = int(grain_info[gid]["parent_id"])
            q_for_parent.setdefault(pid, q)
        beta_phase_map = np.zeros((rh, rw), dtype=np.uint8)
        euler_map = np.zeros((rh, rw, 3), dtype=np.float64)
        for gid in np.unique(grain_id[assigned]):
            mask = assigned & (grain_id == gid)
            row = grain_info.get(int(gid))
            if row is None or not int(row["parent_id"]):
                continue
            q = q_for_parent[int(row["parent_id"])]
            beta_phase_map[mask] = beta_id if beta_id else 255
            euler_map[mask] = Rotation.from_quat(q).as_euler("ZXZ", degrees=False)
        if beta_id:
            beta_color = ipf_colors(euler_map, beta_phase_map, {beta_id: specs[beta_id]}, axis="Z")
        else:
            # IPF key uses the validated BCC symmetry; this synthetic phase id
            # is only a rendering selector, never a measured phase assignment.
            beta_color = ipf_colors(euler_map, np.where(assigned, 1, 0),
                                    {1: {"space_group": 229, "name": "Inferred prior beta"}}, axis="Z")
        beta_color[~assigned] = 0
        boundaries = np.zeros((rh, rw), dtype=np.uint8)
        for a, b in ((parent[:, :-1], parent[:, 1:]), (parent[:-1, :], parent[1:, :])):
            edge = (a != b) & ((a > 0) | (b > 0))
            if a.shape[0] == rh:
                boundaries[:, :-1] |= edge.astype(np.uint8)
            else:
                boundaries[:-1, :] |= edge.astype(np.uint8)
        with h5py.File(outdir / "prior_beta_fields.h5", "w") as out:
            out.attrs["schema_version"] = 1; out.attrs["roi_xywh"] = roi
            out.attrs["orientation_convention"] = ORIENTATION_CONVENTION
            out.attrs["map_type"] = "inferred prior beta from alpha variants; not current measured beta"
            out.create_dataset("Prior_Beta_Grain_ID", data=parent, compression="gzip")
            out.create_dataset("Prior_Beta_IPF_RGB", data=beta_color, compression="gzip")
            out.create_dataset("Prior_Beta_Boundaries", data=boundaries, compression="gzip")
            out.create_dataset("Prior_Beta_OR_Fit_deg", data=fit, compression="gzip")
            out.create_dataset("Variant_ID", data=variant, compression="gzip")
            out.create_dataset("Confidence", data=confidence, compression="gzip")
            out.create_dataset("Unassigned", data=unassigned.astype(np.uint8), compression="gzip")
            out.create_dataset("Measured_Beta_Phase_ID", data=phase == beta_id if beta_id else np.zeros_like(phase, bool),
                               compression="gzip")
        parent_rows = []
        for parent_id in sorted(set(parent[assigned].tolist())):
            if not parent_id: continue
            members = [gid for gid, row in grain_info.items() if int(row["parent_id"]) == int(parent_id)]
            mask = assigned & (parent == parent_id)
            pfit = fit[mask]; pconf = confidence[mask]
            parent_rows.append({
                "parent_id": int(parent_id), "alpha_grain_count": len(members),
                "assigned_pixels": int(mask.sum()), "median_or_fit_deg": float(np.nanmedian(pfit)),
                "p90_or_fit_deg": float(np.nanpercentile(pfit, 90)),
                "mean_confidence": float(np.mean(pconf)),
                "status": "inferred candidate; not measured prior-beta",
            })
        with (outdir / "parent_grains.csv").open("w", encoding="utf-8", newline="") as stream:
            cols = ["parent_id", "alpha_grain_count", "assigned_pixels", "median_or_fit_deg",
                    "p90_or_fit_deg", "mean_confidence", "status"]
            writer = csv.DictWriter(stream, fieldnames=cols); writer.writeheader(); writer.writerows(parent_rows)
        # Copy the alpha-to-parent link audit into the prior-beta run.
        source_links = variant_dir / "alpha_parent_links.csv"
        if source_links.exists():
            (outdir / "alpha_parent_links.csv").write_bytes(source_links.read_bytes())
        residuals = fit[assigned & np.isfinite(fit)]
        coverage = float(assigned.sum() / max(int(alpha_mask.sum()), 1))
        boundary_count = int(boundaries.sum())
        qc = {
            "schema_version": 1, "status": "inferred; MTEX reference not run",
            "algorithm": "V2 alpha grain adjacency + Burgers inverse candidates + symmetry-reduced parent voting",
            "roi_xywh": list(roi), "alpha_pixels": int(alpha_mask.sum()),
            "parent_assigned_coverage": coverage, "unassigned_fraction": float(unassigned.sum() / max(int(alpha_mask.sum()), 1)),
            "assigned_parent_count": len(parent_rows), "parent_boundary_edge_pixel_count": boundary_count,
            "or_residual_median_deg": float(np.median(residuals)) if len(residuals) else None,
            "or_residual_p90_deg": float(np.percentile(residuals, 90)) if len(residuals) else None,
            "or_residual_p95_deg": float(np.percentile(residuals, 95)) if len(residuals) else None,
            "beta_anchor_comparison": "available only when beta grains are adjacent and pass the configured anchor fit; not yet reported as an independent measured validation",
            "parent_volume_fraction": None,
            "warning": "Do not interpret inferred parent coverage as measured prior-beta volume fraction or unique original boundaries.",
        }
        (outdir / "reconstruction_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_prior_maps(outdir, beta_color, boundaries, parent, fit, variant, confidence, unassigned, formats)
    return qc


def _render_prior_maps(outdir, ipf, boundaries, parent, fit, variants, confidence, unassigned, formats):
    for name, arr, title, cmap, label in (
        ("Prior_Beta_IPF.png", ipf, "Reconstructed prior β (inferred)\nNot current measured β phase", None, None),
        ("Prior_Beta_Boundaries.png", boundaries, "Inferred parent boundaries", "gray", "Boundary"),
        ("Prior_Beta_Grain_ID.png", parent, "Prior β parent candidate ID (0=unassigned)", "tab20", "Parent ID"),
        ("Prior_Beta_OR_Fit.png", fit, "α-child / inferred-parent OR residual", "magma", "Residual (°)"),
        ("Variant_ID.png", variants, "α Burgers variant ID (0=unknown)", "tab20", "Variant ID"),
        ("Confidence_Unassigned.png", np.where(unassigned, 0, confidence),
         "Graph-supported confidence; black/zero includes unassigned", "viridis", "Confidence"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4.8), layout="constrained")
        im = ax.imshow(arr, cmap=cmap, interpolation="nearest")
        ax.set_title(title); ax.set_xlabel("ROI pixel X"); ax.set_ylabel("ROI pixel Y")
        if cmap: fig.colorbar(im, ax=ax, label=label)
        for fmt in formats:
            fig.savefig(outdir / f"{Path(name).stem}.{fmt}", dpi=200)
        plt.close(fig)
