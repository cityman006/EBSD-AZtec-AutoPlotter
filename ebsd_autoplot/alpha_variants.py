"""Burgers parent/child orientation tools with explicit ambiguity retention."""
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
from .v3_crystal import ORIENTATION_CONVENTION, disorientation_deg, point_group, normalize_quaternions


def burgers_variant_orientations(c_over_a=1.0):
    """Return 12 unique child-to-parent orientations for ideal Burgers OR.

    The alpha c-axis is parallel to beta {110}; alpha [11-20] is parallel to a
    beta <111> direction lying in that plane. Each beta cubic symmetry action
    is reduced by the alpha hexagonal point group.
    """
    if c_over_a <= 0:
        raise ValueError("c/a must be positive")
    n_beta = np.array([1., 1., 0.]); n_beta /= np.linalg.norm(n_beta)
    d_beta = np.array([1., -1., 1.]); d_beta /= np.linalg.norm(d_beta)
    if abs(float(n_beta @ d_beta)) > 1e-10:
        raise AssertionError("Burgers beta plane/direction are not perpendicular")
    x_alpha = np.array([0.5, np.sqrt(3) / 2, 0.])
    z_alpha = np.array([0., 0., 1.])
    y_alpha = np.cross(z_alpha, x_alpha)
    alpha_basis = np.column_stack((x_alpha, y_alpha, z_alpha))
    y_beta = np.cross(n_beta, d_beta)
    beta_basis = np.column_stack((d_beta, y_beta, n_beta))
    base = Rotation.from_matrix(beta_basis @ alpha_basis.T)
    cubic = point_group(229)
    candidates = [g * base for g in cubic]
    unique = []
    for q in candidates:
        quat = q.as_quat()
        if not any(disorientation_deg(quat, old.as_quat(), 194) < 1e-5 for old in unique):
            unique.append(q)
    unique.sort(key=lambda r: tuple(np.round(r.as_quat(), 9)))
    if len(unique) != 12:
        raise RuntimeError(f"Burgers symmetry reduction produced {len(unique)} variants; expected 12")
    return unique


def burgers_geometry_check(c_over_a=1.0):
    """Return measurable parallelism and variant-pair-angle evidence."""
    variants = burgers_variant_orientations(c_over_a)
    n = np.array([1., 1., 0.]); n /= np.linalg.norm(n)
    d = np.array([1., -1., 1.]); d /= np.linalg.norm(d)
    alpha_n = np.array([0., 0., 1.])
    alpha_d = np.array([0.5, np.sqrt(3) / 2, 0.])
    # The base relation was constructed to satisfy n_beta || c_alpha and
    # d_beta || [11-20]_alpha. Verify the geometry independently.
    # Rebuild the reference relation without symmetry sorting.
    n_beta = np.array([1., 1., 0.]); n_beta /= np.linalg.norm(n_beta)
    d_beta = np.array([1., -1., 1.]); d_beta /= np.linalg.norm(d_beta)
    x_alpha = np.array([0.5, np.sqrt(3) / 2, 0.])
    alpha_basis = np.column_stack((x_alpha, np.cross([0., 0., 1.], x_alpha), [0., 0., 1.]))
    beta_basis = np.column_stack((d_beta, np.cross(n_beta, d_beta), n_beta))
    base = Rotation.from_matrix(beta_basis @ alpha_basis.T)
    got_c = base.apply([0., 0., 1.])
    got_a = base.apply(x_alpha)
    angle_c = float(np.degrees(np.arccos(np.clip(abs(got_c @ n_beta), -1, 1))))
    angle_a = float(np.degrees(np.arccos(np.clip(abs(got_a @ d_beta), -1, 1))))
    angles = sorted({round(disorientation_deg(variants[i].as_quat(), variants[j].as_quat(), 194), 3)
                     for i in range(12) for j in range(i + 1, 12)})
    return {
        "relationship": "{110}_beta // {0001}_alpha; <111>_beta // <11-20>_alpha",
        "parent_to_child_operation": "q_alpha = q_beta * v_i; v_i maps alpha crystal Cartesian to beta crystal Cartesian",
        "beta_plane_alpha_c_misalignment_deg": angle_c,
        "beta_direction_alpha_a_misalignment_deg": angle_a,
        "variant_count": len(variants),
        "unique_alpha_alpha_disorientation_angles_deg": angles,
        "orientation_convention": ORIENTATION_CONVENTION,
        "note": "Twelve child orientations are symmetry-distinct for one fixed parent; one child alone does not determine a unique parent.",
    }


def candidate_parent_orientations(alpha_q, variants):
    """Invert q_alpha=q_beta*v_i to return beta-to-sample candidates."""
    q = Rotation.from_quat(normalize_quaternions(np.asarray(alpha_q).reshape(1, 4))[0])
    return [q * variant.inv() for variant in variants]


def assign_alpha_grains(grains, edges, *, parent_threshold_deg=5.0,
                        beta_anchors=None, beta_anchor_threshold_deg=5.0,
                        alpha_phase_id=1, beta_phase_id=2):
    """Graph-supported candidate voting; isolated/ambiguous grains remain unknown."""
    beta_anchors = beta_anchors or {}
    variants = burgers_variant_orientations()
    alpha = {int(g["grain_id"]): g for g in grains if int(g["phase_id"]) == int(alpha_phase_id)}
    beta = {int(g["grain_id"]): g for g in grains if int(g["phase_id"]) == int(beta_phase_id)}
    candidates = {gid: candidate_parent_orientations(g["q"], variants) for gid, g in alpha.items()}
    neighbors = {gid: [] for gid in alpha}
    for a, b, length in edges:
        if a in alpha and b in alpha:
            neighbors[a].append((b, float(length)))
            neighbors[b].append((a, float(length)))
        elif a in alpha and b in beta:
            neighbors[a].append((b, float(length)))
        elif b in alpha and a in beta:
            neighbors[b].append((a, float(length)))
    output = {}
    for gid, grain in alpha.items():
        check_cancelled()
        scores = np.zeros(12, dtype=np.float64)
        residual_sums = np.zeros(12, dtype=np.float64)
        support_neighbors = [set() for _ in range(12)]
        anchored = np.zeros(12, dtype=bool)
        for other, shared in neighbors[gid]:
            if other in beta:
                measured = beta_anchors.get(other)
                if measured is None:
                    continue
                errs = np.array([disorientation_deg(q.as_quat(), measured, 229) for q in candidates[gid]])
                k = int(np.argmin(errs))
                if errs[k] <= beta_anchor_threshold_deg:
                    scores[k] += shared * 4
                    residual_sums[k] += shared * errs[k]
                    anchored[k] = True
                continue
            for k, candidate in enumerate(candidates[gid]):
                errs = [disorientation_deg(candidate.as_quat(), q.as_quat(), 229) for q in candidates[other]]
                best = float(np.min(errs))
                if best <= parent_threshold_deg:
                    scores[k] += shared
                    residual_sums[k] += shared * best
                    support_neighbors[k].add(other)
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        total_support = len(support_neighbors[best])
        residual = residual_sums[best] / max(scores[best], 1e-12)
        second = scores[order[1]] if len(order) > 1 else 0
        separation = (scores[best] - second) / max(scores[best], 1e-12) if scores[best] > 0 else 0.0
        enough_evidence = anchored[best] or total_support >= 2
        confidence = float(np.clip((scores[best] / max(scores[best] + second, 1e-12)) *
                                   np.exp(-residual / max(parent_threshold_deg, 1e-6)) *
                                   min(1.0, (total_support + int(anchored[best])) / 2), 0, 1))
        assigned = bool(enough_evidence and scores[best] > 0 and confidence >= 0.35)
        output[gid] = {
            "grain_id": gid, "variant_id": best + 1 if assigned else 0,
            "parent_q": candidates[gid][best].as_quat() if assigned else None,
            "parent_candidate_q": candidates[gid][best].as_quat(),
            "fit_deg": residual if scores[best] > 0 else float("nan"),
            "confidence": confidence if assigned else 0.0,
            "status": "assigned" if assigned else ("ambiguous" if scores[best] > 0 else "unknown"),
            "supporting_alpha_neighbors": total_support,
            "beta_anchor_used": bool(anchored[best]),
            "candidate_score_margin": float(separation),
        }
    # Parent IDs require graph continuity as well as close parent orientations.
    ids = sorted(gid for gid, row in output.items() if row["status"] == "assigned")
    parent = {gid: gid for gid in ids}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b, _length in edges:
        if a in parent and b in parent:
            if disorientation_deg(output[a]["parent_q"], output[b]["parent_q"], 229) <= parent_threshold_deg:
                ra, rb = root(a), root(b)
                if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    for gid in ids: output[gid]["parent_id"] = root(gid)
    for gid in output:
        output[gid].setdefault("parent_id", 0)
    # Average parent orientation within graph components after symmetry
    # alignment, then compute a genuine child-vs-parent OR residual.
    sym = point_group(229)
    for parent_id in sorted({output[gid]["parent_id"] for gid in ids}):
        members = [gid for gid in ids if output[gid]["parent_id"] == parent_id]
        ref = normalize_quaternions(np.asarray(output[members[0]]["parent_q"]).reshape(1, 4))[0]
        aligned = []
        for gid in members:
            qrot = Rotation.from_quat(output[gid]["parent_q"])
            equivalents = (qrot * sym).as_quat()
            dots = equivalents @ ref
            q = equivalents[int(np.argmax(np.abs(dots)))]
            if q @ ref < 0: q = -q
            aligned.append(q)
        mean = np.sum(aligned, axis=0); mean /= np.linalg.norm(mean)
        parent_rot = Rotation.from_quat(mean)
        for gid in members:
            vid = output[gid]["variant_id"]
            predicted = parent_rot * variants[vid - 1]
            observed = Rotation.from_quat(alpha[gid]["q"])
            output[gid]["fit_deg"] = float(disorientation_deg(predicted.as_quat(), observed.as_quat(), 194))
            output[gid]["parent_q"] = mean
    return output


@transactional_output()
def analyze_alpha_variants(features_file, v2_dir, outdir, *, parent_threshold_deg=5.0,
                           beta_anchor_threshold_deg=5.0, formats=("png", "pdf", "svg")):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    formats = tuple(dict.fromkeys(str(x).lower().lstrip(".") for x in formats))
    if not formats or set(formats) - {"png", "tiff", "pdf", "svg"}:
        raise ValueError("Variant formats must be PNG, TIFF, PDF, or SVG")
    v2_dir = Path(v2_dir)
    with h5py.File(features_file, "r") as f, h5py.File(v2_dir / "v2_grains.h5", "r") as vf:
        meta = json.loads(f.attrs["metadata_json"])
        specs = {int(k): v for k, v in meta["phase_specs"].items()}
        alpha_ids = [pid for pid, s in specs.items() if s["space_group"] == 194 and
                     any(k in s["name"].casefold() for k in ("alpha", "α", "hcp", "titanium-hex"))]
        beta_ids = [pid for pid, s in specs.items() if s["space_group"] == 229 and
                    any(k in s["name"].casefold() for k in ("beta", "β", "bcc", "titanium-cub"))]
        if len(alpha_ids) != 1 or len(beta_ids) > 1:
            raise ValueError("Variant analysis needs one named HCP α phase and at most one named BCC β phase")
        roi = tuple(map(int, vf.attrs.get("roi_xywh", (0, 0, *f["indexed"].shape[::-1]))))
        x, y, rw, rh = roi
        grain_map = vf["grain_id"][:]
        grain_rows = []
        with (v2_dir / "grains.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                row["grain_id"] = int(row["grain_id"]); row["phase_id"] = int(row["phase_id"])
                row["q"] = np.array([float(row[f"mean_q{k}"]) for k in ("x", "y", "z", "w")])
                row["pixels"] = int(row["pixels"])
                # H5 raster phase IDs are authoritative. V2 grain records stay in
                # their original ROI-local identifiers and coordinates.
                if row["phase_id"] in (alpha_ids[0], *(beta_ids or [])):
                    grain_rows.append(row)
        edge_rows = []
        with (v2_dir / "grain_neighbour_graph.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                edge_rows.append((int(row["grain_a"]), int(row["grain_b"]), float(row["shared_pixel_edges"])))
        anchors = {}
        if beta_ids:
            beta_id = beta_ids[0]
            for row in grain_rows:
                if row["phase_id"] == beta_id:
                    anchors[row["grain_id"]] = row["q"]
        assigned = assign_alpha_grains(grain_rows, edge_rows,
                                       parent_threshold_deg=parent_threshold_deg,
                                       beta_anchors=anchors,
                                       beta_anchor_threshold_deg=beta_anchor_threshold_deg,
                                       alpha_phase_id=alpha_ids[0],
                                       beta_phase_id=beta_ids[0] if beta_ids else -1)
        vmap = np.zeros(grain_map.shape, np.uint8)
        fitmap = np.full(grain_map.shape, np.nan, np.float32)
        confidence = np.zeros(grain_map.shape, np.float32)
        parent_map = np.zeros(grain_map.shape, np.int32)
        for gid, row in assigned.items():
            mask = grain_map == gid
            vmap[mask] = row["variant_id"]
            fitmap[mask] = row["fit_deg"]
            confidence[mask] = row["confidence"]
            parent_map[mask] = row["parent_id"]
        boundary_type = _variant_boundary_types(grain_map, parent_map)
        with h5py.File(outdir / "alpha_variant_fields.h5", "w") as output:
            output.attrs["schema_version"] = 1; output.attrs["roi_xywh"] = roi
            output.attrs["orientation_convention"] = ORIENTATION_CONVENTION
            output.attrs["parent_child_relation"] = "q_alpha = q_beta * v_i"
            output.attrs["variant_boundary_type_codes"] = json.dumps({
                "0": "no boundary", "1": "same inferred parent, distinct alpha variants",
                "2": "different inferred parent candidates", "3": "unknown/unassigned side"})
            output.create_dataset("variant_id_map", data=vmap, compression="gzip")
            output.create_dataset("variant_fit_map_deg", data=fitmap, compression="gzip")
            output.create_dataset("confidence_map", data=confidence, compression="gzip")
            output.create_dataset("parent_id_map", data=parent_map, compression="gzip")
            output.create_dataset("variant_boundary_type_map", data=boundary_type, compression="gzip")
        rows = []
        for gid, row in assigned.items():
            rows.append({k: (json.dumps(v.tolist()) if isinstance(v, np.ndarray) else v)
                         for k, v in row.items()})
        with (outdir / "alpha_variant_grains.csv").open("w", encoding="utf-8", newline="") as stream:
            cols = ["grain_id", "variant_id", "parent_id", "fit_deg", "confidence", "status",
                    "supporting_alpha_neighbors", "beta_anchor_used", "candidate_score_margin",
                    "parent_candidate_q", "parent_q"]
            writer = csv.DictWriter(stream, fieldnames=cols); writer.writeheader(); writer.writerows(rows)
        links = []
        for a, b, length in edge_rows:
            if a in assigned and b in assigned:
                links.append({"grain_a": a, "grain_b": b, "shared_pixel_edges": length,
                              "parent_a": assigned[a]["parent_id"], "parent_b": assigned[b]["parent_id"],
                              "same_assigned_parent": bool(assigned[a]["parent_id"] and
                                                           assigned[a]["parent_id"] == assigned[b]["parent_id"]),
                              "variant_boundary_type": ("same_parent_variant_boundary" if assigned[a]["parent_id"] and
                                  assigned[a]["parent_id"] == assigned[b]["parent_id"] else
                                  "inferred_parent_boundary" if assigned[a]["parent_id"] and assigned[b]["parent_id"] else
                                  "unknown_unassigned")})
        with (outdir / "alpha_parent_links.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(links[0]) if links else
                                    ["grain_a", "grain_b", "shared_pixel_edges", "parent_a", "parent_b", "same_assigned_parent"])
            writer.writeheader(); writer.writerows(links)
    summary = {
        "schema_version": 1, "variant_count_ideal": len(burgers_variant_orientations()),
        "assignment_rule": "graph/anchor-supported; one isolated alpha grain remains unknown",
        "parent_threshold_deg": float(parent_threshold_deg),
        "beta_anchor_threshold_deg": float(beta_anchor_threshold_deg),
        "assigned_grains": sum(row["status"] == "assigned" for row in assigned.values()),
        "ambiguous_grains": sum(row["status"] == "ambiguous" for row in assigned.values()),
        "unknown_grains": sum(row["status"] == "unknown" for row in assigned.values()),
        "scientific_status": "inferred; not MTEX-validated; single-alpha parent is not unique",
        "roi_xywh": list(roi),
    }
    (outdir / "BOR_orientation_test.json").write_text(json.dumps(burgers_geometry_check(), indent=2), encoding="utf-8")
    (outdir / "variant_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _render_variant_maps(outdir, vmap, fitmap, confidence, formats)
    return summary


def _render_variant_maps(outdir, variants, fit, confidence, formats):
    for name, arr, cmap, title, label in (
        ("variant_id_map.png", variants, "tab20", "α child variant ID (0=unknown)", "Variant ID"),
        ("variant_fit_map.png", fit, "magma", "Burgers OR fit | angle (°)", "Misfit (°)"),
        ("variant_confidence_map.png", confidence, "viridis", "Inferred assignment confidence", "Confidence (0–1)"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4.8), layout="constrained")
        image = ax.imshow(arr, cmap=cmap, interpolation="nearest")
        ax.set_title(title + "\nExploratory/inferred; reference validation pending")
        ax.set_xlabel("ROI pixel X"); ax.set_ylabel("ROI pixel Y")
        fig.colorbar(image, ax=ax, label=label)
        for fmt in formats:
            fig.savefig(outdir / f"{Path(name).stem}.{fmt}", dpi=200)
        plt.close(fig)


def _variant_boundary_types(grain_ids, parent_ids):
    """Label both sides of each V2 grain edge using assigned parent evidence."""
    out = np.zeros(grain_ids.shape, dtype=np.uint8)
    for slices_a, slices_b in (( (slice(None), slice(None, -1)), (slice(None), slice(1, None)) ),
                               ( (slice(None, -1), slice(None)), (slice(1, None), slice(None)) )):
        ga, gb = grain_ids[slices_a], grain_ids[slices_b]
        edge = (ga > 0) & (gb > 0) & (ga != gb)
        pa, pb = parent_ids[slices_a], parent_ids[slices_b]
        known = edge & (pa > 0) & (pb > 0)
        same = known & (pa == pb)
        type_a = np.where(same, 1, np.where(known & ~same, 2, np.where(edge, 3, 0))).astype(np.uint8)
        target_a, target_b = out[slices_a], out[slices_b]
        target_a[:] = np.maximum(target_a, type_a)
        target_b[:] = np.maximum(target_b, type_a)
    return out
