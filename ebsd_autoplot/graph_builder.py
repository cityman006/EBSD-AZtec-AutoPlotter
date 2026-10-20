"""Grain adjacency graphs derived from real V2 labels on a physical raster.

Edges represent touching 4-neighbour pixels; this is a 2D section graph, not a
3D grain-contact reconstruction. Disorientation uses existing phase symmetry.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import h5py
import numpy as np
from .crystal import misorientation_deg
from .v2 import _mean_quat_sym
from .storage import check_cancelled, transactional_output


@transactional_output()
def build_graph(features_file, v2_dir, outdir, *, source_id=None, preview=True):
    """Export graph.npz, graph.json, graph_qc.json, optional graph_preview.png.

    Uses every nonzero V2 grain label (including small grains absent from its
    CSV). Node rows map explicitly to original IDs. Edge index is symmetric,
    has no self loops, and shared_boundary_um uses dy for left/right pixel
    neighbours and dx for up/down neighbours. Interphase misorientation is NaN
    with a false validity mask; no numerical crystallographic value is invented.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    with h5py.File(features_file, "r") as src, h5py.File(Path(v2_dir) / "v2_grains.h5", "r") as vf:
        meta = json.loads(src.attrs["metadata_json"])
        from .storage import digest
        pair_hash=vf.attrs.get('source_features_sha256')
        if pair_hash and pair_hash!=digest(features_file):
            raise ValueError('V2 grain cache belongs to a different features file')
        specs = {int(k): v for k, v in meta["phase_specs"].items()}
        dx, dy = float(meta["x_step_um"]), float(meta["y_step_um"])
        if not np.isfinite([dx, dy]).all() or min(dx, dy) <= 0:
            raise ValueError("Graph geometry requires measured positive x/y steps in µm")
        fh, fw = src["indexed"].shape
        x, y, w, h = tuple(map(int, vf.attrs["roi_xywh"]))
        if min(x, y) < 0 or min(w, h) < 1 or x + w > fw or y + h > fh:
            raise ValueError("V2 graph ROI lies outside features")
        ids = vf["grain_id"][:]
        if ids.shape != (h, w) or np.any(ids < 0) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("V2 grain labels must be nonnegative integer IDs matching ROI")
        gids = np.unique(ids[ids > 0]).astype(np.int64)
        if not len(gids):
            raise ValueError("V2 contains no real grain IDs")
        lookup = {int(g): i for i, g in enumerate(gids)}
        sl = (slice(y, y + h), slice(x, x + w))
        features = src["features"][:, sl[0], sl[1]]
        indexed = src["indexed"][sl].astype(bool)
        if np.any((ids > 0) & ~indexed):
            raise ValueError("V2 assigns grain IDs to unindexed source pixels")
        phase = np.rint(features[4]).astype(int)
        kam_valid = src["kam_valid"][sl].astype(bool)
        q = np.moveaxis(features[:4], 0, -1)
        edge_lengths = {}; shared_pixels = {}
        for a, b, length in ((ids[:, :-1], ids[:, 1:], dy), (ids[:-1], ids[1:], dx)):
            use = (a > 0) & (b > 0) & (a != b)
            pairs = np.sort(np.stack((a[use], b[use]), axis=1), axis=1)
            if len(pairs):
                uniq, counts = np.unique(pairs, axis=0, return_counts=True)
                for pair, count in zip(uniq.tolist(), counts.tolist()):
                    key = tuple(pair)
                    edge_lengths[key] = edge_lengths.get(key, 0.) + count * length
                    shared_pixels[key] = shared_pixels.get(key, 0) + count
        # Exact grid perimeter includes ROI exterior and contact with invalid pixels.
        perimeter = {int(g): 0. for g in gids}
        for a, b, length in ((ids[:, :-1], ids[:, 1:], dy), (ids[:-1], ids[1:], dx)):
            for side in (a[a != b], b[a != b]):
                ug, count = np.unique(side[side > 0], return_counts=True)
                for gid, c in zip(ug.tolist(), count.tolist()):
                    perimeter[gid] += c * length
        for side, length in ((ids[:, 0], dy), (ids[:, -1], dy), (ids[0], dx), (ids[-1], dx)):
            ug, counts = np.unique(side[side > 0], return_counts=True)
            for gid, count in zip(ug.tolist(), counts.tolist()):
                perimeter[gid] += count * length
        names = ["pixel_count", "area_um2", "equivalent_diameter_um", "centroid_x_um", "centroid_y_um",
                 "perimeter_um", "aspect_ratio_second_moments", "KAM_mean_deg", "GOS_deg", "BC_mean_source"]
        rows = []; node_phase = []; orientations = []; border = []
        # One sort avoids rescanning the entire raster for every grain.
        valid_pixels = np.flatnonzero(ids.ravel() > 0)
        sorted_pixels = valid_pixels[np.argsort(ids.ravel()[valid_pixels], kind="stable")]
        _, counts = np.unique(ids.ravel()[sorted_pixels], return_counts=True)
        offsets = np.r_[0, np.cumsum(counts)]
        for row_index, gid in enumerate(gids):
            check_cancelled()
            pix = sorted_pixels[offsets[row_index]:offsets[row_index + 1]]
            yy, xx = np.divmod(pix, w)
            phases = np.unique(phase[yy, xx])
            if len(phases) != 1 or int(phases[0]) not in specs or phases[0] == 0:
                raise ValueError(f"Grain {gid} is mixed-phase or lacks phase metadata")
            pid = int(phases[0]); sg = int(specs[pid]["space_group"])
            qq = q[yy, xx]
            if not np.isfinite(qq).all() or np.any(np.linalg.norm(qq, axis=-1) < 1e-12):
                raise ValueError(f"Grain {gid} has invalid indexed orientations")
            avg = _mean_quat_sym(qq, sg)
            dist = misorientation_deg(qq, np.broadcast_to(avg, qq.shape), np.full(len(xx), pid), np.full(len(xx), pid), specs)
            physical = np.column_stack(((xx + x) * dx, (yy + y) * dy))
            # Include individual raster cell second moments (nonzero one-pixel grain).
            centred = physical - physical.mean(0)
            covariance = centred.T @ centred / len(xx) + np.diag([dx*dx/12, dy*dy/12])
            eigenvalues = np.linalg.eigvalsh(covariance)
            aspect = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
            km = features[5, yy, xx]; kval = kam_valid[yy, xx] & np.isfinite(km)
            bc = features[6, yy, xx]; bcval = np.isfinite(bc)
            area = len(xx) * dx * dy
            rows.append([len(xx), area, 2 * np.sqrt(area / np.pi), *physical.mean(0),
                         perimeter[int(gid)], aspect, float(km[kval].mean()) if np.any(kval) else np.nan,
                         float(dist.mean()), float(bc[bcval].mean()) if np.any(bcval) else np.nan])
            node_phase.append(pid); orientations.append(avg)
            border.append(bool(np.any((xx == 0) | (xx == w - 1) | (yy == 0) | (yy == h - 1))))
        nodes = np.asarray(rows, float); node_phase = np.asarray(node_phase, int)
        orientations = np.asarray(orientations, np.float32)
        edges = []; attributes = []
        for (ga, gb), length in sorted(edge_lengths.items()):
            ia, ib = lookup[ga], lookup[gb]
            angle = float(misorientation_deg(orientations[ia:ia + 1], orientations[ib:ib + 1],
                                            node_phase[ia:ia + 1], node_phase[ib:ib + 1], specs)[0])
            distance = float(np.linalg.norm(nodes[ia, 3:5] - nodes[ib, 3:5]))
            attr = [length, shared_pixels[(ga, gb)], angle, distance, int(node_phase[ia] != node_phase[ib])]
            edges.extend(((ia, ib), (ib, ia))); attributes.extend((attr, attr))
        edge_index = np.asarray(edges, np.int64).reshape(-1, 2).T
        edge_attr = np.asarray(attributes, float).reshape(-1, 5)
        degrees = np.bincount(edge_index[0], minlength=len(gids))
        # Components operate on node rows, independently of arbitrary grain IDs.
        parent = np.arange(len(gids))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for ia, ib in edge_index.T:
            parent[root(ib)] = root(ia)
        components = len({root(i) for i in range(len(gids))})
        source_identity = str(meta.get("source_file", meta.get("source", meta.get("path", Path(features_file).resolve())))) + "|" + str(meta.get("scan_path", ""))
        source_id = str(source_id or hashlib.sha256(source_identity.encode()).hexdigest()[:20])
        np.savez_compressed(outdir / "graph.npz", node_ids=gids, node_phase_id=node_phase,
                            node_features=nodes, node_feature_mask=np.isfinite(nodes),
                            node_feature_names=np.asarray(names), orientation_xyzw=orientations,
                            edge_index=edge_index, edge_attr=edge_attr, edge_feature_mask=np.isfinite(edge_attr),
                            edge_feature_names=np.asarray(["shared_boundary_um", "shared_pixel_edges", "disorientation_deg", "centroid_distance_um", "interphase"]),
                            boundary_censored=np.asarray(border), source_id=np.asarray(source_id),
                            metadata_json=np.asarray(json.dumps({'phase_schema':meta['phase_specs']})))
        qc = {"schema_version": 1, "nodes": len(gids), "undirected_edges": len(edge_lengths),
              "cache_pair_identity": 'verified_sha256' if pair_hash else 'unverified_legacy_no_source_hash',
              "segmentation_validation": 'exploratory_external_validation_pending',
              "scientifically_validated": False,
              "directed_edges": len(edges), "self_loops": 0, "isolated_nodes": int(np.count_nonzero(degrees == 0)),
              "connected_components": components, "degree_min": int(degrees.min()), "degree_max": int(degrees.max()),
              "degree_mean": float(degrees.mean()), "physical_length_unit": "µm", "symmetric_edge_index": True,
              "node_ids": "unaltered V2 labels; includes small grains", "edge_rule": "4-neighbour shared raster boundaries",
              "nonfinite_edge_angles": int(np.count_nonzero(~np.isfinite(edge_attr[:, 2]))),
              "scientific_status": "2D experimental grain adjacency; no mechanical prediction or 3D topology claimed"}
        manifest = {"schema_version": 1, "source_id": source_id, "parent_id": source_id,
                    "phase_schema":meta['phase_specs'],
                    "roi_xywh": [x, y, w, h], "step_um_xy": [dx, dy], "node_feature_names": names,
                    "orientation_convention": "symmetry-aligned mean quaternion xyzw, crystal-to-sample",
                    "graph": "graph.npz", "qc": qc,
                    "area_check_um2": float(nodes[:, 1].sum()),
                    "indexed_label_pixels": int(np.count_nonzero(ids)),
                    "boundary_length_method": "vertical pixel face dy; horizontal pixel face dx",
                    "missing_value_policy": "NaN with false mask; interphase disorientation undefined"}
        (outdir / "graph.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (outdir / "graph_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
        if preview:
            _preview(nodes, node_phase, edge_index, dx, dy, outdir)
        return manifest


def _preview(nodes, phases, edges, dx, dy, outdir):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    xy = nodes[:, 3:5]
    segments = xy[edges.T[::2]] if edges.shape[1] else []
    ax.add_collection(LineCollection(segments, colors="0.6", linewidths=.7))
    points = ax.scatter(xy[:, 0], xy[:, 1], c=phases, s=15, cmap="tab20")
    ax.set(xlabel="Scan X (µm)", ylabel="Scan Y (µm)", title="2D grain adjacency: real V2 labels")
    ax.autoscale(); ax.set_aspect("equal"); ax.invert_yaxis()
    fig.colorbar(points, ax=ax, label="Phase ID")
    fig.savefig(outdir / "graph_preview.png", dpi=150); plt.close(fig)
