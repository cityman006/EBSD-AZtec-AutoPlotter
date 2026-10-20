"""Explicitly labelled mechanics imports; EBSD descriptors are never labels.

The input contract deliberately requires units and coordinate provenance. Values
are converted using the declared units, never inferred from their magnitudes.
The supplied affine maps physical source coordinates in micrometres to the EBSD
sample frame in micrometres (it is not a pixel registration matrix).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np

from .storage import check_cancelled, digest, transactional_output

SOURCE_TYPES = {"DIC", "HR_EBSD", "CPFE", "FEM", "EXPERIMENT"}
STRESS_FACTORS = {"Pa": 1e-6, "kPa": 1e-3, "MPa": 1., "GPa": 1e3}
STRAIN_FACTORS = {"1": 1., "%": .01, "microstrain": 1e-6}
COORD_FACTORS = {"m": 1e6, "mm": 1e3, "um": 1., "µm": 1.}


def validate_stress_provenance(metadata, stress_fields):
    """DIC measures displacement/strain, not stress without an external model.

    A stress field paired with DIC requires explicit per-field independent
    stress-source/derivation provenance; merely renaming KAM is never sufficient.
    """
    records = metadata.get("sources") or [metadata]
    for source in records:
        if source.get("source_type") != "DIC":
            continue
        for name in stress_fields:
            origin = source.get("field_provenance", {}).get(name, {})
            if (origin.get("source_type") not in SOURCE_TYPES - {"DIC"}
                    or not str(origin.get("source_id", "")).strip()
                    or not str(origin.get("method", "")).strip()
                    or not isinstance(origin.get("uncertainty"), dict)
                    or not str(origin["uncertainty"].get("description", "")).strip()):
                raise ValueError(f"DIC does not directly measure stress: {name} requires explicit independent field_provenance")


def validate_metadata(metadata):
    """Validate a user supplied mechanics contract without scientific guesses."""
    meta = dict(metadata)
    required = ("source_type", "source_id", "material", "reference_frame", "coordinate_unit",
                "load_step", "time", "loading_condition", "uncertainty",
                "field_units", "field_kinds", "source_to_sample_um")
    missing = [key for key in required if key not in meta or meta[key] is None]
    if missing:
        raise ValueError("Mechanics metadata missing: " + ", ".join(missing))
    if meta["source_type"] not in SOURCE_TYPES:
        raise ValueError("A measured or simulated mechanics source is required; EBSD proxies are not labels")
    for key in ("source_id", "material", "reference_frame", "loading_condition"):
        if not str(meta[key]).strip():
            raise ValueError(f"{key} must be explicit and nonempty")
    if not isinstance(meta["material"], str):
        raise ValueError("material must be an explicit string, never inferred from a file name")
    if meta["coordinate_unit"] not in COORD_FACTORS:
        raise ValueError("coordinate_unit must explicitly be m, mm, um, or µm")
    if not isinstance(meta["time"], dict) or not {"value", "unit"} <= meta["time"].keys():
        raise ValueError("time requires value and unit (use an explicit unknown value if unavailable)")
    uncertainty = meta["uncertainty"]
    if not isinstance(uncertainty, dict) or not str(uncertainty.get("description", "")).strip():
        raise ValueError("uncertainty requires a description; unknown uncertainty must be stated")
    fields = meta["field_units"]
    if not isinstance(fields, dict) or not fields or fields.keys() != meta["field_kinds"].keys():
        raise ValueError("field_units and field_kinds must name the same nonempty label fields")
    for name, unit in fields.items():
        if not isinstance(name, str) or name in {"x", "y", "valid", ".", ".."} or not name.strip() or "/" in name or "\\" in name:
            raise ValueError("Label names must be nonempty simple names, distinct from x, y, valid")
        kind = meta["field_kinds"][name]
        allowed = STRESS_FACTORS if kind == "stress" else STRAIN_FACTORS if kind == "strain" else {}
        if unit not in allowed:
            raise ValueError(f"Unsupported declared {kind} unit {unit!r} for {name}")
    validate_stress_provenance(meta, [name for name in fields if meta["field_kinds"][name] == "stress"])
    affine = np.asarray(meta["source_to_sample_um"], dtype=float)
    if (affine.shape != (3, 3) or not np.isfinite(affine).all()
            or not np.allclose(affine[2], [0, 0, 1]) or abs(np.linalg.det(affine[:2, :2])) < 1e-12):
        raise ValueError("source_to_sample_um must be an invertible 3x3 physical-coordinate affine")
    meta["source_to_sample_um"] = affine.tolist()
    return meta


def _read_columns(path, mapping):
    """Read explicit named CSV/HDF5 or integer NPY columns (no pickle loading)."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(mapping.values()) <= set(reader.fieldnames or []):
                raise ValueError("CSV column mapping contains missing headers")
            rows = list(reader)
        values = {}
        for key, column in mapping.items():
            try:
                values[key] = np.asarray([float(row[column]) if row[column].strip() else np.nan for row in rows])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV column {column!r} contains nonnumeric values") from exc
        return values
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as stream:
            return {key: np.asarray(stream[column], dtype=float).ravel() for key, column in mapping.items()}
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
            raise ValueError("NPY mechanics data must be a numeric N-by-F matrix")
        if any(not isinstance(column, int) or column < 0 or column >= array.shape[1] for column in mapping.values()):
            raise ValueError("NPY column mappings must be valid nonnegative integer indices")
        return {key: np.asarray(array[:, column], dtype=float) for key, column in mapping.items()}
    raise ValueError("Mechanics inputs support CSV, H5/HDF5, and numeric NPY")


@transactional_output()
def import_mechanics(path, outdir, *, metadata, columns, progress=print):
    """Archive aligned point labels and masks without extrapolating a field.

    ``columns`` maps x, y, each label field and optional valid to input columns.
    NaN labels are retained as masked values, never silently imputed to zero.
    """
    path = Path(path)
    meta = validate_metadata(metadata)
    required = {"x", "y"} | set(meta["field_units"])
    if not required <= set(columns) or set(columns) - required - {"valid"}:
        raise ValueError("columns must explicitly map x, y, and exactly the declared label fields")
    check_cancelled()
    data = _read_columns(path, columns)
    lengths = {len(value) for value in data.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("Mechanics columns must have matching nonzero lengths")
    xy = np.column_stack([data["x"], data["y"]]) * COORD_FACTORS[meta["coordinate_unit"]]
    if not np.isfinite(xy).all():
        raise ValueError("Mechanics coordinates must be finite")
    xy = np.column_stack([xy, np.ones(len(xy))]) @ np.asarray(meta["source_to_sample_um"]).T
    if "valid" in data and not np.isin(data["valid"], [0, 1]).all():
        raise ValueError("Explicit valid column must contain only 0 or 1")
    valid = data.get("valid", np.ones(len(xy))).astype(bool)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta.update({"schema_version": 1, "input_sha256": digest(path),
                 "input_name": path.name, "coordinate_output_unit": "um",
                 "label_authority": "external_measured_or_simulated",
                 "scientific_validation": "import_contract_only; source accuracy not validated"})
    counts = {}
    with h5py.File(outdir / "mechanics.h5", "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(meta, ensure_ascii=False)
        stream.create_dataset("coordinates_sample_um", data=xy[:, :2], compression="gzip")
        for name, declared_unit in meta["field_units"].items():
            check_cancelled()
            kind = meta["field_kinds"][name]
            factor = (STRESS_FACTORS if kind == "stress" else STRAIN_FACTORS)[declared_unit]
            values = data[name] * factor
            mask = valid & np.isfinite(values)
            dataset = stream.create_dataset("labels/" + name, data=values, compression="gzip")
            dataset.attrs["unit"] = "MPa" if kind == "stress" else "1"
            dataset.attrs["original_unit"] = declared_unit
            stream.create_dataset("masks/" + name, data=mask, compression="gzip")
            counts[name] = int(mask.sum())
    meta["valid_label_counts"] = counts
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    progress(f"Mechanics import complete: {len(xy)} points; {counts}")
    return outdir / "mechanics.h5"


@transactional_output()
def create_graph_training_bundle(graph_files, mechanics_files, outdir, *, target_names,
                                 max_distance_um, reference_frame, progress=print):
    """Join real M3 grain graphs to independent labels at their centroids.

    This is explicit nearest-point sampling within a physical distance bound,
    not a grain-volume average or an extrapolating field registration. Missing
    features remove a node; missing labels remain masked. Different loading
    conditions are rejected because this minimal bundle lacks load features.
    """
    from scipy.spatial import cKDTree
    graph_files, mechanics_files = list(graph_files), list(mechanics_files)
    names = list(target_names)
    if not graph_files or len(graph_files) != len(mechanics_files) or not names:
        raise ValueError("Pair every graph with one independent mechanics import and select target fields")
    if not np.isfinite(max_distance_um) or max_distance_um <= 0 or not str(reference_frame).strip():
        raise ValueError("Explicit positive max_distance_um and common reference_frame required")
    feature_units_by_name = {"pixel_count": "1", "area_um2": "um2", "equivalent_diameter_um": "um",
                             "centroid_x_um": "um", "centroid_y_um": "um", "perimeter_um": "um",
                             "aspect_ratio_second_moments": "1", "KAM_mean_deg": "deg", "GOS_deg": "deg",
                             "BC_mean_source": "source"}
    all_x, all_y, all_groups, all_edges, all_weights, all_node_ids, provenance, qc = [], [], [], [], [], [], [], []
    expected_names, target_units, expected_load, expected_phase_schema = None, None, None, None
    offset = 0
    for graph_file, mechanics_file in zip(graph_files, mechanics_files):
        check_cancelled()
        with np.load(graph_file, allow_pickle=False) as graph:
            if "metadata_json" in graph:
                graph_metadata = json.loads(str(graph["metadata_json"].item()))
            else:
                companion = Path(graph_file).parent / "graph.json"
                graph_metadata = json.loads(companion.read_text(encoding="utf-8")) if companion.is_file() else {}
            raw_schema = graph_metadata.get("phase_schema")
            if not isinstance(raw_schema, dict) or not raw_schema:
                raise ValueError("Graph lacks explicit phase_schema; legacy graph must be regenerated from its actual phase metadata")
            phase_schema = {str(key): value for key, value in raw_schema.items()}
            for phase_id, spec in phase_schema.items():
                if (not isinstance(spec, dict) or not str(spec.get("name", "")).strip()
                        or "space_group" not in spec or not str(phase_id).isdigit()
                        or int(phase_id) < 1 or int(spec["space_group"]) < 1):
                    raise ValueError("Graph phase_schema must state each original phase ID, name and crystal space_group")
            canonical_schema = json.dumps(phase_schema, sort_keys=True, ensure_ascii=False, allow_nan=False)
            if expected_phase_schema is None:
                expected_phase_schema = canonical_schema
            elif canonical_schema != expected_phase_schema:
                raise ValueError("Graph phase ID meanings or crystal metadata differ; explicit phase remapping is required before training")
            node_phases = np.asarray(graph["node_phase_id"])
            if (not np.issubdtype(node_phases.dtype, np.integer)
                    or any(str(int(pid)) not in phase_schema for pid in np.unique(node_phases))):
                raise ValueError("Graph node phase IDs are absent from its explicit phase_schema")
            feature_names = list(map(str, graph["node_feature_names"]))
            if set(feature_names) - feature_units_by_name.keys():
                raise ValueError("Graph feature units are unknown; explicit schema adaptation required")
            if expected_names is None:
                expected_names = feature_names
            elif feature_names != expected_names:
                raise ValueError("Graph feature schemas do not match")
            x = np.asarray(graph["node_features"], dtype=float)
            use = np.asarray(graph["node_feature_mask"], dtype=bool).all(axis=1) & np.isfinite(x).all(axis=1)
            source = str(graph["source_id"].item())
            centroids = x[:, [feature_names.index("centroid_x_um"), feature_names.index("centroid_y_um")]]
            q = np.asarray(graph["orientation_xyzw"], dtype=float)
            norms = np.linalg.norm(q, axis=-1)
            use &= np.isfinite(q).all(axis=1) & (norms > 1e-12)
            q = q / np.maximum(norms[:, None], 1e-12)
            for row in q:
                for value in row[::-1]:
                    if abs(value) > 1e-12:
                        row *= 1 if value > 0 else -1
                        break
            # Explicit phase and orientation descriptors supplement morphology.
            x = np.column_stack([x, node_phases, q])
            edges = np.asarray(graph["edge_index"], dtype=int)
            weights = np.asarray(graph["edge_attr"][:, 0], dtype=float)
            selected = np.flatnonzero(use)
            if not len(selected):
                raise ValueError("Graph has no nodes with all selected valid features")
            remap = np.full(len(x), -1, dtype=int)
            remap[selected] = np.arange(len(selected)) + offset
            keep_edges = use[edges[0]] & use[edges[1]]
            all_edges.append(remap[edges[:, keep_edges]])
            all_weights.append(weights[keep_edges])
            all_node_ids.append(graph["node_ids"][selected])
        with h5py.File(mechanics_file, "r") as mechanics:
            meta = json.loads(mechanics.attrs["metadata_json"])
            paired_source = str(meta.get("ebsd_source_id", meta.get("source_id", "")))
            if paired_source != source:
                raise ValueError("Mechanics specimen/source mismatch: declare ebsd_source_id matching the paired graph source_id")
            if meta["reference_frame"] != reference_frame:
                raise ValueError("Mechanics reference frame does not match the declared graph sample frame")
            if not str(meta.get("material", "")).strip():
                raise ValueError("Mechanics import lacks explicit material identity; regenerate its metadata")
            load = json.dumps({key: meta[key] for key in ("material", "load_step", "time", "loading_condition")}, sort_keys=True)
            if expected_load is None:
                expected_load = load
            elif load != expected_load:
                raise ValueError("Different loading/time conditions require explicit load features; this adapter does not invent them")
            if set(names) - set(mechanics["labels"]):
                raise ValueError("Mechanics import does not contain every selected target")
            current_units = [str(mechanics["labels/" + name].attrs["unit"]) for name in names]
            if target_units is None:
                target_units = current_units
            elif target_units != current_units:
                raise ValueError("Mechanics imports have incompatible canonical target units")
            xy = mechanics["coordinates_sample_um"][:]
            labels = np.full((len(selected), len(names)), np.nan)
            for column, name in enumerate(names):
                valid = mechanics["masks/" + name][:].astype(bool)
                values = mechanics["labels/" + name][:]
                valid &= np.isfinite(values)
                if valid.any():
                    distance, nearest = cKDTree(xy[valid]).query(centroids[selected], distance_upper_bound=max_distance_um)
                    matched = np.isfinite(distance)
                    labels[matched, column] = values[valid][nearest[matched]]
            provenance.append(meta)
        all_x.append(x[selected])
        all_y.append(labels)
        all_groups.append(np.full(len(selected), source))
        qc.append({"source_id": source, "retained_nodes": len(selected), "dropped_invalid_feature_nodes": int((~use).sum()),
                   "valid_labels_per_target": np.isfinite(labels).sum(axis=0).tolist()})
        offset += len(selected)
        progress(f"Mechanics-to-graph join: {source}, {len(selected)} nodes")
    metadata = {"feature_names": expected_names + ["phase_id", "mean_qx", "mean_qy", "mean_qz", "mean_qw"],
                "feature_units": [feature_units_by_name[name] for name in expected_names] + ["1"] * 5,
                "target_names": names, "target_units": target_units, "synthetic_demo": False,
                "phase_schema": json.loads(expected_phase_schema),
                "label_metadata": {**provenance[0], "source_id": "multi-source paired graph/mechanics dataset", "sources": provenance,
                                   "phase_schema": json.loads(expected_phase_schema)},
                "label_join": {"method": "nearest independently registered mechanics point at grain centroid",
                               "max_distance_um": float(max_distance_um), "reference_frame": reference_frame,
                               "not_a_grain_volume_average": True}}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "training_bundle.npz", features=np.concatenate(all_x), labels=np.concatenate(all_y),
                        label_mask=np.isfinite(np.concatenate(all_y)), groups=np.concatenate(all_groups),
                        edge_index=np.concatenate(all_edges, axis=1), edge_weights=np.concatenate(all_weights),
                        original_node_ids=np.concatenate(all_node_ids), metadata_json=json.dumps(metadata))
    (outdir / "join_qc.json").write_text(json.dumps({"sources": qc, **metadata["label_join"]}, indent=2), encoding="utf-8")
    return outdir / "training_bundle.npz"
