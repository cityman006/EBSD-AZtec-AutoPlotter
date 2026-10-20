"""Auditable CPU supervised research training, grouped evaluation and inference.

No model in this module establishes an EBSD-to-stress physical law. Training
requires separately measured/simulated labels, masks and specimen-level groups.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .mechanics import SOURCE_TYPES, validate_stress_provenance
from .ml_models import GrainGraphRegressor, RidgeRegressor, graph_design
from .storage import cancellation, check_cancelled, digest, transactional_output

SCHEMA_VERSION = 1


def validate_checkpoint_archive(modeldir, required_files, *, required=False):
    """Only complete, digest-verified checkpoint publications may be resumed.

    Older external checkpoints without transaction records retain compatibility;
    newly produced checkpoints mark themselves as requiring a committed receipt.
    """
    modeldir = Path(modeldir)
    records = list((modeldir / ".transactions").glob("*.json"))
    if not records:
        if required:
            raise ValueError("Checkpoint lacks its committed transaction receipt")
        return
    required_files = set(required_files)
    for record in records:
        plan = json.loads(record.read_text(encoding="utf-8"))
        if plan.get("status") != "committed":
            continue
        artifacts = {item["target"]: item for item in plan.get("files", [])}
        if not required_files <= artifacts.keys():
            continue
        for relative in artifacts:
            file = (modeldir / relative).resolve()
            if modeldir.resolve() not in file.parents or not file.is_file() or digest(file) != artifacts[relative]["sha256"]:
                raise ValueError("Checkpoint content differs from its committed receipt: " + relative)
        return
    raise ValueError("Checkpoint publication is incomplete or missing required files")


@transactional_output()
def _write_training_checkpoint(outdir, *, config, state, epoch_metrics):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.savez_compressed(outdir / "checkpoint.npz", **state)
    (outdir / "epoch_metrics.json").write_text(json.dumps(epoch_metrics, indent=2), encoding="utf-8")
    return outdir / "checkpoint.npz"


def group_split(groups, *, seed=0, test_fraction=.2, validation_fraction=.2):
    """Partition whole parent specimens; a source may never cross partitions."""
    groups = np.asarray(groups).astype(str)
    if groups.ndim != 1 or not len(groups) or np.any(np.char.strip(groups) == ""):
        raise ValueError("Explicit nonempty specimen/source group IDs are required")
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("At least 3 independent source groups are required for train/validation/test")
    if not (0 < test_fraction < 1 and 0 < validation_fraction < 1 and test_fraction + validation_fraction < 1):
        raise ValueError("Invalid group split fractions")
    unique = np.random.default_rng(seed).permutation(unique)
    n_test = max(1, int(round(len(unique) * test_fraction)))
    n_val = max(1, int(round(len(unique) * validation_fraction)))
    if n_test + n_val >= len(unique):
        n_test = n_val = 1
    selected = {"test": unique[:n_test], "validation": unique[n_test:n_test + n_val],
                "train": unique[n_test + n_val:]}
    return {key: np.flatnonzero(np.isin(groups, value)) for key, value in selected.items()}


def _validate_provenance(metadata, *, demo):
    metadata = dict(metadata or {})
    if demo:
        if metadata.get("source_type") != "SYNTHETIC":
            raise ValueError("Demo training must declare source_type SYNTHETIC")
    elif metadata.get("source_type") not in SOURCE_TYPES:
        raise ValueError("Training requires independently sourced mechanics labels; EBSD proxies are not labels")
    required = ("source_id", "material", "load_step", "time", "loading_condition", "uncertainty", "reference_frame")
    if any(key not in metadata or metadata[key] is None or metadata[key] == "" for key in required):
        raise ValueError("Label provenance requires source_id, material, load_step, time, loading_condition, uncertainty and reference_frame")
    if not isinstance(metadata["time"], dict) or not {"value", "unit"} <= metadata["time"].keys():
        raise ValueError("Label time must include value and unit")
    if not isinstance(metadata["uncertainty"], dict) or not metadata["uncertainty"].get("description"):
        raise ValueError("Label uncertainty requires an explicit description")
    if any(not isinstance(metadata[key], str) or not metadata[key].strip()
           for key in ("material", "loading_condition", "reference_frame")):
        raise ValueError("Explicit material, loading_condition and reference_frame strings are required")
    if metadata.get("sources"):
        if not isinstance(metadata["sources"], list):
            raise ValueError("Nested label sources must be a list of complete provenance records")
        for source in metadata["sources"]:
            if not isinstance(source, dict) or source.get("sources"):
                raise ValueError("Nested source records must be flat explicit provenance objects")
            _validate_provenance(source, demo=demo)
            if any(source[key] != metadata[key] for key in ("material", "loading_condition", "reference_frame", "load_step", "time")):
                raise ValueError("Mixed material/loading/time/frame sources require an explicit conditional model; this baseline rejects them")
    return metadata


def _array_hash(*arrays):
    result = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        result.update(str((array.dtype.str, array.shape)).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def _metrics(y, prediction, mask, names, units, thresholds, interval):
    result = {}
    for index, name in enumerate(names):
        valid = mask[:, index]
        if not valid.any():
            result[name] = {"label_count": 0, "status": "no_labels"}
            continue
        truth, pred = y[valid, index], prediction[valid, index]
        error = pred - truth
        item = {"label_count": int(valid.sum()), "unit": units[index],
                "MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(np.mean(error ** 2))),
                "observed_max": float(truth.max()), "predicted_max": float(pred.max()),
                "absolute_peak_error": float(abs(truth.max() - pred.max())),
                "validation_residual_interval_halfwidth": float(interval[index]),
                "interval_coverage": float(np.mean(np.abs(error) <= interval[index]))}
        if units[index] == "1":
            true_high, pred_high = np.abs(truth) >= thresholds[index], np.abs(pred) >= thresholds[index]
            intersection = np.count_nonzero(true_high & pred_high)
            union = np.count_nonzero(true_high | pred_high)
            item["high_strain"] = {"threshold_from_training": float(thresholds[index]),
                                  "IoU": float(intersection / union) if union else None,
                                  "precision": float(intersection / pred_high.sum()) if pred_high.any() else None,
                                  "recall": float(intersection / true_high.sum()) if true_high.any() else None}
        result[name] = item
    return result


def _validate_arrays(features, labels, groups, label_mask):
    x = np.asarray(features, dtype=float)
    if labels is None:
        raise ValueError("Supervised training is disabled without actual mechanics labels")
    y = np.asarray(labels, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    groups = np.asarray(groups).astype(str)
    if x.ndim != 2 or y.ndim != 2 or not len(x) or len(x) != len(y) or groups.shape != (len(x),):
        raise ValueError("Expected features N,F, labels N,T and groups N")
    if not np.isfinite(x).all():
        raise ValueError("Feature matrix contains invalid values; use explicit valid-feature masks before training")
    mask = np.isfinite(y) if label_mask is None else np.asarray(label_mask, dtype=bool) & np.isfinite(y)
    if mask.shape != y.shape or not mask.any():
        raise ValueError("Training needs a matching nonempty valid-label mask")
    return x, np.where(mask, y, 0.), groups, mask


@transactional_output()
def train_supervised(features, labels, groups, outdir, *, feature_names, target_names,
                     target_units, feature_units=None, label_mask=None, edge_index=None,
                     edge_weights=None, label_metadata=None, model="gnn", epochs=200,
                     hidden=24, learning_rate=.01, ridge_alpha=1e-3, seed=0,
                     demo=False, progress=print, resume_from=None,
                     checkpoint_root=None, checkpoint_interval=25):
    """Train a real model, choosing the GNN checkpoint using validation only.

    Graph edges must remain within a specimen group. Groups refer to original
    specimens/parent scans, never random patch IDs. Target units are canonical
    MPa (stress) and 1 (strain) following the mechanics importer.
    """
    x, y, groups, mask = _validate_arrays(features, labels, groups, label_mask)
    provenance = _validate_provenance(label_metadata, demo=demo)
    names, targets, units = list(feature_names), list(target_names), list(target_units)
    feature_units = ["1"] * x.shape[1] if feature_units is None else list(feature_units)
    if (len(names) != x.shape[1] or len(set(names)) != len(names) or len(feature_units) != len(names)
            or len(targets) != y.shape[1] or len(set(targets)) != len(targets) or len(units) != len(targets)
            or set(units) - {"MPa", "1"}):
        raise ValueError("Feature/target names and declared canonical units must match matrix columns")
    if not demo:
        validate_stress_provenance(provenance, [name for name, unit in zip(targets, units) if unit == "MPa"])
        if "phase_id" in names:
            schema = provenance.get("phase_schema")
            if not isinstance(schema, dict) or not schema:
                raise ValueError("Real phase_id features require explicit phase_schema; regenerate or explicitly adapt the legacy training bundle")
            provenance["phase_schema"] = {str(key): value for key, value in schema.items()}
            values = x[:, names.index("phase_id")]
            if (np.any(values != np.rint(values))
                    or any(str(int(pid)) not in provenance["phase_schema"] for pid in np.unique(values))):
                raise ValueError("Training phase_id values lack explicit meanings in phase_schema")
    if (model not in {"ridge", "gnn"} or int(epochs) != epochs or epochs < 1
            or int(hidden) != hidden or hidden < 1 or not 0 < learning_rate <= 1
            or int(checkpoint_interval) != checkpoint_interval or checkpoint_interval < 1):
        raise ValueError("Select ridge or gnn with positive epochs and learning rate <= 1")
    if edge_index is not None:
        edges = np.asarray(edge_index)
        graph_design(x, edges, edge_weights=edge_weights)  # validate before indexing
        if edges.size and np.any(groups[edges[0]] != groups[edges[1]]):
            raise ValueError("Graph has cross-source edges; this would leak information across splits")
    split = group_split(groups, seed=seed)
    train, val, test = (split[key] for key in ("train", "validation", "test"))
    for key, indices in split.items():
        if np.any(mask[indices].sum(axis=0) == 0):
            raise ValueError(f"Every target needs actual labels in {key}; source-group split cannot be weakened")
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
    scale[scale < 1e-12] = 1
    ymean = np.array([y[train, c][mask[train, c]].mean() for c in range(y.shape[1])])
    yscale = np.array([y[train, c][mask[train, c]].std() for c in range(y.shape[1])])
    yscale[yscale < 1e-12] = 1
    design = (x - mean) / scale
    if model == "gnn":
        design = graph_design(design, edge_index, edge_weights=edge_weights)
    scaled_y = (y - ymean) / yscale
    config = {"schema_version": SCHEMA_VERSION, "model": model, "feature_names": names,
              "feature_units": feature_units, "target_names": targets, "target_units": units,
              "hidden": int(hidden), "seed": int(seed), "epochs_requested": int(epochs),
              "learning_rate": learning_rate, "ridge_alpha": ridge_alpha,
              "aggregation": "weighted directed incoming neighbour mean; concatenate self" if model == "gnn" else "none",
              "label_provenance": provenance, "synthetic_demo": bool(demo),
              "scientific_status": "RESEARCH_ONLY; external mechanics generalization not validated",
              "dataset_sha256": _array_hash(x, y, mask, groups,
                                            np.asarray(edge_index if edge_index is not None else [], dtype=int),
                                            np.asarray(edge_weights if edge_weights is not None else [], dtype=float)),
              "graph_edges_present": edge_index is not None and np.asarray(edge_index).size > 0,
              "scaler_fit_partition": "train", "checkpoint_selection_partition": "validation",
              "quaternion_convention": "if present: SciPy xyzw active crystal-to-sample; explicit sign representative required",
              "resume_from": str(Path(resume_from).name) if resume_from else None,
              "checkpoint_kind": "final_model", "training_status": "complete"}
    estimator = (RidgeRegressor(ridge_alpha) if model == "ridge" else
                 GrainGraphRegressor(design.shape[1], y.shape[1], hidden=hidden, seed=seed))
    completed_before = 0
    if resume_from:
        old, state = load_checkpoint(resume_from)
        for key in ("model", "feature_names", "feature_units", "target_names", "target_units", "dataset_sha256", "seed", "hidden", "label_provenance"):
            if old[key] != config[key]:
                raise ValueError(f"Resume checkpoint incompatible: {key}")
        estimator.restore({key[6:]: value for key, value in state.items() if key.startswith("model_")})
        completed_before = int(old.get("completed_epoch", 0))
        config["resume_semantics"] = "weights resumed; Adam moments restart; same deterministic group split"
    thresholds = np.array([np.quantile(np.abs(y[train, c][mask[train, c]]), .9) for c in range(y.shape[1])])
    def model_state():
        val_prediction = estimator.predict(design[val]) * yscale + ymean
        residual_interval = np.array([np.quantile(np.abs(y[val, c] - val_prediction[:, c])[mask[val, c]], .95)
                                      for c in range(y.shape[1])])
        result = {"x_mean": mean, "x_scale": scale, "y_mean": ymean, "y_scale": yscale,
                  "residual_interval": residual_interval, "high_strain_thresholds": thresholds}
        result.update({"model_" + key: value for key, value in estimator.state().items()})
        return result
    def save_epoch(epoch, values):
        if checkpoint_root is None or not (epoch == 1 or epoch % checkpoint_interval == 0 or epoch == epochs):
            return
        total_epoch = completed_before + epoch
        target = Path(checkpoint_root) / f"epoch{total_epoch:05d}"
        if target.exists():
            raise FileExistsError("Refusing to overwrite durable training checkpoint: " + str(target))
        epoch_config = {**config, "completed_epoch": total_epoch, "checkpoint_kind": "training_epoch",
                        "training_status": "incomplete_training",
                        "scientific_status": "INCOMPLETE_TRAINING; RESEARCH_ONLY; not a final evaluated model",
                        "resume_semantics": "weights only; Adam moments and RNG are not restored exactly"}
        # Publish this already-computed epoch even if Stop arrived mid-epoch.
        # Cancellation resumes immediately after the small atomic publication.
        with cancellation(None):
            path = _write_training_checkpoint(target, config=epoch_config, state=model_state(), epoch_metrics=values)
        progress(f"Durable research checkpoint epoch {total_epoch}: {path}")
    check_cancelled()
    progress(f"Research {model}: {len(train)} train / {len(val)} validation / {len(test)} test nodes")
    if model == "ridge":
        estimator.fit(design[train], scaled_y[train], mask[train])
        history = []
        save_epoch(1, {"epoch": 1, "method": "closed_form_ridge"})
        check_cancelled()
    else:
        def callback(epoch, values):
            save_epoch(epoch, values)
            check_cancelled()
            if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
                progress(f"Research GNN epoch {epoch}/{epochs}: validation MSE {values['validation_mse_scaled']:.6g}")
        history = estimator.fit(design[train], scaled_y[train], mask[train], epochs=epochs,
                                learning_rate=learning_rate,
                                validation=(design[val], scaled_y[val], mask[val]), callback=callback)
    prediction = estimator.predict(design) * yscale + ymean
    interval = np.array([np.quantile(np.abs(y[val, c] - prediction[val, c])[mask[val, c]], .95)
                         for c in range(y.shape[1])])
    config["completed_epoch"] = completed_before + (1 if model == "ridge" else int(epochs))
    report = {"scientific_status": config["scientific_status"], "synthetic_demo": bool(demo),
              "groups": {key: sorted(set(groups[indices])) for key, indices in split.items()},
              "metrics": {key: _metrics(y[indices], prediction[indices], mask[indices], targets, units, thresholds, interval)
                          for key, indices in split.items()},
              "uncertainty_method": "per-target 95th percentile absolute validation residual, not a validated coverage guarantee",
              "history": history}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (outdir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    state = model_state()
    np.savez_compressed(outdir / "checkpoint.npz", **state)
    np.savez_compressed(outdir / "split_indices.npz", **split, groups=groups)
    with h5py.File(outdir / "evaluation.h5", "w") as stream:
        stream.create_dataset("prediction", data=prediction)
        stream.create_dataset("labels", data=np.where(mask, y, np.nan))
        stream.create_dataset("label_mask", data=mask)
        for key, indices in split.items():
            stream.create_dataset("partitions/" + key, data=indices)
    progress("Research training complete; held-out test metrics written")
    return outdir / "checkpoint.npz"


def load_checkpoint(modeldir):
    modeldir = Path(modeldir)
    if modeldir.is_file():
        modeldir = modeldir.parent
    config = json.loads((modeldir / "config.json").read_text(encoding="utf-8"))
    validate_checkpoint_archive(modeldir, ("config.json", "checkpoint.npz"), required=bool(config.get("checkpoint_kind")))
    if config.get("schema_version") != SCHEMA_VERSION or config.get("model") not in {"gnn", "ridge"}:
        raise ValueError("Unsupported supervised checkpoint schema")
    with np.load(modeldir / "checkpoint.npz", allow_pickle=False) as stream:
        state = {key: stream[key] for key in stream.files}
    return config, state


@transactional_output()
def infer_supervised(modeldir, features, outdir, *, feature_names, feature_units=None,
                     edge_index=None, edge_weights=None, map_shape=None, valid_mask=None,
                     labels=None, label_mask=None, formats=("png",), font="DejaVu Sans", progress=print,
                     inference_metadata=None):
    """Load a compatible checkpoint and archive predictions with research labels."""
    config, state = load_checkpoint(modeldir)
    if config.get("training_status") == "incomplete_training":
        raise ValueError("Checkpoint training is incomplete; resume training before final inference")
    if not config["synthetic_demo"]:
        context = config["label_provenance"]
        required = ("material", "loading_condition", "reference_frame", "load_step", "time")
        if (not isinstance(inference_metadata, dict)
                or any(key not in inference_metadata or inference_metadata[key] is None
                       or not str(inference_metadata[key]).strip() for key in required)
                or not isinstance(inference_metadata["time"], dict)
                or not {"value", "unit"} <= inference_metadata["time"].keys()):
            raise ValueError("Real-model inference requires explicit material, loading_condition, reference_frame, load_step and time metadata")
        for key in required:
            if key not in context or inference_metadata[key] != context[key]:
                raise ValueError("Inference context incompatible with training: " + key)
        if "phase_id" in config["feature_names"]:
            training_schema = context.get("phase_schema")
            prediction_schema = inference_metadata.get("phase_schema")
            if not isinstance(training_schema, dict) or not training_schema:
                raise ValueError("Legacy real model lacks phase_schema; regenerate or explicitly adapt its phase features")
            if not isinstance(prediction_schema, dict) or not prediction_schema:
                raise ValueError("Real phase_id inference requires explicit matching phase_schema")
            prediction_schema = {str(key): value for key, value in prediction_schema.items()}
            if json.dumps(training_schema, sort_keys=True) != json.dumps(prediction_schema, sort_keys=True):
                raise ValueError("Inference phase_schema differs from training; explicit phase remapping is required")
    x = np.asarray(features, dtype=float)
    units = ["1"] * x.shape[1] if feature_units is None and x.ndim == 2 else list(feature_units or [])
    if list(feature_names) != config["feature_names"] or units != config["feature_units"]:
        raise ValueError("Inference feature names/order/units do not match the trained checkpoint")
    if x.ndim != 2 or x.shape[1] != len(config["feature_names"]):
        raise ValueError("Inference feature dimensions mismatch")
    mask = np.isfinite(x).all(axis=1)
    if valid_mask is not None:
        provided = np.asarray(valid_mask, dtype=bool)
        if provided.shape != (len(x),):
            raise ValueError("Inference mask must have one entry per row")
        mask &= provided
    if not mask.any():
        raise ValueError("No valid inference rows")
    if not config["synthetic_demo"] and "phase_id" in config["feature_names"]:
        values = x[mask, config["feature_names"].index("phase_id")]
        if (np.any(values != np.rint(values))
                or any(str(int(pid)) not in config["label_provenance"]["phase_schema"] for pid in np.unique(values))):
            raise ValueError("Inference phase_id values are absent from the trained phase_schema")
    design = (np.where(np.isfinite(x), x, state["x_mean"]) - state["x_mean"]) / state["x_scale"]
    if config["model"] == "gnn":
        if config.get("graph_edges_present") and edge_index is None:
            raise ValueError("This checkpoint requires an explicit compatible graph at inference")
        # Invalid node messages are removed, not imputed into neighbouring nodes.
        if edge_index is not None:
            edges = np.asarray(edge_index)
            graph_design(design, edges, edge_weights=edge_weights)
            retained = mask[edges[0]] & mask[edges[1]]
            edge_index = edges[:, retained]
            if edge_weights is not None:
                edge_weights = np.asarray(edge_weights)[retained]
        design = graph_design(design, edge_index, edge_weights=edge_weights)
        estimator = GrainGraphRegressor(design.shape[1], len(config["target_names"]), hidden=config["hidden"])
    else:
        estimator = RidgeRegressor(config["ridge_alpha"])
    estimator.restore({key[6:]: value for key, value in state.items() if key.startswith("model_")})
    check_cancelled()
    prediction = estimator.predict(design) * state["y_scale"] + state["y_mean"]
    prediction[~mask] = np.nan
    interval = np.broadcast_to(state["residual_interval"], prediction.shape).copy()
    interval[~mask] = np.nan
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": SCHEMA_VERSION, "scientific_status": config["scientific_status"],
              "synthetic_demo": config["synthetic_demo"], "valid_count": int(mask.sum()),
              "target_names": config["target_names"], "target_units": config["target_units"],
              "inference_metadata": inference_metadata,
              "uncertainty_method": "validation residual interval; external calibration unvalidated"}
    with h5py.File(outdir / "predictions.h5", "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(report)
        stream.create_dataset("prediction", data=prediction)
        stream.create_dataset("valid_mask", data=mask)
        stream.create_dataset("interval_halfwidth", data=interval)
    with (outdir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "valid"] + [f"{name} [{unit}]" for name, unit in zip(config["target_names"], config["target_units"])])
        for row, values in enumerate(prediction):
            writer.writerow([row, int(mask[row]), *values])
    if labels is not None:
        truth = np.asarray(labels, dtype=float)
        if truth.ndim == 1:
            truth = truth[:, None]
        if truth.shape != prediction.shape:
            raise ValueError("Evaluation labels must match prediction shape")
        lm = np.isfinite(truth) if label_mask is None else np.asarray(label_mask, dtype=bool) & np.isfinite(truth)
        if lm.shape != truth.shape:
            raise ValueError("Evaluation label mask shape mismatch")
        report["evaluation"] = _metrics(truth, prediction, lm & mask[:, None], config["target_names"],
                                        config["target_units"], state["high_strain_thresholds"], state["residual_interval"])
    if map_shape is not None:
        if len(map_shape) != 2 or np.prod(map_shape) != len(x) or min(map_shape) < 1:
            raise ValueError("map_shape must match the number of prediction rows")
        formats = tuple(dict.fromkeys(str(value).lower() for value in formats))
        if not formats or set(formats) - {"png", "tiff", "pdf", "svg"}:
            raise ValueError("Supported formats: png, tiff, pdf, svg")
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        with plt.rc_context({"font.family": font, "svg.fonttype": "none", "pdf.fonttype": 42}):
            for index, name in enumerate(config["target_names"]):
                check_cancelled()
                for kind, values in (("prediction", prediction[:, index]), ("uncertainty", interval[:, index])):
                    fig, ax = plt.subplots(layout="constrained")
                    image = ax.imshow(values.reshape(map_shape), origin="upper")
                    ax.set_title(f"RESEARCH ONLY: {name} {kind}")
                    fig.colorbar(image, ax=ax, label=config["target_units"][index])
                    for suffix in formats:
                        progress(f"Exporting research {kind} {index} {suffix.upper()}")
                        fig.savefig(outdir / f"target_{index}_{kind}.{suffix}", dpi=180)
                    plt.close(fig)
    (outdir / "prediction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return outdir / "predictions.h5"


def load_training_bundle(path):
    """Read a non-pickled NPZ bundle with explicit metadata, labels and groups."""
    with np.load(path, allow_pickle=False) as stream:
        required = {"features", "labels", "groups", "metadata_json"}
        if not required <= set(stream.files):
            raise ValueError("Training bundle needs features, labels, groups and metadata_json; unlabeled training disabled")
        meta = json.loads(str(stream["metadata_json"].item()))
        result = {"features": stream["features"], "labels": stream["labels"], "groups": stream["groups"]}
        for key in ("label_mask", "edge_index", "edge_weights"):
            if key in stream.files:
                result[key] = stream[key]
    for key in ("feature_names", "feature_units", "target_names", "target_units", "label_metadata"):
        if key not in meta:
            raise ValueError("Training bundle metadata missing " + key)
        result[key] = meta[key]
    result["demo"] = bool(meta.get("synthetic_demo", False))
    return result


def load_grain_graph(v2dir, *, source_id):
    """Build directed grain adjacency/features from real V2 CSVs, without labels."""
    if not str(source_id).strip():
        raise ValueError("Original source/specimen ID is required")
    v2dir = Path(v2dir)
    with (v2dir / "grains.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("V2 contains no reported grains")
    names = ["eq_diameter_um", "area_um2", "GOS_deg", "phase_id", "mean_qx", "mean_qy", "mean_qz", "mean_qw"]
    x = np.asarray([[float(row[key]) for key in names] for row in rows])
    # q and -q encode the same orientation. Select a deterministic sign.
    q = x[:, 4:8]
    norms = np.linalg.norm(q, axis=1)
    if not np.isfinite(norms).all() or np.any(norms < 1e-12):
        raise ValueError("V2 graph contains invalid grain quaternions")
    q /= norms[:, None]
    for row in q:
        for value in row[::-1]:
            if abs(value) > 1e-12:
                row *= 1 if value > 0 else -1
                break
    ids = np.asarray([int(row["grain_id"]) for row in rows])
    lookup = {gid: index for index, gid in enumerate(ids)}
    edges, weights = [], []
    with (v2dir / "grain_neighbour_graph.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            a, b = int(row["grain_a"]), int(row["grain_b"])
            if a in lookup and b in lookup:
                edges.extend([(lookup[a], lookup[b]), (lookup[b], lookup[a])])
                weights.extend([float(row["shared_pixel_edges"])] * 2)
    return {"features": x, "feature_names": names, "feature_units": ["um", "um2", "deg", "1", "1", "1", "1", "1"],
            "grain_ids": ids, "edge_index": np.asarray(edges, dtype=int).reshape(-1, 2).T,
            "edge_weights": np.asarray(weights), "groups": np.full(len(x), str(source_id))}


@transactional_output()
def create_synthetic_training_bundle(outdir, *, seed=42):
    """Create an explicitly synthetic learnable graph for installation QA only."""
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"SYNTHETIC-specimen-{i}" for i in range(8)], 16)
    x = rng.normal(size=(len(groups), 3))
    edges = []
    for start in range(0, len(groups), 16):
        for offset in range(16):
            a, b = start + offset, start + (offset + 1) % 16
            edges.extend([(a, b), (b, a)])
    edges = np.asarray(edges, dtype=int).T
    messages = graph_design(x, edges)[:, 3:]
    y = np.column_stack([200 + 50 * x[:, 0] + 25 * messages[:, 1], .01 * (x[:, 0] - x[:, 2])])
    metadata = {"feature_names": ["synthetic_a", "synthetic_b", "synthetic_c"], "feature_units": ["1"] * 3,
                "target_names": ["synthetic_sigma_xx", "synthetic_epsilon_xx"], "target_units": ["MPa", "1"],
                "synthetic_demo": True, "label_metadata": {"source_type": "SYNTHETIC", "source_id": "CPU-QA-demo",
                "material": "synthetic analytic QA material; no real material model",
                "reference_frame": "synthetic Sample-X/Y", "load_step": 0, "time": {"value": 0, "unit": "s"},
                "loading_condition": "synthetic analytic function; no physical constitutive meaning",
                "uncertainty": {"description": "synthetic exact function; not experimental data"}}}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "synthetic_training_bundle.npz", features=x, labels=y, groups=groups,
                        edge_index=edges, metadata_json=json.dumps(metadata))
    (outdir / "SYNTHETIC_README.txt").write_text("Synthetic CPU installation and regression demo only. No real mechanics validation.\n", encoding="utf-8")
    return outdir / "synthetic_training_bundle.npz"


def _main():
    import argparse
    parser = argparse.ArgumentParser(description="V4 research CPU training; no real material validation")
    sub = parser.add_subparsers(dest="action", required=True)
    demo = sub.add_parser("demo", help="create a synthetic training bundle")
    demo.add_argument("--out", required=True)
    train = sub.add_parser("train", help="train from an independently labelled numeric NPZ bundle")
    train.add_argument("--bundle", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--model", choices=("gnn", "ridge"), default="gnn")
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--resume-from")
    args = parser.parse_args()
    if args.action == "demo":
        print(create_synthetic_training_bundle(args.out))
    else:
        print(train_supervised(**load_training_bundle(args.bundle), outdir=args.out, model=args.model,
                               epochs=args.epochs, resume_from=args.resume_from))


if __name__ == "__main__":
    _main()
