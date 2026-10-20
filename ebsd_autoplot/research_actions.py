"""Small explicit GUI/worker adapters for M4 research operations.

All substantive dependencies are imported only when their action is run. These
adapters never evaluate configuration as code or convert unlabelled EBSD values
into mechanical labels. Root job management owns final GUI success/failure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ACTION_KEYS = {
    "mechanics-import": ({"path", "metadata", "columns"}, {"path", "metadata", "columns"}),
    "graph-training-bundle": ({"graph_files", "mechanics_files", "target_names", "max_distance_um", "reference_frame"},
                              {"graph_files", "mechanics_files", "target_names", "max_distance_um", "reference_frame"}),
    "train-supervised": ({"bundle_path", "model", "epochs", "hidden", "learning_rate", "ridge_alpha", "seed", "resume_from"}, {"bundle_path"}),
    "infer-supervised": ({"bundle_path", "checkpoint", "map_shape", "formats", "font"}, {"bundle_path", "checkpoint"}),
    "diffusion-bundle": ({"index_files", "allowed_phase_ids", "synthetic_demo", "max_patches"}, {"index_files", "allowed_phase_ids"}),
    "train-diffusion": ({"bundle_path", "epochs", "steps", "hidden", "batch_size", "learning_rate", "seed", "resume_from"}, {"bundle_path"}),
    "sample-diffusion": ({"checkpoint", "condition", "count", "seed", "valid_mask_path", "allow_condition_extrapolation"}, {"checkpoint", "condition"}),
    "synthetic-gnn-demo": ({"epochs", "hidden", "seed", "formats", "font"}, set()),
    "synthetic-diffusion-demo": ({"epochs", "steps", "hidden", "seed", "count"}, set()),
}


def _configuration(action, config):
    if action not in ACTION_KEYS:
        raise NotImplementedError("Unknown research action: " + str(action))
    if not isinstance(config, dict):
        raise ValueError("Research action configuration must be a JSON object")
    allowed, required = ACTION_KEYS[action]
    missing = required - config.keys()
    unknown = config.keys() - allowed
    if missing:
        raise ValueError("Missing research configuration: " + ", ".join(sorted(missing)))
    if unknown:
        raise ValueError("Unknown research configuration keys: " + ", ".join(sorted(unknown)))
    result = dict(config)
    for key in ("synthetic_demo", "allow_condition_extrapolation"):
        if key in result and not isinstance(result[key], bool):
            raise ValueError(key + " must be a JSON boolean")
    return result


def _inference_bundle(path):
    """Inference requires features and their schema; labels are not required."""
    with np.load(path, allow_pickle=False) as stream:
        if not {"features", "metadata_json"} <= set(stream.files):
            raise ValueError("Inference bundle requires numeric features and metadata_json")
        metadata = json.loads(str(stream["metadata_json"].item()))
        if not {"feature_names", "feature_units"} <= metadata.keys():
            raise ValueError("Inference metadata requires feature_names and feature_units")
        data = {"features": stream["features"], "feature_names": metadata["feature_names"],
                "feature_units": metadata["feature_units"]}
        if "inference_metadata" in metadata:
            data["inference_metadata"] = metadata["inference_metadata"]
        for key in ("edge_index", "edge_weights", "valid_mask"):
            if key in stream.files:
                data[key] = stream[key]
    return data


def run_research_action(action, cache, run, config):
    """Execute one action under an already allocated independent run directory.

    ``cache`` is accepted for the studio dispatcher contract; all M4 sources are
    explicit in ``config``. Returns JSON-serializable output locations. Unknown
    action/keys raise rather than silently changing a model or data contract.
    """
    values = _configuration(action, config)
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    output, synthetic = {}, False
    if action == "mechanics-import":
        from .mechanics import import_mechanics
        metadata = values["metadata"]
        if isinstance(metadata, (str, Path)):
            metadata = json.loads(Path(metadata).read_text(encoding="utf-8-sig"))
        output["mechanics"] = import_mechanics(values["path"], run / "results", metadata=metadata, columns=values["columns"])
    elif action == "graph-training-bundle":
        from .mechanics import create_graph_training_bundle
        output["bundle"] = create_graph_training_bundle(outdir=run / "results", **values)
    elif action == "train-supervised":
        from .ml_train import load_training_bundle, train_supervised
        bundle = load_training_bundle(values.pop("bundle_path"))
        synthetic = bundle["demo"]
        output["checkpoint"] = train_supervised(**bundle, outdir=run / "results", checkpoint_root=run / "checkpoints", **values)
    elif action == "infer-supervised":
        from .ml_train import infer_supervised, load_checkpoint
        bundle = _inference_bundle(values.pop("bundle_path"))
        checkpoint = values.pop("checkpoint")
        trained, _ = load_checkpoint(checkpoint)
        synthetic = trained["synthetic_demo"]
        output["predictions"] = infer_supervised(checkpoint, outdir=run / "results", **bundle, **values)
    elif action == "diffusion-bundle":
        from .diffusion import create_diffusion_bundle
        synthetic = values.get("synthetic_demo", False)
        output["bundle"] = create_diffusion_bundle(outdir=run / "results", **values)
    elif action == "train-diffusion":
        from .diffusion import load_diffusion_bundle, train_diffusion
        bundle = load_diffusion_bundle(values.pop("bundle_path"))
        synthetic = bool(bundle.get("synthetic_demo", False))
        output["checkpoint"] = train_diffusion(**bundle, outdir=run / "results", checkpoint_root=run / "checkpoints", **values)
    elif action == "sample-diffusion":
        from .diffusion import load_diffusion_checkpoint, sample_diffusion
        checkpoint = values.pop("checkpoint")
        trained, _ = load_diffusion_checkpoint(checkpoint)
        synthetic = trained["synthetic_demo"]
        if "valid_mask_path" in values:
            mask = np.load(values.pop("valid_mask_path"), allow_pickle=False)
            if mask.dtype != np.bool_:
                raise ValueError("valid_mask_path must point to a boolean numeric NPY")
            values["valid_mask"] = mask
        output["generated_microstructures"] = sample_diffusion(checkpoint, run / "results", **values)
    elif action == "synthetic-gnn-demo":
        from .ml_train import (create_synthetic_training_bundle, load_training_bundle,
                               train_supervised, infer_supervised)
        seed = values.get("seed", 42)
        output["bundle"] = create_synthetic_training_bundle(run / "inputs", seed=seed)
        bundle = load_training_bundle(output["bundle"])
        output["checkpoint"] = train_supervised(**bundle, outdir=run / "model", model="gnn",
            epochs=values.get("epochs", 120), hidden=values.get("hidden", 24), seed=seed, checkpoint_root=run / "checkpoints")
        output["predictions"] = infer_supervised(run / "model", bundle["features"], run / "samples",
            feature_names=bundle["feature_names"], feature_units=bundle["feature_units"],
            edge_index=bundle["edge_index"], map_shape=(8, 16),
            formats=tuple(values.get("formats", ("png",))), font=values.get("font", "DejaVu Sans"))
        synthetic = True
    elif action == "synthetic-diffusion-demo":
        from .diffusion import (create_synthetic_diffusion_bundle, load_diffusion_bundle,
                               train_diffusion, sample_diffusion)
        seed = values.get("seed", 42)
        output["bundle"] = create_synthetic_diffusion_bundle(run / "inputs", seed=seed)
        bundle = load_diffusion_bundle(output["bundle"])
        output["checkpoint"] = train_diffusion(**bundle, outdir=run / "model",
            epochs=values.get("epochs", 20), steps=values.get("steps", 8),
            hidden=values.get("hidden", 48), seed=seed, checkpoint_root=run / "checkpoints")
        output["generated_microstructures"] = sample_diffusion(run / "model", run / "samples",
            condition=[.5, .5], count=values.get("count", 2), seed=seed)
        synthetic = True
    return {"action": action, "status": "success", "scientific_status": "RESEARCH_ONLY; independent material validation unavailable",
            "synthetic_demo": synthetic, "outputs": {key: str(value) for key, value in output.items()}}
