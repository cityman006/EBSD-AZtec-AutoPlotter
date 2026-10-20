"""Trainable conditional DDPM research baseline on orientation/phase fields.

This small CPU MLP denoiser demonstrates the complete noise-training/reverse-
diffusion/checkpoint path. It is neither a U-Net nor a verified microstructure
generator. Mechanical conditioning is blocked until an independent validated
surrogate and independent mechanics labels are available and integrated.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .ml_train import group_split, _array_hash, validate_checkpoint_archive
from .storage import cancellation, check_cancelled, transactional_output

MECHANICAL_WORDS = ("stress", "strain", "strength", "mechanic", "yield", "schmid", "tau", "crss")


@transactional_output()
def _write_diffusion_checkpoint(outdir, *, config, state, epoch_metrics):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "diffusion_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.savez_compressed(outdir / "diffusion_checkpoint.npz", **state)
    (outdir / "epoch_metrics.json").write_text(json.dumps(epoch_metrics, indent=2), encoding="utf-8")
    return outdir / "diffusion_checkpoint.npz"


def _check_conditions(names, units):
    if len(names) != len(units) or len(set(names)) != len(names):
        raise ValueError("Condition names and units must be explicit, unique and matching")
    if any(any(word in name.lower() for word in MECHANICAL_WORDS) for name in names):
        raise ValueError("Mechanical conditioning disabled: no independent validated surrogate with mechanics labels is integrated")
    allowed = {"phase_fraction": "1", "temperature": "K", "time": "s"}
    for name, unit in zip(names, units):
        kind = "phase_fraction" if name.startswith("phase_fraction_") else name
        if kind not in allowed or unit != allowed[kind]:
            raise ValueError("Baseline conditions support phase_fraction_<id> [1], temperature [K], time [s] only")


def cosine_schedule(steps):
    if not isinstance(steps, int) or not 4 <= steps <= 1000:
        raise ValueError("DDPM steps must be an integer from 4 to 1000")
    grid = np.arange(steps + 1) / steps
    cumulative = np.cos((grid + .008) / 1.008 * np.pi / 2) ** 2
    cumulative /= cumulative[0]
    beta = np.clip(1 - cumulative[1:] / cumulative[:-1], .0001, .999)
    return beta, np.cumprod(1 - beta)


def _canonical_q(q, valid):
    q = np.asarray(q, dtype=float).copy()
    norms = np.linalg.norm(q, axis=-1)
    if not np.isfinite(q[valid]).all() or np.any(norms[valid] < 1e-12):
        raise ValueError("Every valid pixel requires a finite nonzero xyzw quaternion")
    q[valid] /= norms[valid, None]
    # Stable quaternion sign except at a measure-zero representative seam.
    flattened = q.reshape(-1, 4)
    for component in (3, 2, 1, 0):
        decisive = np.abs(flattened[:, component]) > 1e-12
        undecided = np.all(np.abs(flattened[:, component + 1:]) <= 1e-12, axis=1)
        flip = decisive & undecided & (flattened[:, component] < 0)
        flattened[flip] *= -1
    q[~valid] = 0
    return q


def canonicalize_phase_quaternions(quaternion_xyzw, phase_id, valid_mask, phase_schema):
    """Select a deterministic representative of q * crystal_point_group.

    q maps crystal to sample, hence crystal symmetry acts on its right. Each
    candidate first has a canonical antipodal sign. We maximize quantized xyzw
    lexicographic order (1e-10 resolution), then exact lexicographic order within
    the quantized tie. This changes only the derived representation, not sources.
    """
    from .v3_crystal import point_group
    phase, valid = np.asarray(phase_id), np.asarray(valid_mask, dtype=bool)
    q = _canonical_q(quaternion_xyzw, valid)
    if q.shape != (*phase.shape, 4) or valid.shape != phase.shape:
        raise ValueError("Phase-aware canonicalization needs matching quaternion/phase/mask shapes")
    schema = {str(key): value for key, value in dict(phase_schema).items()}
    flat, pflat, vflat = q.reshape(-1, 4), phase.ravel(), valid.ravel()
    for pid in np.unique(pflat[vflat]):
        try:
            sg = int(schema[str(int(pid))]["space_group"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Real phase {pid} requires explicit space_group 194 or 229") from exc
        if sg not in (194, 229):
            raise ValueError(f"Unsupported crystal symmetry {sg} for phase {pid}; only HCP194/BCC229 are supported")
        symmetry = point_group(sg).as_quat()
        rows = np.flatnonzero(vflat & (pflat == pid))
        for start in range(0, len(rows), 2048):
            check_cancelled()
            selected = rows[start:start + 2048]
            a = flat[selected, None, :]
            b = symmetry[None, :, :]
            vector = a[..., 3:4] * b[..., :3] + b[..., 3:4] * a[..., :3] + np.cross(a[..., :3], b[..., :3])
            scalar = a[..., 3:4] * b[..., 3:4] - np.sum(a[..., :3] * b[..., :3], axis=-1, keepdims=True)
            candidates = np.concatenate([vector, scalar], axis=-1)
            candidates = _canonical_q(candidates, np.ones(candidates.shape[:-1], dtype=bool))
            quantized = np.rint(candidates * 1e10).astype(np.int64)
            eligible = np.ones(candidates.shape[:-1], dtype=bool)
            for component in range(4):
                maximum = np.max(np.where(eligible, quantized[..., component], np.iinfo(np.int64).min), axis=1)
                eligible &= quantized[..., component] == maximum[:, None]
            for component in range(4):
                maximum = np.max(np.where(eligible, candidates[..., component], -np.inf), axis=1)
                eligible &= candidates[..., component] == maximum[:, None]
            choice = np.argmax(eligible, axis=1)
            flat[selected] = candidates[np.arange(len(selected)), choice]
    return q


def encode_fields(quaternion_xyzw, phase_id, *, valid_mask=None, allowed_phase_ids=(1, 2),
                  phase_schema=None):
    q = np.asarray(quaternion_xyzw, dtype=float)
    phase = np.asarray(phase_id)
    if q.ndim != 4 or q.shape[-1] != 4 or phase.shape != q.shape[:-1] or not np.issubdtype(phase.dtype, np.integer):
        raise ValueError("Expected quaternions N,H,W,4 and integer phase IDs N,H,W")
    ids = list(allowed_phase_ids)
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(pid, (int, np.integer)) or pid <= 0 for pid in ids):
        raise ValueError("Allowed phase IDs must be unique positive integers; 0 is reserved for invalid pixels")
    valid = phase != 0 if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != phase.shape or not valid.any() or np.any(~np.isin(phase[valid], ids)):
        raise ValueError("Valid mask must match fields and valid pixels must have a declared legal phase ID")
    if np.any(valid.reshape(len(q), -1).sum(axis=1) == 0):
        raise ValueError("Every training sample requires valid pixels")
    q = (_canonical_q(q, valid) if phase_schema is None else
         canonicalize_phase_quaternions(q, phase, valid, phase_schema))
    onehot = np.stack([np.where(phase == pid, 1., -1.) for pid in ids], axis=-1)
    encoded = np.concatenate([q, onehot], axis=-1)
    mask = np.broadcast_to(valid[..., None], encoded.shape).copy()
    encoded[~mask] = 0
    return encoded, mask, valid


def _time_features(times, steps):
    value = np.asarray(times) / max(steps - 1, 1)
    return np.column_stack([value, np.sin(np.pi * value), np.cos(np.pi * value)])


class _NoiseModel:
    def __init__(self, dimensions, conditions, hidden, seed):
        rng = np.random.default_rng(seed)
        inputs = dimensions + conditions + 3
        self.params = {"w1": rng.normal(0, 1 / np.sqrt(inputs), (inputs, hidden)), "b1": np.zeros(hidden),
                       "w2": rng.normal(0, .02, (hidden, dimensions)), "b2": np.zeros(dimensions)}

    def inputs(self, noisy, time, condition, steps):
        return np.column_stack([noisy, _time_features(time, steps), condition])

    def predict(self, noisy, time, condition, steps):
        x = self.inputs(noisy, time, condition, steps)
        p = self.params
        return np.tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]

    def gradients(self, noisy, time, condition, steps, noise, mask):
        x = self.inputs(noisy, time, condition, steps)
        p = self.params
        h = np.tanh(x @ p["w1"] + p["b1"])
        residual = np.where(mask, h @ p["w2"] + p["b2"] - noise, 0)
        d = 2 * residual / mask.sum()
        dh = (d @ p["w2"].T) * (1 - h * h)
        return float(np.mean(residual[mask] ** 2)), {
            "w1": x.T @ dh, "b1": dh.sum(axis=0), "w2": h.T @ d, "b2": d.sum(axis=0)}


@transactional_output()
def train_diffusion(quaternion_xyzw, phase_id, groups, outdir, *, valid_mask=None,
                    allowed_phase_ids=(1, 2), condition=None, condition_names=None,
                    condition_units=None, steps=32, epochs=50, hidden=48,
                    batch_size=8, learning_rate=.003, seed=0, synthetic_demo=False,
                    dataset_metadata=None, progress=print, resume_from=None,
                    checkpoint_root=None, checkpoint_interval=10):
    """Actually learn epsilon in a DDPM with grouped held-out denoising tests."""
    encoded, masks, valid = encode_fields(quaternion_xyzw, phase_id, valid_mask=valid_mask,
                                          allowed_phase_ids=allowed_phase_ids)
    n, height, width, channels = encoded.shape
    dimensions = height * width * channels
    if dimensions > 4096:
        raise ValueError("CPU baseline supports at most 4096 encoded variables per tile; use smaller physical tiles")
    if np.asarray(groups).shape != (n,):
        raise ValueError("One original specimen/source group is required per diffusion sample")
    if (int(epochs) != epochs or epochs < 1 or int(batch_size) != batch_size or batch_size < 1
            or not 0 < learning_rate <= .1 or int(checkpoint_interval) != checkpoint_interval or checkpoint_interval < 1):
        raise ValueError("Positive integer epochs/batch_size and learning rate <= 0.1 required")
    beta, abar = cosine_schedule(steps)
    ids = list(map(int, allowed_phase_ids))
    dataset_metadata = dict(dataset_metadata or {})
    if "phase_schema" in dataset_metadata:
        if not isinstance(dataset_metadata["phase_schema"], dict):
            raise ValueError("phase_schema must map original phase IDs to crystallographic metadata")
        dataset_metadata["phase_schema"] = {str(key): value for key, value in dataset_metadata["phase_schema"].items()}
    if condition is None:
        condition = np.column_stack([((np.asarray(phase_id) == pid) & valid).sum(axis=(1, 2)) / valid.sum(axis=(1, 2)) for pid in ids])
        condition_names = [f"phase_fraction_{pid}" for pid in ids]
        condition_units = ["1"] * len(ids)
    condition = np.asarray(condition, dtype=float)
    names, units = list(condition_names or []), list(condition_units or [])
    _check_conditions(names, units)
    if not synthetic_demo:
        physical_step = dataset_metadata.get("step_um_xy")
        phase_schema = dataset_metadata.get("phase_schema", {})
        if (physical_step is None or len(physical_step) != 2 or not np.isfinite(physical_step).all()
                or min(physical_step) <= 0 or any(str(pid) not in phase_schema for pid in ids)):
            raise ValueError("Real DDPM training requires measured physical step_um_xy and explicit phase_schema")
    phase_schema = dataset_metadata.get("phase_schema")
    symmetry_rule = "toy synthetic data: antipodal sign only; no crystal symmetry reduction"
    if phase_schema is not None:
        # Validate every declared phase, including those absent from this batch.
        for pid in ids:
            try:
                sg = int(phase_schema[str(pid)]["space_group"])
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"Declared phase {pid} needs space_group 194 or 229") from exc
            if sg not in (194, 229):
                raise ValueError(f"Unsupported crystal symmetry {sg} for phase {pid}")
        encoded, masks, valid = encode_fields(quaternion_xyzw, phase_id, valid_mask=valid_mask,
            allowed_phase_ids=allowed_phase_ids, phase_schema=phase_schema)
        symmetry_rule = "right action q*crystal point group; canonical antipodal sign; max quantized xyzw lexicographic order 1e-10, exact lexicographic tie"
    elif not synthetic_demo:
        raise ValueError("Real DDPM training requires phase-specific crystal symmetry")
    if condition.shape != (n, len(names)) or not np.isfinite(condition).all():
        raise ValueError("Conditions must be a finite N-by-C matrix with explicit names and units")
    for index, name in enumerate(names):
        if name.startswith("phase_fraction_") and (np.any(condition[:, index] < 0) or np.any(condition[:, index] > 1)):
            raise ValueError("Phase fraction conditions must lie in [0,1]")
        if name == "temperature" and np.any(condition[:, index] <= 0):
            raise ValueError("Temperature conditions are absolute Kelvin and must be positive")
    split = group_split(groups, seed=seed)
    train, val, test = (split[key] for key in ("train", "validation", "test"))
    cmean, cscale = condition[train].mean(axis=0), condition[train].std(axis=0)
    cscale[cscale < 1e-12] = 1
    cond = (condition - cmean) / cscale
    data, mask = encoded.reshape(n, -1), masks.reshape(n, -1)
    rng = np.random.default_rng(seed)
    model = _NoiseModel(dimensions, len(names), hidden, seed)
    config = {"schema_version": 1, "model": "conditional_ddpm_numpy_mlp", "steps": steps,
              "shape_hwc": [height, width, channels], "allowed_phase_ids": ids,
              "condition_names": names, "condition_units": units, "hidden": hidden,
              "epochs_requested": int(epochs), "batch_size": int(batch_size), "learning_rate": learning_rate,
              "seed": int(seed), "synthetic_demo": bool(synthetic_demo),
              "dataset_metadata": dataset_metadata,
              "orientation_convention": "SciPy xyzw active crystal-to-sample; unit quaternion; canonical sign",
              "crystal_symmetry_reduced": phase_schema is not None,
              "symmetry_representative_rule": symmetry_rule,
              "phase_encoding": "declared phase one-hot channels mapped to -1,+1; invalid pixels masked",
              "dataset_sha256": _array_hash(data, mask, np.asarray(groups).astype(str), condition),
              "scientific_status": "RESEARCH_ONLY; learned distribution and physical realism not validated",
              "mechanical_conditioning": "disabled; independent validated surrogate and labels unavailable",
              "physics_loss": "not implemented; unit-norm and legal-phase checks are representation checks only",
              "checkpoint_selection_partition": "validation", "condition_scaler_partition": "train",
              "checkpoint_kind": "final_model", "training_status": "complete"}
    completed_before = 0
    if resume_from:
        old, state = load_diffusion_checkpoint(resume_from)
        for key in ("model", "steps", "shape_hwc", "allowed_phase_ids", "condition_names", "condition_units", "hidden", "dataset_sha256", "seed", "dataset_metadata", "symmetry_representative_rule"):
            if config[key] != old[key]:
                raise ValueError("Incompatible DDPM weight-resume checkpoint: " + key)
        model.params = {key[6:]: value for key, value in state.items() if key.startswith("model_")}
        completed_before = int(old.get("completed_epoch", 0))
        config["resume_semantics"] = "weights resumed; optimizer moments and RNG restart; not exact trajectory resume"
    fixed_rng = np.random.default_rng(seed + 1)
    eval_times = fixed_rng.integers(0, steps, size=n)
    eval_noise = fixed_rng.normal(size=data.shape)
    def evaluation(indices):
        times = eval_times[indices]
        noisy = np.sqrt(abar[times, None]) * data[indices] + np.sqrt(1 - abar[times, None]) * eval_noise[indices]
        noisy[~mask[indices]] = 0
        residual = model.predict(noisy, times, cond[indices], steps) - eval_noise[indices]
        return float(np.mean(residual[mask[indices]] ** 2))
    first = {key: np.zeros_like(value) for key, value in model.params.items()}
    second = {key: np.zeros_like(value) for key, value in model.params.items()}
    history, best, best_loss, update = [], None, np.inf, 0
    initial_validation = evaluation(val)
    def model_state():
        state = {"beta": beta, "alpha_bar": abar, "condition_mean": cmean, "condition_scale": cscale,
                 "condition_min": condition[train].min(axis=0), "condition_max": condition[train].max(axis=0)}
        state.update({"model_" + key: value for key, value in model.params.items()})
        return state
    for epoch in range(1, int(epochs) + 1):
        check_cancelled()
        permutation = rng.permutation(train)
        losses = []
        for start in range(0, len(train), int(batch_size)):
            check_cancelled()
            indices = permutation[start:start + int(batch_size)]
            times = rng.integers(0, steps, size=len(indices))
            noise = rng.normal(size=(len(indices), dimensions))
            noisy = np.sqrt(abar[times, None]) * data[indices] + np.sqrt(1 - abar[times, None]) * noise
            noisy[~mask[indices]] = 0
            loss, grads = model.gradients(noisy, times, cond[indices], steps, noise, mask[indices])
            losses.append(loss)
            update += 1
            for key in model.params:
                gradient = np.clip(grads[key], -10, 10)
                first[key] = .9 * first[key] + .1 * gradient
                second[key] = .999 * second[key] + .001 * gradient ** 2
                model.params[key] -= learning_rate * (first[key] / (1 - .9 ** update)) / (
                    np.sqrt(second[key] / (1 - .999 ** update)) + 1e-8)
        val_loss = evaluation(val)
        history.append({"epoch": epoch, "train_noise_mse": float(np.mean(losses)), "validation_noise_mse": val_loss})
        if checkpoint_root is not None and (epoch == 1 or epoch % checkpoint_interval == 0 or epoch == epochs):
            total_epoch = completed_before + epoch
            target = Path(checkpoint_root) / f"epoch{total_epoch:05d}"
            if target.exists():
                raise FileExistsError("Refusing to overwrite durable DDPM checkpoint: " + str(target))
            epoch_config = {**config, "completed_epoch": total_epoch, "checkpoint_kind": "training_epoch",
                            "training_status": "incomplete_training",
                            "scientific_status": "INCOMPLETE_TRAINING; RESEARCH_ONLY; not a final evaluated model",
                            "resume_semantics": "weights only; optimizer moments and RNG are not restored exactly"}
            with cancellation(None):
                path = _write_diffusion_checkpoint(target, config=epoch_config, state=model_state(), epoch_metrics=history[-1])
            progress(f"Durable DDPM checkpoint epoch {total_epoch}: {path}")
        check_cancelled()
        if val_loss < best_loss:
            best_loss, best = val_loss, {key: value.copy() for key, value in model.params.items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            progress(f"DDPM epoch {epoch}/{epochs}: noise MSE {history[-1]['train_noise_mse']:.6g}; validation {val_loss:.6g}")
    model.params = best
    config["completed_epoch"] = completed_before + int(epochs)
    report = {"scientific_status": config["scientific_status"], "initial_validation_noise_mse": initial_validation,
              "selected_validation_noise_mse": evaluation(val), "held_out_test_noise_mse": evaluation(test),
              "groups": {key: sorted(set(np.asarray(groups).astype(str)[indices])) for key, indices in split.items()},
              "history": history, "real_material_validation": False}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "diffusion_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (outdir / "diffusion_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    state = model_state()
    np.savez_compressed(outdir / "diffusion_checkpoint.npz", **state)
    np.savez_compressed(outdir / "diffusion_split.npz", **split, groups=np.asarray(groups).astype(str))
    return outdir / "diffusion_checkpoint.npz"


def load_diffusion_checkpoint(modeldir):
    modeldir = Path(modeldir)
    if modeldir.is_file():
        modeldir = modeldir.parent
    config = json.loads((modeldir / "diffusion_config.json").read_text(encoding="utf-8"))
    validate_checkpoint_archive(modeldir, ("diffusion_config.json", "diffusion_checkpoint.npz"), required=bool(config.get("checkpoint_kind")))
    if config.get("schema_version") != 1 or config.get("model") != "conditional_ddpm_numpy_mlp":
        raise ValueError("Unsupported DDPM checkpoint schema")
    if not config.get("synthetic_demo") and not config.get("crystal_symmetry_reduced"):
        raise ValueError("Legacy real DDPM checkpoint lacks phase-specific symmetry reduction; retraining is required")
    _check_conditions(config["condition_names"], config["condition_units"])
    with np.load(modeldir / "diffusion_checkpoint.npz", allow_pickle=False) as stream:
        state = {key: stream[key] for key in stream.files}
    return config, state


@transactional_output()
def sample_diffusion(modeldir, outdir, *, condition, count=1, seed=0, valid_mask=None,
                     allow_condition_extrapolation=False, progress=print):
    """Run the learned reverse diffusion chain, then enforce representation QC."""
    config, state = load_diffusion_checkpoint(modeldir)
    if config.get("training_status") == "incomplete_training":
        raise ValueError("DDPM checkpoint training is incomplete; resume training before final sampling")
    if int(count) != count or not 1 <= count <= 128:
        raise ValueError("Generate from 1 to 128 samples per independent run")
    condition = np.asarray(condition, dtype=float)
    if condition.shape == (len(config["condition_names"]),):
        condition = np.broadcast_to(condition, (count, len(condition))).copy()
    if condition.shape != (count, len(config["condition_names"])) or not np.isfinite(condition).all():
        raise ValueError("Generation conditions must match the trained names/order and be finite")
    for index, name in enumerate(config["condition_names"]):
        if name.startswith("phase_fraction_") and np.any((condition[:, index] < 0) | (condition[:, index] > 1)):
            raise ValueError("Requested phase fraction must lie in [0,1]")
    ood = np.any((condition < state["condition_min"] - 1e-10) | (condition > state["condition_max"] + 1e-10), axis=1)
    if ood.any() and not allow_condition_extrapolation:
        raise ValueError("Condition outside the training range; explicit research extrapolation opt-in required")
    cond = (condition - state["condition_mean"]) / state["condition_scale"]
    h, w, channels = config["shape_hwc"]
    valid = np.ones((count, h, w), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape == (h, w):
        valid = np.broadcast_to(valid, (count, h, w)).copy()
    if valid.shape != (count, h, w) or np.any(valid.reshape(count, -1).sum(axis=1) == 0):
        raise ValueError("Generation mask must contain valid pixels and match count,H,W or H,W")
    flatmask = np.broadcast_to(valid[..., None], (count, h, w, channels)).reshape(count, -1)
    rng = np.random.default_rng(seed)
    model = _NoiseModel(h * w * channels, len(config["condition_names"]), config["hidden"], seed)
    model.params = {key[6:]: value for key, value in state.items() if key.startswith("model_")}
    current = rng.normal(size=(count, h * w * channels))
    current[~flatmask] = 0
    beta, abar = state["beta"], state["alpha_bar"]
    for step in reversed(range(config["steps"])):
        check_cancelled()
        noise = model.predict(current, np.full(count, step), cond, config["steps"])
        x0 = (current - np.sqrt(1 - abar[step]) * noise) / np.sqrt(abar[step])
        x0 = np.clip(x0, -1, 1)  # bounded encoded representation, not a physical loss
        previous = 1. if step == 0 else abar[step - 1]
        coefficient0 = beta[step] * np.sqrt(previous) / (1 - abar[step])
        coefficient1 = (1 - previous) * np.sqrt(1 - beta[step]) / (1 - abar[step])
        current = coefficient0 * x0 + coefficient1 * current
        if step:
            variance = beta[step] * (1 - previous) / (1 - abar[step])
            current += np.sqrt(variance) * rng.normal(size=current.shape)
        current[~flatmask] = 0
        if step % 8 == 0 or step == config["steps"] - 1:
            progress(f"DDPM reverse step {step}/{config['steps'] - 1}")
    fields = current.reshape(count, h, w, channels)
    q = fields[..., :4]
    norm = np.linalg.norm(q, axis=-1)
    repaired = valid & (norm < 1e-12)
    q[repaired] = [0, 0, 0, 1]
    phase = np.asarray(config["allowed_phase_ids"])[np.argmax(fields[..., 4:], axis=-1)]
    phase[~valid] = 0
    q = (_canonical_q(q, valid) if not config.get("crystal_symmetry_reduced") else
         canonicalize_phase_quaternions(q, phase, valid, config["dataset_metadata"]["phase_schema"]))
    qc = {"scientific_status": config["scientific_status"], "synthetic_demo": config["synthetic_demo"],
          "dataset_metadata": config.get("dataset_metadata", {}),
          "orientation_convention": config["orientation_convention"],
          "symmetry_representative_rule": config.get("symmetry_representative_rule", "legacy synthetic antipodal sign only"),
          "seed": int(seed), "condition_names": config["condition_names"], "requested_conditions": condition.tolist(),
          "condition_outside_training_range": ood.tolist(), "mechanical_conditioning": "disabled",
          "physical_realism_validated": False, "representation_qc": {
              "max_unit_quaternion_norm_error": float(np.abs(np.linalg.norm(q[valid], axis=-1) - 1).max()),
              "illegal_valid_phase_pixels": int(np.count_nonzero(~np.isin(phase[valid], config["allowed_phase_ids"]))),
              "zero_quaternion_repairs": int(repaired.sum())},
          "samples": []}
    from scipy.ndimage import label
    for index in range(count):
        components = sum(label((phase[index] == pid) & valid[index])[1] for pid in config["allowed_phase_ids"])
        fractions = {str(pid): float(np.mean(phase[index][valid[index]] == pid)) for pid in config["allowed_phase_ids"]}
        requested = {name.removeprefix("phase_fraction_"): float(condition[index, column])
                     for column, name in enumerate(config["condition_names"]) if name.startswith("phase_fraction_")}
        discrepancy = {pid: fractions.get(pid, 0.) - fraction for pid, fraction in requested.items()}
        qc["samples"].append({"phase_area_fractions": fractions, "requested_phase_fractions": requested,
                              "phase_fraction_discrepancy_actual_minus_requested": discrepancy,
                              "max_absolute_phase_fraction_discrepancy": max(map(abs, discrepancy.values()), default=None),
                              "phase_fraction_condition_enforced": False,
                              "phase_connected_components_4neighbour": int(components),
                              "valid_pixels": int(valid[index].sum())})
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with h5py.File(outdir / "generated_microstructures.h5", "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(qc)
        # Retain float64 at symmetry-sector boundaries: float32 rounding can
        # move a valid representative across a discontinuous canonical seam.
        stream.create_dataset("quaternion_xyzw", data=q, compression="gzip")
        stream.create_dataset("phase_id", data=phase, compression="gzip")
        stream.create_dataset("valid_mask", data=valid, compression="gzip")
    (outdir / "generation_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    progress("DDPM sampling complete; representation QC passed, material realism remains unvalidated")
    return outdir / "generated_microstructures.h5"


def load_diffusion_bundle(path):
    """Numeric NPZ fields and explicit JSON metadata; no image-only training."""
    with np.load(path, allow_pickle=False) as stream:
        required = {"quaternion_xyzw", "phase_id", "groups", "metadata_json"}
        if not required <= set(stream.files):
            raise ValueError("DDPM bundle requires quaternion_xyzw, phase_id, groups and metadata_json")
        meta = json.loads(str(stream["metadata_json"].item()))
        result = {key: stream[key] for key in ("quaternion_xyzw", "phase_id", "groups")}
        for key in ("valid_mask", "condition"):
            if key in stream.files:
                result[key] = stream[key]
    for key in ("allowed_phase_ids", "condition_names", "condition_units", "synthetic_demo"):
        if key in meta:
            result[key] = meta[key]
    result["dataset_metadata"] = {key: meta[key] for key in ("step_um_xy", "phase_schema", "source") if key in meta}
    return result


@transactional_output()
def create_diffusion_bundle(index_files, outdir, *, allowed_phase_ids,
                            synthetic_demo=False, max_patches=10000, progress=print):
    """Convert native M3 patch databases to a numeric orientation/phase bundle.

    Equal pixel dimensions and physical steps are mandatory. Parent specimen
    groups are retained; display IPF RGB channels are never used as orientations.
    """
    from .ml_dataset import PatchDataset
    dataset = PatchDataset(index_files, channels=("qx", "qy", "qz", "qw", "phase_id"))
    if not 1 <= len(dataset) <= max_patches:
        raise ValueError("Patch count is empty or exceeds the explicit CPU bundle limit")
    quaternions, phases, masks, groups, provenance = [], [], [], [], []
    expected_shape, expected_step, expected_phases = None, None, None
    for index in range(len(dataset)):
        check_cancelled()
        sample = dataset[index]
        channels, validity, meta = sample["channels"], sample["masks"], sample["metadata"]
        phase = channels["phase_id"]
        if not np.isfinite(phase).all() or not np.allclose(phase, np.rint(phase)):
            raise ValueError("Patch phase IDs must be integral categorical values")
        phase = np.rint(phase).astype(int)
        step = meta.get("step_um_xy")
        if step is None or len(step) != 2 or not np.isfinite(step).all() or min(step) <= 0:
            raise ValueError("Patch physical grid steps are required")
        specs = meta.get("source_metadata", {}).get("phase_specs", {})
        phase_schema = {str(pid): {key: specs.get(str(pid), {}).get(key) for key in ("name", "space_group")}
                        for pid in allowed_phase_ids}
        if any(value["name"] is None or value["space_group"] is None for value in phase_schema.values()):
            raise ValueError("Declared phases require names and crystal symmetry in original patch metadata")
        if expected_shape is None:
            expected_shape, expected_step, expected_phases = phase.shape, step, phase_schema
        elif phase.shape != expected_shape or not np.allclose(step, expected_step, rtol=1e-8, atol=0) or phase_schema != expected_phases:
            raise ValueError("DDPM patches must share shape, physical scale and phase meanings; explicit resampling/remapping required")
        q = np.stack([channels[key] for key in ("qx", "qy", "qz", "qw")], axis=-1)
        valid = sample["padding_mask"].copy() & validity["phase_id"]
        for key in ("qx", "qy", "qz", "qw"):
            valid &= validity[key]
        valid &= np.isin(phase, list(allowed_phase_ids))
        if not valid.any():
            raise ValueError("Patch contains no valid declared-phase orientations")
        phase[~valid] = 0
        quaternions.append(q)
        phases.append(phase)
        masks.append(valid)
        groups.append(str(meta["split_group"]))
        provenance.append({key: meta[key] for key in ("patch_id", "source_id", "parent_id", "roi_xywh", "pixel_to_sample_um")})
        if index % 100 == 0:
            progress(f"Preparing numeric diffusion patch {index + 1}/{len(dataset)}")
    quaternions, phases, masks = np.asarray(quaternions), np.asarray(phases), np.asarray(masks)
    encode_fields(quaternions, phases, valid_mask=masks, allowed_phase_ids=allowed_phase_ids)
    if int(np.prod(quaternions.shape[1:3])) * (4 + len(allowed_phase_ids)) > 4096:
        raise ValueError("Patches exceed the 4096-variable CPU baseline limit; export smaller physical tiles")
    metadata = {"allowed_phase_ids": list(allowed_phase_ids), "synthetic_demo": bool(synthetic_demo),
                "step_um_xy": list(expected_step), "phase_schema": expected_phases,
                "source": "native numeric M3 patch database; no RGB image orientation encoding"}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "diffusion_bundle.npz", quaternion_xyzw=quaternions,
                        phase_id=phases, valid_mask=masks, groups=np.asarray(groups), metadata_json=json.dumps(metadata))
    (outdir / "bundle_provenance.json").write_text(json.dumps({**metadata, "patches": provenance}, indent=2), encoding="utf-8")
    return outdir / "diffusion_bundle.npz"


@transactional_output()
def create_synthetic_diffusion_bundle(outdir, *, seed=42):
    """Synthetic quaternion/phase tiles for reproducible CPU smoke testing."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(24, 4, 4, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    yy, xx = np.indices((4, 4))
    phase = np.broadcast_to(1 + (xx + yy) % 2, (len(q), 4, 4)).copy().astype(int)
    groups = np.repeat([f"SYNTHETIC-specimen-{i}" for i in range(6)], 4)
    meta = {"allowed_phase_ids": [1, 2], "synthetic_demo": True, "step_um_xy": [1., 1.],
            "phase_schema": {"1": {"name": "synthetic alpha", "space_group": 194},
                             "2": {"name": "synthetic beta", "space_group": 229}},
            "source": "synthetic quaternion/phase QA; no physical microstructure validation"}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "synthetic_diffusion_bundle.npz", quaternion_xyzw=q, phase_id=phase,
                        groups=groups, valid_mask=np.ones_like(phase, dtype=bool), metadata_json=json.dumps(meta))
    (outdir / "SYNTHETIC_README.txt").write_text("Synthetic numeric orientation/phase DDPM demo; physical realism not validated.\n", encoding="utf-8")
    return outdir / "synthetic_diffusion_bundle.npz"


def _main():
    import argparse
    parser = argparse.ArgumentParser(description="V4 conditional DDPM CPU research baseline")
    sub = parser.add_subparsers(dest="action", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--out", required=True)
    train = sub.add_parser("train")
    train.add_argument("--bundle", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--steps", type=int, default=32)
    train.add_argument("--resume-from")
    sample = sub.add_parser("sample")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--out", required=True)
    sample.add_argument("--condition", type=float, nargs="+", required=True)
    sample.add_argument("--count", type=int, default=1)
    sample.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.action == "demo":
        print(create_synthetic_diffusion_bundle(args.out))
    elif args.action == "train":
        print(train_diffusion(**load_diffusion_bundle(args.bundle), outdir=args.out, epochs=args.epochs,
                              steps=args.steps, resume_from=args.resume_from))
    else:
        print(sample_diffusion(args.checkpoint, args.out, condition=args.condition, count=args.count, seed=args.seed))


if __name__ == "__main__":
    _main()
