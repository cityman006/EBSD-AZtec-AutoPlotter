"""Fresh numerical/contract tests for the explicitly unvalidated M4 baselines."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest


def test_graph_weighted_mean_subunit_lengths_and_scale_invariance():
    from ebsd_autoplot.ml_models.baselines import graph_design
    x=np.array([[2.],[4.],[0.]])
    edges=np.array([[0,1],[2,2]])
    small=graph_design(x,edges,edge_weights=np.array([.1,.1]))
    large=graph_design(x,edges,edge_weights=np.array([10.,10.]))
    assert small[2,1]==pytest.approx(3.)
    assert np.allclose(small,large)
    assert np.array_equal(small[:2,1], [0.,0.])
from ebsd_autoplot.diffusion import (create_diffusion_bundle, create_synthetic_diffusion_bundle,
                                     canonicalize_phase_quaternions,
                                     encode_fields, load_diffusion_bundle,
                                     load_diffusion_checkpoint, sample_diffusion, train_diffusion)
from ebsd_autoplot.mechanics import create_graph_training_bundle, import_mechanics, validate_metadata
from ebsd_autoplot.ml_models import graph_design
from ebsd_autoplot.ml_train import (group_split, infer_supervised, load_checkpoint,
                                    load_grain_graph, load_training_bundle,
                                    train_supervised, create_synthetic_training_bundle)
from ebsd_autoplot.storage import Cancelled, cancellation, validate_receipt


def mechanics_metadata():
    return {"source_type": "CPFE", "source_id": "specimen-A", "material": "synthetic Ti alloy import fixture", "reference_frame": "EBSD Sample-X/Y",
            "coordinate_unit": "mm", "load_step": 2, "time": {"value": 12., "unit": "s"},
            "loading_condition": "uniaxial tensile along Sample-X",
            "uncertainty": {"description": "synthetic test input; real accuracy not asserted"},
            "field_units": {"sigma_xx": "Pa", "epsilon_xx": "%"},
            "field_kinds": {"sigma_xx": "stress", "epsilon_xx": "strain"},
            "source_to_sample_um": [[1, 0, 5], [0, 1, -2], [0, 0, 1]]}


def demo_metadata():
    return {"source_type": "SYNTHETIC", "source_id": "deterministic-test",
            "material": "synthetic analytic QA material",
            "reference_frame": "Sample-X/Y", "load_step": 1,
            "time": {"value": 0., "unit": "s"}, "loading_condition": "synthetic analytic regression",
            "uncertainty": {"description": "synthetic values; no real mechanics validation"}}


def training_data():
    rng = np.random.default_rng(41)
    groups = np.repeat([f"source-{index}" for index in range(6)], 16)
    x = rng.normal(size=(len(groups), 3))
    edges = []
    for start in range(0, len(groups), 16):
        for offset in range(16):
            edges.append((start + offset, start + (offset + 1) % 16))
            edges.append((start + (offset + 1) % 16, start + offset))
    edges = np.asarray(edges).T
    neighbour = graph_design(x, edges)[:, 3:]
    y = np.column_stack([200 + 60 * x[:, 0] + 30 * neighbour[:, 1], .01 * (x[:, 0] - x[:, 2])])
    return x, y, groups, edges


def train_kwargs():
    return {"feature_names": ["a", "b", "c"], "feature_units": ["1"] * 3,
            "target_names": ["sigma_xx", "epsilon_xx"], "target_units": ["MPa", "1"],
            "label_metadata": demo_metadata(), "demo": True, "progress": lambda _: None}


def test_mechanics_csv_declared_units_coordinates_and_masks(tmp_path):
    source = tmp_path / "labels.csv"
    source.write_text("X,Y,S,E,V\n0.1,0.2,200000000,2,1\n0.2,0.3,1000000,,1\n0.3,0.4,0,0,0\n", encoding="utf-8")
    output = tmp_path / "import"
    path = import_mechanics(source, output, metadata=mechanics_metadata(),
                            columns={"x": "X", "y": "Y", "sigma_xx": "S", "epsilon_xx": "E", "valid": "V"},
                            progress=lambda _: None)
    with h5py.File(path, "r") as stream:
        np.testing.assert_allclose(stream["coordinates_sample_um"][0], [105, 198])
        np.testing.assert_allclose(stream["labels/sigma_xx"][:], [200, 1, 0])
        assert stream["labels/epsilon_xx"][0] == .02
        np.testing.assert_array_equal(stream["masks/epsilon_xx"][:], [True, False, False])
        assert stream["labels/sigma_xx"].attrs["unit"] == "MPa"
    assert validate_receipt(output, "ebsd_autoplot.mechanics.import_mechanics")


@pytest.mark.parametrize("suffix", [".npy", ".h5"])
def test_mechanics_npy_and_hdf5_numeric_inputs(tmp_path, suffix):
    source = tmp_path / ("input" + suffix)
    matrix = np.array([[0., 0., 2e6, 1.], [1., 2., 3e6, 2.]])
    keys = ["x", "y", "sigma_xx", "epsilon_xx"]
    if suffix == ".npy":
        np.save(source, matrix)
        columns = dict(zip(keys, range(4)))
    else:
        with h5py.File(source, "w") as stream:
            for index, key in enumerate(keys):
                stream.create_dataset("input/" + key, data=matrix[:, index])
        columns = {key: "input/" + key for key in keys}
    path = import_mechanics(source, tmp_path / "out", metadata=mechanics_metadata(), columns=columns,
                            progress=lambda _: None)
    with h5py.File(path, "r") as stream:
        np.testing.assert_allclose(stream["labels/sigma_xx"][:], [2, 3])


def test_mechanics_never_guesses_units_or_uses_proxy_labels():
    meta = mechanics_metadata()
    del meta["coordinate_unit"]
    with pytest.raises(ValueError, match="missing"):
        validate_metadata(meta)
    meta = mechanics_metadata()
    meta["source_type"] = "KAM_PROXY"
    with pytest.raises(ValueError, match="not labels"):
        validate_metadata(meta)
    meta = mechanics_metadata()
    meta["field_units"]["epsilon_xx"] = "auto"
    with pytest.raises(ValueError, match="Unsupported"):
        validate_metadata(meta)


def test_dic_stress_requires_independent_derivation_provenance(tmp_path):
    meta = mechanics_metadata()
    meta["source_type"] = "DIC"
    with pytest.raises(ValueError, match="DIC does not directly measure stress"):
        validate_metadata(meta)
    meta["field_provenance"] = {"sigma_xx": {"source_type": "FEM", "source_id": "independent-calibrated-model",
        "method": "externally supplied finite-element solution registered to DIC coordinates",
        "uncertainty": {"description": "external model uncertainty is supplied by its author"}}}
    assert validate_metadata(meta)["field_provenance"]["sigma_xx"]["source_type"] == "FEM"
    x, y, groups, _ = training_data()
    kwargs = train_kwargs()
    kwargs.update({"demo": False, "label_metadata": {**demo_metadata(), "source_type": "DIC"}})
    with pytest.raises(ValueError, match="independent field_provenance"):
        train_supervised(x, y, groups, tmp_path / "dic_not_stress", **kwargs)


def test_group_split_parent_separation_and_minimum():
    groups = np.repeat(["parent-A", "parent-B", "parent-C", "parent-D"], 8)
    split = group_split(groups, seed=4)
    sets = [set(groups[indices]) for indices in split.values()]
    assert all(not sets[a] & sets[b] for a in range(3) for b in range(a))
    assert sorted(np.concatenate(list(split.values()))) == list(range(len(groups)))
    with pytest.raises(ValueError, match="3 independent"):
        group_split(["one", "two"])


def test_supervised_missing_labels_and_proxies_are_rejected(tmp_path):
    x, y, groups, edges = training_data()
    with pytest.raises(ValueError, match="without actual"):
        train_supervised(x, None, groups, tmp_path / "unlabeled", **train_kwargs())
    kwargs = train_kwargs()
    kwargs["label_metadata"] = {**demo_metadata(), "source_type": "EBSD_GOS"}
    kwargs["demo"] = False
    with pytest.raises(ValueError, match="not labels"):
        train_supervised(x, y, groups, tmp_path / "proxy", **kwargs)
    assert not (tmp_path / "unlabeled" / "checkpoint.npz").exists()


def test_graph_message_aggregation_independent_small_example():
    x = np.array([[1.], [3.], [8.]])
    edge = np.array([[0, 2, 1], [1, 1, 0]])
    result = graph_design(x, edge, edge_weights=[1, 3, 1])
    np.testing.assert_allclose(result, [[1, 3], [3, 6.25], [8, 0]])


def test_gnn_cross_source_edges_rejected_before_training(tmp_path):
    x, y, groups, _ = training_data()
    with pytest.raises(ValueError, match="cross-source"):
        train_supervised(x, y, groups, tmp_path / "leak", edge_index=np.array([[0], [20]]), **train_kwargs())


def test_ridge_train_only_scaler_and_exact_linear_inference(tmp_path):
    x, _, groups, _ = training_data()
    split = group_split(groups, seed=0)
    x[split["test"], 0] += 10  # scaler must exclude this held-out shift
    y = np.column_stack([100 + 4 * x[:, 0] - 2 * x[:, 1], .01 * x[:, 2]])
    output = tmp_path / "fit"
    train_supervised(x, y, groups, output, model="ridge", ridge_alpha=1e-8, **train_kwargs())
    config, state = load_checkpoint(output)
    np.testing.assert_allclose(state["x_mean"], x[split["train"]].mean(axis=0))
    assert not np.allclose(state["x_mean"], x.mean(axis=0))
    result = infer_supervised(output, x, tmp_path / "prediction", feature_names=config["feature_names"], feature_units=config["feature_units"])
    with h5py.File(result, "r") as stream:
        np.testing.assert_allclose(stream["prediction"][:], y, rtol=1e-7, atol=1e-7)
    report = json.loads((output / "training_report.json").read_text())
    assert "high_strain" in report["metrics"]["test"]["epsilon_xx"]
    assert report["metrics"]["test"]["sigma_xx"]["RMSE"] < 1e-6
    assert validate_receipt(output, "ebsd_autoplot.ml_train.train_supervised")


def test_gnn_is_learned_checkpoint_inference_and_mask(tmp_path):
    x, y, groups, edges = training_data()
    y[::7, 0] = np.nan
    output = tmp_path / "gnn"
    train_supervised(x, y, groups, output, edge_index=edges, epochs=180, hidden=24, **train_kwargs())
    report = json.loads((output / "training_report.json").read_text())
    assert report["history"][-1]["train_mse_scaled"] < report["history"][0]["train_mse_scaled"] * .1
    assert report["metrics"]["test"]["sigma_xx"]["RMSE"] < 35
    config, state = load_checkpoint(output)
    assert np.linalg.norm(state["model_b1"]) > 0
    path = infer_supervised(output, x, tmp_path / "gnn_predict", edge_index=edges,
                            feature_names=config["feature_names"], feature_units=config["feature_units"])
    with h5py.File(output / "evaluation.h5", "r") as evaluation, h5py.File(path, "r") as prediction:
        np.testing.assert_allclose(evaluation["prediction"][:], prediction["prediction"][:])
        assert not evaluation["label_mask"][0, 0]
    with pytest.raises(ValueError, match="names/order/units"):
        infer_supervised(output, x, tmp_path / "bad_schema", feature_names=["c", "a", "b"])


def test_inference_mask_and_real_map_exports(tmp_path):
    x, y, groups, _ = training_data()
    fit = tmp_path / "fit"
    train_supervised(x, y, groups, fit, model="ridge", **train_kwargs())
    invalid = x[:4].copy()
    invalid[0, 0] = np.nan
    out = tmp_path / "maps"
    result = infer_supervised(fit, invalid, out, feature_names=["a", "b", "c"], map_shape=(2, 2),
                              formats=("png", "tiff", "pdf", "svg"), progress=lambda _: None)
    with h5py.File(result, "r") as stream:
        assert np.isnan(stream["prediction"][0]).all()
        assert not stream["valid_mask"][0]
    from PIL import Image
    import xml.etree.ElementTree as ET
    for index in range(2):
        for kind in ("prediction", "uncertainty"):
            Image.open(out / f"target_{index}_{kind}.png").verify()
            Image.open(out / f"target_{index}_{kind}.tiff").verify()
            assert (out / f"target_{index}_{kind}.pdf").read_bytes().startswith(b"%PDF")
            ET.parse(out / f"target_{index}_{kind}.svg")


def test_training_bundle_explicit_metadata_and_unlabelled_gate(tmp_path):
    x, y, groups, edges = training_data()
    meta = {key: value for key, value in train_kwargs().items() if key in {
        "feature_names", "feature_units", "target_names", "target_units", "label_metadata"}}
    meta["synthetic_demo"] = True
    path = tmp_path / "bundle.npz"
    np.savez(path, features=x, labels=y, groups=groups, edge_index=edges, metadata_json=json.dumps(meta))
    bundle = load_training_bundle(path)
    assert bundle["demo"]
    assert bundle["edge_index"].shape == edges.shape
    np.savez(path, features=x, groups=groups, metadata_json=json.dumps(meta))
    with pytest.raises(ValueError, match="unlabeled training disabled"):
        load_training_bundle(path)


def test_training_cancel_preserves_final_output_absence(tmp_path):
    x, y, groups, edges = training_data()
    with cancellation(lambda: True), pytest.raises(Cancelled):
        train_supervised(x, y, groups, tmp_path / "cancelled", edge_index=edges, **train_kwargs())
    assert not (tmp_path / "cancelled" / "checkpoint.npz").exists()


def diffusion_data():
    rng = np.random.default_rng(12)
    q = rng.normal(size=(18, 2, 2, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    phase = np.tile(np.array([[1, 2], [1, 2]]), (18, 1, 1))
    valid = np.ones_like(phase, dtype=bool)
    valid[:, 1, 1] = False
    phase[~valid] = 0
    return q, phase, np.repeat(["specimen-A", "specimen-B", "specimen-C", "specimen-D", "specimen-E", "specimen-F"], 3), valid


def test_diffusion_orientation_and_phase_representation():
    q, phase, _, valid = diffusion_data()
    a, mask, _ = encode_fields(q, phase, valid_mask=valid)
    b, _, _ = encode_fields(-q, phase, valid_mask=valid)
    np.testing.assert_allclose(a, b)
    assert not a[~mask].any()
    phase[0, 0, 0] = 7
    with pytest.raises(ValueError, match="legal phase"):
        encode_fields(q, phase, valid_mask=valid)


def test_ddpm_actual_training_sampling_and_reproducibility(tmp_path):
    q, phase, groups, valid = diffusion_data()
    fit = tmp_path / "ddpm"
    train_diffusion(q, phase, groups, fit, valid_mask=valid, epochs=60, steps=8, hidden=32,
                     batch_size=6, seed=12, synthetic_demo=True, progress=lambda _: None)
    report = json.loads((fit / "diffusion_training_report.json").read_text())
    assert report["selected_validation_noise_mse"] < report["initial_validation_noise_mse"]
    assert np.isfinite(report["held_out_test_noise_mse"])
    assert not report["real_material_validation"]
    arrays = []
    for folder in ("sample1", "sample2"):
        path = sample_diffusion(fit, tmp_path / folder, condition=[2 / 3, 1 / 3], count=2,
                                 valid_mask=valid[0], seed=44, progress=lambda _: None)
        with h5py.File(path, "r") as stream:
            generated = stream["quaternion_xyzw"][:]
            mask = stream["valid_mask"][:]
            np.testing.assert_allclose(np.linalg.norm(generated[mask], axis=-1), 1, atol=1e-6)
            assert not generated[~mask].any()
            assert set(np.unique(stream["phase_id"][:])) <= {0, 1, 2}
            arrays.append(generated)
    np.testing.assert_array_equal(*arrays)
    qc = json.loads((tmp_path / "sample1" / "generation_qc.json").read_text())
    assert qc["representation_qc"]["illegal_valid_phase_pixels"] == 0
    assert not qc["physical_realism_validated"]
    assert validate_receipt(fit, "ebsd_autoplot.diffusion.train_diffusion")
    with pytest.raises(ValueError, match="outside the training range"):
        sample_diffusion(fit, tmp_path / "ood", condition=[.1, .9], progress=lambda _: None)


def test_ddpm_mechanical_conditions_are_blocked(tmp_path):
    q, phase, groups, valid = diffusion_data()
    with pytest.raises(ValueError, match="Mechanical conditioning disabled"):
        train_diffusion(q, phase, groups, tmp_path / "illegal", valid_mask=valid,
                         condition=np.ones((len(q), 1)), condition_names=["yield_stress"], condition_units=["MPa"])


def test_ddpm_bundle_no_arbitrary_png_and_metadata(tmp_path):
    q, phase, groups, valid = diffusion_data()
    path = tmp_path / "texture_fields.npz"
    np.savez(path, quaternion_xyzw=q, phase_id=phase, groups=groups, valid_mask=valid,
             metadata_json=json.dumps({"allowed_phase_ids": [1, 2], "synthetic_demo": True}))
    bundle = load_diffusion_bundle(path)
    assert bundle["quaternion_xyzw"].shape[-1] == 4
    assert bundle["synthetic_demo"]


def test_grain_graph_loader_uses_v2_adjacency_and_sign_canonicalization(tmp_path):
    (tmp_path / "grains.csv").write_text(
        "grain_id,eq_diameter_um,area_um2,GOS_deg,phase_id,mean_qx,mean_qy,mean_qz,mean_qw\n"
        "3,2,3.14159,0.2,1,0,0,0,-1\n5,3,7.06858,0.3,1,0,0,0,1\n", encoding="utf-8")
    (tmp_path / "grain_neighbour_graph.csv").write_text("grain_a,grain_b,shared_pixel_edges\n3,5,4\n", encoding="utf-8")
    graph = load_grain_graph(tmp_path, source_id="real-parent-A")
    np.testing.assert_array_equal(graph["edge_index"], [[0, 1], [1, 0]])
    np.testing.assert_allclose(graph["features"][:, -1], [1, 1])
    assert set(graph["groups"]) == {"real-parent-A"}
    assert "labels" not in graph


def test_real_graph_mechanics_join_masks_distance_and_grouping(tmp_path):
    graph_files, mechanics_files = [], []
    for index in range(3):
        source = tmp_path / f"mechanics{index}.csv"
        source.write_text("X,Y,S,E\n0.5,0,200000000,2\n", encoding="utf-8")
        metadata = mechanics_metadata()
        metadata.update({"coordinate_unit": "um", "source_id": f"source-{index}", "source_to_sample_um": np.eye(3).tolist()})
        mechanics_files.append(import_mechanics(source, tmp_path / f"mechanics{index}", metadata=metadata,
                                                columns={"x": "X", "y": "Y", "sigma_xx": "S", "epsilon_xx": "E"}, progress=lambda _: None))
        graph_file = tmp_path / f"graph{index}.npz"
        np.savez(graph_file, node_feature_names=np.array(["area_um2", "centroid_x_um", "centroid_y_um"]),
                 node_features=np.array([[1., 0., 0.], [1., 5., 0.], [np.nan, 10., 0.]]),
                 node_feature_mask=np.array([[1, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=bool),
                 source_id=f"source-{index}", node_ids=np.array([7, 9, 11]), node_phase_id=np.array([1, 1, 1]),
                 orientation_xyzw=np.tile([0., 0., 0., 1.], (3, 1)), edge_index=np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
                 edge_attr=np.ones((4, 5)), metadata_json=json.dumps({"phase_schema": {"1": {"name": "alpha Ti", "space_group": 194}}}))
        graph_files.append(graph_file)
    path = create_graph_training_bundle(graph_files, mechanics_files, tmp_path / "joined", target_names=["sigma_xx", "epsilon_xx"],
                                         max_distance_um=1., reference_frame="EBSD Sample-X/Y", progress=lambda _: None)
    bundle = load_training_bundle(path)
    assert bundle["features"].shape == (6, 8)
    np.testing.assert_array_equal(bundle["label_mask"][:, 0], [True, False] * 3)
    assert set(bundle["groups"]) == {"source-0", "source-1", "source-2"}
    assert bundle["edge_index"].shape == (2, 6)
    with pytest.raises(ValueError, match="reference frame"):
        create_graph_training_bundle(graph_files, mechanics_files, tmp_path / "wrong_frame", target_names=["sigma_xx"],
                                      max_distance_um=1., reference_frame="UNKNOWN", progress=lambda _: None)
    with pytest.raises(ValueError, match="specimen/source mismatch"):
        create_graph_training_bundle(graph_files, mechanics_files[1:] + mechanics_files[:1], tmp_path / "wrong_specimen",
                                      target_names=["sigma_xx"], max_distance_um=1., reference_frame="EBSD Sample-X/Y", progress=lambda _: None)
    # Independent modality IDs may differ, but the EBSD association must be explicit.
    with h5py.File(mechanics_files[0], "r+") as stream:
        meta = json.loads(stream.attrs["metadata_json"])
        meta.update({"source_id": "CPFE-run-123", "ebsd_source_id": "source-0"})
        stream.attrs["metadata_json"] = json.dumps(meta)
    path = create_graph_training_bundle(graph_files, mechanics_files, tmp_path / "explicit_pairing", target_names=["sigma_xx"],
                                         max_distance_um=1., reference_frame="EBSD Sample-X/Y", progress=lambda _: None)
    assert path.exists()


def test_native_patch_to_diffusion_bundle_preserves_numeric_fields_and_groups(tmp_path):
    entries = []
    for index in range(6):
        path = tmp_path / f"patch{index}.h5"
        phase = np.array([[1, 2], [1, 0]])
        valid = phase > 0
        metadata = {"patch_id": f"patch{index}", "source_id": f"source-{index // 2}", "parent_id": f"source-{index // 2}",
                    "split_group": f"source-{index // 2}", "roi_xywh": [0, 0, 2, 2], "pixel_to_sample_um": np.eye(3).tolist(),
                    "step_um_xy": [1., 1.], "source_metadata": {"phase_specs": {
                        "1": {"name": "alpha", "space_group": 194}, "2": {"name": "beta", "space_group": 229}}}}
        with h5py.File(path, "w") as stream:
            stream.attrs["metadata_json"] = json.dumps(metadata)
            stream.create_dataset("padding_mask", data=valid)
            for key, value in {"qx": np.zeros((2, 2)), "qy": np.zeros((2, 2)), "qz": np.zeros((2, 2)), "qw": valid.astype(float), "phase_id": phase}.items():
                stream.create_dataset("channels/" + key, data=value)
                stream.create_dataset("masks/" + key, data=valid)
        entries.append({"file": path.name, "source_id": metadata["source_id"], "parent_id": metadata["parent_id"],
                        "split_group": metadata["split_group"], "channel_names": ["qx", "qy", "qz", "qw", "phase_id"]})
    index_file = tmp_path / "dataset_index.json"
    index_file.write_text(json.dumps({"schema_version": 1, "entries": entries}), encoding="utf-8")
    bundle_file = create_diffusion_bundle(index_file, tmp_path / "native_bundle", allowed_phase_ids=[1, 2], progress=lambda _: None)
    bundle = load_diffusion_bundle(bundle_file)
    np.testing.assert_array_equal(bundle["phase_id"][0], [[1, 2], [1, 0]])
    assert bundle["dataset_metadata"]["step_um_xy"] == [1., 1.]
    assert set(bundle["groups"]) == {"source-0", "source-1", "source-2"}
    train_diffusion(**bundle, outdir=tmp_path / "native_fit", epochs=2, steps=4, progress=lambda _: None)


def test_synthetic_qa_bundle_creators_are_explicit_and_trainable(tmp_path):
    supervised = create_synthetic_training_bundle(tmp_path / "supervised")
    bundle = load_training_bundle(supervised)
    assert bundle["demo"] and bundle["label_metadata"]["source_type"] == "SYNTHETIC"
    diffusion = create_synthetic_diffusion_bundle(tmp_path / "diffusion")
    fields = load_diffusion_bundle(diffusion)
    assert fields["synthetic_demo"]
    assert len(set(fields["groups"])) >= 3


def test_supervised_resume_rejects_changed_topology_and_allows_compatible(tmp_path):
    x, y, groups, edges = training_data()
    original = tmp_path / "original"
    train_supervised(x, y, groups, original, edge_index=edges, epochs=2, **train_kwargs())
    train_supervised(x, y, groups, tmp_path / "resumed", edge_index=edges, epochs=2, resume_from=original, **train_kwargs())
    with pytest.raises(ValueError, match="dataset_sha256"):
        train_supervised(x, y, groups, tmp_path / "changed_graph", edge_index=edges[:, :-1], epochs=2, resume_from=original, **train_kwargs())


def test_diffusion_resume_rejects_changed_physical_metadata(tmp_path):
    q, phase, groups, valid = diffusion_data()
    original = tmp_path / "original"
    train_diffusion(q, phase, groups, original, valid_mask=valid, epochs=1, steps=4,
                     synthetic_demo=True, dataset_metadata={"step_um_xy": [1, 1]}, progress=lambda _: None)
    with pytest.raises(ValueError, match="dataset_metadata"):
        train_diffusion(q, phase, groups, tmp_path / "wrong_scale", valid_mask=valid, epochs=1, steps=4,
                         synthetic_demo=True, dataset_metadata={"step_um_xy": [2, 2]}, resume_from=original, progress=lambda _: None)


def test_gnn_cancel_retains_committed_epoch_and_can_resume(tmp_path):
    x, y, groups, edges = training_data()
    checkpoints = tmp_path / "run" / "checkpoints"
    first = checkpoints / "epoch00001"
    def stop_after_checkpoint():
        return any((first / ".transactions").glob("*.json"))
    with cancellation(stop_after_checkpoint), pytest.raises(Cancelled):
        train_supervised(x, y, groups, tmp_path / "run" / "results", edge_index=edges, epochs=10,
                         checkpoint_root=checkpoints, **train_kwargs())
    assert not (tmp_path / "run" / "results" / "checkpoint.npz").exists()
    assert validate_receipt(first, "ebsd_autoplot.ml_train._write_training_checkpoint")
    config, state = load_checkpoint(first)
    assert config["completed_epoch"] == 1 and config["training_status"] == "incomplete_training"
    assert config["checkpoint_kind"] == "training_epoch"
    assert "x_mean" in state and "model_w1" in state
    with pytest.raises(ValueError, match="incomplete"):
        infer_supervised(first, x, tmp_path / "premature_inference", feature_names=["a", "b", "c"], edge_index=edges)
    train_supervised(x, y, groups, tmp_path / "resumed" / "results", edge_index=edges, epochs=2,
                     resume_from=first, checkpoint_root=tmp_path / "resumed" / "checkpoints", **train_kwargs())
    resumed, _ = load_checkpoint(tmp_path / "resumed" / "results")
    assert resumed["completed_epoch"] == 3 and resumed["training_status"] == "complete"
    # A reused checkpoint root must fail rather than replacing epoch00001.
    original = (first / "checkpoint.npz").read_bytes()
    with pytest.raises(FileExistsError, match="overwrite durable"):
        train_supervised(x, y, groups, tmp_path / "collision" / "results", edge_index=edges, epochs=2,
                         checkpoint_root=checkpoints, **train_kwargs())
    assert (first / "checkpoint.npz").read_bytes() == original


def test_ddpm_cancel_retains_committed_epoch_and_can_resume(tmp_path):
    q, phase, groups, valid = diffusion_data()
    root = tmp_path / "run" / "checkpoints"
    first = root / "epoch00001"
    with cancellation(lambda: any((first / ".transactions").glob("*.json"))), pytest.raises(Cancelled):
        train_diffusion(q, phase, groups, tmp_path / "run" / "results", valid_mask=valid, epochs=10, steps=4,
                         synthetic_demo=True, checkpoint_root=root, progress=lambda _: None)
    assert not (tmp_path / "run" / "results" / "diffusion_checkpoint.npz").exists()
    assert validate_receipt(first, "ebsd_autoplot.diffusion._write_diffusion_checkpoint")
    config, state = load_diffusion_checkpoint(first)
    assert config["completed_epoch"] == 1 and config["training_status"] == "incomplete_training"
    assert "condition_mean" in state and "model_w1" in state
    with pytest.raises(ValueError, match="incomplete"):
        sample_diffusion(first, tmp_path / "premature_generation", condition=[2/3, 1/3])
    train_diffusion(q, phase, groups, tmp_path / "resumed" / "results", valid_mask=valid, epochs=2, steps=4,
                     synthetic_demo=True, resume_from=first, checkpoint_root=tmp_path / "resumed" / "checkpoints", progress=lambda _: None)
    config, _ = load_diffusion_checkpoint(tmp_path / "resumed" / "results")
    assert config["completed_epoch"] == 3 and config["training_status"] == "complete"


def test_durable_checkpoint_digest_rejects_tampered_weights(tmp_path):
    x, y, groups, edges = training_data()
    root = tmp_path / "checkpoints"
    train_supervised(x, y, groups, tmp_path / "model", edge_index=edges, epochs=1,
                     checkpoint_root=root, **train_kwargs())
    first = root / "epoch00001"
    with (first / "checkpoint.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="differs from its committed receipt"):
        load_checkpoint(first)


def test_real_inference_requires_matching_material_loading_and_frame(tmp_path):
    x, y, groups, _ = training_data()
    kwargs = train_kwargs()
    provenance = {**demo_metadata(), "source_type": "CPFE", "material": "Ti-6Al-4V"}
    kwargs.update(demo=False, label_metadata=provenance)
    model = tmp_path / "model"
    train_supervised(x, y, groups, model, model="ridge", **kwargs)
    infer_kwargs = {"feature_names": ["a", "b", "c"], "feature_units": ["1"] * 3}
    with pytest.raises(ValueError, match="requires explicit material"):
        infer_supervised(model, x, tmp_path / "missing_context", **infer_kwargs)
    context = {key: provenance[key] for key in ("material", "loading_condition", "reference_frame", "load_step", "time")}
    for key in context:
        changed = {**context, key: ("different " + context[key] if isinstance(context[key], str)
                                   else {"value": 99, "unit": "s"} if key == "time" else 99)}
        with pytest.raises(ValueError, match="context incompatible.*" + key):
            infer_supervised(model, x, tmp_path / ("different_" + key), inference_metadata=changed, **infer_kwargs)
    path = infer_supervised(model, x, tmp_path / "valid_context", inference_metadata=context, **infer_kwargs)
    with h5py.File(path, "r") as stream:
        assert json.loads(stream.attrs["metadata_json"])["inference_metadata"] == context
    provenance.pop("material")
    kwargs["label_metadata"] = provenance
    with pytest.raises(ValueError, match="material"):
        train_supervised(x, y, groups, tmp_path / "missing_training_material", **kwargs)


def test_research_adapter_owns_checkpoint_root_and_passes_unlabelled_context(tmp_path):
    from ebsd_autoplot.research_actions import run_research_action
    x, y, groups, _ = training_data()
    kwargs = train_kwargs()
    provenance = {**demo_metadata(), "source_type": "CPFE", "material": "Ti-6Al-4V"}
    metadata = {"feature_names": kwargs["feature_names"], "feature_units": kwargs["feature_units"],
                "target_names": kwargs["target_names"], "target_units": kwargs["target_units"], "label_metadata": provenance}
    bundle = tmp_path / "bundle.npz"
    np.savez(bundle, features=x, labels=y, groups=groups, metadata_json=json.dumps(metadata))
    run = tmp_path / "train_run"
    run_research_action("train-supervised", None, run, {"bundle_path": str(bundle), "model": "ridge"})
    assert (run / "checkpoints" / "epoch00001" / "checkpoint.npz").is_file()
    metadata["inference_metadata"] = {key: provenance[key] for key in ("material", "loading_condition", "reference_frame", "load_step", "time")}
    unlabeled = tmp_path / "unlabeled.npz"
    np.savez(unlabeled, features=x, metadata_json=json.dumps(metadata))
    output = run_research_action("infer-supervised", None, tmp_path / "infer_run",
                                 {"bundle_path": str(unlabeled), "checkpoint": str(run / "results")})
    assert output["status"] == "success" and not output["synthetic_demo"]
    with pytest.raises(ValueError, match="Unknown research configuration"):
        run_research_action("train-supervised", None, tmp_path / "bad_override",
                            {"bundle_path": str(bundle), "checkpoint_root": "not-owned-by-run"})


def test_phase_symmetry_canonicalization_hcp_bcc_arbitrary_ids_invariant():
    from scipy.spatial.transform import Rotation
    from ebsd_autoplot.v3_crystal import point_group
    rng = np.random.default_rng(821)
    q = Rotation.random(12, random_state=rng).as_quat().reshape(3, 2, 2, 4)
    phase = np.tile(np.array([[7, 42], [42, 7]]), (3, 1, 1))
    valid = np.ones(phase.shape, dtype=bool)
    schema = {7: {"name": "alpha Ti", "space_group": 194}, 42: {"name": "beta Ti", "space_group": 229}}
    baseline, _, _ = encode_fields(q, phase, phase_schema=schema, allowed_phase_ids=[7, 42])
    # A right crystal symmetry must not change the learned orientation encoding.
    for pid, sg in ((7, 194), (42, 229)):
        positions = phase == pid
        for operation in point_group(sg):
            equivalent = q.copy()
            equivalent[positions] = (Rotation.from_quat(q[positions]) * operation).as_quat()
            encoded, _, _ = encode_fields(equivalent, phase, phase_schema=schema, allowed_phase_ids=[7, 42])
            np.testing.assert_allclose(encoded, baseline, atol=2e-9, rtol=0)
    # Source arrays have not been normalized/reduced in place.
    original = q.copy()
    reduced = canonicalize_phase_quaternions(q, phase, valid, schema)
    np.testing.assert_array_equal(q, original)
    np.testing.assert_allclose(np.linalg.norm(reduced, axis=-1), 1, atol=1e-12)
    np.testing.assert_allclose(canonicalize_phase_quaternions(reduced, phase, valid, schema), reduced, atol=2e-9)


def test_real_ddpm_requires_supported_phase_symmetry_and_canonical_samples(tmp_path):
    q, phase, groups, valid = diffusion_data()
    # Exercise arbitrary raster IDs rather than assuming alpha=1, beta=2.
    phase = np.where(phase == 1, 7, np.where(phase == 2, 42, 0))
    metadata = {"step_um_xy": [1., 2.], "phase_schema": {
        "7": {"name": "alpha Ti", "space_group": 194}, "42": {"name": "beta Ti", "space_group": 229}}}
    bad = {**metadata, "phase_schema": {"7": {"name": "alpha Ti"}, "42": {"space_group": 229}}}
    with pytest.raises(ValueError, match="needs space_group"):
        train_diffusion(q, phase, groups, tmp_path / "missing_symmetry", valid_mask=valid, allowed_phase_ids=[7, 42],
                         epochs=1, steps=4, dataset_metadata=bad, progress=lambda _: None)
    bad = {**metadata, "phase_schema": {"7": {"space_group": 225}, "42": {"space_group": 229}}}
    with pytest.raises(ValueError, match="Unsupported crystal symmetry"):
        train_diffusion(q, phase, groups, tmp_path / "unsupported_symmetry", valid_mask=valid, allowed_phase_ids=[7, 42],
                         epochs=1, steps=4, dataset_metadata=bad, progress=lambda _: None)
    fit = tmp_path / "real_field_fit"
    train_diffusion(q, phase, groups, fit, valid_mask=valid, allowed_phase_ids=[7, 42],
                     epochs=2, steps=4, dataset_metadata=metadata, progress=lambda _: None)
    config, _ = load_diffusion_checkpoint(fit)
    assert config["crystal_symmetry_reduced"] and "right action" in config["symmetry_representative_rule"]
    output = sample_diffusion(fit, tmp_path / "sample", condition=[2/3, 1/3], count=2, progress=lambda _: None)
    with h5py.File(output, "r") as stream:
        generated, generated_phase, generated_valid = (stream[key][:] for key in ("quaternion_xyzw", "phase_id", "valid_mask"))
    reduced = canonicalize_phase_quaternions(generated, generated_phase, generated_valid, metadata["phase_schema"])
    np.testing.assert_allclose(reduced, generated, atol=1e-6)
    qc = json.loads((tmp_path / "sample" / "generation_qc.json").read_text())
    assert not qc["physical_realism_validated"]
    for index, result in enumerate(qc["samples"]):
        assert result["phase_fraction_condition_enforced"] is False
        for pid, target in (("7", 2/3), ("42", 1/3)):
            actual = float(np.mean(generated_phase[index][generated_valid[index]] == int(pid)))
            assert result["phase_fraction_discrepancy_actual_minus_requested"][pid] == pytest.approx(actual - target)


@pytest.mark.parametrize("schema_case", ["same", "swapped", "missing"])
def test_graph_bundle_requires_identical_explicit_phase_meanings(tmp_path, schema_case):
    schema = {"7": {"name": "alpha Ti", "space_group": 194}, "42": {"name": "beta Ti", "space_group": 229}}
    graphs, mechanics = [], []
    for index in range(3):
        source = tmp_path / f"labels{index}.csv"
        source.write_text("X,Y,S,E\n0,0,200000000,2\n", encoding="utf-8")
        metadata = mechanics_metadata()
        metadata.update(source_id=f"source-{index}", coordinate_unit="um", source_to_sample_um=np.eye(3).tolist())
        mechanics.append(import_mechanics(source, tmp_path / f"mechanics{index}", metadata=metadata,
            columns={"x": "X", "y": "Y", "sigma_xx": "S", "epsilon_xx": "E"}, progress=lambda _: None))
        graph = tmp_path / f"graph{index}.npz"
        fields = {"node_feature_names": np.asarray(["centroid_x_um", "centroid_y_um", "area_um2"]),
                  "node_features": np.asarray([[0., 0., 1.], [1., 0., 1.]]), "node_feature_mask": np.ones((2, 3), bool),
                  "node_phase_id": np.asarray([7, 42]), "node_ids": np.asarray([1, 2]),
                  "orientation_xyzw": np.tile([0., 0., 0., 1.], (2, 1)), "edge_index": np.asarray([[0, 1], [1, 0]]),
                  "edge_attr": np.ones((2, 5)), "source_id": f"source-{index}"}
        if schema_case != "missing" or index != 1:
            selected = {"7": schema["42"], "42": schema["7"]} if schema_case == "swapped" and index == 1 else schema
            fields["metadata_json"] = json.dumps({"phase_schema": selected})
        np.savez(graph, **fields)
        graphs.append(graph)
    kwargs = {"target_names": ["sigma_xx"], "max_distance_um": 2., "reference_frame": "EBSD Sample-X/Y", "progress": lambda _: None}
    if schema_case == "same":
        path = create_graph_training_bundle(graphs, mechanics, tmp_path / "bundle", **kwargs)
        bundle = load_training_bundle(path)
        assert bundle["label_metadata"]["phase_schema"] == schema
        assert list(bundle["features"][:, -5]) == [7., 42.] * 3
    else:
        with pytest.raises(ValueError, match="phase remapping" if schema_case == "swapped" else "lacks explicit phase_schema"):
            create_graph_training_bundle(graphs, mechanics, tmp_path / "bundle", **kwargs)
        assert not (tmp_path / "bundle" / "training_bundle.npz").exists()


def test_real_phase_feature_inference_requires_same_phase_mapping(tmp_path):
    x, y, groups, _ = training_data()
    x[:, 2] = np.tile([7., 42.], len(x) // 2)
    schema = {"7": {"name": "alpha Ti", "space_group": 194}, "42": {"name": "beta Ti", "space_group": 229}}
    provenance = {**demo_metadata(), "source_type": "CPFE", "material": "Ti-6Al-4V", "phase_schema": schema}
    kwargs = {**train_kwargs(), "feature_names": ["a", "b", "phase_id"], "demo": False, "label_metadata": provenance}
    model = tmp_path / "model"
    train_supervised(x, y, groups, model, model="ridge", **kwargs)
    context = {key: provenance[key] for key in ("material", "loading_condition", "reference_frame", "load_step", "time")}
    with pytest.raises(ValueError, match="requires explicit matching phase_schema"):
        infer_supervised(model, x, tmp_path / "missing_mapping", feature_names=kwargs["feature_names"], inference_metadata=context)
    with pytest.raises(ValueError, match="phase_schema differs"):
        infer_supervised(model, x, tmp_path / "swapped_mapping", feature_names=kwargs["feature_names"],
            inference_metadata={**context, "phase_schema": {"7": schema["42"], "42": schema["7"]}})
    output = infer_supervised(model, x, tmp_path / "matching_mapping", feature_names=kwargs["feature_names"],
                              inference_metadata={**context, "phase_schema": schema})
    assert output.is_file()
    bad_values = x.copy()
    bad_values[0, 2] = 99
    with pytest.raises(ValueError, match="phase_id values"):
        infer_supervised(model, bad_values, tmp_path / "unknown_phase", feature_names=kwargs["feature_names"],
                          inference_metadata={**context, "phase_schema": schema})
    missing_provenance = dict(provenance)
    missing_provenance.pop("phase_schema")
    with pytest.raises(ValueError, match="regenerate or explicitly adapt"):
        train_supervised(x, y, groups, tmp_path / "legacy_unknown_mapping", model="ridge",
                         **{**kwargs, "label_metadata": missing_provenance})
