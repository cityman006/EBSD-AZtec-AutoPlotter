"""Independent synthetic checks of V4 quantitative contracts and real exports."""
import hashlib
import json
import csv
import io
import base64
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from ebsd_autoplot.euler_maps import quaternion_euler_maps
from ebsd_autoplot.grain_statistics import (attach_v2_grain_metadata, grain_size_cdf,
                                          grain_statistics, summarize_grain_sizes)
from ebsd_autoplot.quantitative import analyze_quantitative
from ebsd_autoplot.rx_state import RxThresholds, classify_rx, threshold_sensitivity
from ebsd_autoplot.storage import validate_receipt
from ebsd_autoplot.texture_stats import (ODF_ALGORITHM_ID, feature_signature,
    inverse_pole_vectors, odf_haar_weights, odf_metrics, read_texture_statistics)
from ebsd_autoplot.texture_v3 import _odf_grid, _pf_bin_areas, _plot_pf, analyze_texture
from ebsd_autoplot.io import H5Source


SPECS = {7: {"name": "alpha titanium", "space_group": 194, "c_over_a": 1.586},
         42: {"name": "beta titanium", "space_group": 229}}


def synthetic_maps():
    labels = np.zeros((8, 13), np.int32)
    labels[1:3, 1:3] = 1  # four pixels
    labels[1:3, 5:7] = 2
    labels[4:7, 8:12] = 3  # twelve pixels; area dominates median
    phase = np.where(labels > 0, 7, 0).astype(np.uint8)
    indexed = labels > 0
    kam = np.full(labels.shape, np.nan)
    grod = np.full(labels.shape, np.nan)
    for gid, ka, go in ((1, .2, .5), (2, .75, 1.5), (3, 2, 4)):
        kam[labels == gid] = ka; grod[labels == gid] = go
    return labels, phase, indexed, kam, grod


def grain_rows():
    labels, phase, indexed, kam, grod = synthetic_maps()
    return grain_statistics(labels, phase, indexed, 2, 3, kam=kam,
                            kam_valid=indexed, grod=grod, min_pixels=4)


def test_physical_area_perimeter_axes_and_weighted_size():
    rows = grain_rows()
    first = rows[0]
    assert first["area_um2"] == 24
    assert first["ecd_um"] == pytest.approx(2 * np.sqrt(24 / np.pi))
    assert first["perimeter_um"] == 20  # a 4 um by 6 um rectangle
    assert first["centroid_x_um"] == 3  # pixel index centre convention: (1+2)/2*dx
    assert first["centroid_y_um"] == 4.5
    assert first["major_axis_um"] > first["minor_axis_um"] > 0
    summary = summarize_grain_sizes(rows)
    assert summary["ECD_number_median_um"] == pytest.approx(rows[0]["ecd_um"])
    assert summary["ECD_area_weighted_median_um"] == pytest.approx(rows[2]["ecd_um"])
    assert summary["ECD_number_P10_um"] == pytest.approx(rows[0]["ecd_um"])
    assert summary["ECD_area_weighted_P90_um"] == pytest.approx(rows[2]["ecd_um"])
    assert summary["ECD_number_variance_um2"] > 0
    cdf = grain_size_cdf(rows)
    assert cdf[0]["number_CDF"] == pytest.approx(2/3)
    assert cdf[0]["area_CDF"] == pytest.approx(8/20)
    assert cdf[-1]["number_CDF"] == cdf[-1]["area_CDF"] == 1
    assert summarize_grain_sizes([rows[0], rows[2]])["ECD_number_median_um"] == pytest.approx(
        (rows[0]["ecd_um"] + rows[2]["ecd_um"]) / 2)
    labels = np.ones((3, 5), int)
    edge = grain_statistics(labels, labels, labels > 0, 2, 3, min_pixels=1, exclude_edge=True)[0]
    assert edge["edge_truncated"] and not edge["qc_included"]
    assert "edge_truncated" in edge["qc_reason"]
    with pytest.raises(ValueError, match="positive"):
        grain_statistics(labels, labels, labels > 0, -1, 1)


def test_rx_phase_ids_denominators_and_empty_beta():
    labels, phase, indexed, _, _ = synthetic_maps()
    states, summary, assignments = classify_rx(grain_rows(), labels, phase, indexed, SPECS)
    assert assignments == {1: 1, 2: 2, 3: 3}
    assert set(states[indexed]) == {1, 2, 3}
    row = next(row for row in summary if row["phase_id"] == 7 and row["state_code"] == 1)
    assert row["number_fraction_classified"] == pytest.approx(1 / 3)
    assert row["area_fraction_classified"] == pytest.approx(4 / 20)
    assert row["area_fraction_all_ROI"] == pytest.approx(4 / phase.size)
    assert row["area_fraction_indexed_phase"] == pytest.approx(4 / 20)
    assert row["all_ROI_pixel_denominator"] == 104
    beta = next(row for row in summary if row["phase_id"] == 42 and row["state_code"] == 1)
    assert beta["status"].startswith("unavailable")
    assert np.isnan(beta["area_fraction_classified"])


def test_rx_low_indexing_qc_and_threshold_sensitivity():
    labels, phase, indexed, _, _ = synthetic_maps()
    states, summary, _ = classify_rx(grain_rows(), labels, phase, indexed, SPECS,
                                    min_indexed_fraction=.8)
    assert np.all(states == 0)
    assert all(row["status"] == "unavailable_low_indexing" for row in summary)
    rows = threshold_sensitivity(grain_rows(), labels, phase, indexed, SPECS,
        thresholds={7: RxThresholds(rx_gos_deg=.5, rx_kam_deg=.2)}, scales=(.8, 1, 1.2))
    fractions = [r["area_fraction_classified"] for r in rows if r["phase_id"] == 7 and r["state_code"] == 1]
    assert fractions == pytest.approx([0, .2, .2])  # exact-threshold inclusive
    grains = grain_rows(); grains[0]["qc_included"] = False
    _, summary, assignments = classify_rx(grains, labels, phase, indexed, SPECS)
    assert assignments == {2: 2, 3: 3}
    unc = next(r for r in summary if r["phase_id"] == 7 and r["state_code"] == 0)
    assert unc["pixel_count"] == 4
    assert np.isnan(unc["area_fraction_classified"])
    with pytest.raises(ValueError, match="ordered"):
        RxThresholds(rx_gos_deg=3, substructured_gos_deg=2).validate()


def test_actual_aztec_titanium_cubic_name_obeys_beta_QC():
    labels, phase, indexed, _, _ = synthetic_maps()
    # AZtec source names can be Titanium-Cubic, with arbitrary phase IDs.
    states, rows, assignments = classify_rx(grain_rows(), labels, phase, indexed,
        {7: {"name": "Titanium-Cubic", "space_group": 229}})
    assert np.all(states == 0) and assignments == {}
    assert all(row["status"] == "unavailable_insufficient_grains" for row in rows)


def test_euler_known_rotations_roundtrip_and_missing_data():
    angles = np.array([[25., 35., 80.], [90., 0., 0.], [0., 180., 0.]])
    q = Rotation.from_euler("ZXZ", angles, degrees=True).as_quat().reshape(1, 3, 4)
    deg, rgb, good = quaternion_euler_maps(q, np.array([[True, True, False]]))
    assert deg[0, 0] == pytest.approx(angles[0])
    reconstructed = Rotation.from_euler("ZXZ", deg[good], degrees=True)
    assert np.max((reconstructed.inv() * Rotation.from_quat(q[good])).magnitude()) < 1e-6
    assert np.isnan(deg[0, 2]).all() and np.all(rgb[0, 2] == 0)
    assert rgb[0, 0] == pytest.approx(angles[0] / [360, 180, 360])


def _write_unit_source(path, *, unit=None, second_unit=None, split=False):
    path.mkdir(parents=True)
    for start in range(2 if split else 1):
        name = f"scan01_rows{start:06d}-{start:06d}_EBSD.h5" if split else "sample.h5oina"
        with h5py.File(path / name, "w") as f:
            data = f.create_group("1/EBSD/Data")
            data["Euler"] = np.full((2 if split else 4, 3), .1, np.float32)
            data["Phase"] = np.full(2 if split else 4, 7, np.uint8)
            declared = second_unit if split and start else unit
            if declared is not None: data["Euler"].attrs["Units"] = declared
            header = f.create_group("1/EBSD/Header")
            for name, value in (("X Cells", 2), ("Y Cells", 2), ("X Step", .1), ("Y Step", .1)):
                header[name] = [value]
            phase = header.create_group("Phases/7")
            phase["Phase Name"] = [b"alpha titanium"]; phase["Space Group"] = [194]
    return path if split else path / "sample.h5oina"


def test_euler_dataset_units_small_degrees_and_contract(tmp_path):
    source = H5Source(_write_unit_source(tmp_path / "small_degree", unit="degree"))
    assert source.probe_euler_unit() == "degree"  # 0.1 degree must never become 0.1 radian
    assert "explicit" in source.header["euler_unit_evidence"]["sources"][0]["basis"]
    fallback = H5Source(_write_unit_source(tmp_path / "without_attr"))
    assert fallback.probe_euler_unit() == "radian"
    assert "contract" in fallback.header["euler_unit_evidence"]["sources"][0]["basis"]
    bad = H5Source(_write_unit_source(tmp_path / "bad_unit", unit="gradians"))
    with pytest.raises(ValueError, match="Unsupported Euler unit"):
        bad.probe_euler_unit()
    mixed = H5Source(_write_unit_source(tmp_path / "mixed", unit="degrees", second_unit="radians", split=True))
    with pytest.raises(ValueError, match="disagree"):
        mixed.probe_euler_unit()
    conflict_path = _write_unit_source(tmp_path / "conflict", unit="degree")
    with h5py.File(conflict_path, "r+") as f: f["1/EBSD/Data/Euler"].attrs["Unit"] = "radian"
    with pytest.raises(ValueError, match="Conflicting Euler"):
        H5Source(conflict_path).probe_euler_unit()


def test_odf_uniform_peak_and_invalid_grids():
    p1 = p2 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    p = np.linspace(0, np.pi, 5)
    uniform = np.ones((8, 5, 8))
    stats = odf_metrics(p1, p, p2, uniform)
    assert stats["texture_index_J"] == pytest.approx(1)
    assert stats["ODF_entropy_relative_uniform"] == pytest.approx(0)
    concentrated = uniform.copy(); concentrated[2, 2, 2] = 100
    assert odf_metrics(p1, p, p2, concentrated)["texture_index_J"] > 1
    assert odf_metrics(p1, p, p2, uniform*100)["source_normalization_status"].startswith("explicitly")
    with pytest.raises(ValueError, match="nonnegative"):
        odf_metrics(p1, p, p2, -uniform)


def test_haar_exact_cell_mass_and_v3_single_orientation_mrd():
    p1 = p2 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    p = np.linspace(0, np.pi, 3)
    weights = odf_haar_weights(p1, p, p2)
    assert weights.sum() == pytest.approx(1)
    # A polar cap 0<=Phi<=pi/4 has exact Haar probability (1-cos(pi/4))/2.
    assert weights[:, 0, :].sum() == pytest.approx((1-np.cos(np.pi/4))/2)
    q = np.broadcast_to([0., 0., 0., 1.], (50, 4))
    p1, p, p2, mrd, params = _odf_grid(q, np.ones(50), 194, 20, 90, 0)
    assert float(np.sum(mrd * odf_haar_weights(p1, p, p2))) == pytest.approx(1, abs=1e-6)
    assert params["algorithm_id"] == ODF_ALGORITHM_ID
    assert odf_metrics(p1, p, p2, mrd)["texture_index_J"] > 1


def test_v3_uniform_orientation_benchmark():
    orientations = Rotation.random(4000, random_state=13).as_quat()
    p1, p, p2, mrd, params = _odf_grid(orientations, np.ones(len(orientations)), 194, 35, 45, 13)
    metrics = odf_metrics(p1, p, p2, mrd)
    # Independent Haar-uniform rotations should converge towards MRD=1.
    assert metrics["texture_index_J"] == pytest.approx(1, abs=.02)
    assert metrics["ODF_peak_MRD"] < 1.1
    assert params["integral_check"] == pytest.approx(1)


def test_pf_exact_disk_area_rim_poles_and_conservation(tmp_path):
    edges = np.linspace(-np.sqrt(2), np.sqrt(2), 181)
    area = _pf_bin_areas(edges)
    assert area.sum() == pytest.approx(2*np.pi, abs=1e-12)
    assert np.all(area >= 0)
    assert area[90,90] == pytest.approx((edges[91]-edges[90])**2)
    poles = np.broadcast_to([0.9949409123839288,0.10046183785216795,0], (50,3))
    mud = _plot_pf(tmp_path / "rim", poles, title="Rim regression", formats=("png",))
    assert np.nanmax(mud) > 0
    assert np.nansum(mud*area)/(2*np.pi) == pytest.approx(1)
    # Uniform projected disk pole density must use partial rim areas, not an
    # occupied-bin union or equal full-bin-area approximation.
    rng = np.random.default_rng(27)
    az = rng.uniform(0,2*np.pi,300000)
    z = rng.uniform(0,1,300000)
    poles = np.column_stack((np.sqrt(1-z*z)*np.cos(az), np.sqrt(1-z*z)*np.sin(az), z))
    mud = _plot_pf(tmp_path / "uniform", poles, title="Uniform hemisphere", bins=20, formats=("png",))
    area = _pf_bin_areas(np.linspace(-np.sqrt(2), np.sqrt(2), 21))
    use = area > .001
    assert np.mean(np.abs(mud[use]-1)) < .1
    assert np.nansum(mud*area)/(2*np.pi) == pytest.approx(1)


def test_texture_insufficient_beta_no_fake_PF_or_ODF(tmp_path):
    features, _ = write_synthetic_cache(tmp_path / "source")
    with h5py.File(features, "r+") as f:
        f["indexed"][...] = 1
        f["features"][4, ...] = 7
        f["features"][4, -1, -1] = 42
    result = analyze_texture(features, tmp_path / "texture", grid_step_deg=90,
                             bandwidth_deg=30, max_samples=1000, formats=("png",))
    beta = [r for r in result["results"] if r["phase_id"] == 42]
    assert len(beta) == 1 and beta[0]["orientation_samples"] == 1
    assert beta[0]["status"] == "unavailable_insufficient_phase_samples"
    assert np.isnan(beta[0]["PF_peak_MUD"])
    assert not list((tmp_path / "texture").glob("PF_beta*"))
    assert not (tmp_path / "texture" / "ODF_beta_sections").exists()
    assert list((tmp_path / "texture").glob("PF_alpha*"))
    with h5py.File(tmp_path / "texture" / "odf_grid.h5", "r") as f:
        assert "MRD" not in f["phase_42_beta"]
    assert read_texture_statistics(tmp_path / "texture" / "odf_grid.h5", 42)["texture_status"] == beta[0]["status"]


def test_texture_cache_version_provenance_and_configuration(tmp_path):
    features, _ = write_synthetic_cache(tmp_path / "source")
    cache = tmp_path / "texture.h5"
    with h5py.File(cache, "w") as f:
        f.attrs["schema_version"] = 1
    assert "legacy" in read_texture_statistics(cache, 7)["texture_status"]
    p1 = p2 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    p = np.linspace(0, np.pi, 3)
    with h5py.File(cache, "r+") as f:
        f.attrs["odf_algorithm_id"] = ODF_ALGORITHM_ID
        f.attrs["features_signature_json"] = json.dumps(feature_signature(features), sort_keys=True)
        f.attrs["configuration_hash"] = "synthetic_config"
        group = f.create_group("phase_7_alpha"); group.attrs["phase_id"] = 7
        for name, value in (("phi1_rad", p1), ("Phi_rad", p), ("phi2_rad", p2), ("MRD", np.ones((4,3,4)))):
            group[name] = value
    stats = read_texture_statistics(cache, 7, features_file=features)
    assert stats["texture_index_J"] == pytest.approx(1)
    assert stats["texture_status"].startswith("available")
    with h5py.File(cache, "r+") as f: f.attrs["features_signature_json"] = "{}"
    assert "mismatch" in read_texture_statistics(cache, 7, features_file=features)["texture_status"]


def test_inverse_poles_use_inverse_rotation_and_symmetry():
    pytest.importorskip("orix")
    # Crystal c parallel to specimen Z must stay the c-pole under HCP symmetry.
    q = Rotation.identity(2).as_quat()
    v = inverse_pole_vectors(q, 194)
    assert np.abs(v[:, 2]) == pytest.approx(np.ones(2))
    assert np.linalg.norm(v, axis=1) == pytest.approx(np.ones(2))


def test_v2_mean_orientation_and_neighbour_metadata(tmp_path):
    with (tmp_path / "grains.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["grain_id", "phase_id", "mean_qx", "mean_qy", "mean_qz", "mean_qw"])
        writer.writerow([1, 7, 0, 0, 0, 1])
    with (tmp_path / "grain_neighbour_graph.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["grain_a", "grain_b"])
        writer.writerows([[1,2], [2,3], [2,1]])
    rows = attach_v2_grain_metadata(grain_rows(), tmp_path)
    assert rows[0]["mean_qw"] == 1
    assert rows[0]["mean_orientation_status"].startswith("available")
    assert rows[1]["neighbour_count"] == 2
    assert np.isnan(rows[1]["mean_qw"])


def write_synthetic_cache(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    labels, phase, indexed, kam, grod = synthetic_maps()
    features = np.zeros((7, *labels.shape), np.float32)
    euler = np.broadcast_to(np.array([25, 35, 80]), (*labels.shape, 3))
    q = Rotation.from_euler("ZXZ", euler.reshape(-1, 3), degrees=True).as_quat().reshape(*labels.shape, 4)
    features[:4] = np.moveaxis(q, -1, 0)
    features[4] = phase; features[5] = kam
    with h5py.File(root / "features.h5", "w") as f:
        f["features"] = features; f["indexed"] = indexed; f["kam_valid"] = indexed
        f.attrs["metadata_json"] = json.dumps({"phase_specs": SPECS, "angular_units_input": "radian",
            "x_step_um": 2., "y_step_um": 3., "scan_path": "synthetic_scan"})
    v2 = root / "v2"; v2.mkdir()
    with h5py.File(v2 / "v2_grains.h5", "w") as f:
        f["grain_id"] = labels; f["GROD_deg"] = grod
        f.attrs["roi_xywh"] = (0, 0, labels.shape[1], labels.shape[0])
    return root / "features.h5", v2


def test_transactional_analysis_real_formats_and_source_immutability(tmp_path):
    features, v2 = write_synthetic_cache(tmp_path / "source")
    source_hash = hashlib.sha256(features.read_bytes()).hexdigest()
    v2_hash = hashlib.sha256((v2 / "v2_grains.h5").read_bytes()).hexdigest()
    out = tmp_path / "quantitative"
    summary = analyze_quantitative(features, v2, out, min_pixels=4,
        formats=("png", "tiff", "pdf", "svg"), ipf_statistics=False, progress=lambda _: None)
    assert hashlib.sha256(features.read_bytes()).hexdigest() == source_hash
    assert hashlib.sha256((v2 / "v2_grains.h5").read_bytes()).hexdigest() == v2_hash
    assert validate_receipt(out, "ebsd_autoplot.quantitative.analyze_quantitative")
    assert len(summary["phases"]) == 2
    alpha, beta = summary["phases"]
    assert alpha["phase_id"] == 7 and alpha["grain_count"] == 3
    assert alpha["KAM_valid_count"] == 20
    assert beta["phase_id"] == 42 and beta["grain_count"] == 0
    assert np.isnan(alpha["texture_index_J"])
    assert alpha["texture_status"].startswith("unavailable")
    assert alpha["configuration_hash"] == summary["configuration_hash"]
    for path in out.glob("*"):
        if path.suffix in (".png", ".tiff"):
            with Image.open(path) as image: image.verify()
        elif path.suffix == ".pdf": assert path.read_bytes().startswith(b"%PDF-")
        elif path.suffix == ".svg": assert ET.parse(path).getroot().tag.endswith("svg")
    assert len(list(out.glob("*.pdf"))) == 7
    with h5py.File(out / "RX_state.h5", "r") as f:
        assert set(np.unique(f["RX_state"])) == {0, 1, 2, 3}
    with h5py.File(out / "Euler_maps.h5", "r") as f:
        assert f.attrs["unit"] == "degree"
        assert np.isnan(f["phi1_deg"][0, 0])
    native = out / "native_maps"
    with Image.open(native / "RX_state.png") as image:
        assert image.size == (13,8)
        assert set(np.unique(np.asarray(image))) == {0,1,2,3}
    with Image.open(native / "Euler_phi1.tiff") as image:
        array = np.asarray(image)
        assert array.shape == (8,13)
        assert np.isnan(array[0,0]) and array[1,1] == pytest.approx(25)
    with Image.open(native / "Euler_phi1.png") as image:
        array = np.asarray(image)
        assert array.shape == (8,13) and array[0,0] == 65535
        assert float(array[1,1])*360/65534 == pytest.approx(25, abs=.006)
    root = ET.parse(out / "Euler_phi1.svg").getroot()
    embedded = root.findall(".//{http://www.w3.org/2000/svg}image")
    href = embedded[0].attrib["{http://www.w3.org/1999/xlink}href"]
    with Image.open(io.BytesIO(base64.b64decode(href.split(",",1)[1]))) as image:
        assert image.size == (13,8)  # full native raster embedded in vector layout
    with pytest.raises(FileExistsError):
        analyze_quantitative(features, v2, out, min_pixels=4, ipf_statistics=False, progress=lambda _: None)
