"""Lossless extractor shards: local Header counts, explicit global provenance."""
import json

import h5py
import numpy as np
import pytest

from ebsd_autoplot.discovery import discover_scans
from ebsd_autoplot.io import H5Source
from ebsd_autoplot.compute import prepare
from test_pipeline import write_synth


def extractor_montage(tmp_path, *, manifest=True, global_headers=False):
    """Mirror H5OINA_EBSD_Compact_v1 h5_write, using synthetic data only."""
    root = tmp_path / "extractor_parts"
    truth = write_synth(root, split=True)
    entries = []
    yy, xx = np.indices(truth.shape)
    for path in sorted(root.glob("*.h5")):
        start = int(path.name.split("rows")[1].split("-")[0])
        stop = start + truth.shape[0] // 2
        with h5py.File(path, "a") as stream:
            stream.attrs.update(source_name="synthetic specimen.h5oina",
                extractor="H5OINA_EBSD_Compact_v1", global_nx=truth.shape[1],
                global_ny=truth.shape[0], start_row_in_original=start,
                end_row_in_original_exclusive=stop,
                all_arrays_are_original_measured_values=True)
            header = stream["1/EBSD/Header"]
            header["Y Cells"][...] = truth.shape[0] if global_headers else stop-start
            header["Y Cells"].attrs["original_full_y_cells"] = truth.shape[0]
            for name in ("X Cells", "Y Cells"):
                header[name].attrs["Unit"] = "px"
            for name in ("X Step", "Y Step"):
                header[name].attrs["Unit"] = "um"
            data = stream["1/EBSD/Data"]
            data["Euler"].attrs["Unit"] = "rad"
            for key, grid in (("X", xx), ("Y", yy)):
                data[key] = (grid[start:stop] * .05).astype("f4").ravel()
                data[key].attrs["Unit"] = "um"
            eds = stream.create_group("1/EDS")
            stream.copy(header, eds, name="Header")
            edata = eds.create_group("Data")
            for key in ("X", "Y"):
                stream.copy(data[key], edata, name=key)
            edata["Live Time"] = np.full((stop-start)*truth.shape[1], .1, "f4")
            edata["Live Time"].attrs["Unit"] = "s"
            window = edata.create_group("Window Integral")
            window["Ti Ka"] = np.full((stop-start)*truth.shape[1], start+1, "f4")
            window["Ti Ka"].attrs["Unit"] = "cps"
        entries.append(dict(file=path.name, row_start=start, row_end_exclusive=stop))
    document = dict(source_name="synthetic specimen.h5oina", scans=[dict(
        scan_path="/1/EBSD", nx=truth.shape[1], ny=truth.shape[0], parts=entries)])
    if manifest:
        (root / "请上传_分析清单.json").write_text(json.dumps(document), encoding="utf-8")
    return root, truth, document


@pytest.mark.parametrize("manifest", [True, False])
def test_local_headers_preserve_global_rows_and_coordinates(tmp_path, manifest):
    root, truth, _ = extractor_montage(tmp_path, manifest=manifest)
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.glob("*.h5")}
    source = H5Source(discover_scans(root)[0])
    assert (source.ny, source.nx) == truth.shape
    assert np.array_equal(source.read(0, source.ny, ("phase",))["phase"], truth)
    block = source.read(4, 8, ("x", "y"))
    assert np.allclose(block["y"][:, 0], np.arange(4, 8)*.05)
    assert np.allclose(block["x"][0], np.arange(source.nx)*.05)
    assert source.probe_euler_unit() == "radian"
    assert before == {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.glob("*.h5")}


def test_global_headers_with_extractor_evidence_remain_supported(tmp_path):
    root, truth, _ = extractor_montage(tmp_path, global_headers=True)
    source = H5Source(root)
    assert (source.ny, source.nx) == truth.shape


def test_local_and_global_headers_produce_identical_numeric_caches(tmp_path):
    local, _, _ = extractor_montage(tmp_path / 'local')
    global_, _, _ = extractor_montage(tmp_path / 'global', global_headers=True)
    local_cache = prepare(H5Source(local), tmp_path / 'local_cache', batch=4,
                          do_ipf=False, progress=lambda _: None)
    global_cache = prepare(H5Source(global_), tmp_path / 'global_cache', batch=4,
                           do_ipf=False, progress=lambda _: None)
    with h5py.File(local_cache) as left, h5py.File(global_cache) as right:
        assert set(left) == set(right)
        for name in left:
            if isinstance(left[name], h5py.Dataset):
                np.testing.assert_array_equal(left[name][()], right[name][()])


@pytest.mark.parametrize("fault", [
    "missing_global_ny", "global_nx", "global_ny", "start", "stop", "fractional",
    "boolean", "multiple_values", "original_full_y", "header_ny", "manifest_nx",
    "manifest_ny", "missing_part", "overlap", "fractional_row", "boolean_row",
    "source_name", "phase_name", "euler_unit",
])
def test_local_header_contradictions_rejected(tmp_path, fault):
    root, _, document = extractor_montage(tmp_path)
    paths = sorted(root.glob("*.h5"))
    with h5py.File(paths[-1], "a") as stream:
        if fault == "missing_global_ny": del stream.attrs["global_ny"]
        elif fault == "global_nx": stream.attrs["global_nx"] = 17
        elif fault == "global_ny": stream.attrs["global_ny"] = 13
        elif fault == "start": stream.attrs["start_row_in_original"] = 5
        elif fault == "stop": stream.attrs["end_row_in_original_exclusive"] = 11
        elif fault == "fractional": stream.attrs["global_ny"] = 12.5
        elif fault == "boolean": stream.attrs["global_ny"] = True
        elif fault == "multiple_values": stream.attrs["global_ny"] = [12, 12]
        elif fault == "original_full_y": stream["1/EBSD/Header/Y Cells"].attrs["original_full_y_cells"] = 14
        elif fault == "header_ny": stream["1/EBSD/Header/Y Cells"][...] = 7
        elif fault == "source_name": stream.attrs["source_name"] = "another specimen.h5oina"
        elif fault == "phase_name": stream["1/EBSD/Header/Phases/1/Phase Name"][...] = np.array([b"Another phase"])
        elif fault == "euler_unit": stream["1/EBSD/Data/Euler"].attrs["Unit"] = "deg"
    scan = document["scans"][0]
    if fault == "manifest_nx": scan["nx"] = 17
    elif fault == "manifest_ny": scan["ny"] = 13
    elif fault == "missing_part": scan["parts"].pop()
    elif fault == "overlap": scan["parts"].insert(1, dict(scan["parts"][0]))
    elif fault == "fractional_row": scan["parts"][1]["row_start"] = 6.5
    elif fault == "boolean_row": scan["parts"][0]["row_start"] = False
    (root / "请上传_分析清单.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        H5Source(root).probe_euler_unit()


def test_partial_extracted_file_cannot_impersonate_complete_scan(tmp_path):
    root, _, _ = extractor_montage(tmp_path)
    with pytest.raises(ValueError, match="(?i)(partial|montage|range|rows)"):
        H5Source(sorted(root.glob("*.h5"))[0])


def test_local_headers_without_explicit_global_evidence_rejected(tmp_path):
    root, _, _ = extractor_montage(tmp_path)
    for path in root.glob("*.h5"):
        with h5py.File(path, "a") as stream:
            for key in ("global_nx", "global_ny", "start_row_in_original", "end_row_in_original_exclusive"):
                del stream.attrs[key]
            del stream["1/EBSD/Header/Y Cells"].attrs["original_full_y_cells"]
    with pytest.raises(ValueError):
        H5Source(root)
