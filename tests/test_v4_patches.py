"""Independent native-pixel, registration, EDS and real-grain graph invariants."""
import json
from pathlib import Path
import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from ebsd_autoplot.slicing import export_patches, plan_patches
from ebsd_autoplot.eds import load_eds, EDSResult, load_h5oina_eds, save_eds
from ebsd_autoplot.registration import fit_affine, transform_points, resample_map
from ebsd_autoplot.graph_builder import build_graph
from ebsd_autoplot.ml_dataset import PatchDataset, grouped_split
from ebsd_autoplot.storage import validate_receipt


def feature_cache(tmp_path, shape=(7, 11), dx=2., dy=3.):
    path = tmp_path / "features.h5"
    h, w = shape
    raw = np.zeros((7, h, w), np.float32)
    raw[3] = 1; raw[4] = 1
    raw[5] = np.arange(h*w).reshape(h, w) / 100
    raw[6] = np.arange(h*w).reshape(h, w) + 100
    raw[5, 0, 0] = 999  # Stored sentinel retained exactly; mask controls validity.
    with h5py.File(path, "w") as f:
        f["features"] = raw; f["indexed"] = np.ones(shape, np.uint8)
        valid = np.ones(shape, np.uint8); valid[0, 0] = 0; f["kam_valid"] = valid
        f.attrs["metadata_json"] = json.dumps({"x_step_um": dx, "y_step_um": dy,
              "phase_specs": {"1": {"name": "alpha-Ti", "space_group": 194}}, "scan_path": "scan-A"})
    return path, raw


def test_grid_3x5_native_remainder_no_pixel_loss(tmp_path):
    src, raw = feature_cache(tmp_path)
    out = tmp_path / "patches"
    result = export_patches(src, out, grid=(3, 5), preview=False)
    assert result["patch_count"] == 15
    coverage = np.zeros((7, 11), int)
    rebuilt = np.empty_like(raw)
    for entry in result["patches"]:
        x, y, w, h = entry["roi_xywh"]
        coverage[y:y+h, x:x+w] += 1
        with h5py.File(out / entry["file"], "r") as f:
            assert np.array_equal(f["features"][:], raw[:, y:y+h, x:x+w])
            rebuilt[:, y:y+h, x:x+w] = f["features"][:]
            assert f["padding_mask"][:].all()
    assert np.all(coverage == 1)
    assert np.array_equal(rebuilt, raw)
    with h5py.File(out / result["patches"][0]["file"], "r") as first:
        assert first["channels/kam_deg"][0, 0] == 999
        assert not first["masks/kam_deg"][0, 0]
        meta = json.loads(first.attrs["metadata_json"])
        assert meta["modalities"]["eds"] == "unavailable"
        assert not any(k.startswith("eds_") for k in first["channels"])
    assert validate_receipt(out, "ebsd_autoplot.slicing.export_patches")
    with pytest.raises(FileExistsError):
        export_patches(src, out, preview=False)


def test_fixed_pixels_padding_overlap_and_explicit_drop(tmp_path):
    src, raw = feature_cache(tmp_path, (5, 7))
    inc = plan_patches((5, 7), mode="pixels", size_px=(3, 4))
    assert [c[2:4] for c in inc] == [(4, 3), (3, 3), (4, 2), (3, 2)]
    assert len(plan_patches((5, 7), mode="pixels", size_px=(3, 4), edge_policy="drop-explicit")) == 1
    out = tmp_path / "pad"
    result = export_patches(src, out, mode="pixels", size_px=(3, 4), edge_policy="pad-with-mask", preview=False)
    final = result["patches"][-1]
    with h5py.File(out / final["file"], "r") as f:
        assert f["features"].shape == (7, 3, 4)
        assert f["padding_mask"][:].sum() == 6
        assert not f["masks/qw"][-1].any()
        assert np.isnan(f["channels/qw"][-1]).all()
        assert np.array_equal(f["features"][:, :2, :3], raw[:, 3:5, 4:7])
    coverage = np.zeros((5, 7), int)
    for x, y, w, h, _ in plan_patches((5, 7), mode="pixels", size_px=(3, 4), overlap_px=(1, 2)):
        coverage[y:y+h, x:x+w] += 1
    assert coverage.min() >= 1 and coverage.max() > 1
    with pytest.raises(ValueError, match="silently omit"):
        plan_patches((5, 7), mode="pixels", size_px=(3, 4), stride_px=(4, 5))


def test_um_and_roi_no_resizing_or_rounding_hidden(tmp_path):
    src, _ = feature_cache(tmp_path)
    out = tmp_path / "physical"
    result = export_patches(src, out, mode="um", size_um=(7, 5), roi=(1, 2, 5, 4), preview=False)
    assert result["patch_count"] == 4
    assert result["patches"][0]["roi_xywh"] == [1, 2, 3, 3]  # ceil(5/2),ceil(7/3)
    with h5py.File(out / result["patches"][0]["file"], "r") as f:
        meta = json.loads(f.attrs["metadata_json"])
        assert transform_points([[0, 0], [2, 2]], meta["pixel_to_sample_um"]).tolist() == [[2., 6.], [6., 12.]]
    with pytest.raises(ValueError, match="empty"):
        plan_patches((2, 3), grid=(3, 5))
    with pytest.raises(ValueError, match="ROI"):
        plan_patches((7, 11), roi=(-1, 0, 3, 3))


def test_numeric_derived_map_roi_and_dataset_reader(tmp_path):
    src, _ = feature_cache(tmp_path)
    derived = tmp_path / "scientific.h5"
    with h5py.File(derived, "w") as f:
        f["schmid"] = np.full((3, 4), .25, np.float32)
        mask = np.ones((3, 4), bool); mask[1, 1] = False; f["valid"] = mask
        f.attrs["roi_xywh"] = [2, 1, 4, 3]
    desc = {"Schmid": {"path": derived, "dataset": "schmid", "mask_dataset": "valid", "unit": "dimensionless"}}
    out = tmp_path / "derived"
    export_patches(src, out, mode="roi", roi=(2, 1, 4, 3), derived_channels=desc, preview=False)
    data = PatchDataset(out / "dataset_index.json", ["Schmid", "euler_rad"])
    assert len(data) == 1
    sample = data[0]
    assert np.all(sample["channels"]["Schmid"] == .25)
    assert not sample["masks"]["Schmid"][1, 1]
    with pytest.raises(ValueError, match="unavailable"):
        PatchDataset(out / "dataset_index.json", ["eds_Fe"])[0]
    with pytest.raises(ValueError, match="outside"):
        export_patches(src, tmp_path / "bad", mode="roi", derived_channels=desc, preview=False)
    assert not (tmp_path / "bad" / "patch_manifest.json").exists()


def test_eds_windows_metadata_no_fake_elements_and_registration(tmp_path):
    path = tmp_path / "eds.h5"
    spectra = np.arange(2*3*4).reshape(2, 3, 4)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("spectra", data=spectra); ds.attrs["unit"] = "counts"
        en = f.create_dataset("energy", data=[0., .1, .2, .3]); en.attrs["unit"] = "keV"
    assert not load_eds(None).available
    assert not load_eds(path).available
    kwargs = dict(energy_windows={"window": (.1, .2)}, spectrum_path="spectra", energy_axis_path="energy")
    with pytest.raises(ValueError, match="dimensions do not establish"):
        load_eds(path, expected_shape=(2, 3), **kwargs)
    with pytest.raises(ValueError, match="grid mismatch"):
        load_eds(path, expected_shape=(4, 4), **kwargs)
    result = load_eds(path, expected_shape=(2, 3), spatial_metadata={"aligned_to_ebsd": True}, **kwargs)
    assert np.array_equal(result.maps["window"], spectra[..., 1:3].sum(-1))
    assert result.units["window"] == "counts" and result.metadata["energy_unit"] == "keV"
    reg = load_eds(path, expected_shape=(2, 4), registration=[[1, 0, 1], [0, 1, 0], [0, 0, 1]], **kwargs)
    assert not reg.masks["window"][:, 0].any()
    assert np.isnan(reg.maps["window"][:, 0]).all()
    assert np.array_equal(reg.maps["window"][:, 1:], result.maps["window"])


def test_affine_known_geometry_residual_invalid_support():
    landmarks = np.array([[0, 0], [2, 0], [0, 3], [2, 3]], float)
    target = landmarks * [2, 3] + [4, 5]
    matrix, qc = fit_affine(landmarks, target)
    assert np.allclose(transform_points([[1, 1]], matrix), [[6, 8]])
    assert qc["rms_target_px"] < 1e-12 and not qc["independently_validated"]
    with pytest.raises(ValueError, match="collinear"):
        fit_affine([[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])
    field = np.arange(9, dtype=float).reshape(3, 3)
    valid = np.ones((3, 3), bool); valid[1, 1] = False
    out, mask = resample_map(field, (3, 3), np.eye(3), valid)
    assert np.array_equal(out[mask], field[mask])
    assert np.isnan(out[1, 1]) and not mask[1, 1]
    fractional, fractional_mask = resample_map(field, (3, 3), [[1, 0, .5], [0, 1, 0], [0, 0, 1]], valid)
    assert not fractional_mask[1, 1]  # one invalid contributor cannot be averaged as zero.


def test_graph_anisotropic_shared_lengths_and_symmetry(tmp_path):
    src, _ = feature_cache(tmp_path, (2, 3), dx=2, dy=3)
    # Real nonconsecutive labels: 10/20 and 10/30 each dy, 20/30 length=2*dx.
    ids = np.array([[10, 20, 20], [10, 30, 30]], np.int32)
    v2 = tmp_path / "v2"; v2.mkdir()
    with h5py.File(v2 / "v2_grains.h5", "w") as f:
        f["grain_id"] = ids; f.attrs["roi_xywh"] = (0, 0, 3, 2)
    # HCP 60° about c is symmetry-equivalent to identity.
    q60 = Rotation.from_euler("z", 60, degrees=True).as_quat()
    with h5py.File(src, "r+") as f:
        q = f["features"][:4]
        q[:, ids == 20] = q60[:, None]
        f["features"][:4] = q
    out = tmp_path / "graph"
    report = build_graph(src, v2, out, preview=False)
    assert report["qc"]["nodes"] == 3
    assert report["qc"]["undirected_edges"] == 3
    graph = np.load(out / "graph.npz", allow_pickle=False)
    assert json.loads(str(graph['metadata_json'].item()))['phase_schema']==report['phase_schema']
    assert report['phase_schema']['1']['space_group']==194
    assert graph["node_ids"].tolist() == [10, 20, 30]
    pairs = {tuple(e): a for e, a in zip(graph["edge_index"].T.tolist(), graph["edge_attr"])}
    assert pairs[(0, 1)][0] == 3.0 and pairs[(0, 2)][0] == 3.0
    assert pairs[(1, 2)][0] == 4.0
    assert pairs[(0, 1)][2] < .01
    assert np.isclose(graph["node_features"][:, 1].sum(), 36)
    assert graph["node_features"][:, 5].tolist() == [16., 14., 14.]
    assert validate_receipt(out, "ebsd_autoplot.graph_builder.build_graph")


def test_graph_interphase_is_masked_not_invented(tmp_path):
    src, _ = feature_cache(tmp_path, (1, 2))
    with h5py.File(src, "r+") as f:
        meta = json.loads(f.attrs["metadata_json"])
        meta["phase_specs"]["2"] = {"name": "beta-Ti", "space_group": 229}
        f.attrs["metadata_json"] = json.dumps(meta)
        f["features"][4, 0, 1] = 2
    v2 = tmp_path / "v2"; v2.mkdir()
    with h5py.File(v2 / "v2_grains.h5", "w") as f:
        f["grain_id"] = [[8, 9]]; f.attrs["roi_xywh"] = (0, 0, 2, 1)
    build_graph(src, v2, tmp_path / "graph", preview=False)
    graph = np.load(tmp_path / "graph" / "graph.npz", allow_pickle=False)
    assert np.isnan(graph["edge_attr"][:, 2]).all()
    assert not graph["edge_feature_mask"][:, 2].any()
    assert np.all(graph["edge_attr"][:, 4] == 1)


def test_ml_split_grouped_original_scan_reproducible():
    entries = [{"source_id": f"s{s}", "parent_id": f"s{s}", "split_group": f"specimen{s}", "patch_id": f"p{p}"}
               for s in range(6) for p in range(5)]
    split = grouped_split(entries, seed=17)
    assert split == grouped_split(entries, seed=17)
    groups = [{entries[i]["parent_id"] for i in indexes} for indexes in split.values()]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    assert sum(map(len, split.values())) == len(entries)
    with pytest.raises(ValueError, match="Too few"):
        grouped_split(entries[:5])
    with pytest.raises(ValueError, match="independent"):
        grouped_split(entries, group_key="patch_id")


def test_optional_torch_not_imported_for_numpy_dataset():
    import subprocess, sys
    check = subprocess.run([sys.executable, "-c", "import sys; import ebsd_autoplot.ml_dataset; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_registered_eds_numeric_channels_and_unavailable_flag(tmp_path):
    src, _ = feature_cache(tmp_path, (3, 5))
    eds_path = tmp_path / "eds.h5"
    with h5py.File(eds_path, "w") as f:
        f["Ti"] = np.arange(15).reshape(3, 5)
    eds = load_eds(eds_path, {"Ti": "Ti"}, expected_shape=(3, 5), spatial_metadata={"aligned_to_ebsd": True})
    out = tmp_path / "with_eds"
    result = export_patches(src, out, grid=(3, 5), eds=eds, preview=True)
    assert result["modalities"]["eds"] == "available"
    assert result["covered_source_pixels"] == 15 and result["dropped_source_pixels"] == 0
    data = PatchDataset(out / "dataset_index.json", ["eds_Ti"])
    assert [int(data[i]["channels"]["eds_Ti"][0, 0]) for i in range(15)] == list(range(15))
    from PIL import Image
    with Image.open(out / "patch_preview.png") as image:
        image.verify()
    with pytest.raises(ValueError, match="registration"):
        export_patches(src, tmp_path / "unregistered", eds=EDSResult(maps={"Ti": np.ones((3, 5))}, available=True), preview=False)


def test_v2_patch_ids_gos_and_grod_are_real_numeric_maps(tmp_path):
    src, _ = feature_cache(tmp_path, (2, 3))
    v2 = tmp_path / "v2"; v2.mkdir()
    with h5py.File(v2 / "v2_grains.h5", "w") as f:
        f["grain_id"] = [[7, 7, 11], [7, 11, 11]]
        f["GROD_deg"] = [[.1, .2, .3], [.4, .5, .6]]
        f.attrs["roi_xywh"] = [0, 0, 3, 2]
    (v2 / "grains.csv").write_text("grain_id,GOS_deg\n7,0.25\n", encoding="utf-8")
    out = tmp_path / "v2_patches"
    export_patches(src, out, mode="roi", v2_dir=v2, preview=False)
    sample = PatchDataset(out / "dataset_index.json", ["grain_id", "GOS_deg", "GROD_deg"])[0]
    assert sample["channels"]["grain_id"].tolist() == [[7, 7, 11], [7, 11, 11]]
    assert sample["channels"]["GOS_deg"][0, 0] == .25
    assert np.isnan(sample["channels"]["GOS_deg"][0, 2])
    assert not sample["masks"]["GOS_deg"][0, 2]


def test_cancellation_never_exposes_final_partial_database(tmp_path):
    from ebsd_autoplot.storage import cancellation, Cancelled
    src, _ = feature_cache(tmp_path)
    calls = []
    def cancel_after_first_patch():
        calls.append(True)
        return len(calls) >= 3
    out = tmp_path / "cancelled"
    with cancellation(cancel_after_first_patch), pytest.raises(Cancelled):
        export_patches(src, out, preview=False)
    assert not (out / "patch_manifest.json").exists()
    assert not (out / "patches").exists()


def h5oina_eds_fixture(tmp_path, *, unit="counts", shift=0., shape=(3, 4)):
    path=tmp_path/'flat.h5oina';h,w=shape
    values=np.arange(1,h*w+1,dtype=np.float32).reshape(shape)
    with h5py.File(path,'w') as f:
        for modality in ('EBSD','EDS'):
            head=f.create_group(f'1/{modality}/Header');data=f.create_group(f'1/{modality}/Data')
            for key,value,units in [('X Cells',w,'px'),('Y Cells',h,'px'),('X Step',.2,'um'),('Y Step',.3,'um')]:
                ds=head.create_dataset(key,data=[value]);ds.attrs['Unit']=units
            yy,xx=np.indices(shape)
            for key,arr in [('X',xx*.2+(shift if modality=='EDS' else 0)),('Y',yy*.3)]:
                ds=data.create_dataset(key,data=arr.ravel());ds.attrs['Unit']='um'
        data=f['1/EDS/Data']
        for key in ('Live Time','Real Time'):
            times=np.full(shape,2.,np.float32)
            if key=='Live Time':times[0,0]=0;times[1,1]=np.nan
            ds=data.create_dataset(key,data=times.ravel());ds.attrs['Unit']='s'
        window=data.create_group('Window Integral')
        for label,scale in [('Al Kα1',1),('Ti Kα1',10),('V Kα1',.5)]:
            ds=window.create_dataset(label,data=(values*scale).ravel());ds.attrs['Unit']=unit
            ds.attrs['X-ray Line']='Kα1'
    return path,values


def test_h5oina_flat_windows_all_position_checks_rate_and_native_preservation(tmp_path):
    path,values=h5oina_eds_fixture(tmp_path)
    result=load_h5oina_eds(path,count_rate=True,block_rows=1)
    assert result.available and result.metadata['grid_verification']['aligned']
    assert result.metadata['grid_verification']['compared_pixels']==12
    assert result.metadata['registration']['resampled'] is False
    assert np.array_equal(result.maps['Al Kα1'],values)
    assert result.units['Al Kα1']=='counts'
    assert result.units['Al Kα1_cps']=='counts/s'
    assert np.isnan(result.maps['Al Kα1_cps'][0,0])
    assert not result.masks['Al Kα1_cps'][0,0] and not result.masks['Al Kα1_cps'][1,1]
    assert result.maps['Al Kα1_cps'][2,3]==values[2,3]/2
    assert result.timing_units['Live Time']=='s' and result.timing_maps['Real Time'].shape==(3,4)
    phases=np.ones((3,4),int);phases[:,2:]=2
    report=save_eds(result,tmp_path/'saved',phase_map=phases,indexed=np.ones((3,4),bool),phase_names={1:'alpha',2:'beta'})
    stats=[row for row in report['phase_intensity_statistics'] if row['channel']=='Al Kα1']
    assert stats[0]['mean_intensity']==values[:,:2].mean()
    assert stats[1]['mean_intensity']==values[:,2:].mean()
    assert all(row['interpretation'].endswith('not concentration') for row in stats)
    with h5py.File(tmp_path/'saved/eds_fields.h5','r') as f:
        assert np.array_equal(f['channels/Al Kα1'][:],values)
        assert f['channels/Al Kα1'].attrs['unit']=='counts'
        assert f['native_timing/Live Time'].attrs['unit']=='s'
        assert not f['masks/Al Kα1_cps'][0,0]
    assert validate_receipt(tmp_path/'saved','ebsd_autoplot.eds.save_eds')


def test_h5oina_existing_cps_never_divided_twice(tmp_path):
    path,values=h5oina_eds_fixture(tmp_path,unit='cps')
    result=load_h5oina_eds(path,count_rate=True,block_rows=2)
    assert np.array_equal(result.maps['Al Kα1'],values)
    assert result.units['Al Kα1']=='cps' and 'Al Kα1_cps' not in result.maps
    assert 'no second division' in result.metadata['rate_channels']['Al Kα1']['method']


def test_h5oina_shifted_or_unknown_units_require_explicit_registration(tmp_path):
    path,values=h5oina_eds_fixture(tmp_path,shift=.1)
    with pytest.raises(ValueError,match='pixel positions differ'):
        load_h5oina_eds(path,block_rows=1)
    result=load_h5oina_eds(path,registration=np.eye(3))
    assert not result.metadata['grid_verification']['aligned']
    assert result.metadata['registration']['resampled']
    assert np.array_equal(result.maps['Al Kα1'],values)
    # Ensure differences beyond the first row are checked too.
    with h5py.File(path,'r+') as f:
        f['1/EDS/Data/X'][:]=f['1/EBSD/Data/X'][:]
        f['1/EDS/Data/X'][-1]+=.05
    with pytest.raises(ValueError,match='pixel positions differ'):
        load_h5oina_eds(path,block_rows=1)
    with h5py.File(path,'r+') as f:
        f['1/EDS/Data/X'][:]=f['1/EBSD/Data/X'][:]
        del f['1/EDS/Header/X Step'].attrs['Unit']
    with pytest.raises(ValueError,match='spatial length unit'):
        load_h5oina_eds(path)


def test_h5oina_no_eds_unavailable_and_count_unit_cannot_be_guessed(tmp_path):
    src,_=feature_cache(tmp_path)
    missing=load_h5oina_eds(src)
    assert not missing.available and not missing.maps
    saved=save_eds(missing,tmp_path/'unavailable')
    assert not saved['available'] and not (tmp_path/'unavailable/eds_fields.h5').exists()
    path,_=h5oina_eds_fixture(tmp_path,unit='unknown experimental intensity')
    native=load_h5oina_eds(path)
    assert native.units['Al Kα1']=='unknown experimental intensity'
    with pytest.raises(ValueError,match='Cannot infer counts'):
        load_h5oina_eds(path,count_rate=True)


def test_h5oina_known_length_unit_conversion_and_step_mismatch(tmp_path):
    path,values=h5oina_eds_fixture(tmp_path)
    with h5py.File(path,'r+') as f:
        for key in ('X Step','Y Step'):
            ds=f['1/EDS/Header/'+key];ds[:]*=1000;ds.attrs['Unit']='nm'
        for key in ('X','Y'):
            ds=f['1/EDS/Data/'+key];ds[:]*=1000;ds.attrs['Unit']='nm'
    result=load_h5oina_eds(path)
    assert result.metadata['grid_verification']['aligned']
    assert np.array_equal(result.maps['Al Kα1'],values)
    with h5py.File(path,'r+') as f:f['1/EDS/Header/X Step'][:]*=2
    with pytest.raises(ValueError,match='physical step sizes differ'):
        load_h5oina_eds(path)


def test_h5oina_float32_ulp_alignment_tolerance_is_recorded_serializable(tmp_path):
    path,_=h5oina_eds_fixture(tmp_path,shape=(3,300))
    with h5py.File(path,'r+') as f:
        for modality in ('EDS','EBSD'):
            for key in ('X','Y'):
                data=f[f'1/{modality}/Data'];arr=data[key][:].astype(np.float32)
                if modality=='EDS' and key=='X':arr=np.nextafter(arr,np.float32(np.inf))
                del data[key];ds=data.create_dataset(key,data=arr);ds.attrs['Unit']='um'
    result=load_h5oina_eds(path,block_rows=1)
    assert result.metadata['grid_verification']['aligned']
    assert result.metadata['grid_verification']['max_coordinate_error_um']['X']>0
    json.dumps(result.metadata)
    save_eds(result,tmp_path/'float32_export')
    with h5py.File(path,'r+') as f:f['1/EDS/Data/X'][-1]+=.002
    with pytest.raises(ValueError,match='pixel positions differ'):
        load_h5oina_eds(path)
