"""Lazy previews must read bounded hyperslabs and preserve source map meaning."""
import hashlib
import json
import os

import h5py
import numpy as np
import pytest

from ebsd_autoplot import preview_cache as pc


def source(path, h=70, w=90):
    yy, xx = np.indices((h,w))
    features = np.zeros((7,h,w), np.float32)
    features[4] = np.where(xx < w//2, 300, 17)
    features[5] = (xx%6).astype(float)
    features[6] = (xx+yy)%256
    indexed = np.ones((h,w), np.uint8); indexed[0,0] = 0
    kam_valid = indexed.copy(); kam_valid[1,1] = 0
    ipf = np.stack((xx%256, yy%256, (xx+yy)%256), axis=-1).astype(np.uint8)
    with h5py.File(path, "w") as f:
        f["features"] = features; f["indexed"] = indexed; f["kam_valid"] = kam_valid
        for axis in "XYZ": f["ipf_"+axis] = ipf
        f.attrs["metadata_json"] = json.dumps({"phase_specs": {
            300: {"name":"arbitrary phase 300", "rgb":[12,34,56]},
            17: {"name":"arbitrary phase 17", "rgb":[78,90,123]}}})
    return features, indexed, kam_valid, ipf


def test_tiles_native_pixels_masks_phase_ids_and_source_immutability(tmp_path):
    path = tmp_path / "features.h5"
    features, indexed, kam_valid, ipf = source(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = pc.PreviewCache(path, cache_root=tmp_path / "tiles")
    rgb, bounds = cache.get_tile("BC", (7,9,11,13))
    assert bounds == (7,9,11,13) and rgb.shape == (13,11,3)
    assert np.array_equal(rgb[...,0], features[6,9:22,7:18])
    assert np.array_equal(rgb[...,0], rgb[...,1])
    rgb, bounds = cache.get_tile("BC", (7,9,40,48), level=2)
    assert rgb.shape == (12,10,3)
    assert np.array_equal(rgb[...,0], features[6,9:57:4,7:47:4])
    rgb, _ = cache.get_tile("Phase", (0,0,90,70))
    assert np.array_equal(rgb[2,2], [12,34,56])
    assert np.array_equal(rgb[2,-2], [78,90,123])
    assert np.array_equal(rgb[0,0], [0,0,0])
    rgb, _ = cache.get_tile("KAM", (0,0,90,70))
    assert np.array_equal(rgb[1,1], [0,0,0])
    expected = pc.JET_LUT[(np.clip(features[5]/5,0,1)*255).astype(np.uint8)]
    expected[~(indexed.astype(bool)&kam_valid.astype(bool))] = 0
    assert np.array_equal(rgb, expected)
    rgb, _ = pc.get_tile(cache, "IPF_X", (0,0,90,70))
    assert np.array_equal(rgb[2,2], ipf[2,2]) and np.array_equal(rgb[0,0], [0,0,0])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_disk_hit_strict_metadata_invalidation_and_cache_budget(tmp_path, monkeypatch):
    path = tmp_path / "features.h5"; source(path)
    cache_root = tmp_path / "tiles"; cache_root.mkdir()
    personal = cache_root / "user_notes.txt"; personal.write_text("preserve")
    cache = pc.PreviewCache(path, cache_root=cache_root)
    first, _ = cache.get_tile("BC", (2,2,10,10))
    assert cache.last_tile_metadata["persisted"] and not cache.last_tile_metadata["cache_hit"]
    original_read = cache._read_rgb
    monkeypatch.setattr(cache, "_read_rgb", lambda *args: pytest.fail("disk hit reread numeric map"))
    second, _ = cache.get_tile("BC", (2,2,10,10))
    assert np.array_equal(first, second) and cache.last_tile_metadata["cache_hit"]
    monkeypatch.setattr(cache, "_read_rgb", original_read)
    json_path = next((cache_root/pc.CACHE_VERSION).glob("*/*.json"))
    record = json.loads(json_path.read_text()); record["identity"]["field"] = "wrong"
    json_path.write_text(json.dumps(record))
    cache.get_tile("BC", (2,2,10,10))
    assert not cache.last_tile_metadata["cache_hit"]
    old_stamp = cache.source_stamp.copy()
    with h5py.File(path, "r+") as f: f["features"][6,2,2] = 123
    stat = path.stat(); os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1000000))
    current, _ = cache.get_tile("BC", (2,2,10,10))
    assert current[0,0,0] == 123 and cache.source_stamp != old_stamp
    limited = pc.PreviewCache(path, cache_root=cache_root, max_cache_bytes=1)
    limited.get_tile("KAM", (0,0,20,20))
    assert not limited.last_tile_metadata["persisted"]
    assert personal.read_text() == "preserve"


def test_preview_level_native_coordinates_and_missing_fields(tmp_path):
    path = tmp_path / "features.h5"; features, _, _, _ = source(path)
    rgb, metadata = pc.preview(path, "BC", max_side=20, cache_root=tmp_path/"tiles")
    assert metadata["native_shape"] == [70,90] and metadata["level"] == 3
    assert max(rgb.shape[:2]) <= 20
    assert np.array_equal(rgb[1:,:,0], features[6,8::8,::8])
    cache = pc.PreviewCache(path)
    rgb, bounds = cache.get_tile("BC", (-4,-3,10,10))
    assert bounds == (0,0,6,7) and rgb.shape == (7,6,3)
    with pytest.raises(ValueError, match="intersect"): cache.get_tile("BC", (900,900,20,20))
    with pytest.raises(ValueError, match="512"): cache.get_tile("BC", (0,0,20,20), tile_size=513)
    with h5py.File(path,"r+") as f: del f["ipf_Z"]
    with pytest.raises(ValueError, match="never computes"): cache.get_tile("IPF_Z", (0,0,10,10))


def test_huge_instrumented_source_only_bounded_strided_hyperslabs(tmp_path, monkeypatch):
    path = tmp_path / "huge_features.h5"; path.write_bytes(b"instrumented source")
    reads = []
    H, W = 100000, 200000
    class Dataset:
        def __init__(self, kind):
            self.kind = kind; self.shape = (7,H,W) if kind == "features" else (H,W)
        def __getitem__(self, key):
            if self.kind == "features": channel, ys, xs = key
            else: ys, xs = key; channel = None
            assert isinstance(ys, slice) and isinstance(xs, slice)
            rows = np.arange(*ys.indices(H)); cols = np.arange(*xs.indices(W))
            assert max(len(rows),len(cols)) <= 512, "full source grid read"
            reads.append((self.kind, ys, xs, len(rows),len(cols)))
            if self.kind != "features": return np.ones((len(rows),len(cols)), np.uint8)
            assert channel == 6
            return ((rows[:,None]+cols[None,:])%256).astype(np.float32)
    class File:
        attrs = {"metadata_json": '{"phase_specs":{}}'}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def __getitem__(self,key): return Dataset(key)
    monkeypatch.setattr(pc.h5py,"File",lambda *args,**kwargs: File())
    reader = pc.PreviewCache(path, cache_root=tmp_path/"tiles")
    rgb, metadata = reader.preview("BC", max_side=1400)
    assert rgb.shape == (391,782,3) and metadata["level"] == 8
    assert len(reads) == 4  # indexing + BC for two requested coarse tiles
    assert all(key[1].step == 256 and key[2].step == 256 for key in reads)
    assert rgb[7,8,0] == ((7+8)*256)%256
    tile, bounds = reader.get_tile("BC", (123,456,10000,10000), tile_size=512)
    assert tile.shape == (512,512,3) and bounds == (123,456,512,512)
