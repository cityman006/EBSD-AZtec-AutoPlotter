"""Bounded, lazy map tiles from immutable feature caches; no recomputation.

All coordinates are native raster pixel indices. Pyramid level L samples every
2**L source pixels. Only requested tiles are read/persisted. An optional caller
cache_root enables disk reuse; when its budget is full, reads still work and
new cache writes are skipped. No user file, old tile or cache entry is deleted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import h5py
import numpy as np
from PIL import Image

CACHE_VERSION = "ebsd-preview-tiles-v1"
MAX_TILE_SIZE = 512


def _stamp(path):
    path = Path(path).resolve()
    st = path.stat()
    return {"path": os.path.normcase(str(path)), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _jet_lut():
    from matplotlib import colormaps
    return (colormaps['jet'](np.linspace(0,1,256))[...,:3]*255).astype(np.uint8)


JET_LUT = _jet_lut()


def _field(field):
    value = str(field).strip().upper().replace("-", "_")
    if value == "PHASE": return "Phase"
    if value in ("BC", "KAM", "IPF_X", "IPF_Y", "IPF_Z"): return value
    raise ValueError("Preview field must be BC, KAM, Phase, IPF_X, IPF_Y or IPF_Z")


class PreviewCache:
    """Map source plus an optional independently located tile store.

    cache_root must be supplied to persist tiles. max_cache_bytes=None means no
    disk budget; zero disables writes. Invalid/stale entries are ignored, never
    removed. Keys contain source stat evidence, field, level and native bounds.
    """
    def __init__(self, features_file, *, cache_root=None, max_cache_bytes=256*1024*1024):
        self.features_file = Path(features_file).resolve()
        self.cache_root = Path(cache_root).resolve() / CACHE_VERSION if cache_root is not None else None
        if max_cache_bytes is not None and int(max_cache_bytes) < 0:
            raise ValueError("Preview cache budget cannot be negative")
        self.max_cache_bytes = None if max_cache_bytes is None else int(max_cache_bytes)
        self.last_tile_metadata = {}
        self._refresh()

    def _refresh(self):
        self.source_stamp = _stamp(self.features_file)
        with h5py.File(self.features_file, "r") as f:
            self.native_shape = tuple(map(int, f["indexed"].shape))
            if len(self.native_shape) != 2 or min(self.native_shape) <= 0:
                raise ValueError("Preview source must have a nonempty 2D indexing mask")
            if tuple(f["features"].shape) != (7, *self.native_shape):
                raise ValueError("Preview requires the standard seven-channel features cache")
            self.metadata = json.loads(f.attrs["metadata_json"])

    def _check_current(self):
        if _stamp(self.features_file) != self.source_stamp:
            self._refresh()

    def _identity(self, field, bounds, level):
        value = {"cache_version": CACHE_VERSION, "source_stamp": self.source_stamp,
                 "field": field, "level": level, "bounds_xywh": list(bounds),
                 "native_shape": list(self.native_shape)}
        key = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        return key, value

    def _paths(self, key):
        directory = self.cache_root / key[:2]
        return directory / (key + ".png"), directory / (key + ".json")

    def _cached(self, key, identity, shape):
        if self.cache_root is None: return None
        image_path, json_path = self._paths(key)
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if data.get("identity") != identity: return None
            with Image.open(image_path) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (shape[1], shape[0]): return None
                rgb = np.array(image, dtype=np.uint8)
            if hashlib.sha256(rgb.tobytes()).hexdigest() != data.get("rgb_sha256"): return None
            return rgb
        except (OSError, ValueError, KeyError):
            return None

    def _persist(self, key, identity, rgb):
        if self.cache_root is None or self.max_cache_bytes == 0: return False
        # Deliberately use budget refusal instead of deleting anything. Only
        # this algorithm's own namespace is counted; user material is untouched.
        if self.max_cache_bytes is not None:
            used = sum(p.stat().st_size for p in self.cache_root.glob("*/*")
                       if p.is_file() and p.suffix in (".png", ".json")) if self.cache_root.exists() else 0
            # Conservative upper bound for a lossless tile plus its metadata.
            if used + rgb.nbytes + 4096 > self.max_cache_bytes: return False
        image_path, json_path = self._paths(key)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        image_temp = image_path.with_name(key + "." + token + ".png.tmp")
        json_temp = json_path.with_name(key + "." + token + ".json.tmp")
        Image.fromarray(rgb).save(image_temp, format="PNG")
        record = {"identity": identity, "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
        json_temp.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        # Both destinations are opaque keys in our own caller-selected cache.
        # A partial/crashed pair is ignored by strict metadata/hash validation.
        os.replace(image_temp, image_path); os.replace(json_temp, json_path)
        return True

    def _read_rgb(self, field, bounds, level):
        x, y, w, h = bounds
        step = 1 << level
        sl = (slice(y, y+h, step), slice(x, x+w, step))
        with h5py.File(self.features_file, "r") as f:
            indexed = np.asarray(f["indexed"][sl], bool)
            if field.startswith("IPF_"):
                key = "ipf_" + field[-1]
                if key not in f: raise ValueError(f"{field} is unavailable; preview never computes missing IPF")
                ds = f[key]
                if tuple(ds.shape) != (*self.native_shape, 3) or np.dtype(ds.dtype) != np.dtype("uint8"):
                    raise ValueError("Cached IPF must be native-shape uint8 RGB")
                rgb = np.asarray(ds[sl[0], sl[1], :], np.uint8)
                rgb[~indexed] = 0
                return rgb
            channel = {"Phase": 4, "KAM": 5, "BC": 6}[field]
            raw = np.asarray(f["features"][channel, sl[0], sl[1]])
            good = indexed & np.isfinite(raw)
            rgb = np.zeros((*indexed.shape, 3), np.uint8)
            if field == "Phase":
                phase = np.where(good, np.rint(raw), 0).astype(np.int64)
                if np.any(good & (raw != phase)):
                    raise ValueError("Cached phase IDs must be exact integers")
                colors = {int(pid): spec["rgb"] for pid,spec in self.metadata["phase_specs"].items()}
                if set(np.unique(phase[good]).tolist()) - set(colors):
                    raise ValueError("Indexed phase lacks its recorded RGB color")
                for pid, color in colors.items():
                    color = np.asarray(color)
                    if color.shape != (3,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 255)):
                        raise ValueError("Phase RGB must be three recorded byte values")
                    rgb[good & (phase == pid)] = color.astype(np.uint8)
            elif field == "BC":
                value = np.clip(np.where(good, raw, 0), 0, 255).astype(np.uint8)
                rgb[:] = value[..., None]
            else:
                if "kam_valid" not in f: raise ValueError("KAM validity mask is missing")
                good &= np.asarray(f["kam_valid"][sl], bool)
                value = np.clip(np.where(good, raw, 0)/5, 0, 1)
                rgb[:] = JET_LUT[(value*255).astype(np.uint8)]
            rgb[~good] = 0
            return rgb

    def get_tile(self, field, bounds_xywh, *, level=0, tile_size=512):
        field = _field(field)
        if int(level) != level or not 0 <= int(level) <= 30:
            raise ValueError("Pyramid level must be an integer in [0,30]")
        if int(tile_size) != tile_size or not 1 <= int(tile_size) <= MAX_TILE_SIZE:
            raise ValueError("Tile size must be an integer in [1,512]")
        if len(bounds_xywh) != 4 or any(int(v) != v for v in bounds_xywh):
            raise ValueError("Tile bounds must be four integer native pixel coordinates")
        self._check_current()
        x, y, w, h = map(int, bounds_xywh)
        if min(w, h) < 1: raise ValueError("Tile dimensions must be positive")
        H, W = self.native_shape
        right, bottom = min(W, x+w), min(H, y+h)
        x, y = max(0,x), max(0,y)
        stride = 1 << int(level)
        bounds = (x, y, min(right-x, int(tile_size)*stride), min(bottom-y, int(tile_size)*stride))
        if min(bounds[2:]) <= 0: raise ValueError("Tile does not intersect the native scan")
        key, identity = self._identity(field, bounds, int(level))
        shape = ((bounds[3]+stride-1)//stride, (bounds[2]+stride-1)//stride)
        rgb = self._cached(key, identity, shape)
        hit = rgb is not None
        if not hit:
            rgb = self._read_rgb(field, bounds, int(level))
            if _stamp(self.features_file) != self.source_stamp:
                raise ValueError("Source cache changed while reading preview tile; retry with current source")
            try:
                persisted = self._persist(key, identity, rgb)
            except OSError:
                # A read-only/full cache volume must not prevent viewing the
                # immutable numeric cache. Any partial pair fails validation.
                persisted = False
        else: persisted = True
        if _stamp(self.features_file) != self.source_stamp:
            raise ValueError("Source cache changed during preview tile retrieval; retry")
        self.last_tile_metadata = {**identity, "cache_key": key, "cache_hit": hit, "persisted": persisted,
                                   "tile_shape": list(rgb.shape[:2]), "stride": stride}
        return rgb, bounds

    def preview(self, field, max_side=1400):
        if int(max_side) != max_side or not 1 <= int(max_side) <= 4096:
            raise ValueError("Preview max_side must be an integer in [1,4096]")
        self._check_current()
        initial_stamp = dict(self.source_stamp)
        h, w = self.native_shape
        level = max(0, int(np.ceil(np.log2(max(1, max(h,w)/int(max_side))))))
        stride = 1 << level
        out_h, out_w = (h+stride-1)//stride, (w+stride-1)//stride
        rgb = np.empty((out_h, out_w, 3), np.uint8)
        hits = tiles = 0
        for row in range(0, out_h, MAX_TILE_SIZE):
            for col in range(0, out_w, MAX_TILE_SIZE):
                tile, bounds = self.get_tile(field, (col*stride,row*stride,
                    min(MAX_TILE_SIZE,out_w-col)*stride,min(MAX_TILE_SIZE,out_h-row)*stride), level=level)
                rgb[row:row+tile.shape[0],col:col+tile.shape[1]] = tile
                hits += int(self.last_tile_metadata["cache_hit"]); tiles += 1
        if initial_stamp != self.source_stamp or _stamp(self.features_file) != initial_stamp:
            raise ValueError("Source cache changed during preview assembly; retry")
        return rgb, {"field": _field(field), "native_shape": [h,w], "preview_shape": [out_h,out_w],
                     "level": level, "stride": stride, "source_stamp": initial_stamp,
                     "cache_version": CACHE_VERSION, "tile_count": tiles, "disk_cache_hits": hits,
                     "sampling": "native nearest samples at power-of-two stride; no numerical recomputation",
                     "display_ranges": {"BC": [0,255], "KAM_degree": [0,5]},
                     "invalid_pixels": "black; indexing and KAM-valid masks applied",
                     "phase_colors": self.metadata.get("phase_specs", {})}


def preview(cache, field, max_side=1400, *, cache_root=None, max_cache_bytes=256*1024*1024):
    reader = cache if isinstance(cache, PreviewCache) else PreviewCache(cache, cache_root=cache_root, max_cache_bytes=max_cache_bytes)
    return reader.preview(field, max_side=max_side)


def get_tile(cache, field, bounds_xywh, *, level=0, tile_size=512,
             cache_root=None, max_cache_bytes=256*1024*1024):
    reader = cache if isinstance(cache, PreviewCache) else PreviewCache(cache, cache_root=cache_root, max_cache_bytes=max_cache_bytes)
    return reader.get_tile(field, bounds_xywh, level=level, tile_size=tile_size)
