"""Lossless native-pixel EBSD/EDS patch database export.

Only previews are display images. Scientific HDF5 arrays are cropped exactly;
no image resizing, normalisation, interpolation, or implicit missing-data fill.
"""
from __future__ import annotations
import csv
import hashlib
import json
import warnings
from contextlib import ExitStack
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from .compute import CHANNELS
from .eds import EDSResult
from .storage import transactional_output, check_cancelled


def _pair(value, name):
    if value is None or len(value) != 2:
        raise ValueError(f"{name} requires (height, width)")
    vals = tuple(int(x) for x in value)
    if any(isinstance(x, bool) or float(x) != y or y < 1 for x, y in zip(value, vals)):
        raise ValueError(f"{name} must contain positive integers")
    return vals


def plan_patches(shape, *, mode="grid", grid=(3, 5), size_px=None, size_um=None,
                 step_um=None, roi=None, stride_px=None, overlap_px=(0, 0),
                 edge_policy="include"):
    """Return source x,y,w,h crops and requested h,w storage sizes.

    Grid partitions use numpy.array_split remainder semantics: extra rows and
    columns go to the first partitions. include/preserve-variable-shape keeps
    all pixels; pad/pad-with-mask keeps all pixels with invalid right/bottom
    padding; drop/drop-explicit is an explicit, recorded lossy edge choice.
    """
    h, w = _pair(shape, "shape")
    if roi is None:
        x0 = y0 = 0; rw, rh = w, h
    else:
        if len(roi) != 4 or any(float(v) != int(v) for v in roi):
            raise ValueError("ROI requires integer x,y,width,height")
        x0, y0, rw, rh = map(int, roi)
        if min(x0, y0) < 0 or min(rw, rh) < 1 or x0 + rw > w or y0 + rh > h:
            raise ValueError("ROI outside feature grid")
    policies = {"include": "include", "preserve-variable-shape": "include",
                "pad": "pad", "pad-with-mask": "pad", "drop": "drop", "drop-explicit": "drop"}
    if edge_policy not in policies:
        raise ValueError("Edge policy must preserve-variable-shape, pad-with-mask, or drop-explicit")
    policy = policies[edge_policy]
    if mode == "grid":
        gy, gx = _pair(grid, "grid")
        if gy > rh or gx > rw:
            raise ValueError("Grid cannot produce empty patches")
        if stride_px is not None or tuple(overlap_px) != (0, 0):
            raise ValueError("Grid partitions cannot overlap; use fixed pixel/um mode")
        heights = [rh // gy + (i < rh % gy) for i in range(gy)]
        widths = [rw // gx + (i < rw % gx) for i in range(gx)]
        # Default variable crop contains all pixels even for uneven dimensions.
        target = max(heights), max(widths)
        ys = np.cumsum([0] + heights[:-1]).tolist(); xs = np.cumsum([0] + widths[:-1]).tolist()
        crops = [(x0 + xx, y0 + yy, widths[j], heights[i],
                  target if policy == "pad" else (heights[i], widths[j]))
                 for i, yy in enumerate(ys) for j, xx in enumerate(xs)
                 if policy != "drop" or (heights[i], widths[j]) == target]
    elif mode in ("pixels", "fixed_px", "um", "fixed_um"):
        if mode in ("um", "fixed_um"):
            if size_um is None or len(size_um) != 2 or step_um is None:
                raise ValueError("Fixed um requires explicit size_um=(height,width) and source step_um=(dy,dx)")
            sizes, steps = np.asarray(size_um, float), np.asarray(step_um, float)
            if not np.isfinite(sizes).all() or not np.isfinite(steps).all() or np.any(sizes <= 0) or np.any(steps <= 0):
                raise ValueError("Physical patch dimensions and source steps must be positive finite µm")
            ph, pw = map(int, np.ceil(sizes / steps - 1e-12))
        else:
            ph, pw = _pair(size_px, "size_px")
        if len(overlap_px) != 2 or any(float(v) != int(v) or int(v) < 0 for v in overlap_px):
            raise ValueError("Overlap must be nonnegative integer (height,width)")
        overlap = tuple(map(int, overlap_px))
        if overlap[0] >= ph or overlap[1] >= pw:
            raise ValueError("Overlap must be smaller than patch dimensions")
        if stride_px is not None and overlap != (0, 0):
            raise ValueError("Set either stride_px or overlap_px")
        sy, sx = _pair(stride_px, "stride_px") if stride_px is not None else (ph - overlap[0], pw - overlap[1])
        if sy > ph or sx > pw:
            raise ValueError("Stride larger than patch would silently omit interior pixels")
        crops = []
        for yy in range(0, rh, sy):
            for xx in range(0, rw, sx):
                ch, cw = min(ph, rh - yy), min(pw, rw - xx)
                if policy == "drop" and (ch != ph or cw != pw):
                    continue
                crops.append((x0 + xx, y0 + yy, cw, ch, (ph, pw) if policy == "pad" else (ch, cw)))
    elif mode == "roi":
        crops = [(x0, y0, rw, rh, (rh, rw))]
    else:
        raise ValueError(f"Unknown slicing mode: {mode}")
    if not crops:
        raise ValueError("Slicing produced zero patches; change edge policy or patch size")
    return crops


def _padding(arr, shape, fill=0):
    result = np.full((*shape, *arr.shape[2:]), fill, dtype=arr.dtype)
    result[:arr.shape[0], :arr.shape[1]] = arr
    return result


@transactional_output()
def export_patches(features_file, outdir, *, mode="grid", grid=(3, 5), size_px=None,
                   size_um=None, roi=None, stride_px=None, overlap_px=(0, 0),
                   edge_policy="include", v2_dir=None, derived_channels=None,
                   eds=None, source_id=None, split_group=None, preview=True, selected_patches=None):
    """Export a transactional patch database and grouped dataset index.

    derived_channels={name:{'path':h5,'dataset':str,'unit':str,
      'roi_xywh':optional,'mask_dataset':optional}} supplies Schmid/variant/
    recrystallization/etc numeric maps. Their source ROI must contain each patch.
    EDS must be an aligned EDSResult from load_eds. split_group should identify
    the independent specimen; its default is the original scan source_id.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        src = stack.enter_context(h5py.File(features_file, "r"))
        if "features" not in src or src["features"].ndim != 3 or src["features"].shape[0] != 7:
            raise ValueError("Expected native seven-channel features.h5")
        meta = json.loads(src.attrs["metadata_json"])
        h, w = src["indexed"].shape
        if src["features"].shape[1:] != (h, w) or src["kam_valid"].shape != (h, w):
            raise ValueError("Feature grid and validity-mask dimensions differ")
        dy, dx = float(meta["y_step_um"]), float(meta["x_step_um"])
        if not np.isfinite([dx, dy]).all() or min(dx, dy) <= 0:
            raise ValueError("Patch physical metadata requires valid source steps in µm")
        crops = plan_patches((h, w), mode=mode, grid=grid, size_px=size_px, size_um=size_um,
                             step_um=(dy, dx), roi=roi, stride_px=stride_px, overlap_px=overlap_px,
                             edge_policy=edge_policy)
        selected=set(range(len(crops))) if selected_patches is None else set(map(int,selected_patches))
        if not selected or min(selected)<0 or max(selected)>=len(crops):
            raise ValueError('Select one or more valid geometric patch indices')
        # Analysis-run paths change on reanalysis: prefer original source identity.
        identifier = str(meta.get("source_file", meta.get("source", meta.get("path", Path(features_file).resolve())))) + "|" + str(meta.get("scan_path", ""))
        source_id = str(source_id or hashlib.sha256(identifier.encode()).hexdigest()[:20])
        split_group = str(split_group or source_id)
        derived = {}
        for name, desc in (derived_channels or {}).items():
            if not name or "/" in name or "\\" in name:
                raise ValueError("Derived channel names must be simple identifiers")
            f = stack.enter_context(h5py.File(desc["path"], "r"))
            ds = f[desc["dataset"]]
            droi = tuple(map(int, desc.get("roi_xywh", f.attrs.get("roi_xywh", (0, 0, w, h)))))
            if ds.ndim not in (2, 3) or ds.shape[:2] != droi[3:1:-1]:
                raise ValueError(f"Derived channel shape/ROI mismatch: {name}")
            md = f[desc["mask_dataset"]] if desc.get("mask_dataset") else None
            if md is not None and md.shape != ds.shape[:2]:
                raise ValueError(f"Derived channel validity mask shape mismatch: {name}")
            derived[name] = (ds, md,
                             droi, str(desc.get("unit", "unknown (not specified)")))
        vf = None; gos_lookup = {}
        if v2_dir is not None:
            vf = stack.enter_context(h5py.File(Path(v2_dir) / "v2_grains.h5", "r"))
            vroi = tuple(map(int, vf.attrs["roi_xywh"]))
            for name in ("grain_id", "GROD_deg"):
                if name in vf:
                    derived[name] = (vf[name], None, vroi, "grain identifier" if name == "grain_id" else "degree")
            csv_file = Path(v2_dir) / "grains.csv"
            if csv_file.is_file():
                with csv_file.open(encoding="utf-8", newline="") as stream:
                    gos_lookup = {int(row["grain_id"]): float(row["GOS_deg"]) for row in csv.DictReader(stream)}
        if eds is not None and not isinstance(eds, EDSResult):
            raise TypeError("eds must be an explicit aligned EDSResult")
        if eds is not None and eds.available:
            if any(arr.shape != (h, w) for arr in eds.maps.values()):
                raise ValueError("EDS grid differs; register explicitly before slicing")
            if "registration" not in eds.metadata:
                raise ValueError("EDS must carry explicit EBSD registration metadata")
        entries = []; channel_units = {}; modalities = {"ebsd": "available", "eds": "available" if eds and eds.available else "unavailable"}
        for ordinal, (x, y, cw, ch, target) in enumerate(crops):
            if ordinal not in selected:continue
            check_cancelled()
            pid = f"patch_{ordinal:05d}"; filename = f"patches/{pid}.h5"
            path = outdir / filename; path.parent.mkdir(exist_ok=True)
            sl = (slice(y, y + ch), slice(x, x + cw))
            raw = src["features"][:, sl[0], sl[1]]
            indexed = src["indexed"][sl].astype(bool)
            kam_valid = src["kam_valid"][sl].astype(bool)
            values = {}; masks = {}; units = {}
            for i, name in enumerate(CHANNELS):
                values[name] = raw[i]
                masks[name] = np.isfinite(raw[i]) & (kam_valid if i == 5 else indexed if i < 5 else True)
                units[name] = "degree" if i == 5 else "source intensity (uncalibrated)" if i == 6 else "dimensionless" if i < 4 else "phase identifier"
            quat = np.moveaxis(raw[:4], 0, -1)
            qvalid = indexed & np.isfinite(quat).all(-1) & (np.linalg.norm(quat, axis=-1) > 1e-12)
            for name in CHANNELS[:4]:
                masks[name] &= qvalid
            euler = np.full((*indexed.shape, 3), np.nan, np.float32)
            if np.any(qvalid):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Gimbal lock detected")
                    euler[qvalid] = Rotation.from_quat(quat[qvalid]).as_euler("ZXZ").astype(np.float32)
            values["euler_rad"] = euler; masks["euler_rad"] = qvalid; units["euler_rad"] = "radian; Bunge ZXZ crystal-to-sample"
            for name in ("pq", "mad", "ipf_X", "ipf_Y", "ipf_Z", "boundary_right", "boundary_down"):
                if name in src:
                    values[name] = src[name][sl]
                    masks[name] = indexed & (np.isfinite(values[name]).all(-1) if values[name].ndim == 3 else np.isfinite(values[name]))
                    units[name] = str(src[name].attrs.get("Unit", "RGB uint8 display colour" if name.startswith("ipf_") else "category identifier" if name.startswith("boundary_") else "source units (uncalibrated)"))
            for name, (ds, md, (rx, ry, rw, rh), unit) in derived.items():
                if x < rx or y < ry or x + cw > rx + rw or y + ch > ry + rh:
                    raise ValueError(f"Patch lies outside derived channel ROI: {name}; select a common ROI")
                dsl = (slice(y - ry, y - ry + ch), slice(x - rx, x - rx + cw))
                if name in values:
                    raise ValueError(f"Repeated channel name: {name}")
                values[name] = ds[dsl]
                masks[name] = np.isfinite(values[name]).all(-1) if values[name].ndim == 3 else np.isfinite(values[name])
                if md is not None:
                    masks[name] &= md[dsl].astype(bool)
                if name == "grain_id":
                    masks[name] &= values[name] > 0
                units[name] = unit
            if gos_lookup and "grain_id" in values:
                ids = values["grain_id"]
                gos = np.full(ids.shape, np.nan, np.float32)
                for gid in np.unique(ids):
                    if gid in gos_lookup:
                        gos[ids == gid] = gos_lookup[gid]
                values["GOS_deg"] = gos; masks["GOS_deg"] = np.isfinite(gos); units["GOS_deg"] = "degree"
            if eds is not None and eds.available:
                for name, arr in eds.maps.items():
                    key = "eds_" + str(name)
                    if "/" in key or "\\" in key:
                        raise ValueError("EDS labels must not contain path separators")
                    values[key] = arr[sl]; masks[key] = eds.masks[name][sl]; units[key] = eds.units[name]
            padding = np.zeros(target, bool); padding[:ch, :cw] = True
            patch_meta = {"schema_version": 1, "source_id": source_id, "parent_id": source_id,
                          "split_group": split_group, "patch_id": pid, "roi_xywh": [x, y, cw, ch],
                          "shape_hw": list(target), "step_um_xy": [dx, dy],
                          "pixel_to_source_pixels": [[1, 0, x], [0, 1, y], [0, 0, 1]],
                          "pixel_to_sample_um": [[dx, 0, x * dx], [0, dy, y * dy], [0, 0, 1]],
                          "coordinate_origin": "scan first pixel centre; global specimen position unavailable",
                          "padding_right_bottom_px": [target[1] - cw, target[0] - ch],
                          "units": units, "modalities": modalities, "label_status": "unavailable unless explicitly supplied as a derived channel",
                          "source_metadata": meta,
                          "eds_metadata": eds.metadata if eds is not None else {"status": "unavailable"}}
            with h5py.File(path, "x") as dst:
                dst.attrs["metadata_json"] = json.dumps(patch_meta, ensure_ascii=False)
                # Native seven channels are also stored unchanged for exact equality audit.
                feature_data = np.zeros((7, *target), dtype=raw.dtype); feature_data[:, :ch, :cw] = raw
                dst.create_dataset("features", data=feature_data, compression="gzip")
                dst.create_dataset("indexed", data=_padding(indexed, target), compression="gzip")
                dst.create_dataset("kam_valid", data=_padding(kam_valid, target), compression="gzip")
                dst.create_dataset("padding_mask", data=padding, compression="gzip")
                cg, mg = dst.create_group("channels"), dst.create_group("masks")
                for name, arr in values.items():
                    data = _padding(arr, target, np.nan if np.issubdtype(arr.dtype, np.floating) else 0)
                    ds = cg.create_dataset(name, data=data, compression="gzip"); ds.attrs["unit"] = units[name]
                    mg.create_dataset(name, data=_padding(masks[name], target), compression="gzip")
            channel_units.update(units)
            entries.append({"patch_id": pid, "file": filename, "source_id": source_id, "parent_id": source_id,
                            "split_group": split_group, "roi_xywh": [x, y, cw, ch],
                            "shape_hw": list(target), "channel_names": list(values),
                            "indexed_count":int(indexed.sum()),"indexed_fraction":float(indexed.mean()),
                            "physical_bounds_um":[x*dx,y*dy,(x+cw)*dx,(y+ch)*dy],"file_bytes":path.stat().st_size})
        selected_roi = list(roi) if roi is not None else [0, 0, w, h]
        def covered_length(intervals):
            end = -1; total = 0
            for a, b in sorted(intervals):
                total += max(0, b - max(a, end)); end = max(end, b)
            return total
        selected_crops=[c for i,c in enumerate(crops) if i in selected]
        # Rectangle union sweep remains correct for arbitrary enabled tiles.
        yedges=sorted({v for c in selected_crops for v in (c[1],c[1]+c[3])})
        covered=sum((yb-ya)*covered_length([(c[0],c[0]+c[2]) for c in selected_crops if c[1]<=ya and c[1]+c[3]>=yb])
                    for ya,yb in zip(yedges,yedges[1:]))
        manifest = {"schema_version": 1, "scientific_status": "lossless numeric crops; not independently validated with experimental EDS",
                    "source_id": source_id, "parent_id": source_id, "split_group": split_group,
                    "source_shape_hw": [h, w], "source_features": str(Path(features_file).resolve()),
                    "mode": mode, "grid": list(grid), "roi_xywh": selected_roi,
                    "requested_size_px_hw": size_px, "requested_size_um_hw": size_um,
                    "stride_px_hw": stride_px, "overlap_px_hw": list(overlap_px),
                    "edge_policy": edge_policy, "drop_is_explicit": edge_policy in ("drop", "drop-explicit"),
                    "patch_count": len(entries), "patches": entries, "channel_units": channel_units,
                    "selected_geometric_indices": sorted(selected),
                    "covered_source_pixels": covered, "dropped_source_pixels": selected_roi[2] * selected_roi[3] - covered,
                    "overlap_repeated_pixel_instances": sum(c[2] * c[3] for c in selected_crops) - covered,
                    "modalities": modalities, "coordinate_convention": "original pixel centres x,y; no rescale",
                    "split_rule": "Group by split_group/specimen (default original scan), never random patch split",
                    "physical_size_policy": "ceil(size_um/step_um); actual extents stored per patch"}
        (outdir / "patch_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        with (outdir/'patch_manifest.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            fields=['patch_id','file','source_id','split_group','roi_xywh','physical_bounds_um','shape_hw','indexed_count','indexed_fraction','file_bytes']
            writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader()
            for entry in entries:
                writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in entry.items()})
        (outdir / "dataset_index.json").write_text(json.dumps({"schema_version": 1, "entries": entries,
            "source_groups": {source_id: [e["patch_id"] for e in entries]},
            "split_groups": {split_group: [e["patch_id"] for e in entries]}}, indent=2, ensure_ascii=False), encoding="utf-8")
        if preview:
            _preview(src, selected_crops, outdir)
        return manifest


def _preview(src, crops, outdir):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    h, w = src["indexed"].shape
    stride = max(1, int(np.ceil(max(h, w) / 1400)))
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    ax.imshow(src["features"][6, ::stride, ::stride], cmap="gray", origin="upper",
              extent=(-.5, w - .5, h - .5, -.5), interpolation="nearest")
    for i, (x, y, cw, ch, _) in enumerate(crops):
        ax.add_patch(Rectangle((x - .5, y - .5), cw, ch, fill=False, edgecolor="tab:red", linewidth=.7))
        if len(crops) <= 100:
            ax.text(x + cw / 2, y + ch / 2, str(i), ha="center", color="tab:cyan", fontsize=7)
    ax.set(xlabel="Original pixel X", ylabel="Original pixel Y", title="Native-pixel patch partition (preview only)")
    fig.savefig(outdir / "patch_preview.png", dpi=150); plt.close(fig)
