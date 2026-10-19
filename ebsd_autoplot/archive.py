"""Collision-safe dataset and run directories for EBSD analysis outputs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import threading
import uuid
from .storage import atomic_json, file_lock, long_path, transactional_output, check_cancelled


@dataclass(frozen=True)
class OutputContext:
    dataset_id: str
    dataset_dir: Path
    analysis_run_id: str
    analysis_dir: Path
    cache_dir: Path
    render_id: str
    render_dir: Path
    source_version: str
    scope_dir: Path | None = None
    scan_id: str = ''


def _safe_stem(value: str) -> str:
    # Treat manifest names as data: never accept path separators or traversal.
    value = Path(str(value).replace("\\", "/")).name
    value = Path(value).stem or "dataset"
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).rstrip(" .") or "dataset"
    if value.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        value = "_" + value
    if len(value) > 72:
        value = value[:55].rstrip(" .") + "__" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    atomic_json(path,payload)


def _new_dir(parent: Path, prefix: str) -> tuple[str, Path]:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    for _ in range(20):
        run_id = f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"
        target = parent / run_id
        try:
            target.mkdir()
            return run_id, target
        except FileExistsError:
            continue
    raise FileExistsError(f"Unable to reserve unique run directory under {parent}")


def _source_info(source) -> dict:
    source.check_unchanged()
    paths = [p.path for p in source.parts]
    items = []
    for path in paths:
        st = long_path(path).stat()
        items.append({"path": str(path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    identity_basis = json.dumps({'input_path':os.path.normcase(str(source.path.resolve())),
        'source_name':source.spec.source_name.casefold() if long_path(source.path).is_dir() else None},sort_keys=True).encode('utf-8')
    dataset_id = hashlib.sha256(identity_basis).hexdigest()
    version_basis = json.dumps({"scan_path": source.scan_path,'scan_tag':source.spec.scan_tag,
        'parts':[(str(p.path),p.start,p.stop) for p in source.parts],"sizes": [x["size"] for x in items],
        "mtime_ns": [x["mtime_ns"] for x in items]}, sort_keys=True).encode("utf-8")
    return {"dataset_id": dataset_id, "source_version": hashlib.sha256(version_basis).hexdigest(),
            "source_files": items, "fingerprint_kind": "path+size+mtime; not a full content hash"}


def _lock(path: Path):
    return file_lock(path)


def create_context(source, output_root: str | Path) -> OutputContext:
    root = long_path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    info = _source_info(source)
    raw_name = source.spec.source_name
    display = _safe_stem(raw_name)
    candidate = root / display
    with _lock(root / ".datasets.lock"):
        if candidate.exists():
            manifest_path = candidate / "dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            old_paths=[os.path.normcase(x.get('path','')) for x in manifest.get('source',{}).get('source_files',[])]
            same_locator=(os.path.normcase(manifest.get('input_path',''))==os.path.normcase(str(source.path)))
            if long_path(source.path).is_dir():same_locator=same_locator and manifest.get('original_name','').casefold()==source.spec.source_name.casefold()
            if not same_locator and long_path(source.path).is_file():same_locator=old_paths==[os.path.normcase(str(source.path))]
            if same_locator and manifest.get('dataset_id'):info['dataset_id']=manifest['dataset_id']
            if manifest.get("dataset_id") != info["dataset_id"]:
                candidate = root / f"{display}__src_{info['dataset_id'][:8]}"
        suffix=8
        while candidate.exists():
            existing=candidate/'dataset_manifest.json'
            doc=json.loads(existing.read_text(encoding='utf-8')) if existing.exists() else {}
            if doc.get('dataset_id')==info['dataset_id']:break
            suffix+=4
            candidate=root/(f"{display}__src_{info['dataset_id'][:suffix]}" if suffix<=64 else f"{display}__src_{info['dataset_id']}_{uuid.uuid4().hex[:8]}")
        candidate.mkdir(parents=True, exist_ok=True)
        dataset_dir = candidate.resolve()
        if root != dataset_dir and root not in dataset_dir.parents:
            raise ValueError("Dataset path escaped output root")
        manifest_path = dataset_dir / "dataset_manifest.json"
        doc=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {
            "schema_version":2,"dataset_id":info['dataset_id'],"display_name":display,"original_name":raw_name,
            "name_source":source.spec.name_source,"source":info,"input_path":str(source.path),
            "created_at":datetime.now().astimezone().isoformat()}
        scope_dir=dataset_dir; scan_id=source.scan_path
        if source.spec.scan_count>1 or doc.get('scan_registry'):
            registry=doc.setdefault('scan_registry',{})
            key=source.scan_path+'|'+source.spec.scan_tag
            if key not in registry: registry[key]=f'scan_{len(registry)+1:03d}'
            scan_id=registry[key];scope_dir=dataset_dir/'scans'/scan_id
            scope_dir.mkdir(parents=True,exist_ok=True)
            _atomic_json(scope_dir/'scan_manifest.json',{'dataset_id':info['dataset_id'],'scan_id':scan_id,
                'scan_path':source.scan_path,'scan_tag':source.spec.scan_tag})
        _atomic_json(manifest_path,doc)
        analysis_id, analysis_dir = _new_dir(scope_dir / "analysis_runs", "analysis")
        cache_dir = analysis_dir / "analysis_cache"
        cache_dir.mkdir()
        render_id, render_dir = _new_dir(scope_dir / "renders", "render")
    return OutputContext(info["dataset_id"], dataset_dir, analysis_id, analysis_dir,
                         cache_dir, render_id, render_dir, info["source_version"],scope_dir,scan_id)


def write_run_manifest(path: Path, payload: dict) -> None:
    _atomic_json(path, payload)


def update_latest(context: OutputContext, kind: str, run_id: str, relative_path: Path) -> None:
    scope=context.scope_dir or context.dataset_dir
    latest = scope / "latest.json"
    with _lock(scope / ".latest.lock"):
        data = json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else {}
        data[kind] = {"id": run_id, "path": relative_path.relative_to(context.dataset_dir).as_posix(),
                      "updated_at": datetime.now().astimezone().isoformat()}
        _atomic_json(latest, data)


def locate_cache(cache):
    cache=long_path(cache).resolve()
    run=cache.parent.parent
    if cache.parent.name!='analysis_cache' or run.parent.name!='analysis_runs':raise ValueError('Not an archived analysis cache')
    dataset=next((p for p in run.parents if (p/'dataset_manifest.json').is_file()),None)
    if dataset is None:raise ValueError('Cache dataset manifest is missing')
    return dataset,run.parent.parent,run


def preview_legacy(root: str | Path) -> dict:
    """Read-only preview. Only source_metadata.json can establish provenance."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir(): raise NotADirectoryError(root)
    confirmed=[]; known=[]
    for meta_path in root.rglob('source_metadata.json'):
        if any(p in ('.staging','.transactions') for p in meta_path.relative_to(root).parts):continue
        folder=meta_path.parent
        if any(parent in known for parent in folder.parents): continue
        known.append(folder)
        try: metadata=json.loads(meta_path.read_text(encoding='utf-8'))
        except (OSError,ValueError): metadata={}
        source=metadata.get('source')
        if not isinstance(source,str): source=''
        source_path=Path(source) if source else None
        absolute=bool(source_path and source_path.is_absolute() and not any(part=='..' for part in source_path.parts))
        confirmed.append({'directory':str(folder),'source_path':str(source_path) if absolute else None,
                          'scan_id':metadata.get('scan_path'),'status':'confirmed' if absolute else 'unassigned',
                          'file_count':sum(1 for p in folder.rglob('*') if p.is_file())})
    unassigned=[]; ignored_inputs=[]
    for item in root.rglob('*'):
        if any(p in ('.staging','.transactions') for p in item.relative_to(root).parts):continue
        if item.is_file() and not any(folder==item.parent or folder in item.parents for folder in known):
            if item.suffix.lower() in ('.h5oina','.hdf5') or (item.suffix.lower()=='.h5' and item.name.lower() not in ('features.h5','v2_grains.h5')):
                ignored_inputs.append(str(item))
            else: unassigned.append(str(item))
    return {'source_root':str(root),'confirmed':confirmed,'unassigned_files':unassigned,'ignored_measurement_files':ignored_inputs,
            'note':'Preview only; no files were moved or copied.'}


@transactional_output()
def _copy_legacy_tree(source,outdir):
    source=Path(source)
    for src in source.rglob('*'):
        check_cancelled()
        rel=src.relative_to(source)
        if not src.is_file() or any(p in ('.staging','.transactions') for p in rel.parts):continue
        if rel.name in ('render_manifest.json','dataset_manifest.json','analysis_manifest.json'):
            rel=rel.with_name('legacy_'+rel.name)
        dest=outdir/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)


@transactional_output()
def _copy_unknown(files,source_root,outdir):
    for filename in files:
        check_cancelled();src=Path(filename);rel=src.relative_to(Path(source_root))
        dest=outdir/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)


def import_legacy(root: str | Path, output_root: str | Path, *, include_confirmed=True,
                  include_unassigned=False) -> list[Path]:
    """Copy selected legacy material; source tree is retained unchanged for rollback."""
    preview=preview_legacy(root)
    destination=Path(output_root).expanduser().resolve()
    source_root=Path(preview['source_root'])
    if destination==source_root or source_root in destination.parents:
        raise ValueError('Choose a new output root outside the legacy source tree so the preview remains stable')
    destination.mkdir(parents=True,exist_ok=True)
    imported=[]
    if include_confirmed:
        for item in preview['confirmed']:
            if item['status']!='confirmed':continue
            source_path=Path(item['source_path']).resolve()
            dataset_id=hashlib.sha256(json.dumps({'scan_path':item.get('scan_id'),
                'source_paths':[str(source_path)]},sort_keys=True).encode()).hexdigest()
            display=_safe_stem(source_path.name)
            candidate=destination/display
            with _lock(destination/'.datasets.lock'):
                if candidate.exists():
                    mp=candidate/'dataset_manifest.json'
                    existing=json.loads(mp.read_text(encoding='utf-8')) if mp.exists() else {}
                    if existing.get('dataset_id')!=dataset_id:candidate=destination/f'{display}__src_{dataset_id[:8]}'
                candidate.mkdir(parents=True,exist_ok=True)
                if not (candidate/'dataset_manifest.json').exists():
                    _atomic_json(candidate/'dataset_manifest.json',{'schema_version':1,'dataset_id':dataset_id,
                        'display_name':display,'original_name':source_path.name,'name_source':'legacy source_metadata.json',
                        'source':{'source_path':str(source_path),'scan_path':item.get('scan_id'),
                                  'fingerprint_kind':'legacy provenance only; source version unknown'},
                        'created_at':datetime.now().astimezone().isoformat()})
                render_id,render_dir=_new_dir(candidate/'renders','render')
            _copy_legacy_tree(item['directory'],render_dir)
            (render_dir/'logs').mkdir(exist_ok=True)
            if not (render_dir/'publication_style.json').exists():
                _atomic_json(render_dir/'publication_style.json',{'name':'legacy imported','font':'unknown'})
            _atomic_json(render_dir/'render_manifest.json',{'schema_version':1,'dataset_id':dataset_id,
                'scan_id':item.get('scan_id'),'render_id':render_id,'status':'success','analysis_run_id':None,
                'source_path':str(source_path),'style_name':'legacy imported','actual_font':'unknown',
                'formats':sorted({p.suffix.lower().lstrip('.') for p in render_dir.rglob('*') if p.suffix.lower() in ('.png','.tif','.tiff','.pdf','.svg')}),
                'legacy_import':True,'completed_at':datetime.now().astimezone().isoformat()})
            imported.append(render_dir)
    if include_unassigned and preview['unassigned_files']:
        target_parent=destination/'_legacy_unassigned'
        _,collection=_new_dir(target_parent,'collection')
        _copy_unknown(preview['unassigned_files'],preview['source_root'],collection)
        _atomic_json(collection/'legacy_import.json',{'status':'unknown source','source_root':preview['source_root'],
            'file_count':len(preview['unassigned_files']),'imported_at':datetime.now().astimezone().isoformat()})
        imported.append(collection)
    return imported
