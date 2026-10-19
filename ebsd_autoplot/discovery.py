"""Discover independent scans and verified montage provenance, without reading pixels."""
from dataclasses import dataclass, replace
from pathlib import Path, PureWindowsPath
import json
import re
import h5py
from .storage import long_path, canonical_path

ROWS = re.compile(r'^(.*?)_?rows(\d+)-(\d+)_EBSD\.h5$', re.I)

@dataclass(frozen=True)
class ScanSpec:
    input_path: Path
    scan_path: str
    parts: tuple  # (path, first row, exclusive last row), -1 for a whole file
    source_name: str
    name_source: str
    source_paths: tuple = ()  # File-level identity, common to independent scans.
    scan_count: int = 1
    scan_tag: str = ''

def clean_source_name(value, *, from_manifest=True):
    if not isinstance(value, str) or not value.strip(): raise ValueError('Empty source_name')
    value = value.strip()
    normal = value.replace('\\', '/')
    if from_manifest and (PureWindowsPath(value).drive or normal.startswith('/') or '..' in normal.split('/')):
        raise ValueError(f'Unsafe source_name: {value}')
    name = normal.rsplit('/', 1)[-1]
    if name in ('', '.', '..'): raise ValueError(f'Unsafe source_name: {value}')
    return name

def _text(value):
    import numpy as np
    value = np.asarray(value).reshape(-1)[0]
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)

def scan_paths(path):
    with h5py.File(long_path(path), 'r') as f:
        found=[]
        def visit(key, obj):
            if isinstance(obj,h5py.Group) and key.endswith('/EBSD/Data') and 'Euler' in obj and 'Phase' in obj:
                found.append('/'+key.rsplit('/Data',1)[0])
        f.visititems(visit)
    return sorted(found)

def _provenance(path, scan):
    names=[]; tags=[]
    with h5py.File(long_path(path), 'r') as f:
        if scan not in f: raise ValueError(f'Scan {scan} missing in {path.name}')
        for group in (f, f[scan], f[scan+'/Header']):
            for key in ('source_name','original_source_name','original_filename'):
                if key in group.attrs: names.append(clean_source_name(_text(group.attrs[key]),from_manifest=False))
                if key in group and isinstance(group[key],h5py.Dataset): names.append(clean_source_name(_text(group[key][()]),from_manifest=False))
            for key in ('scan_id','scan_identifier'):
                if key in group.attrs: tags.append(_text(group.attrs[key]))
    if len({n.casefold() for n in names})>1: raise ValueError(f'Contradictory source names inside {path.name}: {names}')
    if len(set(tags))>1: raise ValueError(f'Contradictory scan IDs inside {path.name}: {tags}')
    return (names[0] if names else None, tags[0] if tags else None)

def _part_path(root, value):
    if not isinstance(value,str): raise ValueError('Shard filename must be text')
    normal=value.replace('\\','/')
    if PureWindowsPath(value).drive or normal.startswith('/') or '..' in normal.split('/'):
        raise ValueError(f'Unsafe shard filename: {value}')
    path=(root/normal).resolve()
    if root not in path.parents or not long_path(path).is_file(): raise ValueError(f'Shard missing or outside selected folder: {value}')
    return path

def discover_scans(path):
    path=canonical_path(path)
    specs=[]
    if long_path(path).is_file():
        for scan in scan_paths(path): specs.append(ScanSpec(path,scan,((path,0,-1),),path.name,'input filename'))
    elif long_path(path).is_dir():
        manifest=path/'请上传_分析清单.json'
        documents=[]
        for candidate in ([manifest] if manifest.exists() else sorted(path.glob('*.json'))):
            try: data=json.loads(long_path(candidate).read_text(encoding='utf-8-sig'))
            except (OSError,ValueError): continue
            if isinstance(data,dict) and isinstance(data.get('scans'),list): documents.append((candidate,data))
        if len(documents)>1: raise ValueError('Multiple analysis manifests; select a scan-specific folder')
        if documents:
            document,data=documents[0]
            for sc in data['scans']:
                scan=sc['scan_path']; parts=[]; names=[]; tags=[]
                declared=sc.get('source_name',data.get('source_name'))
                if declared is not None: declared=clean_source_name(declared)
                if sc.get('source_name') and data.get('source_name') and clean_source_name(data['source_name']).casefold()!=declared.casefold():
                    raise ValueError('Top-level and scan source_name disagree')
                for part in sc['parts']:
                    fp=_part_path(path,part['file']); first=int(part['row_start']); stop=int(part['row_end_exclusive'])
                    match=ROWS.match(fp.name)
                    if first<0 or stop<=first: raise ValueError('Invalid montage row range')
                    if match and (first!=int(match[2]) or stop!=int(match[3])+1): raise ValueError(f'Manifest/filename row mismatch: {fp.name}')
                    if part.get('scan_path',scan)!=scan:raise ValueError('Shard scan_path contradicts scan manifest')
                    if sc.get('scan_id') and re.fullmatch(r'scan\d+',str(sc['scan_id']),re.I) and match and match[1].rstrip('_').casefold()!=str(sc['scan_id']).casefold():
                        raise ValueError('Shard filename scan ID contradicts manifest')
                    if part.get('source_name'):names.append(clean_source_name(part['source_name']))
                    if part.get('scan_id'):tags.append(str(part['scan_id']))
                    name,tag=_provenance(fp,scan)
                    if name:names.append(name)
                    if tag:tags.append(tag)
                    parts.append((fp,first,stop))
                if not parts:raise ValueError('A montage scan must contain at least one shard')
                if len({x.casefold() for x in names})>1: raise ValueError('Montage source names disagree between shards')
                if declared and names and declared.casefold()!=names[0].casefold(): raise ValueError('Manifest source_name contradicts shard provenance')
                tag=str(sc.get('scan_id',tags[0] if tags else scan))
                if len(set(tags))>1 or (tags and sc.get('scan_id') is not None and tags[0]!=tag): raise ValueError('Montage scan identifiers disagree')
                name=declared or (names[0] if names else path.name+'_montage')
                basis='verified manifest source_name' if declared else ('shard provenance' if names else 'selected directory fallback')
                specs.append(ScanSpec(path,scan,tuple(sorted(parts,key=lambda x:x[1])),name,basis,scan_tag=tag))
        else:
            groups={}
            for fp in sorted(path.glob('*_EBSD.h5')):
                match=ROWS.match(fp.name)
                if not match:continue
                for scan in scan_paths(fp):
                    name,tag=_provenance(fp,scan)
                    tag=tag or match[1].rstrip('_') or scan
                    key=(name or '',scan,tag)
                    groups.setdefault(key,[]).append((fp,int(match[2]),int(match[3])+1))
            for (name,scan,tag),parts in groups.items():
                specs.append(ScanSpec(path,scan,tuple(sorted(parts,key=lambda x:x[1])),name or path.name+'_montage',
                    'shard provenance' if name else 'selected directory fallback',scan_tag=tag))
    if not specs: raise FileNotFoundError('No supported EBSD scans found in the selected input')
    # Common file-level source identity; independent scans from one source share one folder.
    final=[]
    for spec in specs:
        siblings=[s for s in specs if s.source_name.casefold()==spec.source_name.casefold()]
        source_paths=tuple(sorted({p[0] for s in siblings for p in s.parts},key=lambda p:str(p).casefold()))
        final.append(replace(spec,source_paths=source_paths,scan_count=len(siblings)))
    ids=[(s.source_name.casefold(),s.scan_path,s.scan_tag) for s in final]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate or ambiguous scan records in manifest')
    return sorted(final,key=lambda s:(s.source_name.casefold(),s.scan_path,s.scan_tag))
