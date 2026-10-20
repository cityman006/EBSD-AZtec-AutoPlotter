"""Row-streamed Oxford H5OINA and row-split HDF5 ingestion.

Never assumes that input fits in RAM. Manifest row ranges are authoritative.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
import h5py
from .mad_units import source_mad_units
from .discovery import discover_scans, ScanSpec
from .storage import long_path
import numpy as np

ROW_RE = re.compile(r'rows(\d+)-(\d+)_EBSD\.h5$', re.I)
FIELDS = {'euler':'Euler', 'phase':'Phase', 'bc':'Band Contrast', 'pq':'Pattern Quality',
          'mad':'Mean Angular Deviation', 'x':'X','y':'Y'}


def _scalar(g, key, default=None):
    if key not in g: return default
    a=g[key][()]
    if np.asarray(a).size==0:return default
    x=np.asarray(a).reshape(-1)[0]
    if isinstance(x,bytes):return x.decode('utf-8','replace')
    if hasattr(x, 'item'):return x.item()
    return x

def _lattice_parameters(group):
    """Read the lattice recorded in the phase header; never substitute nominal Ti values."""
    def values(key):
        if key not in group or not isinstance(group[key], h5py.Dataset):
            return None
        raw=np.asarray(group[key][()],dtype=float).reshape(-1)
        if not np.all(np.isfinite(raw)):
            return None
        return raw

    for key in ('Lattice Dimensions','Lattice Parameters','Lattice Constants','Lattice'):
        raw=values(key)
        if raw is None or raw.size<6 or np.any(raw[:3]<=0):continue
        a,b,c,alpha,beta,gamma=map(float,raw[:6])
        return {'a':a,'b':b,'c':c,'alpha_deg':alpha,'beta_deg':beta,'gamma_deg':gamma,
                'c_over_a':c/a,'source_key':key,
                'units':str(group[key].attrs.get('Units',group[key].attrs.get('Unit','not specified')))}

    # Oxford H5OINA stores dimensions and angles as separate phase-header
    # datasets.  The dimensions use the source lattice units and the angles
    # follow the H5OINA angular contract (radians); no nominal Ti constants are
    # introduced here.  c/a is dimensionless, so the source length unit does
    # not affect the Schmid calculation.
    dims=None
    for key in ('Lattice Dimensions','Lattice Parameters','Lattice Constants','Lattice'):
        raw=values(key)
        if raw is not None and raw.size>=3 and np.all(raw[:3]>0):
            dims=raw[:3].astype(float);dim_key=key;break
    angles=values('Lattice Angles')
    if dims is not None and angles is not None and angles.size>=3:
        a,b,c=map(float,dims)
        alpha,beta,gamma=np.rad2deg(angles[:3])
        unit=str(group[dim_key].attrs.get('Units',group[dim_key].attrs.get('Unit','not specified')))
        return {'a':a,'b':b,'c':c,'alpha_deg':float(alpha),'beta_deg':float(beta),'gamma_deg':float(gamma),
                'c_over_a':c/a,'source_key':f'{dim_key}+Lattice Angles',
                'units':unit,'angle_units':'radian (H5OINA contract)'}
    return None

@dataclass(frozen=True)
class Part:
    path: Path
    start: int
    stop: int


def _grid_integer(value, label, minimum=0):
    """Read one explicit count; never truncate fractions or infer from length."""
    raw = np.asarray(value)
    if raw.size != 1 or raw.dtype.kind not in 'iuf':
        raise ValueError(f'{label} must contain exactly one integer >= {minimum}')
    number = raw.reshape(-1)[0]
    if not np.isfinite(number) or number < minimum or number != int(number):
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return int(number)


def _part_grid(stream, scan_path, part):
    """Validate global-header or explicitly proven local-header shard layout.

    Compact_v1 preserves measured coordinates and slices arrays, while changing
    Header/Y Cells to the local row count. Only its explicit global dimensions
    and original row interval establish the full grid; array length is not
    evidence for a missing global dimension or an offset.
    """
    head = stream[scan_path+'/Header']
    nx = _grid_integer(head['X Cells'][()], 'Header X Cells', 1)
    ny = _grid_integer(head['Y Cells'][()], 'Header Y Cells', 1)
    keys = ('global_nx', 'global_ny', 'start_row_in_original', 'end_row_in_original_exclusive')
    present = [key in stream.attrs for key in keys]
    if any(present):
        if not all(present):
            raise ValueError(f'Incomplete extractor global grid/range metadata: {part.path.name}')
        gx, gy = (_grid_integer(stream.attrs[key], key, 1) for key in keys[:2])
        first = _grid_integer(stream.attrs[keys[2]], keys[2])
        stop = _grid_integer(stream.attrs[keys[3]], keys[3], 1)
        if first >= stop or stop > gy:
            raise ValueError(f'Extractor row range exceeds global grid: {part.path.name}')
        if part.stop == -1:
            if (first, stop) != (0, gy):
                raise ValueError('Partial extracted rows require the complete montage directory, not one shard')
        elif (first, stop) != (part.start, part.stop):
            raise ValueError(f'Extractor row range contradicts montage/filename: {part.path.name}')
        if nx != gx or ny not in (gy, stop-first):
            raise ValueError(f'Extractor Header dimensions contradict global/local grid: {part.path.name}')
        basis = 'explicit root global grid and original row interval'
    else:
        gx, gy = nx, ny
        basis = 'global Header grid contract'
    if 'original_full_y_cells' in head['Y Cells'].attrs:
        declared = _grid_integer(head['Y Cells'].attrs['original_full_y_cells'], 'original_full_y_cells', 1)
        if declared != gy:
            raise ValueError(f'original_full_y_cells contradicts global grid: {part.path.name}')
    return gx, gy, {'file': str(part.path), 'header_nx': nx, 'header_ny': ny,
                    'global_nx': gx, 'global_ny': gy,
                    'header_layout': 'global' if ny == gy else 'local', 'basis': basis}


class H5Source:
    def __init__(self, path, scan_id=None):
        if isinstance(path,ScanSpec): self.spec=path
        else:
            candidates=discover_scans(path)
            chosen=[s for s in candidates if scan_id in (s.scan_path,s.scan_tag)] if scan_id else candidates
            if len(chosen)!=1:raise ValueError(f'Select a scan explicitly; found {[s.scan_path for s in chosen]}')
            self.spec=chosen[0]
        self.path=self.spec.input_path
        self.parts=[Part(*p) for p in self.spec.parts]
        self.scan_path=self.spec.scan_path
        self.parts.sort(key=lambda z:z.start)
        for p in self.parts:
            if not long_path(p.path).is_file():raise FileNotFoundError(str(p.path))
        self.source_stats={str(p.path):(long_path(p.path).stat().st_size,long_path(p.path).stat().st_mtime_ns) for p in self.parts}
        with h5py.File(long_path(self.parts[0].path),'r') as f:
            if self.scan_path is None:
                candidates=[]
                def visit(k,obj):
                    if isinstance(obj,h5py.Group) and k.endswith('/EBSD/Data') and 'Euler' in obj and 'Phase' in obj:
                        candidates.append('/'+k.rsplit('/Data',1)[0])
                f.visititems(visit)
                if len(candidates)!=1:raise ValueError(f'Expected one EBSD scan; found: {candidates}')
                self.scan_path=candidates[0]
            head=f[self.scan_path+'/Header']
            self.nx,self.ny,_ = _part_grid(f,self.scan_path,self.parts[0])
            if self.spec.manifest_grid is not None and self.spec.manifest_grid != (self.nx,self.ny):
                raise ValueError('Manifest scan grid contradicts validated global grid')
            self.sx=float(_scalar(head,'X Step'))
            self.sy=float(_scalar(head,'Y Step'))
            self.phase_specs={}
            for key,group in head['Phases'].items():
                i=int(key)  # raster Phase values index phase groups; Phase Id may be database ID 6/5
                self.phase_specs[i]={
                    'database_phase_id':int(_scalar(group,'Phase Id',int(key))),
                    'name':str(_scalar(group,'Phase Name','unknown')),
                    'space_group':int(_scalar(group,'Space Group',0)),
                    'rgb': np.asarray(group['Color'][()]).reshape(-1).astype(int).tolist() if 'Color' in group else [120,120,120]
                }
                lattice=_lattice_parameters(group)
                if lattice is not None:self.phase_specs[i]['lattice_parameters']=lattice
            self.header={ 'scan_path':self.scan_path,'nx':self.nx,'ny':self.ny,'x_step_um':self.sx,'y_step_um':self.sy,
                          'phase_specs':self.phase_specs,
                          'specimen_orientation_euler':np.asarray(head['Specimen Orientation Euler'][()]).tolist() if 'Specimen Orientation Euler' in head else None,
                          'scanning_rotation_rad':float(_scalar(head,'Scanning Rotation Angle',0.0)),
                          'source':str(self.path) }
            self.header.update(source_name=self.spec.source_name,name_source=self.spec.name_source,scan_tag=self.spec.scan_tag)
            data=f[self.scan_path+'/Data']; self.available_fields={x: y in data for x,y in FIELDS.items()}
        if len(self.parts)==1 and self.parts[0].stop==-1:
            self.parts=[Part(self.parts[0].path,0,self.ny)]
        expected=0
        grid_evidence=[]
        for p in self.parts:
            if p.start!=expected:raise ValueError(f'Missing or overlapping rows at {expected} to {p.start}')
            expected=p.stop
            with h5py.File(long_path(p.path),'r') as f:
                head=f[self.scan_path+'/Header']; data=f[self.scan_path+'/Data']
                gx,gy,evidence=_part_grid(f,self.scan_path,p)
                if (gx,gy)!=(self.nx,self.ny):raise ValueError('Montage grid dimensions disagree')
                grid_evidence.append(evidence)
                if (float(_scalar(head,'X Step')),float(_scalar(head,'Y Step')))!=(self.sx,self.sy):raise ValueError('Montage step sizes disagree')
                for key,spec in self.phase_specs.items():
                    group=head['Phases'].get(str(key))
                    if group is None or int(_scalar(group,'Space Group',0))!=spec['space_group'] or str(_scalar(group,'Phase Name','unknown'))!=spec['name']:
                        raise ValueError('Montage phase definitions disagree')
                    shard_lattice=_lattice_parameters(group)
                    if shard_lattice!=spec.get('lattice_parameters'):
                        raise ValueError('Montage phase lattice definitions disagree')
                for field,key in FIELDS.items():
                    if (key in data)!=self.available_fields[field]:raise ValueError(f'Shard fields disagree: {key}')
                    if key in data and data[key].shape[0]!=(p.stop-p.start)*self.nx:raise ValueError(f'Shard row count disagrees with manifest: {p.path.name}')
        if expected!=self.ny:raise ValueError(f'Scan has {self.ny} rows; provided partitions cover {expected}')
        self.header['part_grid_layouts']=grid_evidence
        if not (self.sx>0 and self.sy>0):raise ValueError('Invalid scan step')
        unit,evidence=source_mad_units(self.parts,self.scan_path)
        self.header.update(mad_unit=unit,mad_unit_evidence=evidence)
        self.check_unchanged()
    def check_unchanged(self):
        for p in self.parts:
            st=long_path(p.path).stat()
            if (st.st_size,st.st_mtime_ns)!=self.source_stats[str(p.path)]:raise ValueError('Input changed since scan validation')
    def read(self, first:int, stop:int, fields=('euler','phase')) -> dict:
        if first<0 or stop>self.ny or first>=stop:raise ValueError('Bad scan row interval')
        out={}
        for name in fields:
            if not self.available_fields.get(name,False):continue
            is_euler=name=='euler'
            out[name]=np.empty((stop-first,self.nx,3) if is_euler else (stop-first,self.nx),dtype=np.float32 if is_euler or name in ('mad','pq','x','y') else np.uint8)
        for p in self.parts:
            a,b=max(first,p.start),min(stop,p.stop)
            if a>=b:continue
            st=long_path(p.path).stat()
            if (st.st_size,st.st_mtime_ns)!=self.source_stats[str(p.path)]:raise ValueError('Input changed during analysis')
            rel_a=(a-p.start)*self.nx; rel_b=(b-p.start)*self.nx
            with h5py.File(long_path(p.path),'r') as f:
                d=f[self.scan_path+'/Data']
                for name,arr in out.items():
                    sample=d[FIELDS[name]][rel_a:rel_b]
                    arr[a-first:b-first]=sample.reshape(arr[a-first:b-first].shape)
        return out
    def probe_euler_unit(self):
        """Resolve Euler units from metadata, never from angle magnitudes.

        Explicit dataset Unit/Units attributes take precedence. An absent
        attribute follows the Oxford AZtec H5OINA Euler-radian format contract
        (also preserved by row-split exports), with that evidence recorded.
        Unknown attributes, conflicting declarations, or mixed shard units fail
        rather than mixing orientations in one numeric cache.
        """
        self.check_unchanged()
        aliases = {"rad": "radian", "radian": "radian", "radians": "radian",
                   "deg": "degree", "degree": "degree", "degrees": "degree", "°": "degree"}
        sources = []
        units = set()
        for part in self.parts:
            declarations = []
            with h5py.File(long_path(part.path), 'r') as stream:
                dataset = stream[self.scan_path + '/Data/Euler']
                for name, raw in dataset.attrs.items():
                    if str(name).casefold() not in ('unit', 'units'):
                        continue
                    values = np.asarray(raw).reshape(-1)
                    if not values.size:
                        raise ValueError(f'Empty Euler unit attribute in {part.path.name}')
                    for value in values:
                        text = value.decode('utf-8', 'strict') if isinstance(value, (bytes, np.bytes_)) else str(value)
                        normalized = aliases.get(text.strip().casefold())
                        if normalized is None:
                            raise ValueError(f'Unsupported Euler unit {text!r} in {part.path.name}; expected degree or radian')
                        declarations.append((str(name), text, normalized))
            declared_units = {value[2] for value in declarations}
            if len(declared_units) > 1:
                raise ValueError(f'Conflicting Euler Unit/Units attributes in {part.path.name}')
            unit = next(iter(declared_units)) if declared_units else 'radian'
            units.add(unit)
            sources.append({'file': str(part.path), 'dataset': self.scan_path + '/Data/Euler',
                            'unit': unit, 'attributes': [dict(name=n, value=v) for n, v, _ in declarations],
                            'basis': 'explicit dataset attribute' if declarations else
                                     'Oxford AZtec H5OINA Euler-radian format contract; no unit attribute'})
        if len(units) != 1:
            raise ValueError('Montage Euler units disagree across shards; mixed degree/radian data are not accepted')
        unit = next(iter(units))
        self.header['euler_unit_evidence'] = {'unit': unit, 'sources': sources,
            'policy': 'dataset attributes or H5OINA format contract; never numeric magnitude inference'}
        return unit
    def iter_rows(self, batch=64,fields=('euler','phase')):
        for start in range(0,self.ny,batch):
            yield start,min(start+batch,self.ny),self.read(start,min(start+batch,self.ny),fields)
