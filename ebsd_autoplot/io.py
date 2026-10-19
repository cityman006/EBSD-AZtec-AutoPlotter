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

@dataclass(frozen=True)
class Part:
    path: Path
    start: int
    stop: int

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
            self.nx=int(_scalar(head,'X Cells'))
            self.ny=int(_scalar(head,'Y Cells'))
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
        for p in self.parts:
            if p.start!=expected:raise ValueError(f'Missing or overlapping rows at {expected} to {p.start}')
            expected=p.stop
            with h5py.File(long_path(p.path),'r') as f:
                head=f[self.scan_path+'/Header']; data=f[self.scan_path+'/Data']
                if (int(_scalar(head,'X Cells')),int(_scalar(head,'Y Cells')))!=(self.nx,self.ny):raise ValueError('Montage grid dimensions disagree')
                if (float(_scalar(head,'X Step')),float(_scalar(head,'Y Step')))!=(self.sx,self.sy):raise ValueError('Montage step sizes disagree')
                for key,spec in self.phase_specs.items():
                    group=head['Phases'].get(str(key))
                    if group is None or int(_scalar(group,'Space Group',0))!=spec['space_group'] or str(_scalar(group,'Phase Name','unknown'))!=spec['name']:
                        raise ValueError('Montage phase definitions disagree')
                for field,key in FIELDS.items():
                    if (key in data)!=self.available_fields[field]:raise ValueError(f'Shard fields disagree: {key}')
                    if key in data and data[key].shape[0]!=(p.stop-p.start)*self.nx:raise ValueError(f'Shard row count disagrees with manifest: {p.path.name}')
        if expected!=self.ny:raise ValueError(f'Scan has {self.ny} rows; provided partitions cover {expected}')
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
        # Oxford H5OINA stores Euler triplets in radians; reject implausible values
        e=self.read(0,min(8,self.ny),('euler',))['euler']
        q=np.nanpercentile(e.reshape(-1,3),99.9,axis=0)
        if np.any(q > 2*np.pi+0.2):
            return 'degree'
        return 'radian'
    def iter_rows(self, batch=64,fields=('euler','phase')):
        for start in range(0,self.ny,batch):
            yield start,min(start+batch,self.ny),self.read(start,min(start+batch,self.ny),fields)
