"""MAD units are metadata/format contracts, never inferred from magnitudes."""
import h5py
from .storage import long_path
import numpy as np

SPEC_URL='https://github.com/oinanoanalysis/h5oina/blob/master/H5OINAFile.md#ebsd-data'

def normalise_mad_unit(value):
    array=np.asarray(value)
    if array.size!=1:
        raise ValueError('MAD unit must be one unambiguous scalar')
    value=array.reshape(-1)[0]
    if isinstance(value,bytes):value=value.decode('utf-8')
    unit=str(value).strip().lower()
    if unit in ('rad','radian','radians'):return 'rad'
    if unit in ('deg','degree','degrees','°'):return 'deg'
    raise ValueError(f'Unsupported MAD unit {value!r}; supply verified rad or deg metadata')

def source_mad_units(parts,scan_path):
    evidence=[]
    present=[]
    for part in parts:
        with h5py.File(long_path(part.path),'r') as f:
            key=scan_path+'/Data/Mean Angular Deviation'
            present.append(key in f)
            if key not in f:continue
            d=f[key]
            if 'Unit' in d.attrs:
                unit=normalise_mad_unit(d.attrs['Unit'])
                basis='dataset Unit attribute'
            else:
                # This reader accepts the Oxford EBSD schema. The format explicitly
                # specifies MAD in radians, including when cropped files lose Unit.
                unit='rad'
                basis='H5OINA specification (attribute absent)'
            evidence.append({'file':str(part.path),'dataset':key,'unit':unit,
                             'basis':basis,'specification':SPEC_URL})
    if any(present) and not all(present):
        raise ValueError('MAD dataset is absent from some input shards')
    units={e['unit'] for e in evidence}
    if len(units)>1:
        raise ValueError('MAD units differ between shards; normalize explicitly before combining')
    return next(iter(units),None),evidence

def mad_degrees(values,unit):
    unit=normalise_mad_unit(unit)
    return np.rad2deg(values) if unit=='rad' else values
