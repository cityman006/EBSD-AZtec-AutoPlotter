"""Native and authoritative row-shard EDS reads; no image-tile guessing.

Scientific buffers use at most 64 MiB of owned heap per result, then temporary
file-backed memmaps. Call EDSResult.close() after consuming/saving a result.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import tempfile
import h5py
import numpy as np
from .eds import EDSResult, _header_geometry, _flat_rows, _unit, _unit_key, _length_factor_um
from .storage import check_cancelled, long_path
from .registration import validate_affine, resample_map_blocked

_ARRAY_MEMORY_LIMIT = 64 * 1024**2


def allocate_eds_array(result, shape, dtype):
    shape = tuple(int(v) for v in shape);dtype = np.dtype(dtype)
    if not shape or min(shape) < 1 or dtype.hasobject:raise ValueError("Invalid EDS buffer shape/dtype")
    size = math.prod(shape) * dtype.itemsize
    if result._memory_bytes + size <= _ARRAY_MEMORY_LIMIT:
        array = np.empty(shape, dtype=dtype);result._memory_bytes += size
    else:
        handle = tempfile.TemporaryFile(prefix="ebsd_eds_")
        try:array = np.memmap(handle, dtype=dtype, mode="w+", shape=shape)
        except BaseException:handle.close();raise
        array._eds_temporary_file = handle  # Lifetime follows array even if its result is discarded.
        result._temporary_files.append(handle)
    result._owned_arrays.append(array)
    return array


def _rows(value):
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError("EDS block_rows must be a positive integer")
    return int(value)


def _stamp(path):
    path=Path(path).resolve();stat=long_path(path).stat()
    return {"path":str(path),"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}


def _unchanged(stamp):
    if _stamp(stamp['path']) != stamp:raise ValueError("EDS source changed during reading")
    check_cancelled()


def _eds_path(scan_path):
    scan_path='/'+str(scan_path).strip('/')
    if scan_path.rsplit('/',1)[-1] != 'EBSD':raise ValueError("scan_path must identify an EBSD group")
    return scan_path.rsplit('/',1)[0]+'/EDS'


def _attrs(dataset):
    result={}
    for key in ('Atomic Number','X-ray Line'):
        if key in dataset.attrs:
            value=dataset.attrs[key]
            result[key]=value.decode('utf-8') if isinstance(value,bytes) else value.tolist() if isinstance(value,np.ndarray) else value.item() if isinstance(value,np.generic) else value
    return result


def _root_ranges(stream):
    keys=('global_nx','global_ny','start_row_in_original','end_row_in_original_exclusive')
    if not any(key in stream.attrs for key in keys):return None
    if not all(key in stream.attrs for key in keys):raise ValueError("Incomplete explicit original-grid shard provenance")
    vals=[]
    for key in keys:
        raw=np.asarray(stream.attrs[key]).reshape(-1)
        if raw.size!=1 or raw.dtype.kind not in 'iuf' or not np.isfinite(raw[0]) or float(raw[0])!=int(raw[0]):raise ValueError("Invalid original-grid shard provenance")
        vals.append(int(raw[0]))
    nx,ny,start,stop=vals
    if min(nx,ny)<1 or start<0 or stop<=start or stop>ny:raise ValueError("Invalid original-grid shard row range")
    return {'nx':nx,'ny':ny,'start':start,'stop':stop}


def _layout(ds, widths, heights):
    matches=[]
    for h in set(heights):
        w=widths
        if ds.shape in ((h*w,),(h*w,1),(h,w)):matches.append((h,w))
    if len(matches)!=1:raise ValueError(f"Ambiguous or inconsistent EDS raster shape for {ds.name}: {ds.shape}")
    return matches[0]


def _describe(stream, scan_path, part=None, global_shape=None):
    base=_eds_path(scan_path);windows_path=base+'/Data/Window Integral'
    if windows_path not in stream:return None
    windows=stream[windows_path]
    if not isinstance(windows,h5py.Group) or any(not isinstance(v,h5py.Dataset) for v in windows.values()):
        raise ValueError("EDS Window Integral must contain unambiguous named numeric datasets")
    names=sorted(windows)
    if not names:return None
    header_shape,steps,units=_header_geometry(stream[base+'/Header'])
    provenance=_root_ranges(stream)
    if part is not None:
        gh,gw=global_shape;count=part.stop-part.start
        if header_shape[1]!=gw or header_shape[0] not in (gh,count):raise ValueError("EDS shard header differs from authoritative source grid")
        if provenance and (provenance['nx'],provenance['ny'],provenance['start'],provenance['stop'])!=(gw,gh,part.start,part.stop):
            raise ValueError("EDS shard original-grid provenance contradicts manifest row ranges")
        possible=(count,gh)
    else:
        possible=(header_shape[0],)
        if provenance:
            if header_shape[1]!=provenance['nx'] or header_shape[0] not in (provenance['ny'],provenance['stop']-provenance['start']):
                raise ValueError("EDS header contradicts original-grid provenance")
            possible=(*possible,provenance['stop']-provenance['start'])
    native_shape=_layout(windows[names[0]],header_shape[1],possible)
    channels={}
    for name in names:
        ds=windows[name]
        if not np.issubdtype(ds.dtype,np.number):raise ValueError("EDS window is not numeric: "+name)
        if _layout(ds,header_shape[1],possible)!=native_shape:raise ValueError("EDS windows use inconsistent raster layouts")
        channels[name]={'path':ds.name,'dtype':ds.dtype.str,'unit':_unit(ds),'attributes':_attrs(ds)}
    data=stream[base+'/Data'];timing={}
    for name in ('Live Time','Real Time'):
        if name in data:
            ds=data[name]
            if not np.issubdtype(ds.dtype,np.number) or _layout(ds,native_shape[1],(native_shape[0],))!=native_shape:
                raise ValueError("EDS timing layout differs from native window grid")
            timing[name]={'path':ds.name,'dtype':ds.dtype.str,'unit':_unit(ds)}
    for axis in ('X','Y'):
        if axis in data:_layout(data[axis],native_shape[1],(native_shape[0],))
    try:step_um=[steps[i]*_length_factor_um(units[name]) for i,name in enumerate(('X Step','Y Step'))]
    except ValueError:step_um=None
    return {'base':base,'shape':native_shape,'header_shape':header_shape,'steps':steps,'header_units':units,
            'step_um_xy':step_um,'channels':channels,'timing':timing,'original_grid':provenance}


def _schema(desc):
    return json.dumps({'channels':desc['channels'],'timing':desc['timing']},sort_keys=True)


def _allocate_maps(result, desc, shape):
    for name,item in desc['channels'].items():
        result.maps[name]=result.allocate(shape,item['dtype']);result.masks[name]=result.allocate(shape,bool)
        result.units[name]=item['unit']
    for name,item in desc['timing'].items():
        result.timing_maps[name]=result.allocate(shape,item['dtype']);result.timing_masks[name]=result.allocate(shape,bool)
        result.timing_units[name]=item['unit']


def _read_into(result, stream, desc, input_first, input_stop, output_first, block_rows, guard):
    for first in range(input_first,input_stop,block_rows):
        guard();stop=min(input_stop,first+block_rows);dest=slice(output_first+first-input_first,output_first+stop-input_first)
        for fields,masks,key in ((result.maps,result.masks,'channels'),(result.timing_maps,result.timing_masks,'timing')):
            for name,item in desc[key].items():
                raw=_flat_rows(stream[item['path']],first,stop,desc['shape'])
                fields[name][dest]=raw;masks[name][dest]=np.isfinite(raw)&((raw>=0) if key=='timing' else True)
        guard()


def _rates(result, block_rows, guard):
    result.metadata['rate_channels']={}
    for name in list(result.maps):
        unit=result.units[name];key=_unit_key(unit)
        if key in ('cps','counts/s','count/s','cts/s','countss-1'):
            result.metadata['rate_channels'][name]={'method':'source count rate; never divided again','unit':unit};continue
        if key not in ('counts','count','cts','ct'):raise ValueError(f"Cannot infer count-rate conversion from unit {unit!r}: {name}")
        if 'Live Time' not in result.timing_maps:raise ValueError("Count-rate conversion requires measured Live Time")
        factor={'s':1.,'sec':1.,'second':1.,'seconds':1.,'ms':.001,'us':1e-6}.get(_unit_key(result.timing_units['Live Time']))
        if factor is None:raise ValueError("Live Time needs an explicit second-compatible unit")
        rate_name=name+'_cps'
        if rate_name in result.maps:raise ValueError("Count-rate channel name conflicts with native window")
        shape=result.maps[name].shape;rate=result.allocate(shape,np.float64);mask=result.allocate(shape,bool)
        for first in range(0,shape[0],block_rows):
            guard();sl=slice(first,min(shape[0],first+block_rows))
            live=np.asarray(result.timing_maps['Live Time'][sl],float)*factor
            valid=np.asarray(result.masks[name][sl],bool)&np.isfinite(live)&(live>0)
            rate[sl]=np.nan;np.divide(result.maps[name][sl],live,out=rate[sl],where=valid);mask[sl]=valid
        result.maps[rate_name]=rate;result.masks[rate_name]=mask;result.units[rate_name]='counts/s'
        result.metadata['rate_channels'][rate_name]={'method':'native counts / measured live seconds','source_window':name}


def _metadata(path,scan_path,desc):
    return {'status':'available','source':str(path),'scan_path':scan_path,'eds_path':desc['base'],
        'native_shape_hw':list(desc['shape']),'step_um_xy':desc['step_um_xy'],
        'window_channels':desc['channels'],'timing_channels':desc['timing'],
        'quantification':'native X-ray window intensity; not wt% or at%',
        'buffer_policy':'at most 64 MiB owned heap; then temporary file-backed memmaps; call close()',
        'registration':{'method':'native EDS grid only','aligned_to_ebsd':False,'resampled':False}}


def load_native_h5oina_eds(path,scan_path='/1/EBSD',*,count_rate=False,block_rows=32):
    block_rows=_rows(block_rows);stamp=_stamp(path);result=EDSResult()
    try:
        with h5py.File(long_path(path),'r') as stream:
            _unchanged(stamp);desc=_describe(stream,scan_path)
            if desc is None:return EDSResult(metadata={'status':'unavailable','reason':'No native EDS window maps','source_stamp':stamp})
            result.available=True;result.metadata=_metadata(stamp['path'],scan_path,desc);result.metadata['source_stamp']=stamp
            _allocate_maps(result,desc,desc['shape'])
            _read_into(result,stream,desc,0,desc['shape'][0],0,block_rows,lambda:_unchanged(stamp))
        if count_rate:_rates(result,block_rows,lambda:_unchanged(stamp))
        _unchanged(stamp);return result
    except BaseException:result.close();raise


def read_eds_preview(path,scan_path='/1/EBSD',channel=None,max_side=1024):
    max_side=_rows(max_side);stamp=_stamp(path)
    with h5py.File(long_path(path),'r') as stream:
        _unchanged(stamp);desc=_describe(stream,scan_path)
        if desc is None:raise ValueError('Native EDS modality is unavailable')
        names=list(desc['channels']);channel=names[0] if channel is None else str(channel)
        if channel not in desc['channels']:raise KeyError('Unknown native EDS channel: '+channel)
        h,w=desc['shape'];stride=max(1,math.ceil(max(h,w)/max_side))
        sampled=np.empty((math.ceil(h/stride),math.ceil(w/stride)),float)
        ds=stream[desc['channels'][channel]['path']]
        for output,row in enumerate(range(0,h,stride)):
            _unchanged(stamp)
            if ds.shape==(h,w):sampled[output]=ds[row,::stride]
            elif ds.shape==(h*w,1):sampled[output]=ds[row*w:(row+1)*w:stride,0]
            else:sampled[output]=ds[row*w:(row+1)*w:stride]
        _unchanged(stamp)
    return {'image':sampled,'shape_hw':[h,w],'channel':channel,'channels':names,
        'unit':desc['channels'][channel]['unit'],'source_stamp':stamp,'scan_path':scan_path,
        'step_um_xy':desc['step_um_xy'],'registered':False,'sampling_stride':stride,
        'native_dtype':desc['channels'][channel]['dtype'],'preview_only':True}


def _tolerance(dataset,values,step):
    eps=np.finfo(dataset.dtype).eps if np.issubdtype(dataset.dtype,np.floating) else 0.
    magnitude=max(float(np.max(np.abs(values))) if values.size else 0.,step)
    return min(step*.001,max(step*1e-5+1e-10,2*eps*magnitude))


def _coordinates(stream,base,shape,offset,origin,steps,block_rows,guard):
    """Verify one measured regular raster against global or manifest-local rows."""
    data=stream[base+'/Data'];h,w=shape
    global_y=local_y=True;maximum={'X':0.,'Y':0.};units={}
    for axis in ('X','Y'):
        if axis not in data:raise ValueError('Missing measured '+base+' '+axis+' coordinates')
        ds=data[axis];units[axis]=_unit(ds);factor=_length_factor_um(units[axis])
        for first in range(0,h,block_rows):
            guard();stop=min(h,first+block_rows);values=_flat_rows(ds,first,stop,shape).astype(float)*factor
            if not np.isfinite(values).all():raise ValueError('Nonfinite '+base+' '+axis+' coordinates')
            expected=(origin[0]+np.arange(w)*steps[0])[None,:] if axis=='X' else origin[1]+(offset+np.arange(first,stop))[:,None]*steps[1]
            tolerance=_tolerance(ds,values,steps[0 if axis=='X' else 1])
            maximum[axis]=max(maximum[axis],float(np.max(np.abs(values-expected))))
            if axis=='X':
                if not np.allclose(values,expected,rtol=0,atol=tolerance):raise ValueError(base+' X coordinates do not follow the verified raster')
            else:
                local_expected=origin[1]+np.arange(first,stop)[:,None]*steps[1]
                global_y &= bool(np.allclose(values,expected,rtol=0,atol=tolerance))
                local_y &= bool(np.allclose(values,local_expected,rtol=0,atol=tolerance))
    if not global_y and not local_y:raise ValueError(base+' Y coordinates do not follow global or authoritative local row ranges')
    frame='global' if global_y else 'local reset to source origin'
    return {'frame':frame,'local_to_global_y_um':0. if global_y else offset*steps[1],
            'compared_pixels':h*w,'max_difference_from_global_um':maximum,
            'coordinate_units':units,
            'tolerance':'source dtype precision, capped at 0.001 measured pixel step'}


def _pointwise_coordinates(source, stream, desc, offset, frame, source_parts, steps, block_rows, guard):
    """Compare actual physical EDS/EBSD positions, not independent ideal grids.

    Full duplicated EDS rasters are checked across every source row, including
    rows outside that file's authoritative scientific-data contribution.
    """
    height,width=desc['shape'];start=offset;end=offset+height
    report={'aligned':True,'compared_pixels':0,'max_error_um':{'X':0.,'Y':0.},
            'max_tolerance_um':{'X':0.,'Y':0.},'basis':'every measured coordinate after explicit manifest-local Y rebasing'}
    edata=stream[desc['base']+'/Data']
    for part,info in zip(source.parts,source_parts):
        first=max(start,part.start);stop=min(end,part.stop)
        if first>=stop:continue
        with h5py.File(long_path(part.path),'r') as target:
            bdata=target[source.scan_path+'/Data'];bshape=(part.stop-part.start,width)
            for row in range(first,stop,block_rows):
                guard();last=min(stop,row+block_rows)
                for axis in ('X','Y'):
                    eds,bds=edata[axis],bdata[axis]
                    ev=_flat_rows(eds,row-offset,last-offset,desc['shape']).astype(float)*_length_factor_um(_unit(eds))
                    bv=_flat_rows(bds,row-part.start,last-part.start,bshape).astype(float)*_length_factor_um(_unit(bds))
                    if axis=='Y':
                        ev+=frame['local_to_global_y_um'];bv+=info['ebsd_coordinates']['local_to_global_y_um']
                    if not np.isfinite(ev).all() or not np.isfinite(bv).all():raise ValueError('Nonfinite measured EDS/EBSD coordinates')
                    step=steps[0 if axis=='X' else 1]
                    tolerance=max(_tolerance(eds,ev,step),_tolerance(bds,bv,step))
                    error=float(np.max(np.abs(ev-bv)))
                    report['max_error_um'][axis]=max(report['max_error_um'][axis],error)
                    report['max_tolerance_um'][axis]=max(report['max_tolerance_um'][axis],float(tolerance))
                    if not np.allclose(ev,bv,rtol=0,atol=tolerance):report['aligned']=False
                report['compared_pixels']+=(last-row)*width
    if report['compared_pixels']!=height*width:raise ValueError('Coordinate comparison does not cover native EDS rows')
    return report


def load_source_eds(source,*,count_rate=False,block_rows=32,registration=None):
    from .io import H5Source
    if not isinstance(source,H5Source):raise TypeError('source must be a validated H5Source')
    block_rows=_rows(block_rows);source.check_unchanged();check_cancelled()
    shape=(int(source.ny),int(source.nx));expected=0
    for part in source.parts:
        if part.start!=expected or part.stop<=part.start or part.stop>shape[0]:raise ValueError('EDS source row ranges contain an overlap or gap')
        expected=part.stop
    if expected!=shape[0]:raise ValueError('EDS source row ranges do not cover the global scan')
    def guard():source.check_unchanged();check_cancelled()
    descriptors=[];parts=[];missing=[]
    for part in source.parts:
        guard()
        with h5py.File(long_path(part.path),'r') as stream:desc=_describe(stream,source.scan_path,part,shape)
        info={'path':str(part.path),'row_range':[part.start,part.stop],'available':desc is not None,'source_stamp':_stamp(part.path)}
        descriptors.append(desc);parts.append(info)
        if desc is None:missing.append([part.start,part.stop])
    present=[desc for desc in descriptors if desc is not None]
    if present and any(_schema(desc)!=_schema(present[0]) for desc in present[1:]):raise ValueError('EDS channel/timing schema, dtype or units disagree across shards')
    if missing:
        guard()
        return EDSResult(metadata={'status':'unavailable','source':str(source.path),'scan_path':source.scan_path,
            'reason':'EDS missing in one or more authoritative source parts; no missing values were fabricated',
            'source_parts':parts,'missing_row_ranges':missing,'available_row_ranges':[p['row_range'] for p in parts if p['available']]})
    result=EDSResult(available=True)
    try:
        first=source.parts[0];desc0=descriptors[0]
        if desc0['step_um_xy'] is None:raise ValueError('Measured EDS spatial step units are unknown')
        eds_steps=desc0['step_um_xy']
        with h5py.File(long_path(first.path),'r') as stream:
            _,raw_steps,units=_header_geometry(stream[source.scan_path+'/Header'])
            steps=[raw_steps[i]*_length_factor_um(units[name]) for i,name in enumerate(('X Step','Y Step'))]
            ebdata=stream[source.scan_path+'/Data'];eddata=stream[desc0['base']+'/Data']
            origin=[];eds_origin=[]
            for axis in ('X','Y'):
                if axis not in ebdata or axis not in eddata:raise ValueError('Measured EBSD/EDS X/Y coordinates are required')
                origin.append(float(np.asarray(ebdata[axis][0]).reshape(-1)[0])*_length_factor_um(_unit(ebdata[axis])))
                eds_origin.append(float(np.asarray(eddata[axis][0]).reshape(-1)[0])*_length_factor_um(_unit(eddata[axis])))
        if not np.allclose(steps,eds_steps,rtol=1e-5,atol=1e-9) and registration is None:
            raise ValueError('EDS/EBSD measured step differs; explicit registration is required')
        result.metadata=_metadata(source.path,source.scan_path,desc0)
        result.metadata.update(native_shape_hw=list(shape),source_parts=parts,
            source_grid_origin_um_xy=origin,eds_grid_origin_um_xy=eds_origin,
            montage_policy='authoritative source.parts row ranges; no arbitrary 2D tile alignment')
        # Validate and freeze every target frame first, so duplicated full EDS
        # coordinates can also be compared to target rows in other source files.
        for part,info in zip(source.parts,parts):
            guard();count=part.stop-part.start
            with h5py.File(long_path(part.path),'r') as stream:
                eh_shape,eh_steps,eh_units=_header_geometry(stream[source.scan_path+'/Header'])
                current=[eh_steps[i]*_length_factor_um(eh_units[name]) for i,name in enumerate(('X Step','Y Step'))]
                if eh_shape[1]!=shape[1] or eh_shape[0] not in (shape[0],count) or not np.allclose(current,steps,rtol=1e-5,atol=1e-9):
                    raise ValueError('EBSD shard physical header disagrees with source grid')
                info['ebsd_coordinates']=_coordinates(stream,source.scan_path,(count,shape[1]),part.start,origin,steps,block_rows,guard)
                info['ebsd_header_units']=eh_units
        _allocate_maps(result,desc0,shape)
        for part,desc,info in zip(source.parts,descriptors,parts):
            guard()
            print(f'EDS: verifying and reading rows {part.start}:{part.stop} from {part.path.name}', flush=True)
            if desc['step_um_xy'] is None or not np.allclose(desc['step_um_xy'],eds_steps,rtol=1e-5,atol=1e-9):raise ValueError('EDS physical steps disagree across source parts')
            count=part.stop-part.start
            with h5py.File(long_path(part.path),'r') as stream:
                offset=0 if desc['shape'][0]==shape[0] else part.start
                info['eds_coordinates']=_coordinates(stream,desc['base'],desc['shape'],offset,eds_origin,eds_steps,block_rows,guard)
                compared=_pointwise_coordinates(source,stream,desc,offset,info['eds_coordinates'],parts,steps,block_rows,guard)
                info['pointwise_eds_ebsd_coordinates']=compared
                if not compared['aligned'] and registration is None:
                    raise ValueError('EDS/EBSD measured coordinates differ beyond capped precision tolerance; explicit registration is required')
                info.update(native_eds_shape_hw=list(desc['shape']),eds_header_shape_hw=list(desc['header_shape']),
                    eds_header_units=desc['header_units'],
                    input_layout='full global raster cropped by authoritative rows' if desc['shape'][0]==shape[0] else 'manifest row shard')
                a=part.start if desc['shape'][0]==shape[0] else 0
                _read_into(result,stream,desc,a,a+count,part.start,block_rows,guard)
        if count_rate:_rates(result,block_rows,guard)
        if registration is None:
            result.metadata['registration']={'method':'manifest rows + every measured physical EBSD/EDS coordinate',
                'aligned_to_ebsd':True,'resampled':False,'compared_scan_pixels':shape[0]*shape[1]}
        else:
            affine=validate_affine(registration)
            for name in list(result.maps):
                guard();out=result.allocate(shape,float);valid=result.allocate(shape,bool)
                result.maps[name],result.masks[name]=resample_map_blocked(result.maps[name],shape,affine,result.masks[name],block_rows=block_rows,out=out,valid_out=valid)
            result.metadata['registration']={'method':'explicit pixel-centre source-to-EBSD affine','aligned_to_ebsd':True,'independently_validated':False,
                'source_to_ebsd_pixels':affine.tolist(),'resampled':True,'outside':'NaN + false mask'}
        result.metadata['timing_grid']='native EDS mosaic; timing is not resampled'
        guard();return result
    except BaseException:result.close();raise
