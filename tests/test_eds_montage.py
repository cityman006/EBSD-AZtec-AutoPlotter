"""Source-aware EDS row shards use manifest ranges and physical coordinates."""
import json
from dataclasses import replace
from pathlib import Path
import h5py
import numpy as np
import pytest
from ebsd_autoplot.io import H5Source
from ebsd_autoplot.storage import cancellation, Cancelled, digest
from test_pipeline import write_synth


def montage(tmp_path, *, local_coords=False, full_eds=False, unit='cps', big_integer=False):
    base=tmp_path/'base';write_synth(base,h=7,w=5)
    folder=tmp_path/'montage';folder.mkdir();ranges=[(0,2),(2,5),(5,7)]
    truth=np.arange(35,dtype=np.uint64).reshape(7,5)+np.uint64(2**53 if big_integer else 10)
    entries=[]
    for first,stop in ranges:
        path=folder/f'scan01_rows{first:06d}-{stop-1:06d}_EBSD.h5'
        with h5py.File(base/'sample.h5oina','r') as src,h5py.File(path,'w') as dst:
            src.copy('1',dst)
            data=dst['1/EBSD/Data']
            for key in list(data):
                arr=data[key][first*5:stop*5];del data[key];data[key]=arr
            head=dst['1/EBSD/Header']
            for key,value,unit_value in [('X Cells',5,'px'),('Y Cells',7,'px'),('X Step',.2,'um'),('Y Step',.3,'um')]:
                head[key][0]=value;head[key].attrs['Unit']=unit_value
            yy,xx=np.indices((stop-first,5))
            for key,arr in [('X',xx*.2),('Y',(yy+(0 if local_coords else first))*.3)]:
                ds=data.create_dataset(key,data=arr.ravel());ds.attrs['Unit']='um'
            eh=dst.create_group('1/EDS/Header')
            for key,value,unit_value in [('X Cells',5,'px'),('Y Cells',7,'px'),('X Step',.2,'um'),('Y Step',.3,'um')]:
                ds=eh.create_dataset(key,data=[value]);ds.attrs['Unit']=unit_value
            ed=dst.create_group('1/EDS/Data');shape=(7,5) if full_eds else (stop-first,5)
            ey,ex=np.indices(shape)
            for key,arr in [('X',ex*.2),('Y',(ey+(0 if full_eds or local_coords else first))*.3)]:
                ds=ed.create_dataset(key,data=arr.ravel());ds.attrs['Unit']='um'
            values=truth if full_eds else truth[first:stop]
            window=ed.create_group('Window Integral')
            for label,arr in [('Ti Kα1',values),('Al Kα1',values+100)]:
                ds=window.create_dataset(label,data=arr.ravel());ds.attrs['Unit']=unit
            for label in ('Live Time','Real Time'):
                ds=ed.create_dataset(label,data=np.full(np.prod(shape),2.,np.float32));ds.attrs['Unit']='s'
        entries.append({'file':path.name,'row_start':first,'row_end_exclusive':stop})
    (folder/'manifest.json').write_text(json.dumps({'source_name':'original.h5oina','scans':[{'scan_path':'/1/EBSD','parts':entries}]}),encoding='utf-8')
    return H5Source(folder),truth


def test_three_uneven_manifest_parts_preserve_native_integer_fields(tmp_path):
    from ebsd_autoplot.eds import load_source_eds
    source,truth=montage(tmp_path,big_integer=True)
    before={str(part.path):digest(part.path) for part in source.parts}
    result=load_source_eds(source,block_rows=1)
    try:
        assert result.available and result.maps['Ti Kα1'].dtype==np.dtype('uint64')
        assert np.array_equal(result.maps['Ti Kα1'],truth)
        assert result.masks['Ti Kα1'].all() and result.units['Ti Kα1']=='cps'
        assert result.metadata['registration']['aligned_to_ebsd']
        assert result.metadata['source_parts'][1]['row_range']==[2,5]
        assert np.all(result.timing_maps['Live Time']==2.)
    finally:result.close()
    assert before=={str(part.path):digest(part.path) for part in source.parts}


@pytest.mark.parametrize('local,full',[(True,False),(False,True)])
def test_local_coordinate_resets_and_full_global_eds_use_only_authoritative_rows(tmp_path,local,full):
    from ebsd_autoplot.eds import load_source_eds
    source,truth=montage(tmp_path,local_coords=local,full_eds=full)
    with load_source_eds(source,block_rows=2) as result:
        assert np.array_equal(result.maps['Ti Kα1'],truth)
        second=result.metadata['source_parts'][1]
        if local:
            assert second['eds_coordinates']['frame']=='local reset to source origin'
            assert second['eds_coordinates']['local_to_global_y_um']==pytest.approx(.6)
        else:
            assert second['native_eds_shape_hw']==[7,5]
            assert second['input_layout'].startswith('full global')


def test_count_conversion_keeps_raw_counts_and_time_and_existing_cps_unchanged(tmp_path):
    from ebsd_autoplot.eds import load_source_eds
    source,truth=montage(tmp_path,unit='counts')
    with h5py.File(source.parts[1].path,'r+') as f:f['1/EDS/Data/Live Time'][0]=0
    source=H5Source(source.path)
    with load_source_eds(source,count_rate=True,block_rows=1) as result:
        assert np.array_equal(result.maps['Ti Kα1'],truth)
        assert result.units['Ti Kα1']=='counts' and result.units['Ti Kα1_cps']=='counts/s'
        assert result.maps['Ti Kα1_cps'][0,0]==truth[0,0]/2
        assert not result.masks['Ti Kα1_cps'][2,0] and np.isnan(result.maps['Ti Kα1_cps'][2,0])
        assert result.timing_maps['Live Time'][2,0]==0
    cps_root=tmp_path/'cps';cps_root.mkdir();source,truth=montage(cps_root)
    with load_source_eds(source,count_rate=True) as result:
        assert np.array_equal(result.maps['Ti Kα1'],truth) and 'Ti Kα1_cps' not in result.maps


@pytest.mark.parametrize('all_missing',[False,True])
def test_missing_partial_modality_is_explicit_unavailable_no_fake_arrays(tmp_path,all_missing):
    from ebsd_autoplot.eds import load_source_eds
    source,_=montage(tmp_path)
    for part in (source.parts if all_missing else source.parts[1:2]):
        with h5py.File(part.path,'r+') as f:del f['1/EDS']
    source=H5Source(source.path)
    with load_source_eds(source) as result:
        assert not result.available and not result.maps and not result.masks
        assert result.metadata['missing_row_ranges']==([[0,2],[2,5],[5,7]] if all_missing else [[2,5]])
        assert 'no missing values were fabricated' in result.metadata['reason']


@pytest.mark.parametrize('mutation',['unit','dtype','channel','timing_unit','header_step','coordinate','root_range'])
def test_inconsistent_shards_fail_without_silent_alignment_or_unit_guesses(tmp_path,mutation):
    from ebsd_autoplot.eds import load_source_eds
    source,_=montage(tmp_path);part=source.parts[1]
    with h5py.File(part.path,'r+') as f:
        group=f['1/EDS/Data/Window Integral']
        if mutation=='unit':group['Ti Kα1'].attrs['Unit']='counts'
        elif mutation=='dtype':
            data=group['Ti Kα1'][:].astype(np.float32);del group['Ti Kα1'];ds=group.create_dataset('Ti Kα1',data=data);ds.attrs['Unit']='cps'
        elif mutation=='channel':del group['Ti Kα1']
        elif mutation=='timing_unit':f['1/EDS/Data/Live Time'].attrs['Unit']='ms'
        elif mutation=='header_step':f['1/EDS/Header/X Step'][0]=.25
        elif mutation=='coordinate':f['1/EDS/Data/X'][-1]+=.02
        else:
            for key,value in {'global_nx':5,'global_ny':7,'start_row_in_original':1,'end_row_in_original_exclusive':5}.items():f.attrs[key]=value
    if mutation=='root_range':
        # Existing H5Source also rejects contradictory original row metadata.
        with pytest.raises(ValueError):H5Source(source.path)
    else:
        source=H5Source(source.path)
        with pytest.raises(ValueError):load_source_eds(source,block_rows=1)


def test_overlap_gaps_source_mutation_and_cancellation_fail_cleanly(tmp_path):
    from ebsd_autoplot.eds import load_source_eds
    source,_=montage(tmp_path)
    old=source.parts[1];source.parts[1]=replace(old,start=1)
    with pytest.raises(ValueError,match='overlap or gap'):load_source_eds(source)
    source.parts[1]=old
    with h5py.File(old.path,'r+') as f:f.attrs['user_note']='changed since snapshot'
    with pytest.raises(ValueError,match='changed'):load_source_eds(source)
    source=H5Source(source.path);checks=[]
    def cancel():checks.append(True);return len(checks)>8
    with cancellation(cancel),pytest.raises(Cancelled):load_source_eds(source,block_rows=1)


def test_native_preview_is_bounded_raw_and_does_not_claim_registration(tmp_path):
    from ebsd_autoplot.eds import load_native_h5oina_eds,read_eds_preview
    source,truth=montage(tmp_path,full_eds=True)
    path=source.parts[0].path
    # EBSD contains only the first row shard; native EDS is full7x5. A native
    # reader must keep7x5 instead of silently cropping/resizing to EBSD rows.
    with load_native_h5oina_eds(path,block_rows=2) as result:
        assert result.maps['Ti Kα1'].shape==(7,5)
        assert np.array_equal(result.maps['Ti Kα1'],truth)
        assert not result.metadata['registration']['aligned_to_ebsd']
    preview=read_eds_preview(path,'/1/EBSD',channel='Ti Kα1',max_side=3)
    assert preview['shape_hw']==[7,5] and preview['image'].shape==(3,2)
    assert np.array_equal(preview['image'],truth[::3,::3])
    assert preview['unit']=='cps' and preview['step_um_xy']==pytest.approx([.2,.3])
    assert not preview['registered'] and preview['source_stamp']['path']==str(path.resolve())
    with pytest.raises(KeyError):read_eds_preview(path,channel='not an element')


def test_temp_backed_storage_and_close_do_not_close_unowned_arrays(tmp_path,monkeypatch):
    from ebsd_autoplot import eds_montage
    from ebsd_autoplot.eds import EDSResult,load_source_eds
    source,truth=montage(tmp_path)
    monkeypatch.setattr(eds_montage,'_ARRAY_MEMORY_LIMIT',32)
    result=load_source_eds(source,block_rows=1)
    assert isinstance(result.maps['Ti Kα1'],np.memmap)
    assert np.array_equal(result.maps['Ti Kα1'],truth)
    handles=list(result._temporary_files);mapped=result.maps['Ti Kα1']
    unowned=np.arange(8);result.maps['caller']=unowned
    result.close();result.close()
    assert all(handle.closed for handle in handles) and mapped._mmap.closed
    assert np.array_equal(unowned,np.arange(8))
    extra=EDSResult();out=extra.allocate((4,4),float)
    assert isinstance(out,np.memmap);extra.close()


def test_actual_exporter_local_headers_contract_integrates_with_eds(tmp_path):
    from test_montage_local_headers import extractor_montage
    from ebsd_autoplot.eds import load_source_eds
    path,_,_=extractor_montage(tmp_path);source=H5Source(path)
    with load_source_eds(source,block_rows=1) as result:
        assert result.available and next(iter(result.maps.values())).shape==(source.ny,source.nx)
        assert result.metadata['registration']['aligned_to_ebsd']


def test_full_duplicate_eds_coordinates_outside_selected_part_are_checked(tmp_path):
    from ebsd_autoplot.eds import load_source_eds
    source,_=montage(tmp_path,full_eds=True)
    with h5py.File(source.parts[1].path,'r+') as f:
        # This is outside secondpart's authoritative [2,5) selected data rows.
        f['1/EDS/Data/Y'][0]+=.02
    source=H5Source(source.path)
    with pytest.raises(ValueError,match='coordinates'):load_source_eds(source,block_rows=1)


def test_cancelled_memmap_read_releases_only_owned_temporary_handles(tmp_path,monkeypatch):
    from ebsd_autoplot import eds_montage
    from ebsd_autoplot.eds import load_source_eds
    source,_=montage(tmp_path);handles=[];checks=[]
    original=eds_montage.tempfile.TemporaryFile
    def track(*args,**kwargs):
        handle=original(*args,**kwargs);handles.append(handle);return handle
    monkeypatch.setattr(eds_montage,'_ARRAY_MEMORY_LIMIT',0)
    monkeypatch.setattr(eds_montage.tempfile,'TemporaryFile',track)
    def cancel():checks.append(True);return bool(handles)
    with cancellation(cancel),pytest.raises(Cancelled):load_source_eds(source,block_rows=1)
    assert handles and all(handle.closed for handle in handles)


def test_small_step_difference_cannot_accumulate_into_unregistered_far_edge_drift(tmp_path):
    from ebsd_autoplot.eds import load_source_eds
    root=tmp_path/'drift';write_synth(root,h=2,w=2000);path=root/'sample.h5oina'
    yy,xx=np.indices((2,2000))
    with h5py.File(path,'r+') as f:
        head=f['1/EBSD/Header'];data=f['1/EBSD/Data']
        for key,value in [('X Step',1.),('Y Step',1.)]:head[key][0]=value
        for key in ('X Cells','Y Cells'):head[key].attrs['Unit']='px'
        for key in ('X Step','Y Step'):head[key].attrs['Unit']='um'
        for key,values in [('X',xx),('Y',yy)]:
            ds=data.create_dataset(key,data=values.astype(float).ravel());ds.attrs['Unit']='um'
        eds=f.create_group('1/EDS');f.copy(head,eds,name='Header');eds['Header/X Step'][0]=1.000009
        edata=eds.create_group('Data')
        for key,values in [('X',xx*1.000009),('Y',yy.astype(float))]:
            ds=edata.create_dataset(key,data=values.ravel());ds.attrs['Unit']='um'
        ds=edata.create_dataset('Window Integral/Ti Ka',data=np.arange(4000));ds.attrs['Unit']='cps'
    source=H5Source(path)
    with pytest.raises(ValueError,match='coordinates|drift|registration'):load_source_eds(source,block_rows=1)


def one_native_pixel(tmp_path):
    path=tmp_path/'native.h5oina'
    with h5py.File(path,'w') as f:
        head=f.create_group('1/EDS/Header');data=f.create_group('1/EDS/Data')
        for key,value,unit in [('X Cells',1,'px'),('Y Cells',1,'px'),('X Step',1.,'um'),('Y Step',1.,'um')]:
            ds=head.create_dataset(key,data=[value]);ds.attrs['Unit']=unit
        for key in ('X','Y'):
            ds=data.create_dataset(key,data=[0.]);ds.attrs['Unit']='um'
        ds=data.create_dataset('Live Time',data=[.5]);ds.attrs['Unit']='s'
        ds=data.create_dataset('Window Integral/Ti Ka',data=[2]);ds.attrs['Unit']='cps'
    return path


@pytest.mark.parametrize('malformation',['root_bool','header_cells_bool','header_step_bool'])
def test_native_boolean_grid_provenance_is_rejected_by_reader_and_preview(tmp_path,malformation):
    from ebsd_autoplot.eds import load_native_h5oina_eds,read_eds_preview
    path=one_native_pixel(tmp_path)
    with h5py.File(path,'r+') as f:
        if malformation=='root_bool':
            f.attrs.update(global_nx=True,global_ny=True,start_row_in_original=False,end_row_in_original_exclusive=True)
        else:
            key='X Cells' if malformation=='header_cells_bool' else 'X Step'
            head=f['1/EDS/Header'];unit=head[key].attrs['Unit'];del head[key]
            ds=head.create_dataset(key,data=[True]);ds.attrs['Unit']=unit
    with pytest.raises(ValueError):load_native_h5oina_eds(path)
    with pytest.raises(ValueError):read_eds_preview(path)


def test_conflicting_window_unit_declarations_cannot_change_count_rate_semantics(tmp_path):
    from ebsd_autoplot.eds import load_native_h5oina_eds,read_eds_preview
    path=one_native_pixel(tmp_path)
    with h5py.File(path,'r+') as f:f['1/EDS/Data/Window Integral/Ti Ka'].attrs['Units']='counts'
    with pytest.raises(ValueError,match='unit|Unit'):load_native_h5oina_eds(path,count_rate=True)
    with pytest.raises(ValueError,match='unit|Unit'):read_eds_preview(path)


@pytest.mark.parametrize('where,other',[('Header/X Step','nm'),('Header/X Cells','um'),('Data/Live Time','ms'),('Data/Window Integral/Ti Ka','counts')])
def test_conflicting_native_units_in_header_timing_and_intensity_reject_consistently(tmp_path,where,other):
    from ebsd_autoplot.eds import load_native_h5oina_eds,read_eds_preview
    path=one_native_pixel(tmp_path)
    with h5py.File(path,'r+') as f:f['1/EDS/'+where].attrs['Units']=other
    with pytest.raises(ValueError,match='unit|Unit'):load_native_h5oina_eds(path)
    with pytest.raises(ValueError,match='unit|Unit'):read_eds_preview(path)


def test_equivalent_unit_aliases_are_accepted_without_relabeling_source_values(tmp_path):
    from ebsd_autoplot.eds import load_native_h5oina_eds
    path=one_native_pixel(tmp_path)
    with h5py.File(path,'r+') as f:
        for key,alias in [('Header/X Step','µm'),('Header/X Cells','pixel'),('Data/Live Time','seconds'),('Data/Window Integral/Ti Ka','counts/s')]:
            f['1/EDS/'+key].attrs['Units']=alias
    with load_native_h5oina_eds(path,count_rate=True) as result:
        assert result.maps['Ti Ka'][0,0]==2 and result.units['Ti Ka']=='cps'
        assert list(result.maps)==['Ti Ka']  # Existing cps is never divided twice.


@pytest.mark.parametrize('value',[np.nan,np.inf,[1,2],'bad scalar'])
def test_malformed_native_original_grid_scalar_never_passes_snapshot_preview(tmp_path,value):
    from ebsd_autoplot.eds import load_native_h5oina_eds,read_eds_preview
    path=one_native_pixel(tmp_path)
    with h5py.File(path,'r+') as f:
        f.attrs.update(global_nx=1,global_ny=1,start_row_in_original=0,end_row_in_original_exclusive=1)
        f.attrs['global_ny']=value
    with pytest.raises(ValueError):load_native_h5oina_eds(path)
    with pytest.raises(ValueError):read_eds_preview(path)


def test_explicit_registration_declares_target_grid_without_external_validation(tmp_path):
    from ebsd_autoplot.eds import load_source_eds
    source,_=montage(tmp_path)
    for part in source.parts:
        with h5py.File(part.path,'r+') as f:f['1/EDS/Data/X'][:]+=.02
    source=H5Source(source.path)
    with load_source_eds(source,registration=[[1,0,.1],[0,1,0],[0,0,1]],block_rows=1) as result:
        state=result.metadata['registration']
        assert state['aligned_to_ebsd'] and state['resampled']
        assert state['independently_validated'] is False
        assert result.maps['Ti Kα1'].shape==(7,5)
        assert not result.masks['Ti Kα1'][:,0].any()
