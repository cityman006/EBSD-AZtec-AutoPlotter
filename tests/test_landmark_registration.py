"""Source-bound landmark projects and bounded inverse-map sampling."""
import copy
import os

import numpy as np
import pytest

from ebsd_autoplot.registration import (file_stamp, validate_landmark_config,
    resample_map_blocked)
from test_v4_patches import feature_cache


def project(tmp_path):
    cache,_=feature_cache(tmp_path,shape=(7,9))
    source=tmp_path/'EDS 原始数据.h5';source.write_bytes(b'native source sentinel')
    points=[[0,0],[4,0],[0,3],[4,3]]
    return cache,source,dict(schema_version=1,source_path=str(source.resolve()),scan_path='/1/EBSD',
        source_stamp=file_stamp(source),target_cache_stamp=file_stamp(cache),
        source_shape_hw=[4,5],target_shape_hw=[7,9],source_xy=points,
        target_xy=[[x+1,y+2] for x,y in points],matrix=[[1,0,1],[0,1,2],[0,0,1]])


def test_project_recomputes_fit_qc_and_never_claims_external_validation(tmp_path):
    cache,source,cfg=project(tmp_path);cfg['fit_qc']={'independently_validated':True}
    frozen=copy.deepcopy(cfg);actual=validate_landmark_config(cfg,cache_path=cache,source_shape=(4,5))
    assert actual['fit_qc']['rms_target_px']<1e-12
    assert actual['fit_qc']['independently_validated'] is False
    assert actual['coordinate_convention']=='source EDS to target EBSD; x,y pixel centres'
    assert cfg==frozen and source.read_bytes()==b'native source sentinel'
    with pytest.raises(ValueError,match='shape'):
        validate_landmark_config(cfg,cache_path=cache,source_shape=(5,4))


def test_project_rejects_changed_files_and_different_cache(tmp_path):
    cache,source,cfg=project(tmp_path)
    other=tmp_path/'other';other.mkdir();other_cache,_=feature_cache(other,shape=(7,9))
    with pytest.raises(ValueError,match='target cache'):
        validate_landmark_config(cfg,cache_path=other_cache)
    stat=source.stat();os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
    with pytest.raises(ValueError,match='changed'):
        validate_landmark_config(cfg,cache_path=cache)
    cfg['source_stamp']=file_stamp(source)
    stat=cache.stat();os.utime(cache,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
    with pytest.raises(ValueError,match='changed'):
        validate_landmark_config(cfg,cache_path=cache)


@pytest.mark.skipif(os.name!='nt',reason='Windows extended-length path spelling')
def test_same_source_and_target_accept_equivalent_windows_long_paths(tmp_path):
    from ebsd_autoplot.storage import long_path
    cache,source,cfg=project(tmp_path)
    cfg['source_path']=str(long_path(source))
    validated=validate_landmark_config(cfg,cache_path=long_path(cache),source_shape=(4,5))
    assert validated['source_path']==file_stamp(source)['path']
    np.testing.assert_allclose(validated['matrix'],[[1,0,1],[0,1,2],[0,0,1]],atol=1e-12)


@pytest.mark.parametrize('change', ['out_of_bounds','nan','collinear','wrong_matrix','singular','wrong_shape','wrong_schema'])
def test_invalid_or_tampered_landmarks_cannot_be_applied(tmp_path,change):
    cache,_,cfg=project(tmp_path)
    if change=='out_of_bounds':cfg['source_xy'][0][0]=5
    elif change=='nan':cfg['target_xy'][0][0]=float('nan')
    elif change=='collinear':cfg['source_xy']=[[0,0],[1,0],[2,0]];cfg['target_xy']=[[1,1],[2,1],[3,1]]
    elif change=='wrong_matrix':cfg['matrix'][0][2]=2
    elif change=='singular':cfg['matrix'][0]=[0,0,0]
    elif change=='wrong_shape':cfg['target_shape_hw']=[6,9]
    else:cfg['schema_version']=2
    with pytest.raises(ValueError):validate_landmark_config(cfg,cache_path=cache)


def test_blocked_sampling_native_values_masks_and_bounded_coordinates(monkeypatch):
    import ebsd_autoplot.registration as registration
    source=np.arange(35,dtype=float).reshape(5,7);source[2,3]=np.nan
    calls=[];original=registration.map_coordinates
    def bounded(data,coords,*args,**kwargs):
        calls.append(coords.shape);assert coords.shape[1]<=2
        return original(data,coords,*args,**kwargs)
    monkeypatch.setattr(registration,'map_coordinates',bounded)
    values,valid=resample_map_blocked(source,(8,10),[[1,0,2],[0,1,1],[0,0,1]],block_rows=2)
    expected=np.full((8,10),np.nan);expected[1:6,2:9]=source
    np.testing.assert_allclose(values,expected,equal_nan=True)
    np.testing.assert_array_equal(valid,np.isfinite(expected));assert len(calls)==8


def test_blocked_sampling_support_and_cancellation():
    from ebsd_autoplot.storage import cancellation,Cancelled
    data=np.ones((3,4));data[1,1]=np.nan
    values,valid=resample_map_blocked(data,(3,4),[[1,0,.5],[0,1,0],[0,0,1]],block_rows=1)
    assert not valid[1,1] and not valid[1,2] and np.isnan(values[1,1])
    with cancellation(lambda:True),pytest.raises(Cancelled):
        resample_map_blocked(data,(3,4),np.eye(3),block_rows=1)
