import json
import h5py
import numpy as np
import pytest
from PIL import Image
from matplotlib import colormaps
from ebsd_autoplot.io import H5Source
from ebsd_autoplot.compute import prepare
from ebsd_autoplot.render import maps
from test_pipeline import write_synth

@pytest.mark.parametrize('unit,value,expected_deg', [('rad',np.pi/180,1.),('deg',1.,1.),('deg',0.001,0.001)])
def test_mad_display_units(tmp_path,unit,value,expected_deg):
    write_synth(tmp_path/'raw')
    source=tmp_path/'raw'/'sample.h5oina'
    with h5py.File(source,'a') as f:
        d=f['1/EBSD/Data/Mean Angular Deviation'];d[:]=value;d.attrs['Unit']=unit
    out=tmp_path/'out';fp=prepare(H5Source(source),out,do_ipf=False,progress=lambda _:None)
    with h5py.File(fp,'r') as f:
        assert f['mad'].attrs['Unit']==unit
        np.testing.assert_allclose(f['mad'][()],value)
    maps(fp,out,progress=lambda _:None)
    rgb=np.asarray(Image.open(out/'single_maps'/'MAD.png'))
    lut=np.rint(255*colormaps['magma'](np.linspace(0,1,256))[:,:3]).astype('u1')
    expected=lut[int(expected_deg/2*255)]
    np.testing.assert_array_equal(rgb[3,3],expected)
    np.testing.assert_array_equal(rgb[0,0],[0,0,0])
    legend=json.loads((out/'colour_legend_metadata.json').read_text())['MAD']
    assert legend['display_unit']=='deg' and legend['input_unit']==unit
    assert legend['low_deg']==0 and legend['high_deg']==2
    assert (out/'legends'/'MAD_colourbar.png').exists()

def test_missing_attribute_uses_documented_format_not_values(tmp_path):
    write_synth(tmp_path/'raw')
    src=H5Source(tmp_path/'raw'/'sample.h5oina')
    assert src.header['mad_unit']=='rad'
    assert src.header['mad_unit_evidence'][0]['basis']=='H5OINA specification (attribute absent)'

def test_unknown_unit_rejected(tmp_path):
    write_synth(tmp_path/'raw')
    with h5py.File(tmp_path/'raw'/'sample.h5oina','a') as f:
        f['1/EBSD/Data/Mean Angular Deviation'].attrs['Unit']='unknown'
    with pytest.raises(ValueError,match='MAD unit'):
        H5Source(tmp_path/'raw'/'sample.h5oina')

def test_mixed_shard_units_rejected(tmp_path):
    write_synth(tmp_path/'raw',split=True)
    for file,unit in zip(sorted((tmp_path/'raw').glob('*.h5')),['rad','deg']):
        with h5py.File(file,'a') as f:f['1/EBSD/Data/Mean Angular Deviation'].attrs['Unit']=unit
    with pytest.raises(ValueError,match='MAD units differ'):
        H5Source(tmp_path/'raw')

def test_old_features_without_unit_require_regeneration(tmp_path):
    write_synth(tmp_path/'raw')
    fp=prepare(H5Source(tmp_path/'raw'/'sample.h5oina'),tmp_path/'out',do_ipf=False,progress=lambda _:None)
    with h5py.File(fp,'a') as f:
        if 'Unit' in f['mad'].attrs:del f['mad'].attrs['Unit']
    with pytest.raises(ValueError,match='Regenerate'):
        maps(fp,tmp_path/'rerender',progress=lambda _:None)

def test_invalid_mad_is_black_and_raw_values_are_preserved(tmp_path):
    write_synth(tmp_path/'raw')
    source=tmp_path/'raw'/'sample.h5oina'
    with h5py.File(source,'a') as f:
        d=f['1/EBSD/Data/Mean Angular Deviation'];d.attrs['Unit']='deg'
        d[:]=1.;d[1:4]=[np.nan,np.inf,-1.]
    fp=prepare(H5Source(source),tmp_path/'out',do_ipf=False,progress=lambda _:None)
    maps(fp,tmp_path/'out',progress=lambda _:None)
    rgb=np.asarray(Image.open(tmp_path/'out/single_maps/MAD.png'))
    np.testing.assert_array_equal(rgb[0,:4],np.zeros((4,3),dtype='u1'))
    assert rgb[1,1].any()
    with h5py.File(fp,'r') as f:
        assert np.isnan(f['mad'][0,1]) and np.isinf(f['mad'][0,2]) and f['mad'][0,3]==-1
