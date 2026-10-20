import json
from pathlib import Path
import h5py
import numpy as np
import pytest
from ebsd_autoplot.io import H5Source
from ebsd_autoplot.crystal import quat_xyzw, misorientation_deg
from ebsd_autoplot.compute import prepare
from ebsd_autoplot.v2 import run_v2


def write_synth(path,h=12,w=16,split=False):
    path.mkdir(parents=True,exist_ok=True)
    e=np.zeros((h,w,3),np.float32)
    e[:,:,0]=0.15
    e[:,w//2:,0]=0.70
    e[:,:,1]=0.4
    p=np.ones((h,w),np.uint8)
    p[0,0]=0
    bc=np.full((h,w),100,np.uint8)
    parts=[(0,h)] if not split else [(0,h//2),(h//2,h)]
    for a,b in parts:
        fp=path/ (f'scan01_rows{a:06d}-{b-1:06d}_EBSD.h5' if split else 'sample.h5oina')
        with h5py.File(fp,'w') as f:
            dd=f.create_group('1/EBSD/Data')
            dd['Euler']=e[a:b].reshape(-1,3)
            dd['Phase']=p[a:b].reshape(-1)
            dd['Band Contrast']=bc[a:b].reshape(-1)
            dd['Pattern Quality']=np.full((b-a)*w,123,np.float32)
            dd['Mean Angular Deviation']=np.full((b-a)*w,0.4,np.float32)
            hh=f.create_group('1/EBSD/Header')
            hh['X Cells']=[w];hh['Y Cells']=[h];hh['X Step']=[.05];hh['Y Step']=[.05]
            hh['Scanning Rotation Angle']=[0.]
            for i,(name,sg,color) in enumerate([('Titanium-Hexagonal',194,[255,0,0]),('Titanium-Cubic',229,[0,0,255])],1):
                gp=hh.create_group('Phases/'+str(i))
                gp['Phase Id']=[i];gp['Phase Name']=np.array([name.encode()]);gp['Space Group']=[sg];gp['Color']=[color]
                # Synthetic fixture metadata only; production export reads the
                # measured values from the input H5 phase header.
                gp['Lattice Dimensions']=[2.95,2.95,4.68,90,90,120] if sg==194 else [3.30,3.30,3.30,90,90,90]
    return p


def test_symmetry():
    q=quat_xyzw(np.zeros((1,1,3)))
    p=np.ones((1,1),dtype='u1')
    a=misorientation_deg(q,q,p,p,{1:{'space_group':194}})
    assert float(a[0,0]) <1e-4
    assert np.isnan(misorientation_deg(q,q,p,np.zeros_like(p),{1:{'space_group':194}})[0,0])
    # Hexagonal 60-degree c-axis symmetry must be zero crystal disorientation.
    e1=np.zeros((1,1,3));e2=np.zeros((1,1,3));e2[...,0]=np.pi/3
    a=misorientation_deg(quat_xyzw(e1),quat_xyzw(e2),p,p,{1:{'space_group':194}})
    assert a[0,0]<.01


def test_split_and_v2(tmp_path):
    folder=tmp_path/'parts';truth=write_synth(folder,split=True)
    src=H5Source(folder)
    assert src.nx==16 and src.ny==12 and src.probe_euler_unit()=='radian'
    assert np.array_equal(src.read(0,12,('phase',))['phase'],truth)
    out=tmp_path/'out'
    ff=prepare(src,out,batch=4,do_ipf=False,progress=lambda msg:None)
    with h5py.File(ff,'r') as f:
        assert f['features'].shape==(7,12,16)
        assert abs(float(f['features'][4,3,4])-1)<1e-4
        assert float(f['features'][5,5,3])<.001
        # phase0 should not have valid KAM
        assert f['kam_valid'][0,0]==0
        # right-edge phase 1 map differs 0.55 rad, so HAGB
        assert f['boundary_right'][4,7] ==2
        # no artificial seam between rows 5 and 6
        assert f['boundary_down'][5,2]==0
    res=run_v2(ff,out/'v2',roi=[0,0,16,12],min_size=1,progress=lambda msg:None)
    assert res['reported_grains']>=2
    assert (out/'v2'/'grains.csv').exists()


def test_raw_h5(tmp_path):
    folder=tmp_path/'raw';write_synth(folder)
    s=H5Source(folder/'sample.h5oina')
    assert s.ny==12 and s.read(3,7,('euler',))['euler'].shape==(4,16,3)

def test_ipf_never_fake_when_orix_missing():
    import importlib.util
    if importlib.util.find_spec('orix') is None:
        from ebsd_autoplot.crystal import ipf_colors
        with pytest.raises(RuntimeError,match='requires orix'):
            ipf_colors(np.zeros((2,2,3)),np.ones((2,2),dtype=np.uint8),{1:{'space_group':194}},axis='Z')


def test_manifest_phase_group_key(tmp_path):
    folder=tmp_path/'montage';write_synth(folder,split=True)
    p=folder/'scan01_rows000000-000005_EBSD.h5'
    with h5py.File(p,'a') as f:
        f['1/EBSD/Header/Phases/1/Phase Id'][0]=6
        f['1/EBSD/Header/Phases/2/Phase Id'][0]=5
    source=H5Source(folder)
    assert source.phase_specs[1]['space_group']==194
    assert source.phase_specs[1]['database_phase_id']==6
