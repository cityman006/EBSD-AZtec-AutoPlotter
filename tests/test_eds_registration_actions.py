"""Real archive actions preserve raw/cache bytes and publish new EDS runs."""
import copy
import json
import os

import h5py
import numpy as np
import pytest

from ebsd_autoplot.io import H5Source
from ebsd_autoplot.pipeline import analyze
from ebsd_autoplot.cli import parser
from ebsd_autoplot.registration import file_stamp
from ebsd_autoplot.storage import digest,validate_receipt
from ebsd_autoplot.studio_actions import run_action
from test_pipeline import write_synth
from test_v4_patches import h5oina_eds_fixture


def archived(tmp_path,source=None):
    if source is None:
        write_synth(tmp_path/'raw',h=9,w=10);source=H5Source(tmp_path/'raw/sample.h5oina')
    opts=vars(parser().parse_args(['--analysis','v1','--no-ipf','--no-render']))
    run=analyze(source,tmp_path/'out',opts)
    return run/'analysis_cache/features.h5'


def manual_project(tmp_path,cache):
    folder=tmp_path/'原始 EDS';folder.mkdir()
    source,values=h5oina_eds_fixture(folder,unit='cps')
    with h5py.File(cache,'r') as stream:target_shape=list(stream['indexed'].shape)
    points=[[0,0],[3,0],[0,2],[3,2]]
    project=dict(schema_version=1,source_path=str(source.resolve()),scan_path='/1/EBSD',
        source_stamp=file_stamp(source),target_cache_stamp=file_stamp(cache),source_shape_hw=[3,4],
        target_shape_hw=target_shape,source_xy=points,target_xy=[[x+1,y+2] for x,y in points],
        matrix=[[1,0,1],[0,1,2],[0,0,1]])
    return project,source,values


def forbid_reanalysis(monkeypatch):
    import ebsd_autoplot.compute as compute
    def forbidden(*args,**kwargs):raise AssertionError('Original EBSD analysis must never run')
    monkeypatch.setattr(compute,'prepare',forbidden)


def test_landmark_eds_action_all_channels_new_runs_and_source_immutability(tmp_path,monkeypatch):
    cache=archived(tmp_path);cfg,source,values=manual_project(tmp_path,cache)
    before={path:digest(path) for path in (source,cache)};forbid_reanalysis(monkeypatch)
    first=run_action('eds',cache,tmp_path/'out',config={'landmark_registration':cfg,'count_rate':True,'block_rows':1})
    report=json.loads((first/'results/eds_manifest.json').read_text(encoding='utf-8'))
    assert report['available'] and report['units']['Al Kα1']=='cps'
    assert report['metadata']['registration']['method']=='explicit paired-landmark affine'
    assert report['metadata']['registration']['independently_validated'] is False
    with h5py.File(first/'results/eds_fields.h5','r') as stream:
        expected=np.full((9,10),np.nan);expected[2:5,1:5]=values
        np.testing.assert_allclose(stream['channels/Al Kα1'][:],expected,equal_nan=True)
        np.testing.assert_allclose(stream['channels/Ti Kα1'][:],expected*10,equal_nan=True)
        np.testing.assert_array_equal(stream['masks/Al Kα1'][:],np.isfinite(expected))
        assert stream['native_timing/Live Time'].shape==(3,4)
    assert validate_receipt(first/'results','ebsd_autoplot.eds.save_eds')
    original_result=digest(first/'results/eds_fields.h5')
    second=run_action('eds',cache,tmp_path/'out',config={'landmark_registration':cfg})
    assert second!=first and digest(first/'results/eds_fields.h5')==original_result
    assert all(digest(path)==checksum for path,checksum in before.items())


def test_registered_eds_patch_export_and_changed_project_fails_before_publish(tmp_path,monkeypatch):
    cache=archived(tmp_path);cfg,source,_=manual_project(tmp_path,cache);forbid_reanalysis(monkeypatch)
    out=run_action('patches',cache,tmp_path/'out',config={'include_eds':True,'landmark_registration':cfg,
        'grid':[1,1],'preview':False})
    with h5py.File(out/'results/patches/patch_00000.h5','r') as stream:
        assert stream['channels/eds_Al Kα1'].shape==(9,10)
        meta=json.loads(stream.attrs['metadata_json'])
        assert meta['eds_metadata']['registration']['landmark_project']['source_path']==str(source.resolve())
    old=list((tmp_path/'out/studio_runs').iterdir())
    stat=source.stat();os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
    with pytest.raises(ValueError,match='changed'):
        run_action('eds',cache,tmp_path/'out',config={'landmark_registration':cfg})
    new=[p for p in (tmp_path/'out/studio_runs').iterdir() if p not in old]
    assert len(new)==1 and not (new[0]/'results/eds_manifest.json').exists()
    assert json.loads((new[0]/'studio_manifest.json').read_text(encoding='utf-8'))['status']=='failed'


def test_archived_montage_eds_and_patches_use_same_exact_registered_grid(tmp_path,monkeypatch):
    from test_eds_montage import montage
    source,truth=montage(tmp_path/'分片')
    before={part.path:digest(part.path) for part in source.parts}
    cache=archived(tmp_path,source);cache_hash=digest(cache);forbid_reanalysis(monkeypatch)
    eds=run_action('eds',cache,tmp_path/'out')
    with h5py.File(eds/'results/eds_fields.h5','r') as stream:
        np.testing.assert_array_equal(stream['channels/Ti Kα1'][:],truth)
    patches=run_action('patches',cache,tmp_path/'out',config={'include_eds':True,'grid':[1,1],'preview':False})
    with h5py.File(patches/'results/patches/patch_00000.h5','r') as stream:
        np.testing.assert_array_equal(stream['channels/eds_Ti Kα1'][:],truth)
    assert digest(cache)==cache_hash and all(digest(path)==checksum for path,checksum in before.items())


def test_gui_freezes_manual_registration_and_clears_cross_cache_source(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets as W
    from gui_qt import Studio
    cache=archived(tmp_path);cfg,_,_=manual_project(tmp_path,cache)
    app=W.QApplication.instance() or W.QApplication([])
    win=Studio();win.preferences=tmp_path/'preferences.json';win.workspace.load_cache(cache)
    win.eds_registration=cfg;calls=[];win.cached_action=lambda *args:calls.append(copy.deepcopy(args))
    win.import_eds();assert calls[0][0]=='eds'
    assert calls[0][1]['landmark_registration']['source_path']==cfg['source_path']
    cfg['source_xy'][0][0]=1
    assert calls[0][1]['landmark_registration']['source_xy'][0][0]==0
    win.viewer.clear_cache();win.sync_active_view()
    assert win.eds_registration is None
    win.close();app.processEvents()
