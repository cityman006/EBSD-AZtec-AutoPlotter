"""Regression evidence from the independent M1 safety and rendering review."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
import h5py
import numpy as np
import pytest
from PIL import Image
from ebsd_autoplot.batch import execute_batch
from ebsd_autoplot.storage import atomic_json


def batch_manifest(tmp_path, count=1):
    run=tmp_path/'batch';run.mkdir()
    records=[{'input':'unused.h5','source_name':f'synthetic{i}', 'scan_path':'/1/EBSD',
              'scan_id':f's{i}', 'source_stamp':[], 'status':'queued', 'estimated_block_bytes':1}
             for i in range(count)]
    atomic_json(run/'batch_manifest.json',{'status':'queued','records':records,'options':[],
                'memory_budget_mb':16,'output_root':str(tmp_path)})
    return run


def test_callback_failure_reaps_real_child_before_next_sample(tmp_path):
    run=batch_manifest(tmp_path,2);children=[];real_popen=subprocess.Popen;calls=[]
    def factory(argv,**kwargs):
        # The next sample must not start until its predecessor is really gone.
        assert all(child.poll() is not None for child in children)
        if not children:
            control=argv[argv.index('--control-file')+1]
            code=('import json,time,sys;from pathlib import Path; '
                  'p=Path(sys.argv[1]);print("child ready",flush=True); '
                  '\nwhile not json.loads(p.read_text())["cancel"]:time.sleep(.02)\n'
                  'sys.exit(130)')
            args=[sys.executable,'-u','-c',code,control]
        else:args=[sys.executable,'-u','-c','print("next sample complete",flush=True)']
        proc=real_popen(args,**kwargs);children.append(proc);return proc
    def progress(line):
        calls.append(line)
        if len(calls)==1:raise RuntimeError('synthetic progress consumer failure')
    try:
        with patch('ebsd_autoplot.batch.subprocess.Popen',side_effect=factory):
            result=execute_batch(run,progress=progress)
        assert [r['status'] for r in result['records']]==['failed','success']
        assert result['status']=='failed'
        assert all(child.poll() is not None for child in children)
        assert json.loads((run/'s0.control.json').read_text())['cancel']
    finally:
        for child in children:
            if child.poll() is None:child.terminate();child.wait()


def test_interrupt_reaps_owned_child_and_records_true_state(tmp_path):
    run=batch_manifest(tmp_path);children=[];real_popen=subprocess.Popen
    def factory(argv,**kwargs):
        control=argv[argv.index('--control-file')+1]
        code=('import json,time,sys;from pathlib import Path;p=Path(sys.argv[1]);'
              'print("ready",flush=True);\nwhile not json.loads(p.read_text())["cancel"]:time.sleep(.02)')
        proc=real_popen([sys.executable,'-u','-c',code,control],**kwargs);children.append(proc);return proc
    def interrupt(_):raise KeyboardInterrupt('synthetic supervisor interrupt')
    try:
        with patch('ebsd_autoplot.batch.subprocess.Popen',side_effect=factory), pytest.raises(KeyboardInterrupt):
            execute_batch(run,progress=interrupt)
        assert children[0].poll() is not None
        data=json.loads((run/'batch_manifest.json').read_text())
        assert data['status']=='interrupted' and data['records'][0]['status']=='interrupted'
    finally:
        for child in children:
            if child.poll() is None:child.terminate();child.wait()


def test_child_cancel_exit_cannot_report_batch_success(tmp_path):
    run=batch_manifest(tmp_path);real_popen=subprocess.Popen
    def factory(argv,**kwargs):
        return real_popen([sys.executable,'-c','raise SystemExit(130)'],**kwargs)
    with patch('ebsd_autoplot.batch.subprocess.Popen',side_effect=factory):
        data=execute_batch(run,progress=lambda _:None)
    assert not (run/'cancel.json').exists()
    assert data['records'][0]['status']=='cancelled' and data['status']=='cancelled'


def test_cleanup_bounds_unresponsive_owned_child(tmp_path):
    from ebsd_autoplot.batch import _stop_owned_child
    control=tmp_path/'control.json';atomic_json(control,{'cancel':False})
    checkpoint=tmp_path/'retained_checkpoint';checkpoint.write_text('complete')
    child=subprocess.Popen([sys.executable,'-u','-c','import time;print("ready",flush=True);time.sleep(60)'],
                           stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='ready'
        _stop_owned_child(child,control,timeout=.05)
        assert child.poll() is not None
        assert json.loads(control.read_text())['cancel']
        assert checkpoint.read_text()=='complete'
    finally:
        if child.poll() is None:child.terminate();child.wait()
        child.stdout.close()


def test_kam_invalid_finite_nan_masks_and_physical_aspect(tmp_path):
    from ebsd_autoplot.render import maps
    from matplotlib import pyplot as plt
    src=tmp_path/'features.h5';raw=np.zeros((7,2,3),np.float32)
    raw[3]=1;raw[4]=1;raw[5]=1;raw[5,0,0]=999;raw[5,0,1]=np.nan
    with h5py.File(src,'w') as f:
        f['features']=raw;f['indexed']=np.ones((2,3),np.uint8)
        f['kam_valid']=[[0,1,1],[1,1,1]]
        f.attrs['metadata_json']=json.dumps({'x_step_um':2,'y_step_um':6,
            'phase_specs':{'1':{'rgb':[255,0,0]}}})
    aspects=[];subplots=plt.subplots
    def capture(*args,**kwargs):
        fig,axes=subplots(*args,**kwargs)
        # Capture when saving, after imshow has configured the axes.
        save=fig.savefig
        def savefig(*a,**k):
            aspects.extend(ax.get_aspect() for ax in fig.axes)
            return save(*a,**k)
        fig.savefig=savefig
        return fig,axes
    with patch('ebsd_autoplot.render.plt.subplots',side_effect=capture):
        maps(src,tmp_path/'render',dpi=40,style={'maps':['KAM'],'layout':(1,1)},progress=lambda _:None)
    with Image.open(tmp_path/'render/single_maps/KAM.png') as im:
        assert im.getpixel((0,0))==(0,0,0)
        assert im.getpixel((1,0))==(0,0,0)
        assert im.getpixel((2,0))!=(0,0,0)
    assert 3.0 in aspects


def test_busy_qt_actions_keep_active_cancel_handle_and_snapshot(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen');pytest.importorskip('PySide6')
    from gui_qt import Studio
    active=tmp_path/'active';other=tmp_path/'other';active.mkdir();other.mkdir()
    atomic_json(other/'cancel.json',{'cancel':True})
    state=SimpleNamespace(busy=lambda:True,error=Mock(),batch_run=active,
                          active_batch_run=active,active_control=None)
    with patch('gui_qt.W.QFileDialog.getOpenFileName') as dialog:
        Studio.resume_batch(state);Studio.render_only(state)
    assert not dialog.called and state.batch_run==active
    assert state.active_batch_run==active and state.active_control is None
    assert json.loads((other/'cancel.json').read_text())['cancel']


def test_qt_cancel_targets_only_active_task_and_failed_start_recovers(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen');pytest.importorskip('PySide6')
    from gui_qt import Studio,C
    stale=tmp_path/'old_batch';stale.mkdir();current=tmp_path/'current.json'
    state=SimpleNamespace(busy=lambda:True,batch_run=stale,active_batch_run=None,
                          active_control=current,log=SimpleNamespace(appendPlainText=Mock()),
                          run_button=SimpleNamespace(setEnabled=Mock()))
    Studio.cancel(state)
    assert not (stale/'cancel.json').exists()
    assert json.loads(current.read_text())['cancel']
    Studio.process_error(state,C.QProcess.ProcessError.FailedToStart)
    assert state.active_control is None and state.active_batch_run is None
    state.run_button.setEnabled.assert_called_with(True)
