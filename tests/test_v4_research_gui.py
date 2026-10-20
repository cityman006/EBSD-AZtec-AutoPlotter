import os,json,time
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
import pytest
pytest.importorskip('PySide6')
from PySide6 import QtWidgets as W,QtCore as C,QtTest
from gui_qt import Studio


def studio(tmp_path):
    app=W.QApplication.instance() or W.QApplication([])
    win=Studio();win.preferences=tmp_path/'preferences.json';win.out.setText(str(tmp_path));errors=[];win.error=lambda msg:errors.append(str(msg))
    return app,win,errors


def pump(app,predicate,timeout=30):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        app.processEvents()
        if predicate():return
        QtTest.QTest.qWait(10)
    raise AssertionError('Qt process did not reach expected state')


def test_gui_no_labels_training_gate_and_render_cache_gate(tmp_path,monkeypatch):
    app,win,errors=studio(tmp_path)
    assert win.ai_tabs.count()==4
    assert not win.train_gnn.isEnabled() and not win.train_ridge.isEnabled()
    win.render_only();assert 'features.h5' in errors[-1]
    path=tmp_path/'unlabeled.npz';np.savez(path,features=np.zeros((3,2)),groups=np.array(['a','b','c']),metadata_json=np.array('{}'))
    monkeypatch.setattr(W.QFileDialog,'getOpenFileName',lambda *a,**k:(str(path),''))
    win.select_training_bundle()
    assert not win.train_gnn.isEnabled() and not win.train_ridge.isEnabled()
    assert 'No actual labels' in errors[-1]
    win.close();app.processEvents()


def test_qt_research_process_runs_without_raw_cache(tmp_path):
    app,win,errors=studio(tmp_path)
    win.cached_action('synthetic-gnn-demo',{'epochs':2,'hidden':6})
    pump(app,lambda:win.process.state()==C.QProcess.ProcessState.NotRunning)
    app.processEvents()
    manifests=list(tmp_path.glob('studio_runs/*/studio_manifest.json'))
    assert len(manifests)==1
    manifest=json.loads(manifests[0].read_text())
    assert manifest['status']=='success' and manifest['result']['synthetic_demo']
    assert win.task_status.text()=='SUCCESS' and not errors
    win.close();app.processEvents()


def test_pause_keeps_gui_responsive_and_cancel_is_not_success(tmp_path):
    app,win,errors=studio(tmp_path)
    win.cached_action('synthetic-gnn-demo',{'epochs':5000,'hidden':8})
    control=win.active_control
    from ebsd_autoplot.storage import atomic_json
    atomic_json(control,{'cancel':False,'pause':True})
    pump(app,lambda:'PAUSED' in win.log.toPlainText())
    old=win.lang;win.toggle_language();assert win.lang!=old
    assert win.process.state()!=C.QProcess.ProcessState.NotRunning
    win.cancel();pump(app,lambda:win.process.state()==C.QProcess.ProcessState.NotRunning)
    app.processEvents()
    manifest=json.loads(next(tmp_path.glob('studio_runs/*/studio_manifest.json')).read_text())
    assert manifest['status']=='cancelled' and win.task_status.text()=='CANCELLED'
    assert not errors
    win.close();app.processEvents()
