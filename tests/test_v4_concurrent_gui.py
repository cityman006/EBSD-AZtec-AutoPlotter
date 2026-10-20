"""Real Qt workers stay responsive while language/views and parameters change."""
import os,json
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
pytest.importorskip('PySide6')
from PySide6 import QtWidgets as W,QtCore as C
from gui_qt import Studio
from ebsd_autoplot.batch import create_batch
from test_pipeline import write_synth
from test_v4_multiview_gui import cache_file
from test_v4_research_gui import pump

_windows=[]


@pytest.fixture(autouse=True)
def close_owned_windows():
    yield
    app=W.QApplication.instance()
    for win in _windows:
        if win.busy():
            win.cancel();pump(app,lambda:win.process.state()==C.QProcess.ProcessState.NotRunning,timeout=90)
        for view in win.workspace.views:view.generation+=1;view.tile_timer.stop()
        win.close()
    C.QThreadPool.globalInstance().waitForDone(10000)
    if app:app.processEvents()
    _windows.clear()


def window(tmp_path):
    app=W.QApplication.instance() or W.QApplication([])
    win=Studio();win.preferences=tmp_path/'preferences.json';win.out.setText(str(tmp_path/'outputs'))
    _windows.append(win)
    errors=[];win.error=lambda e:errors.append(str(e))
    return app,win,errors


def test_real_parallel_batch_freezes_settings_while_view_language_changes(tmp_path):
    app,win,errors=window(tmp_path)
    paths=[]
    for name in ('a','b'):
        write_synth(tmp_path/name,h=12,w=12);paths.append(tmp_path/name/'sample.h5oina')
    cache=tmp_path/'features.h5';cache_file(cache)
    win.workspace.set_layout(4);win.workspace.load_cache(cache)
    options=['--analysis','v1','--no-ipf','--no-render','--batch','3']
    run=create_batch(paths,tmp_path/'outputs',options,concurrency=2)
    win.launch(['-u','-m','ebsd_autoplot.batch',str(run)],batch_run=run);win.batch_run=run
    def active_two():
        return json.loads((run/'batch_manifest.json').read_text()).get('resource_state',{}).get('active_children')==2
    pump(app,active_two,timeout=90)
    for iteration in range(4):
        win.toggle_language();win.set_theme(not win.dark);win.workspace.set_layout(2 if iteration%2 else 4)
        win.workspace.set_field('BC' if iteration%2 else 'Phase',index=0)
        view=win.workspace.views[0];view.apply_navigation({'center':(500,400),'scale':2.})
        view.navigation_changed.emit(view.navigation_state());app.processEvents()
    win.mode.setCurrentText('v3_schmid');win.block.setValue(1);win.concurrency.setValue(1);win.budget.setValue(16)
    win.out.setText(str(tmp_path/'changed-output'))
    pump(app,lambda:win.process.state()==C.QProcess.ProcessState.NotRunning,timeout=90);app.processEvents()
    manifest=json.loads((run/'batch_manifest.json').read_text())
    assert manifest['options']==options and manifest['concurrency']==2 and manifest['memory_budget_mb']==2048
    assert manifest['status']=='success' and all(r['status']=='success' for r in manifest['records'])
    assert len({r['run_path'] for r in manifest['records']})==2
    assert not (tmp_path/'changed-output').exists()
    assert win.task_status.property('status_code')=='SUCCESS' and not errors
    win.close();C.QThreadPool.globalInstance().waitForDone(10000);app.processEvents()


def test_real_gui_cancel_reaches_all_parallel_children_and_keeps_queue(tmp_path):
    app,win,errors=window(tmp_path)
    paths=[]
    for name in ('a','b','c'):
        write_synth(tmp_path/name,h=12,w=12);paths.append(tmp_path/name/'sample.h5oina')
    run=create_batch(paths,tmp_path/'outputs',['--no-ipf','--no-render'],concurrency=2)
    win.launch(['-u','-m','ebsd_autoplot.batch',str(run)],batch_run=run);win.batch_run=run
    pump(app,lambda:json.loads((run/'batch_manifest.json').read_text()).get('resource_state',{}).get('active_children')==2,timeout=90)
    win.cancel();win.toggle_language()
    pump(app,lambda:win.process.state()==C.QProcess.ProcessState.NotRunning,timeout=90);app.processEvents()
    manifest=json.loads((run/'batch_manifest.json').read_text())
    assert manifest['status']=='cancelled'
    assert [r['status'] for r in manifest['records']]==['cancelled','cancelled','queued']
    assert manifest['resource_state']['active_children']==0
    assert win.task_status.property('status_code')=='CANCELLED' and not errors
    win.close();app.processEvents()


def test_split_utf8_log_bytes_and_localized_dynamic_progress(tmp_path):
    app,win,errors=window(tmp_path)
    win.lang='zh';win.translate()
    code="import sys,time\nb='中文文件名：试样甲\\nDDPM epoch 1/2: noise MSE 0.25; validation 0.5\\n'.encode('utf-8')\nfor v in b:\n sys.stdout.buffer.write(bytes([v]));sys.stdout.buffer.flush();time.sleep(.002)"
    win.launch(['-u','-c',code])
    pump(app,lambda:win.process.state()==C.QProcess.ProcessState.NotRunning,timeout=30);app.processEvents()
    assert '中文文件名：试样甲' in win.log.toPlainText() and '\ufffd' not in win.log.toPlainText()
    assert 'DDPM epoch 1/2: noise MSE 0.25; validation 0.5' in win.log.toPlainText()
    assert win.task_status.text()=='成功' and win.task_status.property('status_code')=='SUCCESS'
    assert not errors
    win.close();app.processEvents()


def test_gui_child_overrides_inherited_non_utf8_encoding(tmp_path,monkeypatch):
    app,win,errors=window(tmp_path)
    monkeypatch.setenv('PYTHONIOENCODING','gbk')
    monkeypatch.setenv('PYTHONUTF8','0')
    win.launch(['-u','-c',"print('中文路径：试样甲')"])
    pump(app,lambda:win.process.state()==C.QProcess.ProcessState.NotRunning,timeout=30);app.processEvents()
    assert '中文路径：试样甲' in win.log.toPlainText()
    assert '\ufffd' not in win.log.toPlainText() and not errors
    win.close();app.processEvents()
