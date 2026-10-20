import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
pytest.importorskip('PySide6')
from PySide6 import QtWidgets as W
from gui_qt import Studio, MapView


def test_qt_studio_theme_language_snapshot_and_roi(tmp_path):
    app=W.QApplication.instance() or W.QApplication([])
    win=Studio();win.preferences=tmp_path/'settings.json'
    win.set_theme(False);win.set_theme(True);win.toggle_language()
    assert win.nav.count()==12 and win.mode.count()>=6
    win.set_roi((2,3,4,5));assert win.roi.text()=='2 3 4 5'
    win.records=[{'nx':3,'ny':3}]
    with pytest.raises(ValueError,match='exceeds'):win.options()
    win.records=[];win.roi.clear()
    snapshot=win.options();win.block.setValue(17)
    assert snapshot[snapshot.index('--batch')+1]=='32'
    win.close();app.processEvents()


def test_context_parameter_pages_are_exclusive_and_keep_slice_values(tmp_path):
    app=W.QApplication.instance() or W.QApplication([])
    win=Studio();win.preferences=tmp_path/'settings.json';win.show();app.processEvents()
    assert win.nav.count()==12
    assert len(win.parameter_pages)==len(win.parameter_forms)==12
    assert win.param_stack.objectName()=='workspaceParameterStack'
    assert win.content_stack.objectName()=='workspaceContentStack'
    win.slice_y.setValue(3);win.slice_x.setValue(5)
    win.out.setText(str(tmp_path/'outputs'));win.paths=['sample.h5oina']
    for row in range(win.nav.count()):
        win.nav.setCurrentRow(row);app.processEvents()
        assert win.param_stack.currentIndex()==row
        assert win.content_stack.currentIndex()==row
        assert win.parameter_pages[row].isVisible()
        assert all(page.isHidden() for index,page in win.parameter_pages.items() if index!=row)
        assert win.panel.windowTitle()
    win.nav.setCurrentRow(6);app.processEvents()
    assert win.slice_x.value()==5 and win.slice_y.value()==3
    assert win.slice_mode.isVisible() and win.slice_y.isVisible() and win.slice_x.isVisible()
    # Slicing controls live only below the slicing workspace; batch-only controls
    # are owned by a different stacked page and cannot reserve visible space.
    assert not win.parameter_pages[6].findChildren(W.QPushButton) or all(
        button not in win.parameter_pages[6].findChildren(W.QPushButton)
        for button in (win.run_button,win.cancel_button,win.resume_button))
    assert not win.run_button.isVisible() and not win.budget.isVisible()
    win.nav.setCurrentRow(7);app.processEvents()
    assert win.eds_block.isVisible() and win.eds_rate.isVisible()
    assert win.out.text()==str(tmp_path/'outputs') and win.paths==['sample.h5oina']
    win.close();app.processEvents()


def test_context_parameter_preferences_restore_after_restart(tmp_path,monkeypatch):
    import gui_qt
    app=W.QApplication.instance() or W.QApplication([])
    (tmp_path/'config').mkdir()
    monkeypatch.setattr(gui_qt,'ROOT',tmp_path)
    first=Studio()
    first.slice_y.setValue(3);first.slice_x.setValue(5)
    first.font.setCurrentText('Times New Roman')
    first.layout_select.setCurrentText('3 × 3')
    first.nav.setCurrentRow(6);first.out.setText(str(tmp_path/'out'));first.save_settings()
    first.close();app.processEvents()
    second=Studio()
    assert (second.slice_y.value(),second.slice_x.value())==(3,5)
    assert second.font.currentText()=='Times New Roman'
    assert second.layout_select.currentText()=='3 × 3'
    assert second.nav.currentRow()==6 and second.out.text()==str(tmp_path/'out')
    second.close();app.processEvents()


def test_context_switch_does_not_mutate_an_active_batch_snapshot(tmp_path,monkeypatch):
    import copy
    import gui_qt
    app=W.QApplication.instance() or W.QApplication([])
    win=Studio();win.preferences=tmp_path/'settings.json'
    win.paths=['sample.h5oina'];win.out.setText(str(tmp_path/'out'));win.nav.setCurrentRow(5)
    captured={}
    expected=win.options()
    def fake_create_batch(paths,output,args,**kwargs):
        captured.update(paths=list(paths),output=output,args=copy.deepcopy(args),kwargs=dict(kwargs))
        return tmp_path/'batch'
    monkeypatch.setattr(gui_qt,'create_batch',fake_create_batch)
    win.launch=lambda args,**kwargs:captured.update(launch_args=list(args),launch_kwargs=dict(kwargs))
    win.start_batch()
    frozen=copy.deepcopy(captured)
    win.nav.setCurrentRow(6);win.slice_x.setValue(7);win.block.setValue(48);win.out.setText(str(tmp_path/'other'))
    assert captured==frozen
    assert captured['args']==expected and captured['paths']==['sample.h5oina']
    win.close();app.processEvents()


def test_cached_action_owns_cancel_control(tmp_path):
    from pathlib import Path
    app=W.QApplication.instance() or W.QApplication([])
    win=Studio();win.preferences=tmp_path/'preferences.json';win.out.setText(str(tmp_path))
    win.viewer.cache=Path('features.h5');captured={}
    win.launch=lambda args,**kw:captured.update(args=args,**kw)
    win.cached_action('quantitative',{'formats':['png']})
    assert captured['control']==win.control
    assert captured['args'][captured['args'].index('--cache')+1]=='features.h5'
    win.viewer.cache=None;win.close();app.processEvents()


def test_lazy_qt_tiles_aspect_switch_and_stale_result(tmp_path):
    import json
    import time
    import h5py
    import numpy as np
    from PySide6 import QtCore as C
    app=W.QApplication.instance() or W.QApplication([])
    path=tmp_path/'features.h5'
    h,w=90,130
    raw=np.zeros((7,h,w),np.float32);raw[4]=7;raw[5]=2;raw[6]=173
    with h5py.File(path,'w') as f:
        f['features']=raw;f['indexed']=np.ones((h,w),np.uint8);f['kam_valid']=np.ones((h,w),np.uint8)
        f.attrs['metadata_json']=json.dumps({'x_step_um':2,'y_step_um':3,
            'phase_specs':{'7':{'rgb':[12,34,56]}}})
    view=MapView();view.resize(600,400);view.show();view.load_cache(path,'BC')
    def wait_tiles():
        limit=time.monotonic()+10
        while not view.tile_items and time.monotonic()<limit:
            app.processEvents();time.sleep(.01)
        assert view.tile_items
    wait_tiles()
    assert view.transform().m22()/view.transform().m11()==pytest.approx(1.5)
    assert view.width_px==w and view.height_px==h
    assert all(max(i.pixmap().width(),i.pixmap().height())<=512 for i in view.tile_items)
    assert view.tile_items[0].pixmap().toImage().pixelColor(3,3).red()==173
    stale=view.generation
    view.load_cache(path,'Phase');wait_tiles()
    assert view.tile_items[0].pixmap().toImage().pixelColor(3,3).getRgb()[:3]==(12,34,56)
    current=view.tile_items[:]
    view.accept_tiles({'generation':stale,'tiles':[]})
    assert view.tile_items==current
    view.generation+=1;view.tile_timer.stop();view.close()
    assert C.QThreadPool.globalInstance().waitForDone(10000)
    app.processEvents()


def test_crashed_zero_exit_is_not_success_and_ddpm_steps_match_backend(tmp_path):
    from PySide6 import QtCore as C
    app=W.QApplication.instance() or W.QApplication([])
    win=Studio();win.preferences=tmp_path/'settings.json'
    win.read_log=lambda:None;win.refresh_queue=lambda:None;win.history=lambda:None
    win.finished(0,C.QProcess.ExitStatus.CrashExit)
    assert win.task_status.property('status_code')=='FAILED'
    assert 'SUCCESS' not in win.log.toPlainText()
    assert win.diff_steps.minimum()==4
    win.close();app.processEvents()


def test_tile_worker_ignores_destroyed_gui_signal_at_shutdown(tmp_path):
    from shiboken6 import delete
    from gui_qt import TileTask
    from test_v4_multiview_gui import cache_file
    app=W.QApplication.instance() or W.QApplication([])
    path=tmp_path/'features.h5';cache_file(path)
    task=TileTask(path,'BC',1,[(0,0,4,4)],0,lambda:True)
    delete(task.signals)
    task.run()  # No GUI remains to receive this obsolete tile result.


def test_qt_real_child_unicode_log_overrides_hostile_gbk_parent(tmp_path,monkeypatch):
    app=W.QApplication.instance() or W.QApplication([])
    win=Studio();win.preferences=tmp_path/'settings.json'
    monkeypatch.setenv('PYTHONIOENCODING','gbk');monkeypatch.setenv('PYTHONUTF8','0')
    expected='中文样品：分析完成'
    try:
        win.launch(['-u','-c',f'print({expected!r},flush=True)'])
        assert win.process.waitForFinished(5000)
        app.processEvents()
        assert win.process.exitCode()==0
        output=win.log.toPlainText()
        assert expected in output and '\ufffd' not in output
    finally:
        if win.busy():win.process.kill();win.process.waitForFinished(5000)
        win.close();app.processEvents()
