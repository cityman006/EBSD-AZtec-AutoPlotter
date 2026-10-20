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
