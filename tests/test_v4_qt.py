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
