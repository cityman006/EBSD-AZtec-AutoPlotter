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
