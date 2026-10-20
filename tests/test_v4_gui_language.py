import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
pytest.importorskip('PySide6')
from PySide6 import QtWidgets as W
from gui_qt import Studio


def test_language_round_trip_keeps_user_state_and_navigation(tmp_path):
    app=W.QApplication.instance() or W.QApplication([])
    win=Studio();win.preferences=tmp_path/'preferences.json'
    win.lang='en';win.translate();win.nav.setCurrentRow(3)
    win.mode.setCurrentText('v3_schmid');win.roi.setText('1 2 30 40')
    win.axis.setText('1 0 0');win.block.setValue(17)
    win.font.setCurrentText('Times New Roman');win.out.setText(str(tmp_path/'个人数据'))
    win.formats['svg'].setChecked(True);win.weight.setCurrentIndex(1)
    before=win.options();nav=win.nav.currentRow();tab=win.bottomtabs.currentIndex()
    win.toggle_language()
    assert win.lang=='zh' and win.nav.currentRow()==nav
    assert win.mode.currentText()=='v3_schmid' and win.options()==before
    assert win.bottomtabs.currentIndex()==tab
    assert win.block.value()==17 and win.roi.text()=='1 2 30 40'
    assert win.open_cache.text()=='打开缓存'
    assert win.budget.parent().layout().labelForField(win.budget).text()=='内存预算'
    assert win.table.horizontalHeaderItem(0).text()=='来源'
    assert 'MTEX' in win.help.text() and any('\u4e00'<=c<='\u9fff' for c in win.help.text())
    win.toggle_language()
    assert win.options()==before and win.nav.currentRow()==nav
    assert win.open_cache.text()=='Open cache' and win.table.horizontalHeaderItem(0).text()=='Source'
    assert not any('\u4e00'<=c<='\u9fff' for c in win.help.text())
    assert win.out.text()==str(tmp_path/'个人数据')
    win.close();app.processEvents()
