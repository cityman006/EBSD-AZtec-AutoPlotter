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


def test_slicing_options_and_dynamic_preview_are_bilingual_without_parameter_changes(tmp_path):
    from test_v4_multiview_gui import cache_file
    app=W.QApplication.instance() or W.QApplication([])
    path=tmp_path/'features.h5';cache_file(path)
    win=Studio();win.preferences=tmp_path/'preferences.json';errors=[];win.error=lambda e:errors.append(str(e))
    win.lang='en';win.translate();win.workspace.load_cache(path)
    win.preview_patches();assert not errors
    assert win.tile_table.rowCount()==15
    win.slice_edge.setCurrentIndex(win.slice_edge.findData('pad-with-mask'))
    expected=win.slice_config()
    win.toggle_language()
    assert win.slice_config()==expected
    assert win.slice_mode.currentText()=='网格划分'
    assert win.slice_edge.currentText()=='填充并保存有效掩膜'
    assert win.tile_table.horizontalHeaderItem(0).text()=='启用'
    assert '个几何切片' in win.tile_note.text()
    win.toggle_language()
    assert win.slice_config()==expected and 'geometric patches' in win.tile_note.text()
    win.close();app.processEvents()


def test_patch_selection_cannot_cross_source_cache_or_changed_file(tmp_path):
    from test_v4_multiview_gui import cache_file
    app=W.QApplication.instance() or W.QApplication([])
    first=tmp_path/'first.h5';second=tmp_path/'second.h5';cache_file(first);cache_file(second)
    win=Studio();win.preferences=tmp_path/'preferences.json';errors=[];actions=[]
    win.error=lambda e:errors.append(str(e));win.cached_action=lambda *a:actions.append(a)
    win.workspace.load_cache(first);win.preview_patches();assert not errors
    win.workspace.load_cache(second);win.export_patches()
    assert not actions and 'Source cache changed' in errors[-1]
    win.preview_patches();win.export_patches();assert len(actions)==1
    stat=second.stat();os.utime(second,ns=(stat.st_atime_ns,stat.st_mtime_ns+10000000))
    win.export_patches();assert len(actions)==1 and 'Source cache changed' in errors[-1]
    # A visible source switch also invalidates the old table/button immediately.
    win.workspace.load_cache(first);win.sync_active_view()
    assert not win.export_patch_button.isEnabled() and win.tile_table.rowCount()==0
    win.close();app.processEvents()
