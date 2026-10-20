import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
pytest.importorskip('PySide6')
import json
import time
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
from PySide6 import QtCore as C, QtWidgets as W, QtTest
from ebsd_autoplot.registration_editor import LandmarkRegistrationDialog, LandmarkView, _WORKERS, read_target_preview
from ebsd_autoplot.registration import file_stamp


def test_registration_editor_public_interface_exists():
    from ebsd_autoplot.registration_editor import LandmarkRegistrationDialog
    assert callable(LandmarkRegistrationDialog.registration_config)


@pytest.fixture(scope='module')
def app():
    return W.QApplication.instance() or W.QApplication([])


def pump(app, predicate, timeout=10):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        app.processEvents()
        if predicate(): return
        QtTest.QTest.qWait(10)
    raise AssertionError('Preview worker did not complete: ' + str(predicate()))


@pytest.fixture
def sources(tmp_path):
    target = tmp_path/'target_features.h5'
    source = tmp_path/'原始EDS.h5oina'
    h, w = 80, 100
    with h5py.File(target, 'w') as f:
        values = np.zeros((7, h, w), np.float32)
        values[6] = np.indices((h, w))[1]
        f['features'] = values; f['indexed'] = np.ones((h, w), np.uint8)
        f.attrs['metadata_json'] = json.dumps({'x_step_um': .2, 'y_step_um': .3})
    with h5py.File(source, 'w') as f:
        for scan in ('1', '2'):
            for modality in ('EBSD', 'EDS'):
                header = f.create_group(f'{scan}/{modality}/Header')
                data = f.create_group(f'{scan}/{modality}/Data')
                for name, value, unit in [('X Cells', w, 'px'), ('Y Cells', h, 'px'), ('X Step', .2, 'um'), ('Y Step', .3, 'um')]:
                    ds = header.create_dataset(name, data=[value]); ds.attrs['Unit'] = unit
                if modality == 'EDS':
                    windows = data.create_group('Window Integral')
                    for channel, scale in (('Ti Kα1', 1.), ('Al Kα1', 2.)):
                        ds = windows.create_dataset(channel, data=(np.arange(h*w, dtype=np.float32)*scale))
                        ds.attrs['Unit'] = 'cps'
    return source, target


@pytest.fixture
def dialog(app, sources):
    source, target = sources
    dialog = LandmarkRegistrationDialog(target, source_path=source)
    dialog.show()
    try:
        pump(app, lambda: all(dialog._metadata.values()) or dialog.details.isVisible())
        assert all(dialog._metadata.values()), dialog.details.toPlainText()
        yield dialog
    finally:
        dialog.reject()
        pump(app, lambda: not any(w.isRunning() for w in _WORKERS))
        app.processEvents()
        dialog.deleteLater(); app.processEvents()


def known_pairs(dialog):
    source = np.array([[10., 10.], [60., 10.], [10., 50.], [60., 50.]])
    target = source + [5., 3.]
    dialog._set_pairs(source, target)
    return source, target


def test_actual_async_previews_numeric_fit_and_apply_config(dialog):
    source, target = known_pairs(dialog)
    assert dialog.fit_landmarks(), dialog.details.toPlainText()
    config = dialog.registration_config()
    np.testing.assert_allclose(config['matrix'], [[1, 0, 5], [0, 1, 3], [0, 0, 1]], atol=1e-10)
    np.testing.assert_array_equal(config['source_xy'], source)
    np.testing.assert_array_equal(config['target_xy'], target)
    assert config['fit_qc']['rms_target_px'] < 1e-10
    assert config['fit_qc']['independently_validated'] is False
    assert config['source_shape_hw'] == config['target_shape_hw'] == [80, 100]
    assert config['source_stamp'] == file_stamp(config['source_path'])
    assert config['target_cache_stamp'] == file_stamp(dialog.cache_path)
    assert dialog.source_view.transform().m22()/dialog.source_view.transform().m11() == pytest.approx(1.5)
    assert 'cps' in dialog.view_info['source'].text()
    dialog.accept()
    assert dialog.result() == W.QDialog.DialogCode.Accepted
    assert dialog.registration_config()['matrix'] == config['matrix']


def test_real_qt_paired_click_uses_native_centres_after_zoom(dialog, app):
    for view in (dialog.source_view, dialog.target_view):
        view.user_zoom = True
        view.resetTransform(); view.scale(4., 6.); view.centerOn(45., 35.)
    for source, target in [((10, 10), (15, 13)), ((60, 10), (65, 13)), ((10, 50), (15, 53))]:
        for view, point in ((dialog.source_view, source), (dialog.target_view, target)):
            position = view.mapFromScene(C.QPointF(*point))
            QtTest.QTest.mouseClick(view.viewport(), C.Qt.MouseButton.LeftButton, pos=position)
            app.processEvents()
    assert dialog.table.rowCount() == 3
    source, target = dialog._pairs()
    np.testing.assert_allclose(source, [[10, 10], [60, 10], [10, 50]], atol=.26)
    np.testing.assert_allclose(target, [[15, 13], [65, 13], [15, 53]], atol=.26)
    assert dialog.fit_landmarks(), dialog.details.toPlainText()
    np.testing.assert_allclose(np.asarray(dialog.registration_config()['matrix'])[:2, 2], [5, 3], atol=.4)


def test_numeric_edits_source_channel_and_scan_invalidate_fit(dialog, app):
    known_pairs(dialog); assert dialog.fit_landmarks()
    dialog.table.item(0, 0).setText('11')
    assert not dialog.apply_button.isEnabled() and dialog._fit is None
    assert dialog.fit_landmarks()
    index = 1-dialog.channel_combo.currentIndex()
    dialog.channel_combo.setCurrentIndex(index)
    assert dialog._fit is None and not dialog.apply_button.isEnabled()
    pump(app, lambda: dialog._metadata['source'] is not None)
    assert dialog.fit_landmarks()
    dialog.scan_path_edit.setText('2/EBSD')
    assert dialog._fit is None and dialog._metadata['source'] is None
    assert dialog.channel_combo.count() == 0
    dialog.load_source(); pump(app, lambda: dialog._metadata['source'] is not None)
    assert dialog._metadata['source']['scan_path'] == '/2/EBSD'
    assert dialog.scan_path_edit.text() == '/2/EBSD' and dialog.fit_landmarks()


@pytest.mark.parametrize('case', ['too_few', 'collinear', 'singular_target', 'nonfinite', 'outside_source', 'outside_target'])
def test_invalid_pairs_never_enable_apply(dialog, case):
    source, target = known_pairs(dialog)
    if case == 'too_few': source, target = source[:2], target[:2]
    elif case == 'collinear': source[:, 1] = source[:, 0]
    elif case == 'singular_target': target[:, 1] = 12
    elif case == 'nonfinite': source[0, 0] = np.nan
    elif case == 'outside_source': source[0, 0] = 100
    else: target[0, 1] = -1
    dialog._set_pairs(source, target)
    assert not dialog.fit_landmarks()
    assert not dialog.apply_button.isEnabled() and dialog._fit is None
    dialog.accept()
    assert dialog.result() != W.QDialog.DialogCode.Accepted


def test_save_load_json_roundtrip_refits_and_never_replaces(dialog, tmp_path, app):
    known_pairs(dialog); assert dialog.fit_landmarks()
    config = dialog.registration_config()
    path = dialog.save_config(tmp_path/'配准结果.json'); original = path.read_bytes()
    with pytest.raises(FileExistsError): dialog.save_config(path)
    assert path.read_bytes() == original
    dialog.clear_pairs(); assert dialog._fit is None
    dialog.load_config(path)
    pump(app, lambda: dialog._fit is not None or dialog.details.isVisible())
    assert dialog._fit is not None, dialog.details.toPlainText()
    np.testing.assert_allclose(dialog.registration_config()['matrix'], config['matrix'])
    assert dialog.table.rowCount() == 4
    bad = dict(config); bad['matrix'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    corrupt = tmp_path/'invalid.json'; corrupt.write_text(json.dumps(bad))
    with pytest.raises(ValueError): dialog.load_config(corrupt)


@pytest.mark.parametrize('kind', ['source', 'target'])
def test_changed_source_or_target_rejects_apply_and_load(dialog, tmp_path, kind):
    known_pairs(dialog); assert dialog.fit_landmarks()
    saved = dialog.save_config(tmp_path/'before.json')
    path = dialog._metadata[kind]['source_stamp']['path']
    old = os.stat(path); os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns+1000000))
    with pytest.raises(ValueError): dialog.registration_config()
    with pytest.raises(ValueError): dialog.load_config(saved)
    dialog.accept(); assert dialog.result() != W.QDialog.DialogCode.Accepted


def test_language_changes_do_not_change_numeric_registration(dialog):
    known_pairs(dialog); assert dialog.fit_landmarks()
    before = dialog.registration_config()
    dialog.set_language('zh')
    assert dialog.windowTitle() == 'EDS 地标配准'
    assert dialog.table.horizontalHeaderItem(0).text() == '源 X'
    assert '外部独立校核' in dialog.matrix_text.toPlainText()
    assert '目标像素' in dialog.matrix_text.toPlainText() and 'target px' not in dialog.matrix_text.toPlainText()
    assert dialog.registration_config() == before
    dialog.set_language('en')
    assert dialog.registration_config() == before
    assert dialog.source_path_edit.text().endswith('原始EDS.h5oina')


def test_bounded_preview_stride_and_pixel_centre_mapping(tmp_path, app):
    path = tmp_path/'large_cache.h5'
    h, w = 2101, 3100
    with h5py.File(path, 'w') as f:
        f.create_dataset('features', shape=(7, h, w), dtype='u1', fillvalue=100, chunks=(1, 64, 64))
        f.create_dataset('indexed', shape=(h, w), dtype='u1', fillvalue=1, chunks=(64, 64))
        f.attrs['metadata_json'] = json.dumps({'x_step_um': .5, 'y_step_um': 1.})
    data = read_target_preview(path)
    assert data['shape_hw'] == [h, w]
    assert data['sampling_stride'] == 4 and max(data['image'].shape) <= 1024
    view = LandmarkView(); view.resize(800, 600); view.set_preview(data); view.show(); app.processEvents()
    item = next(item for item in view.scene().items() if isinstance(item, W.QGraphicsPixmapItem))
    native = item.mapToScene(C.QPointF(.5, .5))
    assert native.x() == native.y() == 0
    assert view.transform().m22()/view.transform().m11() == pytest.approx(2.)
    view.user_zoom = True; view.resetTransform(); view.scale(2., 4.); view.centerOn(1200., 800.)
    point = view.mapFromScene(C.QPointF(1200., 800.))
    assert view.native_position(point) == pytest.approx((1200., 800.), abs=.26)
    view.close(); app.processEvents()


def test_worker_cancel_and_obsolete_results_do_not_reopen_dialog(app, sources, monkeypatch):
    import ebsd_autoplot.eds as eds
    from ebsd_autoplot.storage import check_cancelled
    source, target = sources
    stopped = []
    def slow(*args, **kwargs):
        try:
            for _ in range(100):
                check_cancelled(); time.sleep(.01)
        finally: stopped.append(True)
    monkeypatch.setattr(eds, 'read_eds_preview', slow)
    dialog = LandmarkRegistrationDialog(target, source_path=source)
    dialog.show(); app.processEvents()
    # Ensure the source worker is in its cancellable loop before closing.
    QtTest.QTest.qWait(40); app.processEvents()
    dialog.reject()
    pump(app, lambda: not any(w.isRunning() for w in _WORKERS))
    assert stopped and not dialog.isVisible()
    assert dialog.result() == W.QDialog.DialogCode.Rejected
    assert dialog._metadata['source'] is None and dialog._fit is None
    dialog.deleteLater(); app.processEvents()


def test_pair_delete_clear_and_source_selection_never_reuses_fit(dialog, app, tmp_path):
    known_pairs(dialog); assert dialog.fit_landmarks()
    previous = dialog._metadata['source']
    previous_token = dialog._tokens['source']
    dialog.table.selectRow(1); dialog.delete_selected()
    assert dialog.table.rowCount() == 3 and dialog._fit is None
    assert dialog.fit_landmarks()
    dialog.add_pair()
    assert dialog.table.rowCount() == 4 and not dialog.fit_landmarks()
    dialog.table.selectRow(3); dialog.delete_selected(); assert dialog.fit_landmarks()
    dialog.source_path_edit.setText(str(tmp_path/'a_different_source.h5oina'))
    assert dialog._metadata['source'] is None and dialog._fit is None
    assert dialog.channel_combo.count() == 0 and not dialog.apply_button.isEnabled()
    dialog._preview_loaded('source', previous_token, previous)
    assert dialog._metadata['source'] is None  # Old worker result cannot restore the former source.
    dialog.clear_pairs(); assert dialog.table.rowCount() == 0
    assert not dialog.apply_button.isEnabled()


def test_dictionary_reload_before_target_ready_and_original_sources_unchanged(dialog, app):
    known_pairs(dialog); assert dialog.fit_landmarks()
    config = dialog.registration_config()
    before_source = file_stamp(config['source_path']); before_target = file_stamp(dialog.cache_path)
    other = LandmarkRegistrationDialog(dialog.cache_path)
    try:
        returned = other.load_config(config)
        assert returned == config or returned['matrix'] == config['matrix']
        pump(app, lambda: other._fit is not None or other.details.isVisible())
        assert other._fit is not None, other.details.toPlainText()
        np.testing.assert_allclose(other.registration_config()['matrix'], config['matrix'])
        assert other.table.rowCount() == 4
        assert file_stamp(config['source_path']) == before_source
        assert file_stamp(dialog.cache_path) == before_target
    finally:
        other.reject(); pump(app, lambda: not any(w.isRunning() for w in _WORKERS))
        other.deleteLater(); app.processEvents()


def test_numeric_edit_during_async_project_load_is_not_overwritten(dialog, app):
    known_pairs(dialog); assert dialog.fit_landmarks()
    config = dialog.registration_config()
    dialog.load_config(config)
    dialog.table.item(0, 0).setText('12')
    assert dialog._pending_load is None and dialog._fit is None
    pump(app, lambda: dialog._metadata['source'] is not None)
    assert dialog.table.item(0, 0).text() == '12'
    assert dialog._fit is None  # Explicit Fit is still required after the edit.


def test_application_and_interpreter_exit_join_only_owned_readers(sources):
    _, target = sources
    root = Path(__file__).resolve().parents[1]
    # A real subprocess is necessary: deleting a running QThread can fail only
    # during Qt/Python teardown, after otherwise green in-process assertions.
    code = """
import os,sys,time
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6 import QtCore as C,QtWidgets as W
import ebsd_autoplot.eds as eds
from ebsd_autoplot.registration_editor import LandmarkRegistrationDialog,PreviewWorker,_WORKERS
eds.read_eds_preview=lambda *a,**k:time.sleep(.5)
app=W.QApplication([])
if sys.argv[2]=='application':
    dialog=LandmarkRegistrationDialog(sys.argv[1],source_path=sys.argv[1])
    dialog.show()
    C.QTimer.singleShot(50,app.quit)
    app.exec()
else:
    worker=PreviewWorker('source',1,sys.argv[1],'/1/EBSD')
    _WORKERS.add(worker)
    worker.start()
    time.sleep(.05)
print('safe exit',flush=True)
"""
    for mode in ('application', 'interpreter'):
        result = subprocess.run([sys.executable, '-c', code, str(target), mode], cwd=root,
                                capture_output=True, timeout=20)
        assert result.returncode == 0, result.stderr.decode('utf-8', 'replace')
        assert b'safe exit' in result.stdout
