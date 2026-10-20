"""Explicit landmark registration of a native EDS map to an EBSD cache.

Only bounded, sampled previews are read here. Landmarks and the fitted affine
remain in original pixel-centre coordinates, never display-image coordinates.
No analysis, resampling, raw-source editing or registration inference is done.
"""
from __future__ import annotations

import copy
import atexit
import json
import math
import os
from pathlib import Path
import uuid

import h5py
import numpy as np
from PySide6 import QtCore as C, QtGui as G, QtWidgets as W

from .registration import fit_affine
from .storage import cancellation, check_cancelled, Cancelled


TRANSLATIONS = {
    "EDS landmark registration": "EDS 地标配准",
    "Source EDS H5OINA": "源 EDS H5OINA",
    "Choose source": "选择源文件",
    "Scan path": "扫描路径",
    "EDS window": "EDS 能窗",
    "Read preview": "读取预览",
    "Target EBSD BC cache": "目标 EBSD BC 缓存",
    "Source EDS — native pixels": "源 EDS — 原始像素",
    "Target BC — native pixels": "目标 BC — 原始像素",
    "Source X": "源 X", "Source Y": "源 Y",
    "Target X": "目标 X", "Target Y": "目标 Y",
    "Add row": "添加一行", "Delete selected": "删除所选行", "Clear pairs": "清空点对",
    "Fit affine": "拟合仿射矩阵", "Save JSON": "保存 JSON", "Load JSON": "载入 JSON",
    "Apply registration": "应用配准", "Cancel": "取消",
    "Click a feature in each view to add a matched pair. Drag to pan; wheel to zoom. Coordinates may also be edited in the table.":
        "在两幅图中分别点击同一特征以添加匹配点。拖动可平移，滚轮可缩放，也可在表格中编辑坐标。",
    "At least 3 noncollinear pairs are required. Fit residuals use the fitted landmarks; they are NOT independent validation. Verify on independent features before scientific use.":
        "至少需要 3 对不共线匹配点。拟合残差来自参与拟合的地标，不是独立校核。用于科学分析前，请核对独立特征。",
    "No fitted registration": "尚未拟合配准",
    "Reading bounded previews…": "正在读取有界预览…",
    "Source preview ready; select corresponding landmarks.": "源预览已就绪，请选择对应地标。",
    "Select the matching point in the target BC view.": "请在目标 BC 图中选择对应点。",
    "Select the matching point in the source EDS view.": "请在源 EDS 图中选择对应点。",
    "Landmarks changed; fit again before Apply.": "地标已改变，应用前请重新拟合。",
    "Source selection changed; reload preview and fit again.": "源选择已改变，请重新读取预览并拟合。",
    "Registration loaded and refitted from saved landmarks.": "配准已载入，并已根据保存地标重新拟合。",
    "Preview contrast is relative; source values and units are retained.": "预览仅调整显示对比度，源数值和单位保持不变。",
    "Unknown physical step; view uses native pixel aspect.": "物理步长未知，视图使用原始像素比例。",
    "Registered coordinates are NOT externally validated.": "配准坐标尚未通过外部独立校核。",
    "Poor landmark distribution / extreme transform: inspect independent features.": "地标分布不佳或变换尺度极端，请检查独立特征。",
    "No source selected": "尚未选择源文件",
    "Error": "错误", "Details": "详细信息",
    "Registration cannot be applied. See details below.": "无法应用配准，请查看下方详细信息。",
    "Choose registration JSON": "选择配准 JSON",
    "Save new registration JSON": "另存新的配准 JSON",
    "Source": "源", "Target": "目标",
    "Per-pair residual [target px]": "各点对残差［目标像素］",
    "Design condition": "地标设计矩阵条件数",
    "RMS {rms:.6g} target px | max {maximum:.6g} target px": "RMS {rms:.6g} 目标像素 | 最大值 {maximum:.6g} 目标像素",
}


def _stamp(path):
    from .registration import file_stamp
    return file_stamp(path)


def _validated(config, cache, source_shape=None):
    from .registration import validate_landmark_config
    return validate_landmark_config(config, cache_path=cache, source_shape=source_shape)


def read_target_preview(path, max_side=1024):
    """Read bounded BC/mask hyperslabs without analysing any raw H5 file."""
    if isinstance(max_side, bool) or int(max_side) != max_side or not 1 <= int(max_side) <= 1024:
        raise ValueError('Preview max_side must be an integer from 1 to 1024')
    path = Path(path).resolve()
    before = _stamp(path)
    check_cancelled()
    with h5py.File(path, 'r') as stream:
        shape = tuple(map(int, stream['indexed'].shape))
        if len(shape) != 2 or min(shape) < 1 or stream['features'].shape != (7, *shape):
            raise ValueError('Target must be a standard seven-channel 2D EBSD cache')
        metadata = json.loads(stream.attrs['metadata_json'])
        steps = [float(metadata['x_step_um']), float(metadata['y_step_um'])]
        if not np.isfinite(steps).all() or min(steps) <= 0:
            raise ValueError('Target cache requires explicit positive physical grid steps')
        stride = max(1, int(math.ceil(max(shape) / max_side)))
        image = np.asarray(stream['features'][6, ::stride, ::stride], dtype=float)
        mask = np.asarray(stream['indexed'][::stride, ::stride], dtype=bool)
        image[~mask] = np.nan
    check_cancelled()
    if _stamp(path) != before:
        raise ValueError('Target cache changed while reading its preview')
    return {'image': image, 'shape_hw': list(shape), 'source_stamp': before,
            'sampling_stride': stride, 'step_um_xy': steps, 'unit': 'source BC', 'channel': 'BC'}


# Parentless threads remain referenced until finished even when a dialog closes.
# Cancellation is cooperative; never terminate an HDF5 reader in the middle of I/O.
_WORKERS = set()


def shutdown_preview_workers():
    """Finish only our owned readers before Qt/Python destroys QThread objects.

    Request interruption first for every reader, then wait for their bounded
    read operations to return. Do not terminate an HDF5 I/O thread or affect
    any scientific analysis process. QThread.wait releases the Python GIL.
    """
    from shiboken6 import isValid
    workers = list(_WORKERS)
    for worker in workers:
        if isValid(worker): worker.requestInterruption()
    for worker in workers:
        if isValid(worker) and worker is not C.QThread.currentThread(): worker.wait()
    _WORKERS.difference_update(workers)


atexit.register(shutdown_preview_workers)


class PreviewWorker(C.QThread):
    loaded = C.Signal(str, int, object)
    failed = C.Signal(str, int, str)

    def __init__(self, kind, token, path, scan_path=None, channel=None):
        super().__init__()
        self.kind, self.token, self.path = kind, token, str(path)
        self.scan_path, self.channel = scan_path, channel

    def run(self):
        try:
            with cancellation(self.isInterruptionRequested):
                check_cancelled()
                if self.kind == 'target':
                    result = read_target_preview(self.path)
                else:
                    from .eds import read_eds_preview
                    result = read_eds_preview(self.path, self.scan_path, self.channel, max_side=1024)
                check_cancelled()
            self.loaded.emit(self.kind, self.token, result)
        except Cancelled:
            pass
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(self.kind, self.token, f'{type(exc).__name__}: {exc}')


class LandmarkView(W.QGraphicsView):
    """Sampled pixmap in native pixel-centre coordinates, with physical aspect."""
    point_selected = C.Signal(tuple)
    position_changed = C.Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(W.QGraphicsScene(self))
        self.setDragMode(W.QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setTransformationAnchor(W.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.shape_hw = None
        self.aspect = 1.
        self.user_zoom = False
        self._press = None
        self._markers = []

    def clear_preview(self):
        self.shape_hw = None
        self._markers = []
        self.scene().clear()
        self.scene().setSceneRect(C.QRectF())

    def set_preview(self, data):
        image = np.asarray(data['image'], dtype=float)
        shape = tuple(map(int, data['shape_hw']))
        if (image.ndim != 2 or len(shape) != 2 or min(shape) <= 0
                or max(image.shape) > 1024):
            raise ValueError('Preview must be bounded 2D numeric data with a native shape')
        valid = np.isfinite(image)
        gray = np.zeros(image.shape, dtype=np.uint8)
        if valid.any():
            low, high = np.percentile(image[valid], [1, 99])
            if high > low:
                gray[valid] = (np.clip((image[valid] - low) / (high - low), 0, 1) * 255).astype(np.uint8)
            else:
                gray[valid] = 128
        gray = np.ascontiguousarray(gray)
        qimage = G.QImage(gray.data, gray.shape[1], gray.shape[0], gray.strides[0], G.QImage.Format.Format_Grayscale8).copy()
        self.clear_preview()
        self.shape_hw = shape
        steps = data.get('step_um_xy')
        self.aspect = float(steps[1]) / float(steps[0]) if steps is not None else 1.
        if not np.isfinite(self.aspect) or self.aspect <= 0:
            raise ValueError('Preview physical steps must be positive and finite')
        stride = int(data['sampling_stride'])
        if stride < 1 or image.shape != tuple((size + stride - 1) // stride for size in shape):
            raise ValueError('Preview sampling stride disagrees with native grid')
        item = self.scene().addPixmap(G.QPixmap.fromImage(qimage))
        item.setTransform(G.QTransform.fromScale(stride, stride))
        # QPixmap pixel centres are k+0.5; sampled native centres are k*stride.
        item.setPos(-stride / 2, -stride / 2)
        self.scene().setSceneRect(-.5, -.5, shape[1], shape[0])
        self.user_zoom = False
        self.fit_native()

    def fit_native(self):
        if self.shape_hw is None:
            return
        h, w = self.shape_hw
        scale = min(max(1, self.viewport().width()-8) / w,
                    max(1, self.viewport().height()-8) / (h*self.aspect))
        self.resetTransform()
        self.scale(scale, scale*self.aspect)
        self.centerOn((w-1)/2, (h-1)/2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.user_zoom:
            self.fit_native()

    def wheelEvent(self, event):
        self.user_zoom = True
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        self.scale(factor, factor)
        event.accept()

    def native_position(self, position):
        if self.shape_hw is None:
            return None
        point = self.mapToScene(position)
        if not self.sceneRect().contains(point):
            return None
        h, w = self.shape_hw
        return (float(np.clip(point.x(), 0, w-1)), float(np.clip(point.y(), 0, h-1)))

    def mousePressEvent(self, event):
        self._press = event.position().toPoint() if event.button() == C.Qt.MouseButton.LeftButton else None
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        pressed = self._press
        self._press = None
        super().mouseReleaseEvent(event)
        if pressed is not None and (pressed-event.position().toPoint()).manhattanLength() <= 4:
            point = self.native_position(event.position().toPoint())
            if point is not None:
                self.point_selected.emit(point)

    def mouseMoveEvent(self, event):
        point = self.native_position(event.position().toPoint())
        if point is not None:
            self.position_changed.emit(point)
        super().mouseMoveEvent(event)

    def set_landmarks(self, points, pending=None):
        for item in self._markers:
            self.scene().removeItem(item)
        self._markers = []
        if self.shape_hw is None:
            return
        pen = G.QPen(G.QColor('#ffcc33'))
        pen.setCosmetic(True)
        radius = max(self.shape_hw) / 160
        for index, point in enumerate(points):
            x, y = point
            dot = self.scene().addEllipse(x-radius, y-radius, radius*2, radius*2, pen)
            label = self.scene().addSimpleText(str(index+1))
            label.setBrush(G.QBrush(G.QColor('#ffcc33')))
            label.setFlag(W.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            label.setPos(x+radius, y+radius)
            self._markers.extend((dot, label))
        if pending is not None:
            x, y = pending
            self._markers.append(self.scene().addEllipse(x-radius, y-radius, radius*2, radius*2, pen))


class LandmarkRegistrationDialog(W.QDialog):
    """Fit a user-declared source-EDS -> target-cache affine; no raw writes.

    Constructing starts bounded asynchronous readers. After Accepted, call
    registration_config(); it rechecks source stamps and returns a fresh copy.
    Error details remain inline so failed fits do not leave a modal error loop.
    """
    def __init__(self, cache_path, lang='en', parent=None, source_path=None, scan_path='/1/EBSD'):
        super().__init__(parent)
        self.lang = 'zh' if str(lang).lower().startswith('zh') else 'en'
        self.cache_path = Path(cache_path).resolve()
        self._initial_target_stamp = _stamp(self.cache_path)
        self._metadata = {'source': None, 'target': None}
        self._tokens = {'source': 0, 'target': 0}
        self._workers = set()
        self._closed = False
        self._fit = None
        self._pending_load = None
        self._pending_points = {'source': None, 'target': None}
        self._bindings = []
        self._status_key = 'No fitted registration'
        app = W.QApplication.instance()
        if app is not None and not getattr(app, '_ebsd_registration_shutdown_installed', False):
            app.aboutToQuit.connect(shutdown_preview_workers)
            app._ebsd_registration_shutdown_installed = True
        self.resize(1240, 900)
        self._build(source_path, scan_path)
        self._start_preview('target', self.cache_path)
        if source_path is not None:
            self.load_source()

    def t(self, source):
        return TRANSLATIONS.get(source, source) if self.lang == 'zh' else source

    def _label(self, text):
        label = W.QLabel(self.t(text))
        self._bindings.append((label.setText, text))
        return label

    def _button(self, text, slot):
        button = W.QPushButton(self.t(text))
        button.clicked.connect(slot)
        self._bindings.append((button.setText, text))
        return button

    def _build(self, source_path, scan_path):
        self.setWindowTitle(self.t('EDS landmark registration'))
        layout = W.QVBoxLayout(self)
        form = W.QGridLayout()
        form.addWidget(self._label('Target EBSD BC cache'), 0, 0)
        target = W.QLineEdit(str(self.cache_path)); target.setReadOnly(True)
        form.addWidget(target, 0, 1, 1, 3)
        form.addWidget(self._label('Source EDS H5OINA'), 1, 0)
        self.source_path_edit = W.QLineEdit(str(source_path or ''))
        form.addWidget(self.source_path_edit, 1, 1, 1, 2)
        form.addWidget(self._button('Choose source', self.choose_source), 1, 3)
        form.addWidget(self._label('Scan path'), 2, 0)
        self.scan_path_edit = W.QLineEdit(scan_path)
        form.addWidget(self.scan_path_edit, 2, 1)
        form.addWidget(self._label('EDS window'), 2, 2)
        self.channel_combo = W.QComboBox(); form.addWidget(self.channel_combo, 2, 3)
        form.addWidget(self._button('Read preview', self.load_source), 3, 3)
        layout.addLayout(form)
        instruction = self._label('Click a feature in each view to add a matched pair. Drag to pan; wheel to zoom. Coordinates may also be edited in the table.')
        instruction.setWordWrap(True); layout.addWidget(instruction)
        panels = W.QSplitter(C.Qt.Orientation.Horizontal)
        self.source_view, self.target_view = LandmarkView(), LandmarkView()
        self.view_info = {}
        for kind, view, title in (('source', self.source_view, 'Source EDS — native pixels'),
                                  ('target', self.target_view, 'Target BC — native pixels')):
            panel = W.QWidget(); box = W.QVBoxLayout(panel)
            box.addWidget(self._label(title)); box.addWidget(view)
            info = W.QLabel(); info.setWordWrap(True); box.addWidget(info); self.view_info[kind] = info
            view.point_selected.connect(lambda point, k=kind: self._clicked(k, point))
            view.position_changed.connect(lambda point, k=kind: self.coordinates.setText(f'{self.t(k.title())}: X {point[0]:.4f}, Y {point[1]:.4f} px'))
            panels.addWidget(panel)
        layout.addWidget(panels, 3)
        self.coordinates = W.QLabel(); layout.addWidget(self.coordinates)
        self.table = W.QTableWidget(0, 4)
        self.table.setSelectionBehavior(W.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMaximumHeight(180)
        self._headers(); layout.addWidget(self.table)
        controls = W.QHBoxLayout()
        controls.addWidget(self._button('Add row', lambda: self.add_pair()))
        controls.addWidget(self._button('Delete selected', self.delete_selected))
        controls.addWidget(self._button('Clear pairs', self.clear_pairs))
        self.fit_button = self._button('Fit affine', self.fit_landmarks)
        controls.addWidget(self.fit_button)
        controls.addWidget(self._button('Save JSON', self.choose_save))
        controls.addWidget(self._button('Load JSON', self.choose_load))
        layout.addLayout(controls)
        self.matrix_text = W.QPlainTextEdit(); self.matrix_text.setReadOnly(True); self.matrix_text.setMaximumHeight(150)
        layout.addWidget(self.matrix_text)
        warning = self._label('At least 3 noncollinear pairs are required. Fit residuals use the fitted landmarks; they are NOT independent validation. Verify on independent features before scientific use.')
        warning.setWordWrap(True); layout.addWidget(warning)
        self.status = W.QLabel(self.t('Reading bounded previews…')); self.status.setWordWrap(True); layout.addWidget(self.status)
        self.details = W.QPlainTextEdit(); self.details.setReadOnly(True); self.details.setMaximumHeight(65); self.details.hide(); layout.addWidget(self.details)
        buttons = W.QHBoxLayout(); buttons.addStretch()
        self.apply_button = self._button('Apply registration', self.accept)
        self.apply_button.setEnabled(False); buttons.addWidget(self.apply_button)
        buttons.addWidget(self._button('Cancel', self.reject)); layout.addLayout(buttons)
        self.source_path_edit.textChanged.connect(self._source_input_changed)
        self.scan_path_edit.textChanged.connect(self._source_input_changed)
        self.source_path_edit.editingFinished.connect(self.load_source)
        self.scan_path_edit.editingFinished.connect(self.load_source)
        self.channel_combo.currentIndexChanged.connect(self._channel_changed)
        self.table.itemChanged.connect(self._landmarks_changed)
        self._invalidate_fit('No fitted registration')

    def _headers(self):
        self.table.setHorizontalHeaderLabels([self.t(s) for s in ('Source X', 'Source Y', 'Target X', 'Target Y')])

    def set_language(self, lang):
        self.lang = 'zh' if str(lang).lower().startswith('zh') else 'en'
        self.setWindowTitle(self.t('EDS landmark registration'))
        for setter, source in self._bindings:
            setter(self.t(source))
        self._headers()
        self._show_fit()
        for kind in ('source', 'target'):
            self._update_info(kind)
        self._status(self._status_key)

    def _status(self, key):
        self._status_key = key
        self.status.setText(self.t(key))

    def _error(self, exc):
        self._status('Registration cannot be applied. See details below.')
        self.details.setPlainText(f'{type(exc).__name__}: {exc}')
        self.details.show()

    def _invalidate_fit(self, message='Landmarks changed; fit again before Apply.'):
        self._fit = None
        self.apply_button.setEnabled(False)
        self.matrix_text.setPlainText(self.t('No fitted registration'))
        self._status(message); self.details.hide()

    def _source_input_changed(self, *_, clear_channels=True):
        self._tokens['source'] += 1
        self._metadata['source'] = None
        for worker in self._workers:
            if worker.kind == 'source': worker.requestInterruption()
        self.source_view.clear_preview()
        if clear_channels:
            blocker = C.QSignalBlocker(self.channel_combo)
            self.channel_combo.clear()
            del blocker
        self._pending_load = None
        self._pending_points = {'source': None, 'target': None}
        self._invalidate_fit('Source selection changed; reload preview and fit again.')
        self._update_info('source')

    def _channel_changed(self, *_):
        self._source_input_changed(clear_channels=False)
        self.load_source()

    def choose_source(self):
        path, _ = W.QFileDialog.getOpenFileName(self, self.t('Choose source'), '', 'H5OINA (*.h5oina *.h5)',
                                               options=W.QFileDialog.Option.DontUseNativeDialog)
        if path:
            self.source_path_edit.setText(path)
            self.load_source()

    def load_source(self):
        path = self.source_path_edit.text().strip()
        if not path:
            return
        self._invalidate_fit('Reading bounded previews…')
        scan_path = '/' + self.scan_path_edit.text().strip().strip('/')
        self._start_preview('source', path, scan_path, self.channel_combo.currentData())

    def _start_preview(self, kind, path, scan_path=None, channel=None):
        self._tokens[kind] += 1
        for worker in self._workers:
            if worker.kind == kind: worker.requestInterruption()
        self._metadata[kind] = None
        worker = PreviewWorker(kind, self._tokens[kind], path, scan_path, channel)
        self._workers.add(worker); _WORKERS.add(worker)
        worker.loaded.connect(self._preview_loaded)
        worker.failed.connect(self._preview_failed)
        worker.finished.connect(self._worker_finished)
        worker.finished.connect(lambda w=worker: _WORKERS.discard(w))
        worker.start()

    @C.Slot()
    def _worker_finished(self):
        self._workers.discard(self.sender())

    @C.Slot(str, int, object)
    def _preview_loaded(self, kind, token, data):
        if self._closed or token != self._tokens[kind]:
            return
        try:
            stamp = data['source_stamp']
            if _stamp(stamp['path']) != stamp:
                raise ValueError('Source changed while preview was being delivered')
            if kind == 'target' and stamp != self._initial_target_stamp:
                raise ValueError('Target cache changed; close and reopen the registration editor')
            view = self.source_view if kind == 'source' else self.target_view
            view.set_preview(data)
            self._metadata[kind] = data
            if kind == 'source':
                # The reader canonicalizes '/1/EBSD' and '1/EBSD' identically.
                # Retain that verified spelling for later provenance checks.
                scan_blocker = C.QSignalBlocker(self.scan_path_edit)
                self.scan_path_edit.setText(data['scan_path'])
                del scan_blocker
                blocker = C.QSignalBlocker(self.channel_combo)
                self.channel_combo.clear()
                for channel in data['channels']:
                    self.channel_combo.addItem(str(channel), str(channel))
                self.channel_combo.setCurrentIndex(self.channel_combo.findData(data['channel']))
                del blocker
                self._status('Source preview ready; select corresponding landmarks.')
            self._update_info(kind); self._update_markers()
            if self._pending_load is not None and all(self._metadata.values()):
                config = _validated(self._pending_load, self.cache_path, data['shape_hw'] if kind == 'source' else self._metadata['source']['shape_hw'])
                self._pending_load = None
                self._set_pairs(config['source_xy'], config['target_xy'])
                if self.fit_landmarks():
                    self._status('Registration loaded and refitted from saved landmarks.')
        except Exception as exc:
            self._metadata[kind] = None; self._pending_load = None
            self._invalidate_fit(); self._error(exc)

    @C.Slot(str, int, str)
    def _preview_failed(self, kind, token, error):
        if not self._closed and token == self._tokens[kind]:
            self._metadata[kind] = None; self._pending_load = None
            self._invalidate_fit(); self._error(RuntimeError(error))

    def _update_info(self, kind):
        data = self._metadata[kind]
        if data is None:
            self.view_info[kind].setText(self.t('No source selected'))
            return
        h, w = data['shape_hw']; steps = data.get('step_um_xy')
        text = f'{w} × {h} px | {data["channel"]} [{data["unit"]}]'
        text += (f' | Δx {steps[0]:g}, Δy {steps[1]:g} µm' if steps is not None else
                 '\n' + self.t('Unknown physical step; view uses native pixel aspect.'))
        self.view_info[kind].setText(text + '\n' + self.t('Preview contrast is relative; source values and units are retained.'))

    def _clicked(self, kind, point):
        self._pending_points[kind] = tuple(point)
        if all(p is not None for p in self._pending_points.values()):
            self.add_pair(self._pending_points['source'], self._pending_points['target'])
            self._pending_points = {'source': None, 'target': None}
        else:
            self._status('Select the matching point in the target BC view.' if kind == 'source' else
                         'Select the matching point in the source EDS view.')
        self._update_markers()

    def add_pair(self, source_xy=None, target_xy=None):
        row = self.table.rowCount()
        blocker = C.QSignalBlocker(self.table)
        self.table.insertRow(row)
        values = (*source_xy, *target_xy) if source_xy is not None and target_xy is not None else ('', '', '', '')
        for column, value in enumerate(values):
            text = format(float(value), '.12g') if value != '' else ''
            self.table.setItem(row, column, W.QTableWidgetItem(text))
        del blocker
        self._landmarks_changed()

    def _set_pairs(self, source, target):
        blocker = C.QSignalBlocker(self.table)
        self.table.setRowCount(0)
        for src, dst in zip(source, target):
            row = self.table.rowCount(); self.table.insertRow(row)
            for col, value in enumerate((*src, *dst)):
                self.table.setItem(row, col, W.QTableWidgetItem(format(float(value), '.17g')))
        del blocker
        self._pending_points = {'source': None, 'target': None}
        self._landmarks_changed()

    def delete_selected(self):
        rows = {index.row() for index in self.table.selectionModel().selectedRows()}
        if not rows and self.table.currentRow() >= 0:
            rows = {self.table.currentRow()}
        blocker = C.QSignalBlocker(self.table)
        for row in sorted(rows, reverse=True): self.table.removeRow(row)
        del blocker
        self._landmarks_changed()

    def clear_pairs(self):
        self.table.setRowCount(0)
        self._pending_points = {'source': None, 'target': None}
        self._landmarks_changed()

    def _pairs(self):
        values = []
        for row in range(self.table.rowCount()):
            try:
                values.append([float(self.table.item(row, column).text()) for column in range(4)])
            except (ValueError, AttributeError) as exc:
                raise ValueError(f'Landmark row {row+1} requires four finite numeric coordinates') from exc
        array = np.asarray(values, dtype=float).reshape(-1, 4)
        if not np.isfinite(array).all():
            raise ValueError('Landmark coordinates must be finite')
        return array[:, :2], array[:, 2:]

    def _landmarks_changed(self, *_):
        # User edits during an async JSON reload supersede its deferred table
        # restoration. A late preview must never overwrite those new points.
        self._pending_load = None
        self._invalidate_fit()
        self._update_markers()

    def _update_markers(self):
        try:
            source, target = self._pairs()
        except ValueError:
            source, target = [], []
        self.source_view.set_landmarks(source, self._pending_points['source'])
        self.target_view.set_landmarks(target, self._pending_points['target'])

    def _check_sources(self):
        if not all(self._metadata.values()):
            raise ValueError('Wait for both bounded source and target previews before fitting')
        source, target = self._metadata['source'], self._metadata['target']
        if (_stamp(source['source_stamp']['path']) != source['source_stamp']
                or _stamp(self.cache_path) != target['source_stamp']
                or target['source_stamp'] != self._initial_target_stamp):
            raise ValueError('Source or target changed after preview; reload before fitting or applying')
        if (Path(self.source_path_edit.text().strip()).resolve() != Path(source['source_stamp']['path']).resolve()
                or self.scan_path_edit.text().strip() != source['scan_path']
                or self.channel_combo.currentData() != source['channel']):
            raise ValueError('Source selection differs from the loaded preview')
        return source, target

    def fit_landmarks(self):
        self._invalidate_fit()
        try:
            source, target = self._check_sources()
            src, dst = self._pairs()
            for values, metadata in ((src, source), (dst, target)):
                h, w = metadata['shape_hw']
                if np.any(values < 0) or np.any(values > [w-1, h-1]):
                    raise ValueError('Landmark outside native pixel-centre bounds')
            matrix, qc = fit_affine(src, dst)
            condition = float(np.linalg.cond(np.column_stack((src, np.ones(len(src))))))
            scales = np.linalg.svd(matrix[:2, :2], compute_uv=False)
            qc.update(landmark_design_condition=condition, affine_singular_values=scales.tolist(),
                      poor_distribution_warning=bool(condition > 1e6 or scales.max()/scales.min() > 1e3))
            config = {'schema_version': 1, 'source_path': source['source_stamp']['path'],
                'scan_path': source['scan_path'], 'source_channel': source['channel'],
                'source_stamp': source['source_stamp'], 'target_cache_stamp': target['source_stamp'],
                'source_shape_hw': source['shape_hw'], 'target_shape_hw': target['shape_hw'],
                'source_xy': src.tolist(), 'target_xy': dst.tolist(), 'matrix': matrix.tolist(), 'fit_qc': qc}
            self._fit = _validated(config, self.cache_path, source['shape_hw'])
            # Central validation rederives the fit QC. Preserve additional UI
            # diagnostics without changing its residuals or validation claim.
            self._fit['fit_qc'].update({key: qc[key] for key in
                ('landmark_design_condition', 'affine_singular_values', 'poor_distribution_warning')})
            self.apply_button.setEnabled(True); self._show_fit()
            return True
        except Exception as exc:
            self._error(exc)
            return False

    def _show_fit(self):
        if self._fit is None:
            self.matrix_text.setPlainText(self.t('No fitted registration'))
            return
        matrix = np.asarray(self._fit['matrix']); qc = self._fit['fit_qc']
        lines = [np.array2string(matrix, precision=9),
                 self.t('RMS {rms:.6g} target px | max {maximum:.6g} target px').format(
                     rms=qc['rms_target_px'], maximum=qc['max_target_px']),
                 self.t('Per-pair residual [target px]') + ': ' + ', '.join(f'{v:.6g}' for v in qc['residuals_target_px']),
                 self.t('Design condition') + f' {qc.get("landmark_design_condition", float("nan")):.6g}',
                 self.t('Registered coordinates are NOT externally validated.')]
        self.matrix_text.setPlainText('\n'.join(lines))
        self._status('Poor landmark distribution / extreme transform: inspect independent features.'
                     if qc.get('poor_distribution_warning') else
                     'Registered coordinates are NOT externally validated.')

    def registration_config(self):
        source, _ = self._check_sources()
        if self._fit is None:
            raise ValueError('Fit valid landmarks before applying or saving registration')
        config = _validated(self._fit, self.cache_path, source['shape_hw'])
        for key in ('landmark_design_condition', 'affine_singular_values', 'poor_distribution_warning'):
            config['fit_qc'][key] = self._fit['fit_qc'][key]
        return copy.deepcopy(config)

    def save_config(self, path):
        config = self.registration_config()
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError('Registration JSON already exists; choose a new filename')
        temporary = path.with_name('.registration-' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('x', encoding='utf-8') as stream:
                json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.flush(); os.fsync(stream.fileno())
            if os.name == 'nt': os.rename(temporary, path)
            else: os.link(temporary, path); temporary.unlink()
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def load_config(self, path):
        """Load a validated project dictionary or a saved JSON and refit async."""
        if isinstance(path, dict):
            value = copy.deepcopy(path)
        else:
            path = Path(path)
            if path.stat().st_size > 1024*1024:
                raise ValueError('Registration JSON exceeds 1 MiB; choose a landmark configuration')
            value = json.loads(path.read_text(encoding='utf-8-sig'))
        config = _validated(value, self.cache_path)
        self._invalidate_fit()
        for editor, text in ((self.source_path_edit, config['source_path']), (self.scan_path_edit, config['scan_path'])):
            blocker = C.QSignalBlocker(editor); editor.setText(text); del blocker
        blocker = C.QSignalBlocker(self.channel_combo)
        self.channel_combo.clear()
        if config.get('source_channel'):
            self.channel_combo.addItem(config['source_channel'], config['source_channel'])
        del blocker
        self._pending_load = config
        self._start_preview('source', config['source_path'], config['scan_path'], config.get('source_channel'))
        return copy.deepcopy(config)

    def choose_save(self):
        path, _ = W.QFileDialog.getSaveFileName(self, self.t('Save new registration JSON'), '', 'JSON (*.json)',
                                               options=W.QFileDialog.Option.DontUseNativeDialog | W.QFileDialog.Option.DontConfirmOverwrite)
        if path:
            try: self.save_config(path)
            except Exception as exc: self._error(exc)

    def choose_load(self):
        path, _ = W.QFileDialog.getOpenFileName(self, self.t('Choose registration JSON'), '', 'JSON (*.json)',
                                               options=W.QFileDialog.Option.DontUseNativeDialog)
        if path:
            try: self.load_config(path)
            except Exception as exc: self._error(exc)

    def _shutdown_workers(self):
        self._closed = True
        for kind in self._tokens: self._tokens[kind] += 1
        for worker in self._workers: worker.requestInterruption()

    def accept(self):
        try:
            self.registration_config()
        except Exception as exc:
            self.apply_button.setEnabled(False); self._error(exc)
            return
        self._shutdown_workers()
        super().accept()

    def reject(self):
        self._shutdown_workers()
        super().reject()

    def closeEvent(self, event):
        self._shutdown_workers()
        super().closeEvent(event)
