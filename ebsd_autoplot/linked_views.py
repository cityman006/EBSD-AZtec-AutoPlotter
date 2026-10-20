"""Reusable 1/2/4 map panels with native-coordinate navigation links.

MapView owns pixels, tile workers and numerical hover. This widget only routes
navigation/cursor/ROI signals between views displaying the identical unchanged
cache. view_factory must return a QWidget with the MapView hooks documented in
the constructor; no numeric map or scientific result is computed here.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

from PySide6 import QtCore as C, QtWidgets as W

MAP_FIELDS = ("BC", "Phase", "KAM", "IPF_X", "IPF_Y", "IPF_Z")


def _identity(path):
    if path is None: return None
    path = Path(path).resolve()
    try: stat = path.stat()
    except OSError: return None
    return (os.path.normcase(str(path)), stat.st_size, stat.st_mtime_ns)


class LinkedMapWorkspace(W.QWidget):
    """MapView hooks: load_cache(path,field), clear_cache(), navigation_state(),
    apply_navigation(dict), set_linked_cursor(tuple|None), set_linked_roi(tuple|None),
    plus navigation_changed(object), cursor_moved(object), roi_selected(tuple).

    language may be a callable returning 'zh'/'en', or a language string.
    All four views persist across layout changes. load_cache loads all four;
    load_view_cache permits independent sources. Failed fields are cleared and
    explicitly unavailable, never substituted by another map.
    """
    active_view_changed = C.Signal(int)
    field_changed = C.Signal(int, str)
    roi_selected = C.Signal(tuple)
    coordinate = C.Signal(str)
    load_error = C.Signal(int, str)

    def __init__(self, view_factory, language=None, parent=None):
        super().__init__(parent)
        self.language = language
        self.views = [view_factory() for _ in range(4)]
        self.paths = [None]*4
        self._identities = [None]*4
        self._cursors = [None]*4; self._rois = [None]*4
        self._errors = [None]*4
        self._active = 0; self._count = 1; self._linked = True
        self._routing = False
        self.frames = []; self.map_selectors = []; self.status_labels = []; self.view_labels = []
        layout = W.QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        toolbar = W.QHBoxLayout()
        self.layout_label = W.QLabel()
        self.layout_selector = W.QComboBox(); self.layout_selector.addItems(["1","2","4"])
        self.layout_selector.currentTextChanged.connect(lambda text: self.set_layout(int(text)))
        self.link_checkbox = W.QCheckBox(); self.link_checkbox.setChecked(True)
        self.link_checkbox.toggled.connect(self.set_linked)
        toolbar.addWidget(self.layout_label); toolbar.addWidget(self.layout_selector)
        toolbar.addWidget(self.link_checkbox); toolbar.addStretch()
        layout.addLayout(toolbar)
        self.grid = W.QGridLayout(); layout.addLayout(self.grid, 1)
        for index, view in enumerate(self.views):
            frame = W.QFrame(self); frame.setFrameShape(W.QFrame.Shape.StyledPanel)
            frame.setStyleSheet("QFrame[activeView='true'] {border: 2px solid #3b82f6;} QFrame[activeView='false'] {border: 1px solid #777;}")
            panel = W.QVBoxLayout(frame); panel.setContentsMargins(3,3,3,3)
            header = W.QHBoxLayout(); label = W.QLabel()
            selector = W.QComboBox()
            for field in MAP_FIELDS: selector.addItem(field, field)
            selector.setCurrentIndex(selector.findData(("BC","Phase","KAM","IPF_Z")[index]))
            selector.currentIndexChanged.connect(lambda _, i=index: self._change_field(i, self.map_selectors[i].currentData()))
            header.addWidget(label); header.addStretch(); header.addWidget(selector)
            panel.addLayout(header); panel.addWidget(view, 1)
            status = W.QLabel(); status.setWordWrap(True); panel.addWidget(status)
            self.frames.append(frame); self.map_selectors.append(selector)
            self.status_labels.append(status); self.view_labels.append(label)
            view.installEventFilter(self)
            if hasattr(view, "viewport"): view.viewport().installEventFilter(self)
            signal = getattr(view, "navigation_changed", None)
            if signal is not None: signal.connect(lambda state, i=index: self._navigation(i, state))
            signal = getattr(view, "cursor_moved", None)
            if signal is not None: signal.connect(lambda position, i=index: self._cursor(i, position))
            signal = getattr(view, "roi_selected", None)
            if signal is not None: signal.connect(lambda roi, i=index: self._roi(i, roi))
            signal = getattr(view, "coordinate", None)
            if signal is not None: signal.connect(lambda message, i=index: self.coordinate.emit(message) if i == self._active else None)
        self.selectors = self.map_selectors
        self.set_layout(1); self.retranslate()

    @property
    def active_view(self): return self.views[self._active]

    @property
    def active_index(self): return self._active

    @property
    def view_count(self): return self._count

    @property
    def linked(self): return self._linked

    def _text(self, english, chinese):
        language = self.language() if callable(self.language) else self.language
        return chinese if str(language or "en").lower().startswith("zh") else english

    def retranslate(self, language=None):
        if language is not None: self.language = language
        self.layout_label.setText(self._text("Views", "视图数量"))
        self.link_checkbox.setText(self._text("Link pan / zoom / cursor / ROI", "联动平移 / 缩放 / 光标 / ROI"))
        for index in range(4):
            selector = self.map_selectors[index]
            blocker = C.QSignalBlocker(selector)
            for item in range(selector.count()):
                field = selector.itemData(item)
                selector.setItemText(item, self._text("Phase", "相位") if field == "Phase" else field)
            del blocker
            suffix = self._text("active", "当前") if index == self._active else ""
            self.view_labels[index].setText(self._text(f"View {index+1}", f"视图 {index+1}") + (f" ({suffix})" if suffix else ""))
            self._update_status(index)

    def eventFilter(self, watched, event):
        if event.type() in (C.QEvent.Type.MouseButtonPress, C.QEvent.Type.FocusIn, C.QEvent.Type.Wheel):
            for index, view in enumerate(self.views):
                if watched is view or (hasattr(view, "viewport") and watched is view.viewport()):
                    self.set_active_view(index); break
        return super().eventFilter(watched, event)

    def set_active_view(self, index):
        index = int(index)
        if not 0 <= index < self._count: return
        changed = index != self._active
        self._active = index
        for i, frame in enumerate(self.frames):
            self.views[i].setProperty('linked_active',i==index)
            frame.setProperty("activeView", i == index)
            frame.style().unpolish(frame); frame.style().polish(frame)
        self.retranslate()
        if changed: self.active_view_changed.emit(index)

    def set_layout(self, count):
        count = int(count)
        if count not in (1,2,4): raise ValueError("Linked maps support 1, 2 or 4 views")
        snapshots = [(self._identities[i], self._state(view)) for i, view in enumerate(self.views)]
        previous_routing = self._routing
        self._routing = True
        self._count = count
        try:
            while self.grid.count(): self.grid.takeAt(0)
            columns = 1 if count == 1 else 2
            for index, frame in enumerate(self.frames):
                frame.setVisible(index < count)
                if index < count: self.grid.addWidget(frame, index//columns, index%columns)
            # Resolve the resized viewports before restoring native cameras.
            # Suppress resize-triggered navigation until every panel is ready.
            self.grid.activate(); self.layout().activate()
            for index, (identity, state) in enumerate(snapshots):
                if identity is not None and identity == self._identities[index] and state is not None:
                    self._apply_navigation(index, state)
        finally: self._routing = previous_routing
        blocker = C.QSignalBlocker(self.layout_selector)
        self.layout_selector.setCurrentText(str(count)); del blocker
        self.set_active_view(min(self._active, count-1))

    set_view_count = set_layout

    def set_linked(self, enabled):
        self._linked = bool(enabled)
        blocker = C.QSignalBlocker(self.link_checkbox)
        self.link_checkbox.setChecked(self._linked); del blocker
        if not self._linked:
            self._cursors = [None]*4
            for index in range(4): self._set_overlay(index, "cursor", None)
        else:
            state = self._state(self.active_view)
            if state is not None: self._navigation(self._active, state)
            self._cursor(self._active, self._cursors[self._active])
            if self._rois[self._active] is not None: self._roi(self._active, self._rois[self._active])

    def _state(self, view):
        method = getattr(view, "navigation_state", None)
        return method() if method is not None else None

    def _apply_navigation(self, index, state):
        method = getattr(self.views[index], "apply_navigation", None)
        if method is not None:
            blocker = C.QSignalBlocker(self.views[index])
            method(dict(state)); del blocker

    def _set_overlay(self, index, kind, value):
        method = getattr(self.views[index], "set_linked_"+kind, None)
        if method is not None:
            blocker = C.QSignalBlocker(self.views[index])
            method(value); del blocker

    def _clear_one(self, index, *, keep_path=False):
        self._identities[index] = None; self._cursors[index] = None; self._rois[index] = None
        if not keep_path: self.paths[index] = None
        self._set_overlay(index, "cursor", None); self._set_overlay(index, "roi", None)
        view = self.views[index]
        blocker = C.QSignalBlocker(view)
        method = getattr(view, "clear_cache", None)
        if method is not None: method()
        else:
            if hasattr(view, "scene") and view.scene() is not None: view.scene().clear()
            view.cache = None; view.width_px = view.height_px = 0
        del blocker

    def clear_cache(self):
        for index in range(4):
            self._clear_one(index); self._errors[index] = None; self._update_status(index)

    def _invalidate_changed_sources(self):
        for index, expected in enumerate(self._identities):
            if expected is not None and _identity(getattr(self.views[index], "cache", None)) != expected:
                self._clear_one(index, keep_path=True)
                self._errors[index] = "__source_changed__"
                self._update_status(index)

    def _peers(self, index):
        self._invalidate_changed_sources()
        identity = self._identities[index]
        if identity is None: return []
        return [i for i, other in enumerate(self._identities) if other == identity]

    def _navigation(self, index, state):
        if self._routing: return
        self.set_active_view(index)
        if not self._linked: return
        if not isinstance(state, dict): return
        try:
            center = tuple(map(float, state["center"])); scale = float(state["scale"])
            if len(center) != 2 or not all(math.isfinite(v) for v in (*center,scale)) or scale <= 0: return
        except (KeyError, TypeError, ValueError): return
        peers = self._peers(index)
        self._routing = True
        try:
            for peer in peers:
                if peer != index: self._apply_navigation(peer, {**state, "center": center, "scale": scale})
        finally: self._routing = False

    def _cursor(self, index, position):
        if self._routing: return
        peers = self._peers(index)
        if not peers: return
        if position is not None:
            if len(position) != 2 or not all(math.isfinite(float(v)) for v in position): return
            position = tuple(map(float, position))
        self._cursors[index] = position
        self._routing = True
        try:
            for peer in peers if self._linked else [index]:
                self._cursors[peer] = position
                self._set_overlay(peer, "cursor", position)
        finally: self._routing = False

    def _roi(self, index, roi):
        if self._routing: return
        peers = self._peers(index)
        if not peers: return
        if roi is not None:
            if len(roi) != 4 or not all(math.isfinite(float(v)) for v in roi) or min(roi[2:]) <= 0: return
            roi = tuple(roi)
        self._rois[index] = roi
        self.set_active_view(index)
        self._routing = True
        try:
            for peer in peers if self._linked else [index]:
                self._rois[peer] = roi
                self._set_overlay(peer, "roi", roi)
        finally: self._routing = False
        if roi is not None: self.roi_selected.emit(roi)

    def _update_status(self, index):
        if self._errors[index]:
            raw = self._errors[index]
            if raw == "__source_changed__":
                self.status_labels[index].setText(self._text("Source changed; reload cache", "源缓存已改变，请重新加载"))
            else:
                self.status_labels[index].setText(self._text("Unavailable — see error details", "不可用，详见错误信息"))
            self.status_labels[index].setToolTip(raw if raw != "__source_changed__" else "")
            return
        self.status_labels[index].setToolTip("")
        if self._identities[index] is None:
            self.status_labels[index].setText(self._text("Open an analysis cache", "打开分析缓存")); return
        view = self.views[index]; meta = getattr(view, "meta", {})
        width, height = getattr(view, "width_px",0), getattr(view,"height_px",0)
        dx, dy = meta.get("x_step_um"), meta.get("y_step_um")
        if dx is not None and dy is not None:
            text = f"X 0–{max(0,width-1)*float(dx):g} µm | Y 0–{max(0,height-1)*float(dy):g} µm | Δx {float(dx):g}, Δy {float(dy):g} µm"
        else: text = self._text("Native pixels", "原始像素") + f": {width} × {height}"
        self.status_labels[index].setText(text)

    def load_view_cache(self, index, path, field=None, *, preserve_navigation=False):
        index = int(index)
        if not 0 <= index < 4: raise IndexError(index)
        path = Path(path).resolve()
        field = field or self.map_selectors[index].currentData()
        if field not in MAP_FIELDS: raise ValueError(field)
        same_source = self._identities[index] is not None and _identity(path) == self._identities[index]
        previous_state = self._state(self.views[index]) if preserve_navigation and same_source else None
        previous_cursor = self._cursors[index] if same_source else None
        previous_roi = self._rois[index] if same_source else None
        self._clear_one(index); self.paths[index] = path; self._errors[index] = None
        blocker = C.QSignalBlocker(self.map_selectors[index])
        self.map_selectors[index].setCurrentIndex(self.map_selectors[index].findData(field)); del blocker
        view = self.views[index]; blocker = C.QSignalBlocker(view)
        try:
            view.load_cache(path, field=field)
            identity = _identity(path)
            if identity is None: raise FileNotFoundError(path)
            self._identities[index] = identity
            if previous_state is not None: self._apply_navigation(index, previous_state)
            if same_source:
                self._cursors[index] = previous_cursor; self._rois[index] = previous_roi
                self._set_overlay(index, "cursor", previous_cursor)
                self._set_overlay(index, "roi", previous_roi)
            success = True
        except Exception as exc:
            self._clear_one(index, keep_path=True)
            self._errors[index] = str(exc)
            self.load_error.emit(index, str(exc)); success = False
        finally: del blocker
        self._update_status(index)
        return success

    def load_cache(self, path, field=None):
        identity = _identity(path)
        successful = []
        for index in range(4):
            same = identity is not None and identity == self._identities[index]
            successful.append(self.load_view_cache(index, path, field, preserve_navigation=same))
        if self._linked and successful[self._active]:
            state = self._state(self.active_view)
            if state is not None: self._navigation(self._active, state)
        return successful

    def _change_field(self, index, field):
        self.set_active_view(index)
        path = self.paths[index]
        if path is not None:
            self.load_view_cache(index, path, field, preserve_navigation=True)
        self.field_changed.emit(index, field)

    def set_field(self, field, index=None):
        index = self._active if index is None else int(index)
        if field not in MAP_FIELDS: raise ValueError(field)
        if self.map_selectors[index].currentData() == field: return
        self.map_selectors[index].setCurrentIndex(self.map_selectors[index].findData(field))
