"""Catalog inventory and real Qt language round trips without science imports."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ebsd_autoplot.i18n import (ALIASES, CATALOG, SOURCE_ROLE, bind_combo_item,
    bind_item, bind_text, canonical_source, install_qt_translator,
    missing_ui_strings, skip_translation, tr, translate_widget_tree)


def test_translation_module_does_not_import_scientific_or_qt_dependencies():
    script = "import sys; from ebsd_autoplot.i18n import tr; assert tr('Settings','zh')=='设置'; assert not any(n in sys.modules for n in ('PySide6','numpy','matplotlib','h5py','scipy'))"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_bilingual_catalog_and_legacy_aliases_have_complete_english_output():
    for source, chinese in CATALOG.items():
        assert isinstance(source, str) and source.strip() and chinese.strip()
        assert tr(source, "en") == source
        assert tr(source, "zh-CN") == chinese
    for alias, source in ALIASES.items():
        assert source in CATALOG
        assert canonical_source(alias) == source
        assert tr(alias, "en") == source
        assert tr(alias, "zh") == CATALOG[source]
    assert tr("Open cache / 打开缓存", "en") == "Open cache"


def test_unknown_values_identifiers_fonts_json_and_formatted_user_data_are_unchanged():
    for value in ("v3_schmid", "IPF_X", "BC", "MAD", "KAM", "PNG", "TIFF", "PDF", "SVG",
                  "Times New Roman", "Arial", "DejaVu Sans", r"E:\试验数据\sample.h5",
                  '{"source_id":"中文试样", "formats":["pdf","svg"]}', "HDF5 OSError: locked source 47"):
        assert tr(value, "zh") == value
        assert tr(value, "en") == value
    value = tr("{count} scans | {points:,} measured grid points", "zh", count=3, points=12000)
    assert value == "3 个扫描 | 12,000 个测量网格点"
    assert tr("View {number}: {error}", "zh", number=2, error="raw HDF5 path E:/中文.h5") == "视图 2：raw HDF5 path E:/中文.h5"


def test_ast_catalog_covers_m1_and_latest_m4_gui_literals():
    root = Path(__file__).parents[1]
    assert missing_ui_strings((root / "gui_qt.py").read_text(encoding="utf-8")) == []
    linked = root / "ebsd_autoplot" / "linked_views.py"
    if linked.exists():
        assert missing_ui_strings(linked.read_text(encoding="utf-8")) == []
    # Include the newest M2-M4 GUI source if this is the local review worktree.
    latest = root.parent.parent / "gui_qt.py"
    if latest.exists() and latest != root / "gui_qt.py":
        assert missing_ui_strings(latest.read_text(encoding="utf-8")) == []
    # A positive control ensures that the inventory detects newly unregistered UI.
    assert missing_ui_strings("label=W.QLabel('Unregistered explanatory paragraph')") == ["Unregistered explanatory paragraph"]


def test_m2_m4_explanatory_catalog_not_only_navigation():
    required = ["RX / Euler / Grain Size (cache)", "RX phase rules", "Phase ID → thresholds JSON",
        "No labels: supervised GNN disabled", "No labels: supervised mechanics disabled",
        "Synthetic GNN demo — software test only", "Synthetic diffusion demo — not real material",
        "Inference uses a checkpoint and a schema-compatible numeric feature bundle; labels are unnecessary for inference. Set workflow to infer-supervised and provide checkpoint + bundle_path.",
        "No mechanics truth is inferred from EBSD/EDS. Import DIC strain or declared HR-EBSD/CPFE/FEM labels through Dataset Builder. DIC alone does not establish full stress. Ridge is a numeric baseline, not U-Net/FNO.",
        "Conditional DDPM of numeric xyzw orientation/phase fields. Tiny CPU research baseline (≤4096 encoded variables/tile); no validated mechanical inverse design."]
    for text in required:
        assert text in CATALOG and tr(text, "zh") != text


@pytest.fixture(scope="module")
def qt_app():
    W = pytest.importorskip("PySide6.QtWidgets")
    app = W.QApplication.instance() or W.QApplication([])
    yield app


def test_widget_tree_roundtrip_preserves_inputs_selection_data_and_signals(qt_app):
    from PySide6 import QtCore as C, QtWidgets as W
    window = W.QMainWindow()
    central = W.QWidget(); layout = W.QVBoxLayout(central); window.setCentralWidget(central)
    label = W.QLabel("No labels: supervised GNN disabled"); label.setToolTip("Actual labels are required"); layout.addWidget(label)
    button = W.QPushButton("Open cache / 打开缓存"); layout.addWidget(button)
    line = W.QLineEdit("Settings"); line.setPlaceholderText("Same specimen must share this ID"); layout.addWidget(line)
    editor = W.QPlainTextEdit('{"mode":"v3_schmid","source":"中文用户数据"}'); layout.addWidget(editor)
    logs = W.QPlainTextEdit("Research GNN epoch 1/20: raw scientific log"); logs.setReadOnly(True); layout.addWidget(logs)
    checkbox = W.QCheckBox("Include verified EDS"); checkbox.setChecked(True); layout.addWidget(checkbox)
    combo = W.QComboBox(); combo.addItem("Area weighted", {"value": "area"}); combo.addItem("Grain weighted", "grain")
    bind_combo_item(combo, 0, "Area weighted"); bind_combo_item(combo, 1, "Grain weighted"); combo.setCurrentIndex(1); layout.addWidget(combo)
    font = W.QComboBox(); font.addItems(["Times New Roman", "Settings"]); font.setCurrentIndex(1); layout.addWidget(font)
    modes = W.QComboBox(); modes.addItems(["v1", "v3_schmid"]); modes.setCurrentIndex(1); layout.addWidget(modes)
    nav = W.QListWidget(); nav.addItems(["Dashboard", "Settings", "Crystallography"]); nav.setCurrentRow(2); layout.addWidget(nav); window.nav = nav
    history = W.QListWidget(); history.addItem("Settings"); history.item(0).setData(C.Qt.ItemDataRole.UserRole, r"E:\中文\Settings.h5"); layout.addWidget(history)
    table = W.QTableWidget(1, 2); table.setHorizontalHeaderLabels(["Source", "State"])
    table.setItem(0, 0, W.QTableWidgetItem("Settings")); table.setItem(0, 1, W.QTableWidgetItem("RUNNING")); layout.addWidget(table)
    tabs = W.QTabWidget(); tabs.addTab(W.QWidget(), "Data / Queue"); tabs.addTab(W.QWidget(), "Logs"); tabs.setCurrentIndex(1); layout.addWidget(tabs)
    dock = W.QDockWidget("Parameters / 参数", window); dock.setWidget(W.QWidget()); window.addDockWidget(C.Qt.DockWidgetArea.RightDockWidgetArea, dock)
    emissions = []
    combo.currentIndexChanged.connect(lambda _: emissions.append("combo")); nav.currentRowChanged.connect(lambda _: emissions.append("nav"))
    checkbox.toggled.connect(lambda _: emissions.append("checked")); tabs.currentChanged.connect(lambda _: emissions.append("tab"))
    for language in ("zh", "zh", "en", "zh", "en"):
        translate_widget_tree(window, language)
        assert label.text() == tr("No labels: supervised GNN disabled", language)
        assert button.text() == tr("Open cache", language)
        assert dock.windowTitle() == tr("Parameters", language)
        assert line.placeholderText() == tr("Same specimen must share this ID", language)
        assert combo.currentText() == tr("Grain weighted", language)
        assert combo.currentData() == "grain" and combo.itemData(0) == {"value": "area"}
        assert combo.itemData(0, SOURCE_ROLE) == "Area weighted"
        assert font.currentText() == "Settings" and modes.currentText() == "v3_schmid"
        assert line.text() == "Settings" and "中文用户数据" in editor.toPlainText()
        assert logs.toPlainText() == "Research GNN epoch 1/20: raw scientific log"
        assert history.item(0).text() == "Settings" and history.item(0).data(C.Qt.ItemDataRole.UserRole) == r"E:\中文\Settings.h5"
        assert table.item(0, 0).text() == "Settings" and table.item(0, 1).text() == "RUNNING"
        assert table.horizontalHeaderItem(0).text() == tr("Source", language)
        assert nav.currentRow() == 2 and tabs.currentIndex() == 1 and checkbox.isChecked()
        assert nav.item(2).text() == tr("Crystallography", language)
        assert not emissions
    window.close()


def test_dynamic_bindings_refresh_templates_and_do_not_restore_stale_user_labels(qt_app):
    from PySide6 import QtWidgets as W
    parent = W.QWidget(); label = W.QLabel(parent)
    bind_text(label, "{count} scans | {points:,} measured grid points", "en", count=4, points=1200)
    translate_widget_tree(parent, "zh")
    assert label.text() == "4 个扫描 | 1,200 个测量网格点"
    translate_widget_tree(parent, "en")
    assert label.text() == "4 scans | 1,200 measured grid points"
    bind_text(label, "Open cache", "en")
    label.setText("my actual specimen 文件名.h5")
    translate_widget_tree(parent, "zh")
    assert label.text() == "my actual specimen 文件名.h5"
    untouched = W.QLabel("Settings", parent); skip_translation(untouched)
    translate_widget_tree(parent, "zh")
    assert untouched.text() == "Settings"
    parent.close()


def test_opt_in_list_item_and_bundled_standard_qt_translation(qt_app):
    from PySide6 import QtCore as C, QtWidgets as W
    parent = W.QWidget(); listing = W.QListWidget(parent); item = W.QListWidgetItem("Settings", listing)
    item.setData(C.Qt.ItemDataRole.UserRole, "unchanged_semantic_value")
    bind_item(item, "Settings", "en"); translate_widget_tree(parent, "zh")
    assert item.text() == "设置" and item.data(C.Qt.ItemDataRole.UserRole) == "unchanged_semantic_value"
    assert isinstance(install_qt_translator(qt_app, "zh"), bool)
    assert install_qt_translator(qt_app, "en") is True
    parent.close()
