"""English/Chinese presentation strings and optional Qt bindings.

Importing this module uses only the standard library. Scientific identifiers,
user values, file names, JSON and backend logs are not translation keys. Qt is
loaded lazily by the widget helpers. Display translation never changes UserRole
values used by a processing option.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager

CATALOG = {
    "KAM unavailable": "KAM 不可用",
    "EBSD-AZtec AutoPlotter V4 — Materials Intelligence Studio": "EBSD-AZtec AutoPlotter V4 — 材料智能分析工作室",
    "Phase": "相位", "Views": "视图数量", "Link pan / zoom / cursor / ROI": "联动平移 / 缩放 / 光标 / ROI",
    "active": "当前", "Source changed; reload cache": "源缓存已改变，请重新加载",
    "Unavailable — see error details": "不可用，详见错误信息", "Open an analysis cache": "打开分析缓存",
    "Native pixels": "原始像素", "Linked maps support 1, 2 or 4 views": "联动地图支持 1、2 或 4 个视图",
    "View {number}": "视图 {number}",
    "Dashboard": "总览", "Data Manager": "数据管理", "EBSD Maps": "EBSD 图谱",
    "Crystallography": "晶体学", "Grain & Recrystallization": "晶粒与再结晶",
    "Batch Processing": "批量处理", "Slicing & Dataset": "切片与数据集",
    "Multimodal EDS": "多模态 EDS", "GNN / Mechanics / Diffusion": "GNN / 力学 / 扩散研究",
    "Publication Studio": "论文出图", "Results History": "结果历史", "Settings": "设置",
    "Workspace": "工作区", "Parameters": "参数", "Queue & Logs": "队列与日志",
    "Import files": "导入文件", "Import folder": "导入文件夹", "Open cache": "打开缓存",
    "Import H5 / H5OINA — AZtec color not calibrated": "导入 H5 / H5OINA — 尚未校准 AZtec 配色",
    "Analysis": "分析模式", "ROI pixels": "ROI 像素范围", "Load axis X Y Z": "加载轴 X Y Z",
    "Block rows": "每块处理行数", "Memory budget": "内存预算", "ODF weighting": "ODF 权重",
    "Parallel analyses": "并行分析数",
    "Maximum simultaneous analysis processes. The total estimated memory of active jobs must fit the memory budget; this is not an operating-system hard limit.":
        "同时运行的分析进程上限。运行任务的预计内存总量必须符合内存预算；该预算不是操作系统级的硬性内存限制。",
    "Parallel analyses are bounded by process count and total estimated memory. Failures do not stop other samples; the batch snapshot is frozen.":
        "并行分析同时受进程数和预计内存总量限制。单个样本失败不会阻止其他样本；批次参数快照保持固定。",
    "Invalid linked navigation": "无效的关联导航目标",
    "Selected cache has no calibrated orientation IPF data": "所选缓存不含已校准的取向 IPF 数据",
    "Export font": "导出字体", "Formats": "导出格式", "Maps": "图谱", "Panel layout": "拼图布局",
    "Output root": "输出根目录", "Choose output": "选择输出目录", "Run batch": "运行批量分析",
    "Cancel task": "取消任务", "Pause / Resume cached action": "暂停 / 继续缓存任务",
    "Resume batch": "恢复批次", "Render Only": "仅使用缓存重绘", "Refresh history": "刷新历史",
    "RX / Euler / Grain Size (cache)": "再结晶代理 / Euler 角 / 晶粒尺寸（缓存）",
    "RX phase rules": "各相再结晶代理规则", "Phase ID → thresholds JSON": "相位 ID → 阈值 JSON",
    "X Y WIDTH HEIGHT (blank = full scan)": "X Y 宽度 高度（留空表示整个扫描）",
    "Light / Dark": "浅色 / 深色", "Language": "语言", "English": "英语", "Chinese": "中文",
    "Source": "来源", "Scan": "扫描", "Nx × Ny": "Nx × Ny", "Step µm": "步长 µm",
    "Phases": "相", "Shards": "分块文件数", "Est. cache MB": "预计缓存 MB", "State": "状态",
    "Data / Queue": "数据 / 队列", "Logs": "日志", "History": "历史",
    "Mode": "模式", "Rows / height": "行数 / 高度", "Columns / width": "列数 / 宽度",
    "Edges": "边缘策略", "Fixed-size overlap": "固定尺寸切片重叠率", "Split specimen ID": "划分用试样 ID",
    "Same specimen must share this ID": "同一试样必须使用相同的 ID",
    "Include verified EDS": "包含已核对的 EDS", "Include cached grain fields (ROI must fit)": "包含缓存晶粒场（必须覆盖所选 ROI）",
    "3 rows × 5 columns = 15 geometric patches; not 15 specimens": "3 行 × 5 列 = 15 个几何切片；不代表 15 个独立试样",
    "Preview patch bounds": "预览切片边界", "Export numeric patches": "导出数值切片",
    "Build grain graph": "构建晶粒图", "Slicing / Graph": "切片 / 晶粒图",
    "Enabled": "启用", "Patch": "切片", "Pixel x y w h": "像素 x y 宽 高",
    "Physical x0 y0 x1 y1 µm": "物理范围 x0 y0 x1 y1 µm", "Approx indexed %": "预计标定比例 %",
    "Est bytes": "预计字节数", "Patch preview": "切片预览",
    "Read original H5OINA window intensities after verifying EBSD/EDS X/Y coordinates. Units remain cps or counts; no wt.% or at.%.":
        "核对 EBSD/EDS 的 X/Y 坐标后，读取原始 H5OINA 能窗强度。单位保持为 cps 或 counts；不转换为 wt.% 或 at.%。",
    "Convert explicit counts to counts/s when live time valid": "有效采集时间已确认时，将明确的 counts 转为 counts/s",
    "Import EDS + phase statistics": "导入 EDS 并统计各相",
    "Research only. Use original specimen/scan groups. At least 3 independent groups are required; patches from one scan are not independent samples.":
        "仅供研究。按原始试样或扫描分组，至少需要 3 个独立组；同一扫描的切片不能视为独立样本。",
    "No patch databases selected": "尚未选择切片数据库", "Select dataset_index.json files": "选择 dataset_index.json 文件",
    "Export protected GroupSplit": "导出防止来源泄漏的 GroupSplit", "Validated workflow": "带输入校验的工作流",
    "JSON configuration": "JSON 配置", "Load workflow JSON": "载入工作流 JSON",
    "Run configured workflow": "执行已配置的工作流", "Dataset Builder": "数据集构建",
    "One-hop learned mean-message GNN. Node responses require independently measured/simulated labels matched to the same specimen.":
        "使用一跳均值消息聚合的可训练 GNN。节点响应必须由同一试样独立测量或模拟的标签提供。",
    "Build graph from selected cache": "从所选缓存构建晶粒图", "Select labeled training bundle NPZ": "选择含标签的训练数据 NPZ",
    "No labels: supervised GNN disabled": "尚无标签：已禁用监督式 GNN", "Epochs": "训练轮数",
    "Optional complete checkpoint directory": "可选：完整 checkpoint 所在目录", "Weight resume": "从权重继续训练",
    "Train GNN with labels": "使用标签训练 GNN", "Synthetic GNN demo — software test only": "合成 GNN 演示 — 仅验证软件流程",
    "GNN Graphs": "GNN 晶粒图",
    "No mechanics truth is inferred from EBSD/EDS. Import DIC strain or declared HR-EBSD/CPFE/FEM labels through Dataset Builder. DIC alone does not establish full stress. Ridge is a numeric baseline, not U-Net/FNO.":
        "EBSD/EDS 不会被自动当作真实力学标签。请通过数据集构建导入 DIC 应变或明确来源的 HR-EBSD/CPFE/FEM 标签。DIC 本身不能确定完整应力。Ridge 仅为数值基线，未实现 U-Net/FNO。",
    "No labels: supervised mechanics disabled": "尚无标签：已禁用监督式力学训练", "Select independently labeled NPZ": "选择独立标签数据 NPZ",
    "Fit CPU ridge baseline with labels": "使用标签拟合 CPU Ridge 基线",
    "Inference uses a checkpoint and a schema-compatible numeric feature bundle; labels are unnecessary for inference. Set workflow to infer-supervised and provide checkpoint + bundle_path.":
        "推理需要 checkpoint 和结构兼容的数值特征数据，不要求推理标签。选择 infer-supervised 工作流，并提供 checkpoint 与 bundle_path。",
    "Mechanics Prediction": "力学预测研究",
    "Conditional DDPM of numeric xyzw orientation/phase fields. Tiny CPU research baseline (≤4096 encoded variables/tile); no validated mechanical inverse design.":
        "对数值 xyzw 取向场与相位场进行条件 DDPM 研究。该小型 CPU 基线每块最多支持 4096 个编码变量，尚未验证力学逆向设计能力。",
    "Select numeric diffusion bundle NPZ": "选择数值扩散数据 NPZ", "No numeric dataset selected": "尚未选择数值数据集",
    "Diffusion steps": "扩散步数", "Train research DDPM": "训练研究型 DDPM",
    "Checkpoint directory": "checkpoint 目录", "Declared conditions": "明确声明的条件",
    "Generate numeric research samples": "生成数值研究样本", "Synthetic diffusion demo — not real material": "合成扩散演示 — 不代表真实材料",
    "Diffusion": "扩散研究", "AI Research": "AI 研究",
    "Select patch dataset indexes": "选择切片数据集索引", "Workflow configuration": "工作流配置",
    "Labeled training bundle": "含标签的训练数据", "Numeric DDPM bundle": "数值 DDPM 数据",
    "No actual labels/groups/schema: supervised training remains disabled": "缺少实际标签、来源组或数据结构：监督训练仍保持禁用",
    "Mechanics label provenance is missing or unsupported": "力学标签来源信息缺失或不受支持",
    "Actual labels are required": "必须提供实际标签", "DDPM needs numeric orientation/phase fields and independent groups": "DDPM 需要数值取向场、相位场及独立来源组",
    "SYNTHETIC DEMO ONLY": "仅为合成演示", "Label schema found; full QC is checked in worker": "已找到标签结构；完整质量检查将在任务进程中执行",
    "research QC checked in worker": "研究质量检查将在任务进程中执行",
    "Import scans → choose analysis → run → inspect immutable results. Ctrl-drag ROI; wheel zoom; drag pan.":
        "导入扫描 → 选择分析 → 运行 → 检查独立归档结果。按住 Ctrl 拖动选择 ROI；滚轮缩放，拖动平移。",
    "Independent H5 files remain separate. Montage row shards are grouped using validated provenance.":
        "各独立 H5 文件分别处理。Montage 行分块依据已核对的来源信息进行分组。",
    "Standard IPF uses cached symmetry-aware orientation colors. Euler RGB is a separate representation.":
        "标准 IPF 使用缓存中考虑晶体对称性的取向颜色。Euler RGB 是另一种表示方式。",
    "Select V3 texture, Schmid, variants or prior-beta. External MTEX validation pending.":
        "可选择 V3 织构、Schmid 因子、取向变体或原始 β 重构；仍需开展 MTEX 外部校核。",
    "Quantitative proxy tools are enabled in M2; no metallurgical RX certification.":
        "M2 提供定量代理判据工具；不能据此认定材料已经发生冶金学意义上的再结晶。",
    "One analysis child at a time; failures do not stop subsequent samples. Frozen batch snapshot.":
        "每次只运行一个分析子任务；单个样本失败不会阻止后续样本。批次使用创建时固定的参数快照。",
    "Numeric slicing tools are enabled in M3; geometric patches are not independent specimens.":
        "M3 提供数值切片工具；几何切片不等于独立试样。",
    "EDS window intensity is not wt.% or at.%. EDS availability is checked separately.":
        "EDS 能窗强度不等于 wt.% 或 at.%。系统会单独检查 EDS 是否可用。",
    "Research tools require QC and explicit labels. Supervised mechanics unavailable without labels.":
        "研究工具需要质量检查和明确标签。没有标签时禁用监督式力学训练。",
    "Choose font and formats; Render Only uses the selected cache without scientific recomputation.":
        "选择字体和导出格式。Render Only 使用所选缓存重绘，不重新进行科学分析计算。",
    "Double-click an archived analysis to preview its cache. Existing results are never overwritten.":
        "双击归档分析可预览其缓存。已有结果不会被覆盖。",
    "Optional dependencies: PySide6 for Qt; torch only for AI. Tk fallback remains gui.py.":
        "可选依赖：Qt 界面使用 PySide6；AI 的可选工具使用 torch。仍可通过 gui.py 使用 Tk 界面。",
    "Select at least one export format": "请至少选择一种导出格式",
    "Select maps and a layout with enough cells": "请选择图谱，并选用格数足够的拼图布局",
    "ROI requires X Y WIDTH HEIGHT within scan": "ROI 必须为扫描范围内的 X Y 宽度 高度",
    "ROI exceeds a selected scan": "ROI 超出了某个所选扫描的范围",
    "Import one or more scans first": "请先导入一个或多个扫描", "A task is already running": "已有任务正在运行",
    "Open an archived features.h5 first": "请先打开归档的 features.h5", "Open features.h5 first": "请先打开 features.h5",
    "Rows and columns must be integers": "行数和列数必须为整数",
    "Preview limited to 10,000 patches; use a coarser grid": "预览最多支持 10,000 个切片；请使用更粗的网格",
    "Preview the current patch settings before export": "导出前请先预览当前切片设置",
    "EBSD Studio": "EBSD 分析工作室", "EBSD inputs": "EBSD 输入文件", "EBSD folder / montage": "EBSD 文件夹 / Montage 数据",
    "Select batch_manifest.json": "选择 batch_manifest.json", "Select features.h5": "选择 features.h5",
    "IDLE": "空闲", "RUNNING": "运行中", "SUCCESS": "成功", "FAILED": "失败", "CANCELLED": "已取消",
    "PAUSED": "已暂停", "RESUMED": "已继续", "pending": "等待中", "running": "运行中", "success": "成功",
    "failed": "失败", "cancelled": "已取消", "interrupted": "已中断", "queued": "排队中", "paused": "已暂停",
    "ready": "就绪", "unavailable": "不可用", "incomplete_training": "训练尚未完成", "Research only": "仅供研究",
    "FAILED: process could not start": "失败：无法启动任务进程",
    "Cancellation requested; preserving checkpoints": "已请求取消；将保留已提交的 checkpoint",
    "Pause is available for active cached/research actions; batches support safe Cancel/Resume.":
        "正在运行的缓存或研究任务支持暂停；批量任务支持安全取消后恢复。",
    "Pause requested": "已请求暂停", "Resume requested": "已请求继续",
    "Wait for input validation": "请等待输入校验完成", "Wait for safe task cancellation, then close": "请等待任务安全取消后再关闭",
    "ERROR: ": "错误：", "Process error: ": "进程错误：", " | KAM unavailable": " | KAM 不可用", "KAM unavailable": "KAM 不可用",
    "Error": "错误", "Warning": "提示", "Details": "详细信息", "Technical details": "原始技术详情",
    "The task failed. See technical details and logs.": "任务失败。请查看原始技术详情和日志。",
    "OK": "确定", "Cancel": "取消", "Close": "关闭", "Open": "打开", "Save": "保存", "Yes": "是", "No": "否",
    "Area weighted": "按面积加权", "Grain weighted": "按晶粒数加权", "Grid": "网格划分",
    "area": "按面积加权", "grain": "按晶粒数加权", "ERROR": "错误", "Cache (*.h5)": "缓存 (*.h5)",
    "View {number}: {error}": "视图 {number}：{error}",
    "The task failed. Open technical details for the original error.": "任务失败。请打开原始技术详情查看错误信息。",
    "Fixed pixel size": "固定像素尺寸", "Fixed physical size (µm)": "固定物理尺寸（µm）",
    "Preserve variable edge shape": "保留边缘实际尺寸", "Pad with validity mask": "填充并保存有效掩膜",
    "Drop incomplete edge patches": "明确丢弃不完整边缘切片",
    "{count} scans | {points:,} measured grid points": "{count} 个扫描 | {points:,} 个测量网格点",
    "{count} patch databases selected; specimen IDs must be checked": "已选择 {count} 个切片数据库；请核对试样 ID",
    "{status} | {name}": "{status} | {name}", "{name} | research QC checked in worker": "{name} | 研究质量检查将在任务进程中执行",
    "FAILED: exit {code}; {status}": "失败：退出码 {code}；{status}",
    "FAILED / CANCELLED: exit {code}; inspect manifest and log": "失败 / 已取消：退出码 {code}；请检查清单和日志",
    "{count} geometric patches | {width} × {height} pixels | {width_um:.3f} × {height_um:.3f} µm. Exact indexing fractions are saved at export.":
        "{count} 个几何切片 | {width} × {height} 像素 | {width_um:.3f} × {height_um:.3f} µm。导出时将保存准确的标定比例。",
    "Research GNN epoch {epoch}/{total}: validation MSE {loss}": "研究型 GNN 第 {epoch}/{total} 轮：验证 MSE {loss}",
    "DDPM epoch {epoch}/{total}: noise MSE {loss}; validation {validation}": "DDPM 第 {epoch}/{total} 轮：噪声 MSE {loss}；验证 {validation}",
    'Source cache changed. Preview the current patch settings before export.': '源缓存已改变。导出前请重新预览当前切片设置。',
}

ALIASES = {
    "Open cache / 打开缓存": "Open cache", "Workspace / 工作区": "Workspace", "Analysis / 分析": "Analysis",
    "Choose output / 选择输出": "Choose output", "Resume batch / 恢复批次": "Resume batch",
    "Render Only / 缓存重绘": "Render Only", "Refresh history / 刷新历史": "Refresh history",
    "Parameters / 参数": "Parameters", "Queue & Logs / 队列与日志": "Queue & Logs", "中文 / English": "Language",
    "Preview patch bounds / 预览切片": "Preview patch bounds", "Export numeric patches / 导出数值": "Export numeric patches",
    "Build grain graph / 晶粒建图": "Build grain graph", "EBSD 地图": "EBSD Maps", "AI 研究": "GNN / Mechanics / Diffusion",
    "导入文件": "Import files", "导入文件夹": "Import folder", "运行批量分析": "Run batch", "取消任务": "Cancel task",
}
PROTECTED = frozenset({"BC", "Phase", "KAM", "IPF_X", "IPF_Y", "IPF_Z", "MAD", "PQ", "EDS",
    "PNG", "TIFF", "PDF", "SVG", "v1", "v1v2", "v2", "v3_texture", "v3_schmid", "v3_variants", "v3_prior_beta", "v4_quantitative"})
SOURCE_ROLE_OFFSET = 731
SOURCE_ROLE = 0x0100 + SOURCE_ROLE_OFFSET


def canonical_source(text):
    """Resolve only catalog entries and explicit legacy bilingual aliases."""
    if not isinstance(text, str):
        return text
    if text in CATALOG or text in PROTECTED:
        return text
    if text in ALIASES:
        return ALIASES[text]
    matches = [key for key, value in CATALOG.items() if value == text]
    return matches[0] if len(matches) == 1 else text


def tr(text, lang="en", **fmt):
    """Translate an exact UI key, then interpolate values without translating them.

    Unknown technical text is returned verbatim; no guessed error translation.
    JSON and user strings with braces remain untouched when no fmt is supplied.
    """
    source = canonical_source(text)
    translated = CATALOG.get(source, source) if str(lang).lower().startswith("zh") else source
    return translated.format(**fmt) if fmt and isinstance(translated, str) else translated


def _qt():
    from PySide6 import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


@contextmanager
def _block_signals(obj):
    previous = obj.blockSignals(True)
    try:
        yield
    finally:
        obj.blockSignals(previous)


def skip_translation(widget, skip=True):
    """Exclude raw logs or a user-data-only widget from automatic capture."""
    widget.setProperty("ebsd_i18n_skip", bool(skip))
    return widget


def bind_text(widget, source, lang="en", property_name="text", **fmt):
    """Bind a static string or a dynamic template to a Qt presentation property."""
    bindings = getattr(widget, "_ebsd_i18n_bindings", {})
    value = tr(source, lang, **fmt)
    bindings[property_name] = {"source": canonical_source(source), "fmt": dict(fmt), "last": value}
    widget._ebsd_i18n_bindings = bindings
    setter = getattr(widget, "set" + property_name[0].upper() + property_name[1:])
    setter(value)
    return widget


def bind_combo_item(combo, index, source, lang="en"):
    """Translate display text only; keep the existing semantic UserRole value."""
    C, _, _ = _qt()
    role = int(C.Qt.ItemDataRole.UserRole) + SOURCE_ROLE_OFFSET
    combo.setItemData(index, canonical_source(source), role)
    with _block_signals(combo):
        combo.setItemText(index, tr(source, lang))


def bind_item(item, source, lang="en"):
    """Opt-in translation for a navigation/list/header item; never its user data."""
    C, _, _ = _qt()
    role = int(C.Qt.ItemDataRole.UserRole) + SOURCE_ROLE_OFFSET
    item.setData(role, canonical_source(source))
    item.setText(tr(source, lang))


def _property(widget, name, lang):
    getter = getattr(widget, name, None)
    setter = getattr(widget, "set" + name[0].upper() + name[1:], None)
    if not callable(getter) or not callable(setter):
        return
    current = getter()
    bindings = getattr(widget, "_ebsd_i18n_bindings", {})
    binding = bindings.get(name)
    if binding is None or current != binding["last"]:
        source = canonical_source(current)
        if source not in CATALOG:
            bindings.pop(name, None)  # A caller replaced the label with user data.
            widget._ebsd_i18n_bindings = bindings
            return
        binding = {"source": source, "fmt": {}, "last": current}
        bindings[name] = binding
    value = tr(binding["source"], lang, **binding["fmt"])
    setter(value)
    binding["last"] = value
    widget._ebsd_i18n_bindings = bindings


def translate_widget_tree(window, lang="en"):
    """Translate catalogued presentation text without rebuilding any widget.

    Editors/line-edit values, table cells, logs and unregistered combo items are
    deliberately not inspected. Selection, currentData, check states and signal
    wiring remain intact. Call again after creating a dynamic tab/header.
    """
    C, G, W = _qt()
    role = int(C.Qt.ItemDataRole.UserRole) + SOURCE_ROLE_OFFSET
    objects = [window, *window.findChildren(C.QObject)]
    for widget in objects:
        if widget.property("ebsd_i18n_skip"):
            continue
        with _block_signals(widget):
            for name in ("toolTip", "statusTip", "whatsThis", "placeholderText", "windowTitle"):
                _property(widget, name, lang)
            if isinstance(widget, (W.QLabel, W.QAbstractButton, G.QAction)):
                _property(widget, "text", lang)
            if isinstance(widget, (W.QGroupBox, W.QMenu)):
                _property(widget, "title", lang)
            if isinstance(widget, W.QComboBox):
                for index in range(widget.count()):
                    source = widget.itemData(index, role)
                    if source is not None:
                        widget.setItemText(index, tr(source, lang))
            if isinstance(widget, W.QTabWidget):
                for index in range(widget.count()):
                    page = widget.widget(index)
                    source = getattr(page, "_ebsd_i18n_tab_source", None)
                    if source is None:
                        candidate = canonical_source(widget.tabText(index))
                        if candidate in CATALOG:
                            source = candidate
                            page._ebsd_i18n_tab_source = source
                    if source is not None:
                        widget.setTabText(index, tr(source, lang))
            if isinstance(widget, W.QTableWidget):
                for index in range(widget.columnCount()):
                    item = widget.horizontalHeaderItem(index)
                    if item is not None:
                        source = item.data(role) or canonical_source(item.text())
                        if source in CATALOG:
                            bind_item(item, source, lang)
            if isinstance(widget, W.QListWidget):
                # Only registered static entries; no history/file-name guessing.
                navigation = getattr(window, "nav", None) is widget
                for index in range(widget.count()):
                    item = widget.item(index)
                    source = item.data(role)
                    if source is None and navigation:
                        candidate = canonical_source(item.text())
                        if candidate in CATALOG:
                            source = candidate
                    if source is not None:
                        bind_item(item, source, lang)
    window.setProperty("ebsd_i18n_language", lang)


def install_qt_translator(app, lang="en"):
    """Install bundled Qt standard-dialog Chinese text when available, no download."""
    C, _, _ = _qt()
    old = getattr(app, "_ebsd_qt_translator", None)
    if old is not None:
        app.removeTranslator(old)
        app._ebsd_qt_translator = None
    if not str(lang).lower().startswith("zh"):
        return True
    translator = C.QTranslator(app)
    location = C.QLibraryInfo.path(C.QLibraryInfo.LibraryPath.TranslationsPath)
    loaded = translator.load("qtbase_zh_CN", location)
    if loaded:
        app.installTranslator(translator)
        app._ebsd_qt_translator = translator
    return bool(loaded)


def ui_literal_strings(source):
    """AST inventory of static presentation literals, excluding data/code IDs."""
    tree = ast.parse(source)
    found = set()
    constructors = {"QLabel", "QPushButton", "QCheckBox", "QDockWidget", "QGroupBox", "QRadioButton", "QAction", "QMenu"}
    setters = {"setWindowTitle", "setToolTip", "setStatusTip", "setWhatsThis", "setPlaceholderText", "addToolBar", "addRow", "error", "appendPlainText"}
    def add(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.strip():
            found.add(node.value)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in constructors | setters | {"ValueError", "tr", "t", "_text"} and node.args:
                add(node.args[0])
            elif name in {"addTab", "add_tab", "getOpenFileName", "getOpenFileNames", "getExistingDirectory", "warning", "critical", "information", "ui"} and len(node.args) > 1:
                add(node.args[1])
            elif name == "setHorizontalHeaderLabels" and node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                for child in node.args[0].elts:
                    add(child)
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if "NAV" in names and isinstance(node.value, (ast.List, ast.Tuple)):
                for pair in node.value.elts:
                    if isinstance(pair, (ast.List, ast.Tuple)) and pair.elts:
                        add(pair.elts[0])
            if "descriptions" in names and isinstance(node.value, ast.Dict):
                for child in node.value.values:
                    add(child)
    return found


def missing_ui_strings(source):
    return sorted(text for text in ui_literal_strings(source)
                  if canonical_source(text) not in CATALOG and text not in PROTECTED)
