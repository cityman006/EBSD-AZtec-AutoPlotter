# V4.0 Materials Intelligence Studio 使用说明（开发版）

当前版本 `4.0.0.dev1`。这是可运行的研究开发版；界面中的 SUCCESS 仅表示该次软件任务正常完成，不代表材料科学结论已通过独立验证。

## 启动与迁移

在本项目目录双击 `START_QT_WINDOWS.cmd` 打开 V4 Qt 界面。首次使用在安装 Python 3.12 x64 后运行 `INSTALL_QT_WINDOWS.cmd`。旧 Tk 界面仍使用 `START_WINDOWS.cmd`。

复制到另一台 Windows 电脑时，复制源代码、脚本和说明文档；在新电脑重新运行安装脚本建立 `.venv`。原电脑的 `.venv` 和快捷方式不能保证可迁移。基础软件与 NumPy 研究模型不要求 CUDA、MATLAB 或 MTEX；Torch 仅为可选 Dataset/DataLoader 适配器依赖。不要把实验数据、缓存或模型权重误传到公共仓库。

## 批量 EBSD 分析

1. 用“导入文件 / Import files”选择多个 H5/H5OINA，或“导入文件夹 / Import folder”选择目录，也可以拖入文件。数据表显示实际扫描、尺寸、步长、相位及分片数。同一扫描的 montage 分片合为一个任务。
2. 设置 Output root，选择 Analysis。`v1` 为基础地图；`v1v2` 加晶粒；`v4_quantitative` 生成再结晶代理指标、晶粒尺寸、Euler 和逐相数值表。`v3_texture`、`v3_schmid`、`v3_variants`、`v3_prior_beta` 保留。
3. 大图先运行 V1，再打开缓存并 Ctrl 拖拽选择 ROI。V2/定量分析有约 220 万像素安全上限；不要把分块读取误认为全部晶粒算法均支持无限大的全图。
4. 字体从已安装字体中选择；PNG、TIFF、PDF、SVG 可以分别勾选。选图数量不能超过拼图格数。Block rows 控制行块大小；Memory budget 是估算门禁，不是系统强制内存上限。
5. 点击“运行批量分析”。当前批量采用一个子进程依次处理各扫描；运行时改参数不会影响已经冻结的批次。单个失败后继续其他样本。

`output_root/batches/batch_.../` 保存批次清单、日志、成功/失败表；定量批次另有 `Summary.csv`。每个扫描具有独立 analysis run 和 render run，旧结果不覆盖。Summary 中保留来源、范围、单位、算法设置、QC 和不可比提示；相同设置不代表不同材料天然可比。

## 缓存浏览与独立重绘

“Open cache / 打开缓存”选择归档的 `analysis_cache/features.h5`。它保存数值取向、相位、KAM、BC 及 mask，不是多余的原始数据副本。顶部选择 BC、Phase、KAM、IPF-X/Y/Z；滚轮缩放、拖拽平移，Ctrl 拖拽 ROI。状态栏显示原始像素和物理坐标。视图使用有界预览与异步、多层磁盘瓦片；不通过截图重算科学数值。

只改字体、选图或排版时点击 Render Only，直接使用缓存，在原归档下创建新的独立 render 目录，不重读原始 H5。单图 PNG/TIFF 保留原始栅格；PDF/SVG 中图像仍是栅格，文字和坐标为矢量。它们不把每个 EBSD 像素变成独立矢量对象。AZtec 颜色一致性尚未独立校准。

“RX / Euler / Grain Size (cache)”在选中的归档缓存上另建 `studio_runs/quantitative_.../results/`。RX phase rules 为按**实际相位 ID**配置的 JSON；默认阈值只是研究代理，需结合材料、热处理与独立实验校准。未索引、小样本、无效 KAM 或缺少可靠 β 统计会保留 unavailable/unclassified，不填入示例值。Euler RGB 不是 IPF 颜色。PF 的 MUD、ODF 的 MRD 与 IPF 方向散点有不同含义；散点不能称为密度。

## 切片、EDS 与图数据库

在 Slicing 工具中选择 grid（默认 3 行×5 列）、pixels 或 µm，明确边缘保留/带 mask 填充/显式丢弃和重叠设置，先 Preview，再逐片勾选并 Export。3×5 表示一个扫描内的 15 个几何区域，不是 15 个独立试样。相邻切片及同一试样必须使用相同 Split specimen ID，防止训练/测试泄漏。

导出的 H5 保留原始七通道数值、mask、坐标和来源；索引为 `dataset_index.json` 和 CSV。EDS 必须通过网格与全坐标核对，或使用显式配准。H5OINA 的 Window Integral 若原单位为 cps 就保留 cps，不再次除以采集时间；不从强度推断 wt.% 或 at.%。跨 montage 的 EDS 自动拼接和交互式地标编辑尚未实现。

Graph 动作以 V2 实际晶粒和四邻域共享边界建图，保存晶粒 ID、物理长度、取向、QC 与原始相位含义。小晶粒、孤立节点、界面角度缺失不会被伪造为有效力学数据。

## AI Research 四个页签

- Dataset Builder：选择多个 patch index 导出按母样本隔离的 GroupSplit。真实训练至少需要三个独立来源组。高级工作流通过明确 JSON 导入力学标签、配对图与标签、构建数值扩散数据或执行预测；协议见 [M4 文档](V4_M4.md)。
- GNN Graphs：建立图，选择有独立标签与来源协议的 NPZ 后训练；无标签时按钮禁用。Synthetic GNN demo 仅验证软件流程。
- Mechanics Prediction：真实拟合的 CPU ridge 基线与已保存模型预测；不是 U-Net/FNO。应力/应变真值必须来自明确的外部来源和单位，DIC 本身不提供完整应力。真实预测必须匹配训练的材料、加载条件、坐标系、载荷步与时间。
- Diffusion：训练小型条件 DDPM，或根据已完成的模型生成数值四元数/相位场。支持真实相位的 HCP/BCC 对称等价约化，但不代表生成组织已验证。每片最多 4096 个编码变量；目标相分数不保证被满足。力学性能条件与材料反向设计当前禁用。

Pause / Resume cached action 保持当前研究子进程状态；Cancel 安全中断，不报告 SUCCESS。训练在每个完整 epoch checkpoint 以事务方式保存。选择该 checkpoint 可继续**权重训练**，Adam 和随机数轨迹不是精确恢复。中间模型不可当作最终模型推理。批量分析使用 Cancel 和 Resume batch；恢复时重新检查来源与完整结果摘要。

## 验证边界

当前有自动数值、档案、GUI 子进程、取消/恢复和真实 H5OINA 小样本导出检查。尚缺独立 MTEX/AZtec 数值校核、再结晶冶金标准标定、真实力学标签训练/盲测、生成组织真实性评估、GPU 与实际 4K 多屏测试。完整阶段状态及证据见 [V4_PROGRESS.md](../V4_PROGRESS.md)。不要把上述未验证项目用于正式材料性能结论。
