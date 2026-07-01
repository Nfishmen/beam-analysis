[README.md](https://github.com/user-attachments/files/29558937/README.md)
# 🏗️ 梁结构受力分析系统 (Beam Structure Analysis System)

基于 **YOLO 目标检测** + **材料力学计算** 的梁结构自动识别与分析系统。

上传一张梁结构示意图，系统自动识别梁类型、支座、载荷，计算剪力/弯矩/挠度/应力，并生成可视化报告。

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. (可选) 生成训练数据 + 训练 YOLO 模型
python train_all.py

# 3. 启动 Web 服务
python 启动.bat          # Windows 双击运行
# 或
python app.py            # 命令行启动 → 浏览器访问 http://127.0.0.1:5000
```

---

## 📋 功能特性

| 功能 | 说明 |
|------|------|
| **YOLO 检测** | 识别梁、固定支座、铰支座、滚动支座、集中力、分布载荷（6类） |
| **力学分析** | 自动计算支座反力、剪力(SFD)、弯矩(BMD)、挠度、弯曲应力 |
| **可视化** | 生成 SFD/BMD/挠度/斜率/应力分布图 + 截面示意图 |
| **Web 界面** | Flask 拖拽上传，实时展示分析结果 |
| **Demo 模式** | 无 YOLO 模型时自动降级为演示模式，使用内置算例 |
| **合成数据生成** | 程序化生成 400 张训练图 + 100 张验证图（含 YOLO 标注） |

---

## 🧠 YOLO 检测类别

| ID | 类别 | 说明 |
|----|------|------|
| 0 | `beam` | 梁主体 |
| 1 | `fixed_support` | 固定支座 |
| 2 | `pinned_support` | 铰支座 |
| 3 | `roller_support` | 滚动支座 |
| 4 | `point_load` | 集中力 |
| 5 | `distributed_load` | 分布载荷 |

---

## 📐 力学计算

支持的梁类型：
- **简支梁** (Simply Supported)
- **悬臂梁** (Cantilever)
- **两端固定梁** (Fixed-Fixed)
- **外伸梁** (Overhanging)

支持的截面类型：
- 矩形 (Rectangle)
- 圆形 (Circle)
- 工字钢 (I-Beam)
- 空心矩形 (Hollow Rect)

计算方法：静力平衡方程求解反力 → 数值积分求剪力/弯矩 → 双积分法求挠度 → 弯曲应力 σ = My/I

---

## 📁 项目结构

```
beam_analysis/
├── app.py                      # Flask Web 服务入口
├── main.py                     # 命令行演示入口（无需 Web）
├── pipeline.py                 # 完整分析流水线：检测→解析→计算→可视化
├── generate_training_data.py   # 合成训练数据生成器
├── generate_test_images.py     # 测试用梁图生成器
├── train_all.py                # 一键：生成数据 + 训练 YOLO
├── verify_mechanics.py         # 力学计算验证（对比解析解）
├── thermo_exam.py              # 热力学考试 Word 文档生成器 (v1)
├── thermo_exam_v2.py           # 热力学考试 Word 文档生成器 (v2 紧凑版)
├── requirements.txt            # Python 依赖清单
├── 启动.bat                    # Windows 一键启动脚本
├── .gitignore                  # Git 忽略规则
├── best.pt                     # 训练好的 YOLO 模型权重
├── yolo11n.pt                  # YOLO11 Nano 预训练权重
│
├── src/                        # 核心源码模块
│   ├── __init__.py
│   ├── detector.py             # YOLO 检测器封装
│   ├── geometry.py             # 检测结果→力学模型解析器
│   ├── mechanics.py            # 材料力学计算引擎
│   └── visualize.py            # Matplotlib 可视化（SFD/BMD/挠度/应力）
│
├── templates/
│   └── index.html              # Web 前端页面（拖拽上传 + 结果展示）
│
├── static/
│   ├── uploads/                # 用户上传的原始图片
│   └── results/                # 生成的分析图表 PNG
│
├── yolo/
│   └── dataset/
│       ├── data.yaml           # YOLO 数据集配置
│       ├── images/
│       │   ├── train/          # 训练图片（400 张合成梁图）
│       │   └── val/            # 验证图片（100 张合成梁图）
│       └── labels/
│           ├── train/          # 训练标注（YOLO 格式 .txt）
│           └── val/            # 验证标注（YOLO 格式 .txt）
│
└── runs/
    └── detect/                 # YOLO 训练输出（权重、日志、指标）
```

---

## 🔍 各文件夹作用说明

| 文件夹 | 作用 |
|--------|------|
| **`src/`** | **核心算法模块** — 包含 YOLO 检测器 (`detector.py`)、几何解析器 (`geometry.py`)、材料力学计算引擎 (`mechanics.py`)、可视化工具 (`visualize.py`)。这是系统的"大脑"，所有计算逻辑都在这里。 |
| **`templates/`** | **Web 前端页面** — 存放 Flask 渲染的 HTML 模板 (`index.html`)，提供拖拽上传、JSON 数据展示、图表预览的完整交互界面。 |
| **`static/`** | **静态资源目录** — `uploads/` 存放用户上传的梁图原图，`results/` 存放系统生成的 SFD/BMD/应力分布等分析报告图片。二者都通过 `/static/...` 路由对外提供访问。 |
| **`yolo/`** | **YOLO 数据集** — 存放训练 YOLO 目标检测模型所需的全部数据：`images/` 是合成梁结构图，`labels/` 是对应的 YOLO 格式标注文件（每行一个目标框），`data.yaml` 定义 6 个检测类别和数据集路径。 |
| **`runs/`** | **YOLO 训练输出** — Ultralytics 训练过程中自动生成：模型权重 (`best.pt`, `last.pt`)、训练曲线、验证结果、ONNX 导出等。每次训练会产生一个带编号的子目录。 |
| **`__pycache__/`** | Python 字节码缓存，自动生成，可忽略。 |

### 根目录关键文件补充

| 文件 | 作用 |
|------|------|
| `app.py` | Flask Web 服务器，提供上传接口 `/upload` 和前端页面 |
| `pipeline.py` | 完整流水线：接收图片 → YOLO检测 → 几何解析 → 力学计算 → 生成图表，是 `app.py` 的后端核心 |
| `main.py` | 命令行演示脚本，无需启动 Web 即可测试简支梁/悬臂梁两种工况 |
| `generate_training_data.py` | 程序化生成 500 张带标注的合成梁图（400训练+100验证），支持 5 种梁型、随机载荷、样式变化 |
| `train_all.py` | 一键训练脚本：先调用数据生成，再调用 Ultralytics YOLO 进行 100 epoch 训练 |
| `verify_mechanics.py` | 力学验证：用 3 个标准工况（简支梁中点集中力、悬臂梁均布载荷、简支梁全跨均布载荷）对比数值解与解析解，确保计算正确 |
| `best.pt` | 已训练完成的 YOLO 检测模型权重文件 |
| `yolo11n.pt` | YOLO11 Nano 预训练权重，作为迁移学习的起点 |
| `启动.bat` | Windows 批处理，自动打开浏览器并启动 Flask 服务 |
