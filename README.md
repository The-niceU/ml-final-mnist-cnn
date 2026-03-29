# MNIST Handwritten Digit Classification

基于 PyTorch 的 MNIST 手写数字分类课程项目，支持本地 IDX 数据读取、多次重复实验、Accuracy/F1 统计与命令行复现。

## 项目简介

本项目用于机器学习课程期末实验展示，目标是构建一个可复现、可扩展、可发布的图像分类基线系统。项目采用卷积神经网络（CNN）在 MNIST 数据集上完成 10 类手写数字识别。

## 功能特性

- 读取本地 MNIST 原始 IDX 文件（`*.ubyte`）
- 基于 PyTorch 的 CNN 训练与评估
- 多次重复实验并统计均值与标准差
- 输出 `Accuracy` 与 `Macro-F1`
- 命令行参数化配置，便于复现实验

## 项目结构

```text
.
├─ .github/workflows/ci.yml          # GitHub Actions 持续集成
├─ data/
│  └─ MNIST/raw/                     # 本地数据目录（建议不上传）
├─ src/mnist_project/
│  ├─ __init__.py
│  ├─ cli.py                         # 命令行入口
│  ├─ data.py                        # IDX 数据加载
│  ├─ engine.py                      # 训练与评估流程
│  └─ model.py                       # CNN 模型定义
├─ tests/
│  ├─ conftest.py
│  └─ test_data_loading.py
├─ mnist_final_cpu.py                # 兼容旧入口
├─ pyproject.toml
├─ requirements.txt
├─ REPORT.md                         # 个人课程报告（本仓库新增）
├─ 说明.ipynb                        # 教师提供文件
└─ 说明.txt                          # 教师提供文件
```

## 环境配置

推荐 Python 版本：`3.9+`

```bash
pip install -r requirements.txt
pip install -e .
```

## 数据准备

将以下文件放入 `data/MNIST/raw/`：

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

> 说明：`*.gz` 为压缩包，`*.ubyte` 为解压后的原始文件；训练脚本默认读取 `*.ubyte`。

## 快速开始

### 1) 标准实验（CPU）

```bash
python -m mnist_project.cli --data-root data/MNIST/raw --device cpu --runs 5 --epochs 5
```

### 2) 快速自检（低耗时）

```bash
python -m mnist_project.cli --data-root data/MNIST/raw --device cpu --runs 1 --epochs 1 --quick-train-samples 2000 --quick-test-samples 1000
```

### 3) 兼容旧脚本

```bash
python mnist_final_cpu.py --data-root data/MNIST/raw
```

## 常用参数

- `--runs`：重复实验次数
- `--epochs`：每次实验训练轮数
- `--batch-size`：训练批大小
- `--test-batch-size`：测试批大小
- `--lr`：学习率
- `--device`：`cpu` / `cuda` / `auto`
- `--seed`：随机种子

## 评估指标

- `Accuracy`：总体分类正确率
- `F1-score (macro)`：对每个类别计算 F1 后取算术平均，适合多分类整体性能评估

## 测试

```bash
pytest -q
```

## 结果展示建议（GitHub）

建议在仓库中增加以下内容以提升展示质量：

1. 训练损失曲线（Loss Curve）
2. 测试集混淆矩阵（Confusion Matrix）
3. 错分样本可视化（Misclassified Cases）
4. 消融实验表格（如学习率、批大小、轮数）

## 学术说明

- `说明.ipynb`、`说明.txt` 为课程教师提供材料，不属于本仓库作者原创研究内容。
- 本仓库中的实验实现、工程化重构与报告文档（如 `REPORT.md`）为作者个人实践成果。

## 发布建议

- 建议不上传 `data/MNIST/raw/` 原始数据文件
- 仓库标签建议：`mnist`、`pytorch`、`cnn`、`machine-learning`
- 首次发布可创建 `v1.0.0` Release，并附实验截图

## 致谢

- MNIST dataset: Yann LeCun et al.
- Course staff for assignment guidance and baseline materials.
