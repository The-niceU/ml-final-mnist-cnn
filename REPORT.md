# 机器学习期末项目报告（个人版）

> 课程：机器学习引论  
> 题目：基于 CNN 的 MNIST 手写数字识别  
> 作者：`请填写姓名`  
> 学号：`请填写学号`  
> 日期：`2026-03-29`

---

## 1. 摘要

本项目面向 MNIST 手写数字分类任务，构建了一个可复现的卷积神经网络（CNN）基线系统。实验在本地 CPU（可扩展到 CUDA）环境下运行，通过多次重复实验统计 `Accuracy` 与 `Macro-F1` 的均值与标准差，以降低偶然性影响。结果表明，所构建方法在较低工程复杂度下能够稳定取得较好的分类性能。

---

## 2. 问题定义

### 2.1 任务描述

给定输入图像 $x \in \mathbb{R}^{1\times 28 \times 28}$，预测类别标签 $y \in \{0,1,\dots,9\}$，这是典型的 10 类多分类问题。

### 2.2 数据集说明

MNIST 数据集包含：

- 训练集：60000 张灰度图像
- 测试集：10000 张灰度图像
- 图像尺寸：$28\times 28$

本项目读取 `IDX` 原始文件（`*.ubyte`），并完成归一化预处理（像素值缩放至 $[0,1]$）。

---

## 3. 方法设计

### 3.1 模型结构

采用两层卷积 + 两层全连接的 CNN：

1. `Conv2d(1, 32, kernel=3)` + ReLU + MaxPool
2. `Conv2d(32, 64, kernel=3)` + ReLU + MaxPool
3. `Linear(64*5*5, 128)` + ReLU
4. `Linear(128, 10)`

### 3.2 损失函数与优化器

- 损失函数：交叉熵损失（CrossEntropyLoss）
- 优化器：Adam
- 学习率：$1\times 10^{-3}$

### 3.3 评估指标

- 分类准确率（Accuracy）
- 宏平均 F1（Macro-F1）

其中，宏平均 F1 公式为：

$$
F1_{macro} = \frac{1}{K}\sum_{k=1}^{K} F1_k, \quad K=10
$$

---

## 4. 实验设置

### 4.1 运行环境

- 操作系统：Windows
- 深度学习框架：PyTorch
- Python：建议 3.9+
- 设备：CPU（可选 CUDA）

### 4.2 超参数

- `runs=5`
- `epochs=5`
- `batch_size=64`
- `test_batch_size=1000`
- `learning_rate=0.001`
- `seed=42`

### 4.3 复现实验命令

```bash
python -m mnist_project.cli --data-root data/MNIST/raw --device cpu --runs 5 --epochs 5
```

快速自检：

```bash
python -m mnist_project.cli --data-root data/MNIST/raw --device cpu --runs 1 --epochs 1 --quick-train-samples 2000 --quick-test-samples 1000
```

---

## 5. 实验结果

> 请将你实际运行输出填入以下表格。

| 指标     | 平均值 | 标准差 |
| -------- | -----: | -----: |
| Accuracy | `0.9912 ` | `0.0013` |
| Macro-F1 | `0.9911` | `0.0013` |

### 5.1 结果分析（示例写法）

1. 多次重复实验后，Accuracy 与 Macro-F1 的方差较小，说明训练流程稳定。
2. 该 CNN 结构参数量适中，在 CPU 环境下训练时间可接受。
3. 部分类别（如 `4` 与 `9`）可能存在混淆，后续可通过数据增强或更深网络改进。

---

## 6. 消融与扩展（可选加分）

可从以下方向扩展：

1. **学习率消融**：比较 `1e-2 / 1e-3 / 1e-4`
2. **训练轮数消融**：比较 `3 / 5 / 10` 轮
3. **批大小消融**：比较 `32 / 64 / 128`
4. **结构改进**：加入 BatchNorm、Dropout
5. **可视化增强**：混淆矩阵、错分样本图

---

## 7. 结论

本文完成了一个工程化的 MNIST 分类基线，具备如下特点：

- 可复现：固定随机种子与命令行参数
- 可验证：统一指标输出并支持重复实验统计
- 可维护：训练流程、模型与数据读取模块化

后续将引入更系统的超参数搜索与更丰富的可视化分析，以提升模型精度与报告说服力。

---

## 8. 个人工作说明

- 课程教师提供材料：`说明.ipynb`、`说明.txt`
- 个人完成内容：项目工程化重构、训练评估代码、README 展示文档与本报告

---

## 9. 参考资料

1. LeCun, Y., Cortes, C., & Burges, C. J. C. The MNIST Database of Handwritten Digits.
2. Paszke, A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library.
3. scikit-learn documentation: F1-score and classification metrics.
