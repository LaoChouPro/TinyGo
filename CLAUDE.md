# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于深度学习的围棋AI项目，包含CNN模型训练、数据获取和对弈系统。项目主要由以下几个核心组件构成：

1. **CNN训练系统** (`CNN/`) - 基于卷积神经网络的围棋策略网络训练
2. **数据获取模块** (`DownloadData/`) - 从KataGo Archive下载和处理围棋对局数据
3. **训练数据** (`Training_data/`, `Raw_data/`) - 处理后的围棋对局数据集
4. **对弈界面** - 基于tkinter的GUI围棋对弈程序

## 常用命令

### CNN模型训练
```bash
# 基础训练（使用默认数据路径）
cd CNN
python -m CNN.train --board-size 19 --output-dir ./output --epochs 10

# 快速测试训练
python -m CNN.train --board-size 19 --output-dir ./output/debug --epochs 1 --steps-per-epoch 200 --eval-steps 50 --batch-size 128

# 恢复训练
python -m CNN.train --board-size 19 --resume ./output/checkpoint_latest.pt

# 指定特定数据文件训练
python -m CNN.train --board-size 19 --data-paths ../Training_data/1.data ../Training_data/10.data
```

### 对弈程序
```bash
# 命令行对弈
cd CNN
python -m CNN.play --checkpoint ./output/checkpoint_latest.pt --human-color B

# GUI对弈
python -m CNN.go_gui --checkpoint ./output/checkpoint_latest.pt --board-size 19
```

### 数据获取和处理
```bash
# 下载KataGo数据
cd DownloadData
python3 katago_downloader.py --dir download_target --max-files 100 --delay 2.0

# 筛选19x19对局
python3 filter_19x19_large_scale.py

# 验证数据质量
python3 validate_data_quality.py
```

## 架构说明

### CNN训练系统架构

**核心组件：**
- `datasets.py` - 数据加载和预处理，支持.data格式的序列预测训练
- `model.py` - CNN模型定义，包含残差块和策略头
- `trainer.py` - 训练循环管理，支持SGD优化器和检查点保存
- `train.py` - 训练入口点，处理命令行参数和配置
- `config.py` - 训练配置管理，自动检测数据文件路径

**数据格式：**
- 训练数据存储在 `../Training_data/` 目录下的 `.data` 文件中
- 数据格式：`[{"B":[x,y]}, {"W":[x,y]}, ...]` 表示对局序列
- 坐标系统：1-indexed坐标（标准围棋记谱法）
- 训练策略：序列预测，每步棋生成一个训练样本

**模型架构：**
- 输入：多通道特征平面（当前棋盘状态）
- 网络：6个残差块的CNN架构
- 输出：19x19位置的概率分布

### 数据处理流水线

**数据获取：**
- `katago_downloader.py` - 从KataGo Archive下载SGF文件
- `filter_19x19_large_scale.py` - 筛选19x19对局
- `validate_data_quality.py` - 数据质量验证

**数据转换：**
- 原始SGF文件 → `.data` 格式的序列数据
- 自动检测和加载 `Training_data/` 目录下的所有数据文件

### 对弈系统

**命令行对弈 (`play.py`)：**
- 支持人机对弈
- 可选择执黑/执白
- 显示top-k AI建议

**GUI对弈 (`go_gui.py`)：**
- 基于tkinter的图形界面
- 19x19标准围棋棋盘
- 实时AI对弈

## 重要配置

### 训练参数
- 默认批大小：256
- 学习率：0.05 (SGD with momentum)
- 设备：自动检测CUDA/CPU
- 数据工作进程：4

### 数据路径配置
- 训练数据：`../Training_data/*.data` (自动检测)
- 原始数据：`../Raw_data/` (SGF文件)
- 输出目录：`./output/` (可配置)

## 版本历史

项目经历了重要的数据格式和训练策略修正：
- **v2.2.0**: 实现正确的序列预测训练，解决AI只从边缘下棋的问题
- **v2.1.0**: 尝试SGF文件方案（已废弃）
- **v2.0.0**: 适配Training_data数据结构

## 依赖要求

主要Python依赖：
- torch (PyTorch)
- numpy
- tkinter (GUI)
- pathlib, argparse, logging (标准库)

## 注意事项

1. **数据格式重要性**：确保使用正确的.data格式进行序列预测训练
2. **模型路径**：训练完成后模型保存在 `output/checkpoint_latest.pt`
3. **设备兼容性**：代码自动检测CUDA可用性，可在CPU/GPU上运行
4. **中文交互**：项目配置要求使用中文与用户交流