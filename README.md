# TinyGo - 围棋AI项目

基于深度学习的围棋AI项目，包含CNN模型训练、数据获取和对弈系统。

## 项目概述

TinyGo是一个完整的围棋AI系统，主要包含以下组件：

- **CNN训练系统** - 基于卷积神经网络的围棋策略网络训练
- **数据获取模块** - 从KataGo Archive下载和处理围棋对局数据
- **对弈界面** - 命令行和GUI界面的围棋对弈程序

## 项目结构

```
TinyGo/
├── CNN/                    # CNN训练系统
│   ├── datasets.py         # 数据加载和预处理
│   ├── model.py           # CNN模型定义
│   ├── train.py           # 训练入口
│   ├── play.py            # 命令行对弈
│   ├── go_gui.py          # GUI对弈程序
│   └── output/            # 模型输出目录
├── DownloadData/          # 数据获取工具
│   ├── katago_downloader.py      # KataGo数据下载器
│   ├── filter_19x19_large_scale.py # 数据筛选工具
│   └── validate_data_quality.py   # 数据质量验证
├── CLAUDE.md             # Claude Code 项目指导文档
└── README.md             # 项目说明文档
```

## 快速开始

### 1. 训练模型

```bash
cd CNN
python -m CNN.train --board-size 19 --output-dir ./output --epochs 10
```

### 2. 命令行对弈

```bash
cd CNN
python -m CNN.play --checkpoint ./output/sequential_training/checkpoint_best.pt
```

### 3. GUI对弈

```bash
cd CNN
python -m CNN.go_gui --checkpoint ./output/sequential_training/checkpoint_best.pt
```

## 技术特性

- **CNN架构**: 6层残差网络的深度学习模型
- **序列预测**: 基于围棋对局序列的正确训练策略
- **数据管道**: 完整的数据获取、筛选和验证流程
- **多接口支持**: 命令行和图形界面对弈程序
- **自动检查点**: 支持训练中断恢复和最佳模型保存

## 模型性能

项目包含一个预训练的最佳模型 `checkpoint_best.pt`，该模型经过11个epoch的训练，能够进行合理的围棋对弈。

## 数据来源

- **KataGo Archive**: 高质量的19x19围棋对局数据
- **数据量**: 超过80万个有效对局
- **格式**: SGF格式转换为序列训练数据

## 系统要求

- Python 3.8+
- PyTorch
- NumPy
- tkinter (GUI界面)

## 许可证

本项目采用开源许可证，欢迎学习和贡献。

---

**作者**: LaoChouPro
**项目地址**: https://github.com/LaoChouPro/TinyGo