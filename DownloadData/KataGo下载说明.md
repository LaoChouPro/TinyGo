# KataGo 19x19自我对弈数据下载方案

## 项目概述

我已经深入研究了KataGo Archive网站的结构，并开发了一个完整的批量下载方案，用于获取19x19的KataGo自我对弈完整对局数据。

## 数据源信息

**网站地址**: https://katagoarchive.org/kata1/traininggames/

**数据特点**:
- **棋盘大小**: 19x19（我验证了解压后的SGF文件，发现有些是17x17，需要筛选）
- **数据格式**: SGF (Smart Game Format) + tar.bz2压缩
- **对局类型**: 自我对弈训练对局
- **时间范围**: 2020-12-08 到 2021-10-20
- **数据量**: 约300+个压缩包，总计30GB+
- **包含信息**: MCTS分析数据、胜率预测、分数估计等

## 文件命名规则

格式: `YYYY-MM-DDsgfs.tar.bz2`
- 例如: `2021-10-20sgfs.tar.bz2`
- 每个压缩包包含当天的所有训练对局

## 下载工具特性

### 主要功能
1. **智能断点续传** - 支持网络中断后继续下载
2. **429错误防护** - 自动处理请求过于频繁的问题
3. **进度跟踪** - 实时显示下载进度和状态
4. **选择性下载** - 支持日期范围和文件数量限制
5. **自动解压** - 可选的自动解压功能
6. **日志记录** - 完整的下载日志

### 防止429错误的策略
- **请求间隔**: 默认2秒间隔
- **指数退避**: 遇到429错误时自动增加等待时间
- **最大重试**: 每个文件最多重试3次
- **会话管理**: 使用持久会话减少连接开销

## 使用方法

### 基本用法

```bash
# 下载所有文件
python katago_downloader.py

# 指定下载目录
python katago_downloader.py --dir my_katago_data

# 设置请求间隔（避免429错误）
python katago_downloader.py --delay 3.0

# 限制下载文件数量（测试用）
python katago_downloader.py --max-files 10
```

### 高级用法

```bash
# 下载指定日期范围
python katago_downloader.py --start-date 2021-01-01 --end-date 2021-03-31

# 下载最近30天的数据
python katago_downloader.py --start-date 2021-09-20 --max-files 30

# 增加重试次数和延迟
python katago_downloader.py --max-retries 5 --delay 5.0

# 自动解压下载的文件
python katago_downloader.py --extract
```

### 命令行参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--dir` | `-d` | `katago_games` | 下载目录 |
| `--delay` | | `2.0` | 请求间隔秒数 |
| `--max-retries` | | `3` | 最大重试次数 |
| `--start-date` | | | 开始日期 (YYYY-MM-DD) |
| `--end-date` | | | 结束日期 (YYYY-MM-DD) |
| `--max-files` | | | 最大下载文件数量 |
| `--extract` | | `False` | 自动解压下载的文件 |

## 数据验证

我已经下载并验证了样本数据：

### SGF文件格式示例
```
(;FF[4]GM[1]SZ[19]PB[kata1-b40c256-s10289624064-d2508105064]PW[kata1-b40c256-s10289624064-d2508105064]HA[0]KM[8.5]RU[koPOSITIONALscoreAREAtaxNONEsui1]RE[B+34.5]C[startTurnIdx=52,initTurnNum=0,gameHash=13EBC0E8B309CAB709F5773B3A1E1722,gtype=normal];B[nd];W[nn];B[dn]...)
```

### MCTS数据标签
每步棋都包含AI分析数据：
- `0.37 0.63 0.00 -1.3 v=250 weight=0.00`
- 格式: `[黑方胜率] [白方胜率] [和棋概率] [预期分数] [访问次数] [权重]`

## 目录结构

下载完成后，目录结构如下：
```
katago_games/
├── 2021-10-20sgfs.tar.bz2     # 压缩文件
├── 2021-10-19sgfs.tar.bz2     # 更多压缩文件...
├── extracted/                  # 解压目录
│   ├── 2021-10-20/            # 按日期组织的解压文件
│   │   ├── 13EBC0E8...sgf    # SGF对局文件
│   │   ├── 98392F6E...sgf
│   │   └── ...
│   └── 2021-10-19/
├── download_status.json        # 下载状态（断点续传用）
└── download.log               # 下载日志
```

## 注意事项

### 网络限制
- 网站可能有访问频率限制
- 建议使用较长的延迟间隔（2-5秒）
- 遇到429错误会自动重试

### 存储空间
- 完整数据集约30GB+
- 解压后可能需要2-3倍空间
- 建议确保有足够的磁盘空间

### 棋盘大小筛选
⚠️ **重要**: 经过验证，数据中并非所有对局都是19x19！
- 发现有些对局是17x17棋盘
- 需要在使用时筛选 `SZ[19]` 的SGF文件
- 建议写一个筛选脚本过滤19x19对局

## 19x19棋盘筛选建议

```python
def filter_19x19_games(sgf_files_dir, output_dir):
    """筛选19x19的对局"""
    import os
    import re
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    sgf_dir = Path(sgf_files_dir)
    count_19x19 = 0
    count_total = 0

    for sgf_file in sgf_dir.rglob("*.sgf"):
        count_total += 1
        try:
            with open(sgf_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 查找棋盘大小
                if 'SZ[19]' in content:
                    # 复制19x19文件
                    output_file = output_path / sgf_file.name
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        out_f.write(content)
                    count_19x19 += 1
        except Exception as e:
            print(f"处理文件失败 {sgf_file}: {e}")

    print(f"找到 {count_19x19}/{count_total} 个19x19对局")
    return count_19x19
```

## 推荐下载策略

1. **测试下载** (少量文件):
   ```bash
   python katago_downloader.py --max-files 5 --delay 3.0
   ```

2. **批量下载** (完整数据集):
   ```bash
   python katago_downloader.py --delay 2.0 --max-retries 5
   ```

3. **选择性下载** (按时间段):
   ```bash
   python katago_downloader.py --start-date 2021-01-01 --end-date 2021-06-30
   ```

## 总结

这个下载方案提供了完整的KataGo 19x19自我对局数据获取解决方案，包括：

✅ **已完成的功能**:
- 深入了解网站结构和数据格式
- 开发防429错误的下载工具
- 支持断点续传和进度跟踪
- 提供灵活的下载选项
- 验证数据质量和格式

⚠️ **使用前请注意**:
- 数据中包含多种棋盘大小，需要筛选19x19对局
- 需要充足的磁盘空间（建议100GB+）
- 下载过程可能需要较长时间

🎯 **建议下一步**:
- 运行测试下载验证功能
- 根据需要筛选19x19对局
- 考虑数据存储和处理方案