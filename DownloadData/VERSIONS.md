# 项目版本记录

## v1.0.0 - 2025-10-07

### 新增功能
- 深入研究KataGo Archive网站结构 (https://katagoarchive.org/kata1/)
- 开发完整的KataGo 19x19自我对弈数据下载方案
- 创建防429错误的智能下载工具 (katago_downloader.py)
- 实现断点续传和进度跟踪功能
- 支持选择性下载（日期范围、文件数量限制）
- 验证数据格式和质量

### 技术特性
- **数据源**: https://katagoarchive.org/kata1/traininggames/
- **数据格式**: SGF (Smart Game Format) + tar.bz2压缩
- **数据量**: 约300+个压缩包，30GB+
- **时间范围**: 2020-12-08 到 2021-10-20
- **防429策略**: 请求间隔、指数退避、自动重试
- **断点续传**: 支持网络中断后继续下载

### 文件清单
- `explore_katago_archive.py` - 网站结构探索工具
- `read_katago_readme.py` - README文件读取工具
- `explore_traininggames.py` - 训练数据目录探索
- `test_download.py` - 下载测试和数据验证
- `katago_downloader.py` - 主要下载工具
- `KataGo下载说明.md` - 完整使用说明
- `katago_test_data/` - 测试下载的样本数据

### 重要发现
- 数据中包含多种棋盘大小（19x19和17x17）
- 需要筛选 `SZ[19]` 的SGF文件以获取19x19对局
- SGF文件包含完整的MCTS分析数据
- 文件命名规则: `YYYY-MM-DDsgfs.tar.bz2`

### 使用示例
```bash
# 测试下载
python katago_downloader.py --max-files 5 --delay 3.0

# 批量下载
python katago_downloader.py --delay 2.0 --max-retries 5

# 选择性下载
python katago_downloader.py --start-date 2021-01-01 --end-date 2021-06-30
```