#!/usr/bin/env python3
"""
KataGo 19x19自我对弈数据批量下载器
从 https://katagoarchive.org/kata1/traininggames/ 下载训练对局数据

使用方法:
    python katago_downloader.py [选项]

特性:
- 支持断点续传
- 自动避免429错误（请求过于频繁）
- 进度跟踪和日志记录
- 支持选择性下载
- 自动解压和整理
"""

import requests
import os
import sys
import time
import tarfile
import json
import argparse
from datetime import datetime, timedelta
from urllib.parse import urljoin
from pathlib import Path

class KataGoDownloader:
    def __init__(self, download_dir="katago_games", delay=2.0, max_retries=3):
        self.base_url = "https://katagoarchive.org/kata1/traininggames/"
        self.download_dir = Path(download_dir)
        self.delay = delay  # 请求间隔（秒）
        self.max_retries = max_retries
        self.session = requests.Session()

        # 设置请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://katagoarchive.org/kata1/',
            'Accept': 'application/octet-stream, application/x-bzip2, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        })

        # 创建目录
        self.download_dir.mkdir(exist_ok=True)
        self.extract_dir = self.download_dir / "extracted"
        self.extract_dir.mkdir(exist_ok=True)

        # 状态文件
        self.status_file = self.download_dir / "download_status.json"
        self.log_file = self.download_dir / "download.log"

        # 加载下载状态
        self.status = self.load_status()

        print(f"KataGo下载器初始化完成")
        print(f"下载目录: {self.download_dir.absolute()}")
        print(f"请求间隔: {self.delay}秒")
        print(f"最大重试次数: {self.max_retries}")

    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    def load_status(self):
        """加载下载状态"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.log(f"加载状态文件失败: {e}")

        return {
            'completed_files': [],
            'failed_files': [],
            'total_files': 0,
            'total_size': 0,
            'downloaded_size': 0,
            'last_download': None
        }

    def save_status(self):
        """保存下载状态"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.status, f, indent=2, ensure_ascii=False)

    def get_file_list(self):
        """获取可用文件列表"""
        self.log("获取文件列表...")

        try:
            # 访问traininggames的index.html
            index_url = self.base_url + "index.html"
            response = self.session.get(index_url, timeout=30)
            response.raise_for_status()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            files = []
            for link in soup.find_all('a'):
                href = link.get('href', '')
                text = link.get_text(strip=True)

                # 查找tar.bz2文件
                if href.endswith('.tar.bz2') and 'sgfs' in href:
                    # 提取文件大小信息
                size_str = ""
                prev_sibling = link.previous_sibling
                if prev_sibling and '[' in str(prev_sibling):
                    import re
                    size_match = re.search(r'\[([^\]]+)\]', str(prev_sibling))
                    if size_match:
                        size_str = size_match.group(1)

                files.append({
                    'name': href,
                    'size': size_str,
                    'url': urljoin(self.base_url, href)
                })

            # 按日期排序
            files.sort(key=lambda x: x['name'])

            self.log(f"找到 {len(files)} 个数据文件")
            return files

        except Exception as e:
            self.log(f"获取文件列表失败: {e}")
            return []

    def parse_size(self, size_str):
        """解析文件大小字符串"""
        if not size_str:
            return 0

        size_str = size_str.strip().replace('&nbsp;', '').upper()
        if 'M' in size_str:
            return float(size_str.replace('M', '')) * 1024 * 1024
        elif 'K' in size_str:
            return float(size_str.replace('K', '')) * 1024
        elif 'G' in size_str:
            return float(size_str.replace('G', '')) * 1024 * 1024 * 1024
        else:
            return 0

    def download_file(self, file_info):
        """下载单个文件"""
        filename = file_info['name']
        url = file_info['url']
        file_path = self.download_dir / filename

        # 检查是否已经下载
        if filename in self.status['completed_files'] and file_path.exists():
            self.log(f"跳过已下载的文件: {filename}")
            return True

        self.log(f"开始下载: {filename} ({file_info.get('size', 'unknown size')})")

        for attempt in range(self.max_retries):
            try:
                # 添加延迟避免429错误
                if attempt > 0 or self.status['last_download']:
                    time.sleep(self.delay)

                # 发送请求
                response = self.session.get(url, stream=True, timeout=60)
                response.raise_for_status()

                # 获取文件大小
                total_size = int(response.headers.get('content-length', 0))

                # 检查是否支持断点续传
                mode = 'ab' if file_path.exists() and file_path.stat().st_size > 0 else 'wb'
                initial_pos = file_path.stat().st_size if mode == 'ab' else 0

                if mode == 'ab' and initial_pos > 0:
                    headers = {'Range': f'bytes={initial_pos}-'}
                    response = self.session.get(url, headers=headers, stream=True, timeout=60)
                    response.raise_for_status()
                    self.log(f"断点续传: {filename} (从 {initial_pos} 字节开始)")

                # 下载文件
                downloaded = initial_pos
                with open(file_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # 显示进度
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\r{filename}: {progress:.1f}% ({downloaded}/{total_size} bytes)", end="", flush=True)
                            else:
                                print(f"\r{filename}: {downloaded} bytes", end="", flush=True)

                print()  # 换行

                # 验证下载
                if total_size > 0 and downloaded != total_size:
                    raise Exception(f"文件大小不匹配: 期望 {total_size}, 实际 {downloaded}")

                self.log(f"下载完成: {filename} ({downloaded} bytes)")

                # 更新状态
                self.status['completed_files'].append(filename)
                self.status['downloaded_size'] += downloaded - initial_pos
                self.status['last_download'] = datetime.now().isoformat()
                self.save_status()

                return True

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = min(60, self.delay * (2 ** attempt))
                    self.log(f"遇到429错误，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                elif e.response.status_code == 403:
                    self.log(f"访问被拒绝 (403): {filename}")
                    break
                else:
                    self.log(f"HTTP错误 {e.response.status_code}: {filename}")

            except Exception as e:
                self.log(f"下载失败 (尝试 {attempt + 1}/{self.max_retries}): {filename} - {e}")
                if attempt < self.max_retries - 1:
                    wait_time = self.delay * (2 ** attempt)
                    self.log(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)

        # 下载失败
        if filename not in self.status['failed_files']:
            self.status['failed_files'].append(filename)
        self.save_status()
        return False

    def extract_file(self, filename):
        """解压文件"""
        file_path = self.download_dir / filename

        if not file_path.exists():
            self.log(f"文件不存在，跳过解压: {filename}")
            return False

        self.log(f"开始解压: {filename}")

        try:
            with tarfile.open(file_path, 'r:bz2') as tar:
                members = tar.getmembers()
                sgf_files = [m for m in members if m.isfile() and m.name.endswith('.sgf')]

                self.log(f"压缩包包含 {len(sgf_files)} 个SGF文件")

                # 创建解压目录
                extract_subdir = self.extract_dir / filename.replace('.tar.bz2', '')
                extract_subdir.mkdir(exist_ok=True)

                # 解压SGF文件
                for member in sgf_files:
                    try:
                        tar.extract(member, path=extract_subdir)
                    except Exception as e:
                        self.log(f"解压文件失败 {member.name}: {e}")

                self.log(f"解压完成: {filename} -> {extract_subdir}")
                return True

        except Exception as e:
            self.log(f"解压失败: {filename} - {e}")
            return False

    def download_range(self, start_date=None, end_date=None, max_files=None):
        """下载指定范围的文件"""
        files = self.get_file_list()

        if not files:
            self.log("没有找到可下载的文件")
            return

        # 更新统计信息
        self.status['total_files'] = len(files)
        self.status['total_size'] = sum(self.parse_size(f.get('size', '0')) for f in files)
        self.save_status()

        # 过滤文件
        filtered_files = []
        for file_info in files:
            filename = file_info['name']

            # 从文件名提取日期
            try:
                date_str = filename.replace('sgfs.tar.bz2', '')
                file_date = datetime.strptime(date_str, '%Y-%m-%d')

                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue

                filtered_files.append(file_info)
            except ValueError:
                self.log(f"无法解析文件日期: {filename}")
                continue

        # 限制文件数量
        if max_files:
            filtered_files = filtered_files[:max_files]

        self.log(f"计划下载 {len(filtered_files)} 个文件")

        if not filtered_files:
            self.log("没有符合条件的新文件")
            return

        # 开始下载
        success_count = 0
        failed_count = 0

        for i, file_info in enumerate(filtered_files, 1):
            self.log(f"进度: {i}/{len(filtered_files)}")

            if self.download_file(file_info):
                success_count += 1

                # 可选：自动解压
                if file_info['name'] in self.status['completed_files']:
                    self.extract_file(file_info['name'])
            else:
                failed_count += 1

        # 总结
        self.log("=" * 60)
        self.log("下载完成!")
        self.log(f"成功: {success_count} 个文件")
        self.log(f"失败: {failed_count} 个文件")
        self.log(f"总计下载大小: {self.status['downloaded_size'] / (1024**3):.2f} GB")
        self.log("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='KataGo 19x19自我对弈数据下载器')
    parser.add_argument('--dir', '-d', default='katago_games', help='下载目录 (默认: katago_games)')
    parser.add_argument('--delay', type=float, default=2.0, help='请求间隔秒数 (默认: 2.0)')
    parser.add_argument('--max-retries', type=int, default=3, help='最大重试次数 (默认: 3)')
    parser.add_argument('--start-date', help='开始日期 (格式: YYYY-MM-DD)')
    parser.add_argument('--end-date', help='结束日期 (格式: YYYY-MM-DD)')
    parser.add_argument('--max-files', type=int, help='最大下载文件数量')
    parser.add_argument('--extract', action='store_true', help='自动解压下载的文件')

    args = parser.parse_args()

    # 解析日期
    start_date = None
    end_date = None

    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            print(f"无效的开始日期格式: {args.start_date}")
            return

    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            print(f"无效的结束日期格式: {args.end_date}")
            return

    # 创建下载器
    downloader = KataGoDownloader(
        download_dir=args.dir,
        delay=args.delay,
        max_retries=args.max_retries
    )

    # 开始下载
    downloader.download_range(
        start_date=start_date,
        end_date=end_date,
        max_files=args.max_files
    )

if __name__ == "__main__":
    main()