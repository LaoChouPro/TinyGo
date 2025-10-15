#!/usr/bin/env python3
"""
KataGo 19x19对局数据筛选器 - 大规模版本
筛选所有19x19且完整的对局数据
"""
import os
import re
import shutil
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

class KataGoFilter:
    def __init__(self, source_dir, output_dir, workers=8):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.workers = workers
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'valid_19x19': 0,
            'invalid_size': 0,
            'incomplete': 0,
            'errors': 0
        }

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建日志文件
        self.log_file = open(self.output_dir / 'filter_log.txt', 'w', encoding='utf-8')

    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        self.log_file.write(log_msg + '\n')
        self.log_file.flush()

    def validate_sgf(self, sgf_content):
        """
        验证SGF文件的完整性和有效性
        """
        try:
            # 检查基本SGF格式
            if not sgf_content.startswith('(') or not sgf_content.endswith(')'):
                return False, "SGF格式不正确"

            # 检查棋盘大小
            size_match = re.search(r'SZ\[(\d+)\]', sgf_content)
            if not size_match:
                return False, "无法找到棋盘大小"

            board_size = int(size_match.group(1))
            if board_size != 19:
                return False, f"棋盘大小不是19x19，而是{board_size}x{board_size}"

            # 检查是否有对局数据
            if not re.search(r'[BW]\[', sgf_content):
                return False, "没有找到对局数据"

            # 检查对局结果
            result_match = re.search(r'RE\[(.+?)\]', sgf_content)
            if not result_match:
                return False, "没有找到对局结果"

            # 检查是否有MCTS分析数据（可选）
            has_mcts = bool(re.search(r'MV\[(.+?)\]', sgf_content))

            return True, f"有效19x19对局，MCTS分析: {'有' if has_mcts else '无'}"

        except Exception as e:
            return False, f"验证时出错: {str(e)}"

    def process_single_file(self, sgf_path):
        """
        处理单个SGF文件
        """
        try:
            with open(sgf_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 验证文件
            is_valid, message = self.validate_sgf(content)

            if is_valid:
                # 创建相对路径保持目录结构
                relative_path = sgf_path.relative_to(self.source_dir)
                output_path = self.output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # 复制有效文件
                shutil.copy2(sgf_path, output_path)
                return 'valid_19x19', message
            else:
                if "棋盘大小不是19x19" in message:
                    return 'invalid_size', message
                elif "完整" in message:
                    return 'incomplete', message
                else:
                    return 'errors', message

        except Exception as e:
            return 'errors', f"处理文件时出错: {str(e)}"

    def scan_files(self):
        """
        扫描所有SGF文件
        """
        self.log("开始扫描SGF文件...")
        sgf_files = list(self.source_dir.rglob("*.sgf"))
        self.stats['total_files'] = len(sgf_files)
        self.log(f"找到 {self.stats['total_files']} 个SGF文件")
        return sgf_files

    def filter_files(self):
        """
        批量筛选文件
        """
        sgf_files = self.scan_files()

        if not sgf_files:
            self.log("没有找到SGF文件")
            return

        self.log(f"开始筛选，使用 {self.workers} 个并行工作线程...")
        start_time = time.time()

        # 创建进度跟踪
        processed_count = 0

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.process_single_file, sgf_path): sgf_path
                for sgf_path in sgf_files
            }

            # 处理完成的任务
            for future in as_completed(future_to_file):
                sgf_path = future_to_file[future]
                processed_count += 1
                self.stats['processed_files'] += 1

                try:
                    result_type, message = future.result()
                    self.stats[result_type] += 1

                    # 每1000个文件输出一次进度
                    if processed_count % 1000 == 0:
                        elapsed_time = time.time() - start_time
                        rate = processed_count / elapsed_time
                        eta = (self.stats['total_files'] - processed_count) / rate if rate > 0 else 0
                        self.log(f"进度: {processed_count}/{self.stats['total_files']} "
                               f"({processed_count/self.stats['total_files']*100:.1f}%) "
                               f"速度: {rate:.1f} 文件/秒, "
                               f"预计剩余时间: {eta/60:.1f}分钟")

                except Exception as e:
                    self.stats['errors'] += 1
                    self.log(f"处理文件 {sgf_path} 时出错: {str(e)}")

        # 输出最终统计
        elapsed_time = time.time() - start_time
        self.log("=" * 60)
        self.log("筛选完成！")
        self.log(f"总处理时间: {elapsed_time/60:.2f}分钟")
        self.log(f"平均处理速度: {self.stats['processed_files']/elapsed_time:.1f} 文件/秒")
        self.log("=" * 60)
        self.log(f"总文件数: {self.stats['total_files']}")
        self.log(f"已处理文件: {self.stats['processed_files']}")
        self.log(f"有效19x19对局: {self.stats['valid_19x19']}")
        self.log(f"非19x19棋盘: {self.stats['invalid_size']}")
        self.log(f"不完整对局: {self.stats['incomplete']}")
        self.log(f"处理错误: {self.stats['errors']}")
        self.log(f"19x19对局占比: {self.stats['valid_19x19']/self.stats['total_files']*100:.2f}%")

        # 计算输出目录大小
        output_size = self.calculate_directory_size(self.output_dir)
        self.log(f"输出目录大小: {output_size/1024/1024/1024:.2f}GB")

        self.log_file.close()

    def calculate_directory_size(self, directory):
        """
        计算目录大小
        """
        total_size = 0
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

def main():
    source_directory = "./large_scale_download.py/extracted"
    output_directory = "./data"

    print("KataGo 19x19对局数据筛选器 - 大规模版本")
    print("=" * 60)

    # 检查源目录
    if not os.path.exists(source_directory):
        print(f"错误: 源目录 {source_directory} 不存在")
        return

    # 创建筛选器
    filter_instance = KataGoFilter(source_directory, output_directory, workers=8)

    # 开始筛选
    filter_instance.filter_files()

    print("\n筛选完成！结果已保存到:", output_directory)

if __name__ == "__main__":
    main()