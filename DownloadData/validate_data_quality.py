#!/usr/bin/env python3
"""
KataGo 19x19对局数据质量验证工具
验证筛选后的数据质量和完整性
"""
import os
import re
import random
from datetime import datetime
from pathlib import Path
import json

class DataQualityValidator:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.validation_results = {
            'total_files': 0,
            'valid_sgf': 0,
            'invalid_format': 0,
            'missing_size': 0,
            'invalid_size': 0,
            'missing_moves': 0,
            'missing_result': 0,
            'has_mcts': 0,
            'sample_files': [],
            'board_size_distribution': {},
            'move_count_distribution': {},
            'result_distribution': {},
            'mcts_coverage': 0
        }

    def validate_single_file(self, sgf_path):
        """验证单个SGF文件的质量"""
        try:
            with open(sgf_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 基本格式检查
            if not content.startswith('(') or not content.endswith(')'):
                return 'invalid_format', None

            # 检查棋盘大小
            size_match = re.search(r'SZ\[(\d+)\]', content)
            if not size_match:
                return 'missing_size', None

            board_size = int(size_match.group(1))

            # 检查是否有对局步骤
            moves = re.findall(r'[BW]\[([a-zA-Z]{2})\]', content)
            if not moves:
                return 'missing_moves', None

            # 检查对局结果
            result_match = re.search(r'RE\[(.+?)\]', content)
            if not result_match:
                return 'missing_result', None

            # 检查MCTS分析数据
            has_mcts = bool(re.search(r'MV\[(.+?)\]', content))

            # 提取统计信息
            result = result_match.group(1)
            move_count = len(moves)

            return 'valid', {
                'board_size': board_size,
                'move_count': move_count,
                'result': result,
                'has_mcts': has_mcts,
                'file_size': len(content),
                'sample_content': content[:200] + '...' if len(content) > 200 else content
            }

        except Exception as e:
            return 'invalid_format', None

    def validate_sample(self, sample_size=1000):
        """验证样本文件"""
        print("开始数据质量验证...")

        # 获取所有SGF文件
        sgf_files = list(self.data_dir.rglob("*.sgf"))
        self.validation_results['total_files'] = len(sgf_files)

        if not sgf_files:
            print("没有找到SGF文件")
            return

        # 随机抽样
        sample_files = random.sample(sgf_files, min(sample_size, len(sgf_files)))

        print(f"总文件数: {len(sgf_files)}")
        print(f"验证样本数: {len(sample_files)}")
        print("-" * 60)

        valid_count = 0
        for i, sgf_path in enumerate(sample_files):
            result, data = self.validate_single_file(sgf_path)

            if result == 'valid':
                valid_count += 1
                self.validation_results['valid_sgf'] += 1

                # 统计棋盘大小分布
                size = data['board_size']
                self.validation_results['board_size_distribution'][size] = \
                    self.validation_results['board_size_distribution'].get(size, 0) + 1

                # 统计步数分布
                moves = data['move_count']
                move_range = self.get_move_range(moves)
                self.validation_results['move_count_distribution'][move_range] = \
                    self.validation_results['move_count_distribution'].get(move_range, 0) + 1

                # 统计结果分布
                result_str = data['result']
                self.validation_results['result_distribution'][result_str] = \
                    self.validation_results['result_distribution'].get(result_str, 0) + 1

                # 统计MCTS覆盖率
                if data['has_mcts']:
                    self.validation_results['has_mcts'] += 1

                # 保存样本文件信息
                if len(self.validation_results['sample_files']) < 10:
                    self.validation_results['sample_files'].append({
                        'file': str(sgf_path.relative_to(self.data_dir)),
                        'board_size': data['board_size'],
                        'move_count': data['move_count'],
                        'result': data['result'],
                        'has_mcts': data['has_mcts'],
                        'file_size': data['file_size']
                    })

            else:
                self.validation_results[result] += 1

            # 显示进度
            if (i + 1) % 100 == 0:
                print(f"已验证: {i + 1}/{len(sample_files)} "
                      f"({(i + 1)/len(sample_files)*100:.1f}%)")

        # 计算MCTS覆盖率
        if self.validation_results['valid_sgf'] > 0:
            self.validation_results['mcts_coverage'] = \
                self.validation_results['has_mcts'] / self.validation_results['valid_sgf'] * 100

    def get_move_range(self, move_count):
        """将步数转换为范围区间"""
        if move_count < 50:
            return "0-49"
        elif move_count < 100:
            return "50-99"
        elif move_count < 150:
            return "100-149"
        elif move_count < 200:
            return "150-199"
        elif move_count < 250:
            return "200-249"
        else:
            return "250+"

    def generate_report(self):
        """生成验证报告"""
        print("\n" + "=" * 60)
        print("KataGo 19x19对局数据质量验证报告")
        print("=" * 60)

        total = self.validation_results['total_files']
        valid = self.validation_results['valid_sgf']

        print(f"\n📊 基本统计:")
        print(f"  总文件数: {total:,}")
        print(f"  有效SGF: {valid:,} ({valid/len(self.validation_results['sample_files'])*100:.1f}%)")
        print(f"  无效格式: {self.validation_results['invalid_format']:,}")
        print(f"  缺少棋盘大小: {self.validation_results['missing_size']:,}")
        print(f"  无效棋盘大小: {self.validation_results['invalid_size']:,}")
        print(f"  缺少对局步骤: {self.validation_results['missing_moves']:,}")
        print(f"  缺少对局结果: {self.validation_results['missing_result']:,}")

        print(f"\n🎯 棋盘大小分布:")
        for size, count in sorted(self.validation_results['board_size_distribution'].items()):
            percentage = count / valid * 100 if valid > 0 else 0
            print(f"  {size}x{size}: {count:,} ({percentage:.1f}%)")

        print(f"\n🎲 对局步数分布:")
        for move_range, count in sorted(self.validation_results['move_count_distribution'].items()):
            percentage = count / valid * 100 if valid > 0 else 0
            print(f"  {move_range}步: {count:,} ({percentage:.1f}%)")

        print(f"\n🏆 对局结果分布 (前10):")
        sorted_results = sorted(self.validation_results['result_distribution'].items(),
                              key=lambda x: x[1], reverse=True)
        for result, count in sorted_results[:10]:
            percentage = count / valid * 100 if valid > 0 else 0
            print(f"  {result}: {count:,} ({percentage:.1f}%)")

        print(f"\n🤖 MCTS分析数据:")
        print(f"  包含MCTS数据: {self.validation_results['has_mcts']:,}")
        print(f"  MCTS覆盖率: {self.validation_results['mcts_coverage']:.1f}%")

        print(f"\n📝 样本文件示例:")
        for sample in self.validation_results['sample_files']:
            print(f"  {sample['file']}")
            print(f"    棋盘: {sample['board_size']}x{sample['board_size']}, "
                  f"步数: {sample['move_count']}, "
                  f"结果: {sample['result']}, "
                  f"MCTS: {'有' if sample['has_mcts'] else '无'}")

        # 数据质量评估
        print(f"\n✅ 数据质量评估:")
        if valid / len(self.validation_results['sample_files']) > 0.95:
            print("  🌟 优秀：数据质量非常高，适合机器学习训练")
        elif valid / len(self.validation_results['sample_files']) > 0.90:
            print("  👍 良好：数据质量较高，可用于训练")
        elif valid / len(self.validation_results['sample_files']) > 0.80:
            print("  ⚠️  一般：数据质量中等，建议进一步清理")
        else:
            print("  ❌ 较差：数据质量较低，需要大量清理工作")

        # MCTS数据评估
        if self.validation_results['mcts_coverage'] > 80:
            print("  🤖 MCTS数据覆盖率优秀，适合深度学习训练")
        elif self.validation_results['mcts_coverage'] > 50:
            print("  🤖 MCTS数据覆盖率良好")
        else:
            print("  ⚠️  MCTS数据覆盖率较低，可能影响训练效果")

    def save_report(self, output_file="validation_report.json"):
        """保存详细报告到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, ensure_ascii=False, indent=2)
        print(f"\n📄 详细报告已保存到: {output_file}")

def main():
    data_directory = "./data"

    print("KataGo 19x19对局数据质量验证工具")
    print("=" * 60)

    # 检查数据目录
    if not os.path.exists(data_directory):
        print(f"错误: 数据目录 {data_directory} 不存在")
        return

    # 创建验证器
    validator = DataQualityValidator(data_directory)

    # 执行验证
    validator.validate_sample(sample_size=2000)

    # 生成报告
    validator.generate_report()

    # 保存详细报告
    validator.save_report()

if __name__ == "__main__":
    main()