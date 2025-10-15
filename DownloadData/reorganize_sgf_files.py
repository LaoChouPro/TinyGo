#!/usr/bin/env python3
"""
KataGo SGF文件重组工具
将所有分散在子目录中的SGF文件重命名为1.sgf, 2.sgf, ... n.sgf
并统一存放到data文件夹根目录中
"""
import os
import shutil
from pathlib import Path
import time

class SGFOrganizer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.temp_dir = self.data_dir / "temp_reorganization"
        self.processed_count = 0
        self.error_count = 0

    def collect_all_sgf_files(self):
        """收集所有SGF文件"""
        print("正在收集所有SGF文件...")
        sgf_files = list(self.data_dir.rglob("*.sgf"))
        print(f"找到 {len(sgf_files)} 个SGF文件")
        return sgf_files

    def create_temp_directory(self):
        """创建临时目录"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        print(f"创建临时目录: {self.temp_dir}")

    def reorganize_files(self):
        """重组文件"""
        sgf_files = self.collect_all_sgf_files()

        if not sgf_files:
            print("没有找到SGF文件")
            return

        print(f"开始重组 {len(sgf_files)} 个文件...")
        self.create_temp_directory()

        start_time = time.time()

        for i, sgf_path in enumerate(sgf_files, 1):
            try:
                # 新文件名: 1.sgf, 2.sgf, 3.sgf, ...
                new_filename = f"{i}.sgf"
                new_path = self.temp_dir / new_filename

                # 复制文件到临时目录
                shutil.copy2(sgf_path, new_path)
                self.processed_count += 1

                # 显示进度
                if i % 10000 == 0:
                    elapsed_time = time.time() - start_time
                    rate = i / elapsed_time
                    print(f"已处理: {i:,}/{len(sgf_files):,} "
                          f"({i/len(sgf_files)*100:.1f}%) "
                          f"速度: {rate:.1f} 文件/秒")

            except Exception as e:
                print(f"处理文件 {sgf_path} 时出错: {str(e)}")
                self.error_count += 1

        print(f"文件重组完成!")
        print(f"成功处理: {self.processed_count:,}")
        print(f"处理错误: {self.error_count}")

    def clean_old_directories(self):
        """清理旧的子目录"""
        print("正在清理旧的子目录...")

        # 删除所有子目录（除了临时目录）
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name != "temp_reorganization":
                try:
                    shutil.rmtree(item)
                    print(f"删除目录: {item.name}")
                except Exception as e:
                    print(f"删除目录 {item} 时出错: {str(e)}")

    def move_files_to_root(self):
        """将文件从临时目录移动到根目录"""
        print("正在将文件移动到根目录...")

        temp_files = list(self.temp_dir.glob("*.sgf"))
        print(f"临时目录中有 {len(temp_files)} 个文件")

        for i, temp_file in enumerate(temp_files, 1):
            try:
                target_path = self.data_dir / temp_file.name
                shutil.move(str(temp_file), str(target_path))

                if i % 50000 == 0:
                    print(f"已移动: {i:,}/{len(temp_files):,} "
                          f"({i/len(temp_files)*100:.1f}%)")

            except Exception as e:
                print(f"移动文件 {temp_file} 时出错: {str(e)}")
                self.error_count += 1

        # 删除临时目录
        try:
            self.temp_dir.rmdir()
            print("删除临时目录")
        except Exception as e:
            print(f"删除临时目录时出错: {str(e)}")

    def verify_organization(self):
        """验证重组结果"""
        print("\n验证重组结果...")

        # 检查根目录中的SGF文件
        root_sgf_files = list(self.data_dir.glob("*.sgf"))
        print(f"根目录SGF文件数量: {len(root_sgf_files):,}")

        # 检查是否还有子目录
        subdirs = [item for item in self.data_dir.iterdir() if item.is_dir()]
        print(f"剩余子目录数量: {len(subdirs)}")

        # 检查文件命名格式
        invalid_names = []
        for sgf_file in root_sgf_files[:1000]:  # 检查前1000个文件
            if not sgf_file.name.isdigit() and not sgf_file.name.endswith('.sgf'):
                invalid_names.append(sgf_file.name)

        if invalid_names:
            print(f"发现无效文件名: {len(invalid_names)} 个")
            print(f"示例: {invalid_names[:5]}")
        else:
            print("✅ 文件命名格式正确")

        # 检查文件大小
        total_size = sum(f.stat().st_size for f in root_sgf_files)
        print(f"总数据大小: {total_size/1024/1024/1024:.2f}GB")

        return len(root_sgf_files), len(subdirs), len(invalid_names)

    def run_reorganization(self):
        """执行完整的重组流程"""
        print("KataGo SGF文件重组工具")
        print("=" * 60)
        print(f"目标目录: {self.data_dir}")
        print("-" * 60)

        # 1. 重组文件到临时目录
        self.reorganize_files()

        if self.error_count > 0:
            print(f"⚠️  重组过程中有 {self.error_count} 个错误")
            return

        # 2. 清理旧目录
        self.clean_old_directories()

        # 3. 移动文件到根目录
        self.move_files_to_root()

        # 4. 验证结果
        file_count, dir_count, invalid_count = self.verify_organization()

        print("\n" + "=" * 60)
        print("重组完成!")
        print(f"✅ SGF文件数量: {file_count:,}")
        print(f"✅ 子目录数量: {dir_count}")
        print(f"✅ 无效文件名: {invalid_count}")
        print(f"✅ 处理错误: {self.error_count}")

        if dir_count == 0 and invalid_count == 0 and self.error_count == 0:
            print("🎉 重组完全成功！所有文件已按照 1.sgf, 2.sgf, ... n.sgf 格式存放在data文件夹中")
        else:
            print("⚠️  重组完成，但存在一些问题，请检查上述输出")

def main():
    data_directory = "./katago_download_test/data"

    print("KataGo SGF文件重组工具")
    print("=" * 60)
    print("⚠️  警告: 此操作将重组data文件夹中的所有文件")
    print("  - 所有SGF文件将被重命名为 1.sgf, 2.sgf, ... n.sgf")
    print("  - 所有子目录将被删除")
    print("  - 所有文件将移动到data文件夹根目录")
    print("-" * 60)
    print("自动确认操作...")

    # 检查目录存在
    if not os.path.exists(data_directory):
        print(f"错误: 目录 {data_directory} 不存在")
        return

    # 创建重组器并执行
    organizer = SGFOrganizer(data_directory)
    organizer.run_reorganization()

if __name__ == "__main__":
    main()