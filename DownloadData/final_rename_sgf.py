#!/usr/bin/env python3
"""
最终SGF文件重命名工具
将所有SGF文件重新命名为1.sgf, 2.sgf, 3.sgf, ... n.sgf
"""
import os
import shutil
from pathlib import Path

def rename_sgf_files():
    data_dir = Path("./katago_download_test/data")

    print("开始最终重命名操作...")

    # 获取所有sgf文件并按数字排序
    sgf_files = list(data_dir.glob("*.sgf"))
    print(f"找到 {len(sgf_files)} 个SGF文件")

    # 创建临时目录
    temp_dir = data_dir / "temp_final_rename"
    temp_dir.mkdir(exist_ok=True)

    # 按数字大小排序
    def extract_number(filename):
        import re
        match = re.search(r'(\d+)\.sgf$', filename)
        return int(match.group(1)) if match else 0

    sgf_files.sort(key=lambda x: extract_number(x.name))

    print("开始重命名文件...")

    # 重命名并复制到临时目录
    for i, sgf_file in enumerate(sgf_files, 1):
        new_name = f"{i}.sgf"
        new_path = temp_dir / new_name

        try:
            shutil.copy2(sgf_file, new_path)

            if i % 10000 == 0:
                print(f"已处理: {i:,}/{len(sgf_files):,}")

        except Exception as e:
            print(f"处理文件 {sgf_file} 时出错: {e}")

    print("删除原始文件...")
    # 删除原始文件
    for sgf_file in sgf_files:
        try:
            sgf_file.unlink()
        except Exception as e:
            print(f"删除文件 {sgf_file} 时出错: {e}")

    print("移动文件回原位置...")
    # 移动文件回原位置
    for temp_file in temp_dir.glob("*.sgf"):
        target_path = data_dir / temp_file.name
        shutil.move(str(temp_file), str(target_path))

    print("清理临时目录...")
    # 删除临时目录
    temp_dir.rmdir()

    print("重命名完成!")

    # 验证结果
    final_files = list(data_dir.glob("*.sgf"))
    final_files.sort(key=lambda x: int(x.stem))

    print(f"最终文件数量: {len(final_files)}")
    print("前10个文件:")
    for f in final_files[:10]:
        print(f"  {f.name}")

    print("最后10个文件:")
    for f in final_files[-10:]:
        print(f"  {f.name}")

if __name__ == "__main__":
    rename_sgf_files()