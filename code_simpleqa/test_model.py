import os
import json
import glob
import argparse
from tqdm import tqdm

def rename_files_in_folder(folder_path, dry_run=False):
    """
    重命名文件夹中的文件:
    1. 如果文件名以simpleqa或trivia结尾，只需去掉文件名开始的simpleqa_
    2. 否则，根据JSON文件中的数据数量决定新文件名：
       - 数据量>5000：以trivia结尾
       - 数据量<=5000：以simpleqa结尾
       同时删除文件名开始的simpleqa_
    
    参数:
        folder_path: 要处理的文件夹路径
        dry_run: 如果为True，只显示会发生什么，不实际重命名
    """
    # 确保文件夹路径存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return
    
    # 获取文件夹中的所有文件
    all_files = glob.glob(os.path.join(folder_path, "*.*"))
    
    if not all_files:
        print(f"在 {folder_path} 中未找到文件")
        return
    
    print(f"找到 {len(all_files)} 个文件，{'预览' if dry_run else '开始'}处理...")
    
    # 重命名文件
    renamed_count = 0
    skipped_count = 0
    error_count = 0
    
    for file_path in tqdm(all_files, desc="处理文件"):
        try:
            # 获取文件名和扩展名
            file_dir, file_name = os.path.split(file_path)
            base_name, ext = os.path.splitext(file_name)
            
            # 检查是否以simpleqa或trivia结尾
            if base_name.endswith("simpleqa") or base_name.endswith("trivia"):
                # 情况1: 只需去掉文件名开始的simpleqa_
                if base_name.startswith("simpleqa_"):
                    new_base_name = base_name[len("simpleqa_"):]
                    new_file_name = new_base_name + ext
                    new_file_path = os.path.join(file_dir, new_file_name)
                    
                    # 如果新文件名不同于原文件名，则重命名
                    if new_file_path != file_path:
                        if not dry_run:
                            os.rename(file_path, new_file_path)
                        tqdm.write(f"{'将重命名' if dry_run else '已重命名'}: {file_name} -> {new_file_name}")
                        renamed_count += 1
                    else:
                        skipped_count += 1
                else:
                    # 文件名不以simpleqa_开头，无需重命名
                    skipped_count += 1
            
            # 情况2: 需要根据JSON文件内容决定新文件名
            elif ext.lower() == '.json':
                # 读取JSON文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        data_count = len(data) if isinstance(data, list) else 0
                        
                        # 根据数据量决定后缀
                        suffix = "trivia" if data_count > 5000 else "simpleqa"
                        
                        # 构建新文件名
                        if base_name.startswith("simpleqa_"):
                            # 去掉前缀并添加后缀
                            new_base_name = base_name[len("simpleqa_"):].rstrip("_")
                            if not new_base_name.endswith(suffix):
                                new_base_name = f"{new_base_name}_{suffix}"
                        else:
                            # 如果不是以simpleqa_开头，只需添加后缀（如果不存在）
                            new_base_name = base_name
                            if not new_base_name.endswith(suffix):
                                new_base_name = f"{new_base_name}_{suffix}"
                        
                        new_file_name = new_base_name + ext
                        new_file_path = os.path.join(file_dir, new_file_name)
                        
                        # 如果新文件名不同于原文件名，则重命名
                        if new_file_path != file_path:
                            if not dry_run:
                                os.rename(file_path, new_file_path)
                            tqdm.write(f"{'将重命名' if dry_run else '已重命名'}: {file_name} -> {new_file_name} (数据量: {data_count})")
                            renamed_count += 1
                        else:
                            skipped_count += 1
                    
                    except json.JSONDecodeError:
                        tqdm.write(f"警告: {file_path} 不是有效的JSON文件，跳过")
                        skipped_count += 1
            
            # 对于其他类型的文件，保持不变
            else:
                skipped_count += 1
            
        except Exception as e:
            tqdm.write(f"处理 {file_path} 时出错: {e}")
            error_count += 1
    
    action = "预览" if dry_run else "重命名"
    print(f"\n{action}完成!")
    print(f"- {'将重命名' if dry_run else '已重命名'}: {renamed_count} 个文件")
    print(f"- 已跳过: {skipped_count} 个文件")
    print(f"- 出错: {error_count} 个文件")
    
    if dry_run and renamed_count > 0:
        print("\n这只是预览模式，没有实际重命名文件。如需实际重命名，请去掉--dry-run参数。")

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='根据文件名和内容重命名文件')
    parser.add_argument('folder', help='包含要重命名文件的文件夹路径')
    parser.add_argument('--dry-run', action='store_true', help='预览模式，不实际重命名文件')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 执行重命名
    rename_files_in_folder(args.folder, args.dry_run)

if __name__ == "__main__":
    main()