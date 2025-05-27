import os
import json
import glob
from tqdm import tqdm

def convert_prompt_to_chat_format(prompt):
    """将普通prompt转换为chat格式"""
    return {
        "role": "user",
        "content": prompt
    }

def modify_json_file(file_path, dry_run=True):
    """修改单个JSON文件中的prompt格式"""
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否是列表格式
        if not isinstance(data, list):
            print(f"警告: {file_path} 不是列表格式，跳过")
            return False
        
        # 显示文件内容示例
        print(f"\n文件: {file_path}")
        print("原始数据示例:")
        print(json.dumps(data[0] if data else {}, ensure_ascii=False, indent=2))
        
        # 检查是否需要修改
        needs_modification = False
        for item in data:
            if isinstance(item, dict) and 'prompt' in item:
                if not isinstance(item['prompt'], dict) or 'role' not in item['prompt']:
                    needs_modification = True
                    break
        
        if not needs_modification:
            print("该文件的prompt已经是chat格式，无需修改")
            return False
        
        # 询问是否修改
        if not dry_run:
            response = input("\n是否要修改这个文件? (y/n): ").strip().lower()
            if response != 'y':
                print("跳过修改")
                return False
        
        # 修改数据
        modified_data = []
        for item in data:
            if isinstance(item, dict) and 'prompt' in item:
                if not isinstance(item['prompt'], dict) or 'role' not in item['prompt']:
                    # 转换prompt格式
                    item['prompt'] = convert_prompt_to_chat_format(item['prompt'])
            modified_data.append(item)
        
        # 显示修改后的示例
        print("\n修改后的数据示例:")
        print(json.dumps(modified_data[0] if modified_data else {}, ensure_ascii=False, indent=2))
        
        if not dry_run:
            # 备份原文件
            
            
            # 保存修改后的文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(modified_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n文件已修改并保存")
            
        else:
            print("\n这是预览模式，未实际修改文件")
        
        return True
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False

def process_directory(directory_path):
    """处理目录下的所有JSON文件"""
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    if not json_files:
        print(f"在 {directory_path} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 统计信息
    modified_count = 0
    skipped_count = 0
    error_count = 0
    
    # 逐个处理文件
    for file_path in tqdm(json_files, desc="处理文件"):
        print(f"\n正在处理: {os.path.basename(file_path)}")
        try:
            # 先显示文件内容预览
            print("\n=== 文件预览 ===")
            modify_json_file(file_path, dry_run=True)
            
            # 询问是否修改当前文件
            response = input("\n是否修改这个文件? (y/n/q 退出): ").strip().lower()
            
            if response == 'q':
                print("\n用户请求退出，终止处理")
                break
            elif response == 'y':
                if modify_json_file(file_path, dry_run=False):
                    modified_count += 1
                    print(f"文件 {os.path.basename(file_path)} 已修改")
                else:
                    skipped_count += 1
                    print(f"文件 {os.path.basename(file_path)} 无需修改")
            else:
                skipped_count += 1
                print(f"已跳过文件 {os.path.basename(file_path)}")
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            error_count += 1
    
    # 打印统计信息
    print("\n处理完成!")
    print(f"- 已修改: {modified_count} 个文件")
    print(f"- 已跳过: {skipped_count} 个文件")
    print(f"- 出错: {error_count} 个文件")

def main():
    print("JSON文件prompt格式修改工具")
    print("------------------------\n")
    
    # 获取用户输入
    directory_path = input("请输入JSON文件所在文件夹路径: ").strip()
    
    if not directory_path:
        print("错误: 未提供文件夹路径")
        return
    
    if not os.path.exists(directory_path):
        print(f"错误: 文件夹 '{directory_path}' 不存在")
        return
    
    # 处理文件夹
    process_directory(directory_path)

if __name__ == "__main__":
    main()