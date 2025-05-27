from lxml import html
import pandas as pd
import html as ihtml  # 用于 unescape &#39; 等实体
import time
import os
import json
import glob
from tqdm import tqdm  # 添加进度条支持

def parse_html_to_dataframe(html_path):
    """
    Parse an evaluation HTML file and return a pandas DataFrame
    with Prompt, Sampled, Correct Answer, Extracted Answer, and Score.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("<hr>")
    data = []

    for block in blocks:
        if "<h3>Prompt conversation" not in block:
            continue

        try:
            tree = html.fromstring(block)

            # Prompt
            prompt_pre = tree.xpath('//h3[text()="Prompt conversation"]/following-sibling::div//pre')
            prompt = prompt_pre[0].text_content().strip() if prompt_pre else ""

            # Sampled
            sampled_pre = tree.xpath('//h3[text()="Sampled message"]/following-sibling::div//pre')
            sampled = sampled_pre[0].text_content().strip() if sampled_pre else ""

            # Results
            p_tags = tree.xpath('//h3[text()="Results"]/following-sibling::p')
            correct = extracted = score = ""
            for p in p_tags:
                text = p.text_content().strip()
                if text.startswith("Correct Answer:"):
                    correct = text.replace("Correct Answer:", "").strip()
                elif text.startswith("Extracted Answer:"):
                    extracted = text.replace("Extracted Answer:", "").strip()
                elif text.startswith("Score:"):
                    score = text.replace("Score:", "").strip()

            # Unescape HTML entities
            prompt = ihtml.unescape(prompt)
            sampled = ihtml.unescape(sampled)
            correct = ihtml.unescape(correct)
            extracted = ihtml.unescape(extracted)

            # 提取最后一个问题（如果有）
            last_question = ""
            lines = prompt.split('\n')
            for i in range(len(lines)-1, -1, -1):
                if lines[i].startswith("Q:"):
                    question_text = lines[i][2:].strip()
                    break_idx = i
                    while break_idx < len(lines) - 1 and not lines[break_idx+1].startswith("A:"):
                        break_idx += 1
                        question_text += " " + lines[break_idx].strip()
                    last_question = question_text.strip()
                    break

            data.append({
                "prompt": prompt,
                "question": last_question,
                "sampled": sampled,
                "correct_answer": correct,
                "extracted_answer": extracted,
                "is_correct": 1 if score.lower() == 'true' else 0
            })

        except Exception as e:
            print(f"解析块时出错: {e}")
            continue

    return data

def process_html_folder(input_folder, output_folder=None):
    """
    处理输入文件夹中的所有HTML文件，将结果保存为JSON文件
    
    参数:
        input_folder: 包含HTML文件的文件夹路径
        output_folder: JSON文件输出文件夹路径，如果为None则与input_folder相同
    
    返回:
        处理的文件数量
    """
    # 如果未指定输出文件夹，则使用输入文件夹
    if output_folder is None:
        output_folder = input_folder
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有HTML文件
    html_files = glob.glob(os.path.join(input_folder, "*.html"))
    
    if not html_files:
        print(f"在 {input_folder} 中未找到HTML文件")
        return 0
    
    print(f"找到 {len(html_files)} 个HTML文件，开始处理...")
    
    # 处理每个文件
    processed_count = 0
    for html_file in tqdm(html_files, desc="处理HTML文件"):
        start_time = time.time()
        
        try:
            # 获取基本文件名（不含路径和扩展名）
            base_name = os.path.splitext(os.path.basename(html_file))[0]
            
            # 解析HTML文件
            data = parse_html_to_dataframe(html_file)
            
            if not data:
                print(f"警告: {html_file} 未提取到数据")
                continue
            
            # 准备输出文件路径
            json_file = os.path.join(output_folder, f"{base_name}.json")
            
            # 保存为JSON
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            processed_count += 1
            
            # 计算处理时间
            elapsed = time.time() - start_time
            tqdm.write(f"已处理: {html_file} -> {json_file} (用时: {elapsed:.2f}秒, 提取: {len(data)} 条记录)")
            
        except Exception as e:
            tqdm.write(f"处理 {html_file} 失败: {e}")
    
    print(f"\n处理完成! 成功将 {processed_count}/{len(html_files)} 个HTML文件转换为JSON文件")
    print(f"输出目录: {output_folder}")
    
    return processed_count

def generate_summary(folder_path):
    """生成所有JSON文件的汇总报告"""
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print(f"在 {folder_path} 中未找到JSON文件")
        return
    
    # 汇总数据
    summary = {
        "total_files": len(json_files),
        "total_questions": 0,
        "correct_answers": 0,
        "incorrect_answers": 0,
        "accuracy": 0.0,
        "files": []
    }
    
    print("正在生成汇总报告...")
    
    # 处理每个JSON文件
    all_data = []
    for json_file in tqdm(json_files, desc="分析JSON文件"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                continue
                
            # 统计正确答案数量
            correct = sum(item.get("is_correct", 0) for item in data)
            total = len(data)
            accuracy = correct / total if total > 0 else 0
            
            # 添加到汇总
            summary["total_questions"] += total
            summary["correct_answers"] += correct
            summary["files"].append({
                "file": os.path.basename(json_file),
                "total_questions": total,
                "correct_answers": correct,
                "accuracy": accuracy
            })
            
            # 将数据添加到全局列表
            for item in data:
                item["file"] = os.path.basename(json_file)
                all_data.append(item)
                
        except Exception as e:
            print(f"处理 {json_file} 时出错: {e}")
    
    # 计算总体准确率
    if summary["total_questions"] > 0:
        summary["accuracy"] = summary["correct_answers"] / summary["total_questions"]
    
    # 保存汇总报告
    summary_path = os.path.join(folder_path, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 保存所有数据合并为一个文件
    all_data_path = os.path.join(folder_path, "all_results.json")
    with open(all_data_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    # 转换为CSV格式方便查看
    df = pd.DataFrame(all_data)
    csv_path = os.path.join(folder_path, "all_results.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"\n汇总报告已生成:")
    print(f"- 汇总JSON: {summary_path}")
    print(f"- 全部数据JSON: {all_data_path}")
    print(f"- 全部数据CSV: {csv_path}")
    print(f"\n总体统计:")
    print(f"- 文件总数: {summary['total_files']}")
    print(f"- 问题总数: {summary['total_questions']}")
    print(f"- 正确答案: {summary['correct_answers']}")
    print(f"- 总体准确率: {summary['accuracy']:.2%}")

def main():
    print("HTML文件转JSON工具")
    print("-----------------\n")
    
    # 获取用户输入
    input_folder = input("请输入HTML文件所在文件夹路径: ").strip()
    if not input_folder:
        print("错误: 未提供输入文件夹路径")
        return
    
    if not os.path.exists(input_folder):
        print(f"错误: 文件夹 '{input_folder}' 不存在")
        return
    
    output_folder = input("请输入JSON输出文件夹路径 (留空则与输入文件夹相同): ").strip()
    if not output_folder:
        output_folder = input_folder
    
    # 处理文件
    processed_count = process_html_folder(input_folder, output_folder)
    
    # 如果成功处理了文件，生成汇总报告
    if processed_count > 0:
        generate_summary_option = input("\n是否生成汇总报告? (y/n): ").strip().lower()
        if generate_summary_option == 'y':
            generate_summary(output_folder)

if __name__ == "__main__":
    main()