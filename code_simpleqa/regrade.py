import os
import re
import json
import pandas as pd
import glob
from bs4 import BeautifulSoup
import time
from sampler.chat_completion_sampler import ChatCompletionSampler

# 从simpleqa_eval.py导入评分函数
from simpleqa_eval import SimpleQAEval

def extract_data_from_simpleqa_html(html_file):
    """从SimpleQA HTML文件中提取问题、标准答案和模型回答"""
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    data = []
    
    # 寻找所有评估案例
    # 每个案例包含用户问题、模型回答、正确答案和评分
    hr_tags = soup.find_all('hr')
    
    # 如果没有找到hr标签，可能是不同的格式
    if not hr_tags:
        # 尝试找到所有问题段落
        results_sections = soup.find_all('h3', string='Results')
        
        for results_section in results_sections:
            # 向前找用户问题
            user_div = None
            for element in reversed(list(results_section.find_all_previous())):
                if element.name == 'div' and 'message user' in element.get('class', []):
                    user_div = element
                    break
            
            # 找模型回答
            assistant_div = None
            for element in reversed(list(results_section.find_all_previous())):
                if element.name == 'div' and 'message assistant' in element.get('class', []):
                    assistant_div = element
                    break
            
            # 提取正确答案和评分
            correct_answer_p = results_section.find_next('p', string=lambda s: s and 'Correct Answer:' in s)
            extracted_answer_p = results_section.find_next('p', string=lambda s: s and 'Extracted Answer:' in s)
            score_p = results_section.find_next('p', string=lambda s: s and 'Score:' in s)
            
            # 解析问题
            if user_div:
                content_div = user_div.find('div', class_='content')
                if content_div and content_div.find('pre'):
                    user_text = content_div.find('pre').text.strip()
                    # 获取最后一个问题
                    questions = re.findall(r'Q: (.*?)\nA:', user_text + '\nA:', re.DOTALL)
                    if questions:
                        last_question = questions[-1].strip()
                    else:
                        last_question = "问题未找到"
                else:
                    last_question = "问题未找到"
            else:
                last_question = "问题未找到"
            
            # 解析模型回答
            if assistant_div:
                content_div = assistant_div.find('div', class_='content')
                if content_div and content_div.find('pre'):
                    model_answer = content_div.find('pre').text.strip()
                else:
                    model_answer = "模型回答未找到"
            else:
                model_answer = "模型回答未找到"
            
            # 解析正确答案
            if correct_answer_p:
                correct_answer = correct_answer_p.text.replace('Correct Answer:', '').strip()
            else:
                correct_answer = "正确答案未找到"
            
            # 解析现有评分
            if score_p:
                existing_score = score_p.text.replace('Score:', '').strip()
            else:
                existing_score = "未找到"
            
            # 添加到数据列表
            data.append({
                'problem': last_question,
                'answer': correct_answer,
                'model_response': model_answer,
                'existing_score': existing_score
            })
    
    # 如果还是没找到数据，直接搜索所有问题/答案对
    if not data:
        user_divs = soup.find_all('div', class_='message user')
        assistant_divs = soup.find_all('div', class_='message assistant')
        correct_answer_ps = soup.find_all('p', string=lambda s: s and 'Correct Answer:' in s)
        
        # 确保长度匹配
        min_len = min(len(user_divs), len(assistant_divs), len(correct_answer_ps))
        
        for i in range(min_len):
            # 解析问题
            content_div = user_divs[i].find('div', class_='content')
            if content_div and content_div.find('pre'):
                user_text = content_div.find('pre').text.strip()
                # 获取最后一个问题
                questions = re.findall(r'Q: (.*?)\nA:', user_text + '\nA:', re.DOTALL)
                if questions:
                    last_question = questions[-1].strip()
                else:
                    last_question = "问题未找到"
            else:
                last_question = "问题未找到"
            
            # 解析模型回答
            content_div = assistant_divs[i].find('div', class_='content')
            if content_div and content_div.find('pre'):
                model_answer = content_div.find('pre').text.strip()
            else:
                model_answer = "模型回答未找到"
            
            # 解析正确答案
            correct_answer = correct_answer_ps[i].text.replace('Correct Answer:', '').strip()
            
            # 查找评分
            score_p = None
            next_element = correct_answer_ps[i].find_next('p')
            if next_element and 'Score:' in next_element.text:
                existing_score = next_element.text.replace('Score:', '').strip()
            else:
                existing_score = "未找到"
            
            # 添加到数据列表
            data.append({
                'problem': last_question,
                'answer': correct_answer,
                'model_response': model_answer,
                'existing_score': existing_score
            })
    
    return data

def create_grader_model():
    """创建评分模型"""
    print("初始化评分模型...")
    # 替换为您的评分模型配置
    grader_model = ChatCompletionSampler(
        model="qwen3-32b",
        model_name="qwen3-32b-grader",
        url="http://172.27.98.243:8001/v1",  # 根据您的实际情况修改API地址
        use_chat=True,
        enable_thinking=False,
        max_tokens=30000,
        model_config="qwen3"
    )
    return grader_model

def regrade_responses(html_files, output_file):
    """重新评分从HTML文件中提取的模型回答"""
    # 创建评分模型
    grader_model = create_grader_model()
    
    # 初始化SimpleQAEval (只使用其中的grade_sample方法)
    evaluator = SimpleQAEval(grader_model=grader_model, num_examples=1)
    
    all_results = []
    total_files = len(html_files)
    
    for i, html_file in enumerate(html_files):
        print(f"处理文件 {i+1}/{total_files}: {os.path.basename(html_file)}")
        data = extract_data_from_simpleqa_html(html_file)
        
        if not data:
            print(f"从文件中未提取到数据: {html_file}")
            continue
        
        print(f"  从文件中提取了 {len(data)} 个评估案例")
        
        # 处理提取的数据
        file_results = []
        for j, item in enumerate(data):
            try:
                question = item.get('problem', '')
                correct_answer = item.get('answer', '')
                model_response = item.get('model_response', '')
                existing_score = item.get('existing_score', '')
                
                if not question or not correct_answer or not model_response:
                    print(f"  跳过项目 {j+1}: 数据不完整")
                    continue
                
                # 使用grade_sample进行评分
                print(f"  评分项目 {j+1}/{len(data)}")
                grade = evaluator.grade_sample(
                    question=question,
                    target=correct_answer,
                    predicted_answer=model_response,
                    model_name="extracted_from_html"
                )
                
                grade_text = {'A': 'CORRECT', 'B': 'INCORRECT', 'C': 'NOT_ATTEMPTED'}.get(grade, 'UNKNOWN')
                original_score_matches = existing_score.lower() == 'true' and grade == 'A' or existing_score.lower() == 'false' and grade != 'A'
                
                # 将结果添加到列表
                result = {
                    'file': os.path.basename(html_file),
                    'question': question,
                    'correct_answer': correct_answer,
                    'model_response': model_response,
                    'original_score': existing_score,
                    'new_grade': grade,
                    'new_grade_text': grade_text,
                    'scores_match': original_score_matches
                }
                file_results.append(result)
                
            except Exception as e:
                print(f"  评分项目 {j+1} 时出错: {e}")
        
        # 添加到总结果
        all_results.extend(file_results)
        
        # 每处理5个文件保存一次中间结果
        if (i+1) % 5 == 0 or i == total_files - 1:
            temp_df = pd.DataFrame(all_results)
            temp_df.to_csv(f"{output_file}_temp.csv", index=False)
            print(f"已保存中间结果到 {output_file}_temp.csv")
    
    # 创建最终结果DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("警告：未提取到任何数据，请检查HTML文件格式")
        return None, None
    
    # 计算汇总统计
    total = len(results_df)
    correct = len(results_df[results_df['new_grade'] == 'A'])
    incorrect = len(results_df[results_df['new_grade'] == 'B'])
    not_attempted = len(results_df[results_df['new_grade'] == 'C'])
    scores_match = len(results_df[results_df['scores_match'] == True])
    scores_differ = total - scores_match
    
    # 计算指标
    is_correct_rate = correct / total if total > 0 else 0
    is_incorrect_rate = incorrect / total if total > 0 else 0
    is_not_attempted_rate = not_attempted / total if total > 0 else 0
    is_given_attempted = is_correct_rate + is_incorrect_rate
    accuracy_given_attempted = is_correct_rate / is_given_attempted if is_given_attempted > 0 else 0
    f1_score = 2 * accuracy_given_attempted * is_correct_rate / (accuracy_given_attempted + is_correct_rate) if (accuracy_given_attempted + is_correct_rate) > 0 else 0
    scores_match_rate = scores_match / total if total > 0 else 0
    
    # 创建汇总信息
    summary = {
        'total_examples': total,
        'correct': correct,
        'incorrect': incorrect,
        'not_attempted': not_attempted,
        'is_correct_rate': is_correct_rate,
        'is_incorrect_rate': is_incorrect_rate,
        'is_not_attempted_rate': is_not_attempted_rate,
        'is_given_attempted': is_given_attempted,
        'accuracy_given_attempted': accuracy_given_attempted,
        'f1_score': f1_score,
        'scores_match': scores_match,
        'scores_differ': scores_differ,
        'scores_match_rate': scores_match_rate
    }
    
    # 保存结果
    results_df.to_csv(f"{output_file}.csv", index=False)
    
    # 保存汇总信息
    with open(f"{output_file}_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 打印汇总信息
    print("\n评估汇总:")
    print(f"总样本数: {total}")
    print(f"正确: {correct} ({is_correct_rate:.2%})")
    print(f"错误: {incorrect} ({is_incorrect_rate:.2%})")
    print(f"未尝试: {not_attempted} ({is_not_attempted_rate:.2%})")
    print(f"Accuracy Given Attempted: {accuracy_given_attempted:.3f}")
    print(f"F1 Score: {f1_score:.3f}")
    print(f"与原评分一致: {scores_match} ({scores_match_rate:.2%})")
    print(f"与原评分不一致: {scores_differ} ({1-scores_match_rate:.2%})")
    
    return results_df, summary

def main():
    # 设置输入和输出路径
    html_dir = input("请输入HTML文件目录 (默认为 'html_results'): ") or "html_results"
    output_file = input("请输入输出文件名前缀 (默认为 'regrade_results'): ") or "regrade_results"
    
    # 获取所有HTML文件
    html_files = glob.glob(os.path.join(html_dir, "*.html"))
    
    if not html_files:
        print(f"在 {html_dir} 目录中未找到HTML文件")
        return
    
    print(f"找到 {len(html_files)} 个HTML文件")
    
    # 处理文件
    results_df, summary = regrade_responses(html_files, output_file)
    
    if results_df is not None:
        print(f"\n结果已保存到 {output_file}.csv")
        print(f"汇总信息已保存到 {output_file}_summary.json")
    else:
        print("处理完成，但未生成结果文件")

if __name__ == "__main__":
    main()