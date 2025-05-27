import pandas as pd
import os
import argparse
from tqdm import tqdm

def parquet_to_csv(input_file, output_file=None, chunksize=None):
    """
    将Parquet文件转换为CSV文件
    
    参数:
    input_file (str): 输入的Parquet文件路径
    output_file (str, 可选): 输出的CSV文件路径, 如果不指定则使用相同的文件名但扩展名为.csv
    chunksize (int, 可选): 分块处理的行数，用于处理大文件
    """
    # 如果没有指定输出文件，则使用输入文件名修改扩展名
    if output_file is None:
        file_base = os.path.splitext(input_file)[0]
        output_file = f"{file_base}.csv"
    
    print(f"正在将 {input_file} 转换为 {output_file}...")
    
    # 如果文件较大，使用分块处理
    if chunksize:
        # 先获取总行数来估计进度
        total_rows = pd.read_parquet(input_file, columns=[]).shape[0]
        chunks_count = (total_rows // chunksize) + 1
        
        # 分块读取并写入
        for i, chunk in enumerate(tqdm(pd.read_parquet(input_file, chunksize=chunksize), 
                                       total=chunks_count, 
                                       desc="处理进度")):
            # 第一个块，写入带有列头的文件
            if i == 0:
                chunk.to_csv(output_file, index=False)
            # 后续块，追加到文件，不包含列头
            else:
                chunk.to_csv(output_file, mode='a', header=False, index=False)
    else:
        # 对于小文件，一次性读取并写入
        df = pd.read_parquet(input_file)
        print(f"数据行数: {len(df)}, 列数: {len(df.columns)}")
        df.to_csv(output_file, index=False)
    
    print(f"转换完成! 已保存到 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将Parquet文件转换为CSV文件")
    parser.add_argument("input", help="输入的Parquet文件路径")
    parser.add_argument("-o", "--output", help="输出的CSV文件路径", default=None)
    parser.add_argument("-c", "--chunksize", help="分块处理的行数", type=int, default=None)
    
    args = parser.parse_args()
    
    parquet_to_csv(args.input, args.output, args.chunksize)