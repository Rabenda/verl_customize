import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 1. 外部参数设置 =================
train_batch_size = 512 
rollout_n = 4
chunk_size = train_batch_size * rollout_n  # 自动计算分界线 (4096)

# file_path = '/workspace/repo/verl/lengths_qwen3_8b_gsm8k_5step_grpo.csv'
file_path = '/workspace/repo/verl/lengths_qwen3_4b_aime_2step_grpo_rn4.csv'  
# =================================================

try:
    df = pd.read_csv(file_path, header=None, names=['length'])
    total_rows = len(df)
    print(f"成功读取文件: {file_path}")
    print(f"总记录数: {total_rows} | 预期每个Batch记录数: {chunk_size}")
except Exception as e:
    print(f"读取文件失败: {e}")
    exit()

sns.set_theme(style="whitegrid")

#  for each chunk_size getting a cut and do the loop
for i in range(0, total_rows, chunk_size):
    batch_idx = (i // chunk_size) + 1
    start_idx = i
    end_idx = min(i + chunk_size, total_rows)
    current_chunk_len = end_idx - start_idx
    
    chunk = df.iloc[start_idx:end_idx]
    
    plt.figure(figsize=(12, 6))

    if current_chunk_len > 1:
        sns.histplot(chunk['length'], bins=50, kde=True, color='skyblue', edgecolor='black')
    else:
        plt.hist(chunk['length'], color='skyblue', edgecolor='black')
    
    avg_len = chunk['length'].mean()
    max_len = chunk['length'].max()
    
    plt.title(f'Decoding Length Distribution (Step {batch_idx})\n'
              f'[Config: BatchSize({train_batch_size}) * RolloutN({rollout_n}) = {chunk_size}]', fontsize=14)
    plt.xlabel('Decoding Length (Tokens)', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    
    plt.axvline(avg_len, color='green', linestyle='--', alpha=0.8, label=f'Average: {avg_len:.1f}')
    plt.axvline(max_len, color='red', linestyle=':', alpha=0.8, label=f'Max: {max_len}')
    
    plt.legend()

    output_filename = f'dist_step_{batch_idx}_bz{train_batch_size}_n{rollout_n}.png'
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    
    print(f"已生成 Step {batch_idx} 的图表: {output_filename} (包含 {current_chunk_len} 条记录)")

print("\n数据处理完成！")