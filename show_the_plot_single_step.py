import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
file_path = '/workspace/lengths_qwen3_1.7b_gsm8k_hehua.csv' 

try:
    df = pd.read_csv(file_path, header=None, names=['length'])
except Exception as e:
    print(f"读取文件失败: {e}")
    exit()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

sns.histplot(df['length'], bins=50, kde=True, color='skyblue', edgecolor='black')

plt.title('Distribution of Model Decoding Lengths', fontsize=15)
plt.xlabel('Decoding Length (Tokens)', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)

max_len = df['length'].max()
plt.axvline(max_len, color='red', linestyle='--', alpha=0.7, label=f'Max Length: {max_len}')
plt.legend()

plt.tight_layout()
plt.savefig('length_distribution.png')
print(f"图像已保存为 'length_distribution.png'。最大长度为: {max_len}")
plt.show()