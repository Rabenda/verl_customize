"""
四个 single-turn CSV 的 first step 分布合在一张图里，支持折线图或直方图：
- qwen3-4b gsm8k / math，qwen3-4b-instruct gsm8k / math
默认细粒度直方图；加 --line 为折线图。横坐标均截取到 4096；直方图用 100 格。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_batch_size = 256
rollout_n = 4
chunk_size = train_batch_size * rollout_n  # 1024

DIR = os.path.dirname(os.path.abspath(__file__))

# 四个 single-turn CSV（Search-R1）
FILES = [
    ("lengths_model_search_r1_qwen3_4b_nprobe128.csv", "Qwen3-4B", "Search-R1 nprobe128"),
    ("lengths_model_search_r1_qwen3_4b_nprobe32.csv", "Qwen3-4B", "Search-R1 nprobe32"),
    ("lengths_model_search_r1_qwen3_4b_instruct_nprobe32.csv", "Qwen3-4B-instruct", "Search-R1 nprobe32"),
    ("lengths_model_search_r1_qwen3_32b_nprobe32.csv", "Qwen3-32B", "Search-R1 nprobe32"),
]


def load_df(file_path):
    """支持无表头单列 length 或 表头 length,turn。"""
    with open(file_path) as f:
        first_line = f.readline().strip()
    if first_line.lower().startswith("length") and "turn" in first_line.lower():
        df = pd.read_csv(file_path)
        if "turn" not in df.columns:
            df["turn"] = 0
    else:
        df = pd.read_csv(file_path, header=None, names=["length"])
    return df


def main():
    parser = argparse.ArgumentParser(description="First step decoding length: default histogram, --line for line plot.")
    parser.add_argument(
        "--line",
        action="store_true",
        help="Draw line plot instead of (default) fine-grained histogram.",
    )
    args = parser.parse_args()
    use_histogram = not args.line

    # 提高了对比度的柔和配色（色相分明，饱和度适中）
    colors_solid = [
        (0.30, 0.45, 0.65),  # 灰蓝色 (Base - GSM8K)
        (0.75, 0.45, 0.35),  # 铁锈红 (Base - MATH)
        (0.25, 0.60, 0.45),  # 灰绿色 (Instruct - GSM8K)
        (0.65, 0.45, 0.65),  # 柔紫色 (Instruct - MATH)
    ]
    # 透明度从 0.5 提高到 0.6，避免颜色显得太脏
    colors_alpha = [(c[0], c[1], c[2], 0.6) for c in colors_solid]
    
    # 用线形区分 Base 和 Instruct 模型
    # Base 用实线 '-', Instruct 用短虚线 '--'
    line_styles = ['-', '-', '--', '--']

    all_lengths = []
    labels = []
    first_step_series = []

    for (fname, model, dataset) in FILES:
        path = os.path.join(DIR, fname)
        if not os.path.isfile(path):
            print(f"跳过不存在的文件: {path}")
            continue
        df = load_df(path)
        n_take = min(chunk_size, len(df))
        chunk = df.iloc[:n_take]
        lengths = chunk["length"].values
        first_step_series.append(lengths)
        all_lengths.extend(lengths)
        labels.append(f"{model} | {dataset}")

    if not first_step_series:
        print("没有可用的数据。")
        return

    x_max_plot = 4096
    if use_histogram:
        n_bins = 100
        bins = np.linspace(0, x_max_plot, n_bins + 1)
    else:
        all_arr = np.array(all_lengths)
        x_min, x_max = all_arr.min(), all_arr.max()
        bins = np.linspace(x_min, max(x_max, x_max_plot), 50)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    if use_histogram:
        for lengths, label, color, ls in zip(first_step_series, labels, colors_alpha, line_styles):
            ax.hist(
                lengths,
                bins=bins,
                alpha=color[3],
                color=color[:3],
                edgecolor=color[:3],  # 边框颜色与主体一致
                linestyle=ls,         # 应用虚线/实线
                linewidth=1.8,        # 稍微加粗边框，让虚线更清晰
                label=label,
            )
    else:
        for lengths, label, color, ls in zip(first_step_series, labels, colors_solid[: len(first_step_series)], line_styles):
            counts, _ = np.histogram(lengths, bins=bins)
            ax.plot(
                bin_centers, 
                counts, 
                color=color, 
                linestyle=ls,         
                label=label, 
                linewidth=1.5       
            )

    ax.set_xlim(0, x_max_plot)
    ax.set_xlabel("Decoding Length (Tokens)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Decoding Length (Single Turn)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    out_name = "plot_four_single_turn_first_step.png" if use_histogram else "plot_four_single_turn_first_step_line.png"
    out = os.path.join(DIR, out_name)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"已生成: {out}")


if __name__ == "__main__":
    main()