"""
四个 doc_lengths_*.csv 的 first step 分布合在一张图里（zhengding 风格）：
- Qwen3-4B Search-R1 nprobe128 / nprobe32
- Qwen3-4B-instruct Search-R1 nprobe32
- Qwen3-32B Search-R1 nprobe32

默认细粒度直方图；加 --line 为折线图。
横坐标默认截到 4096（与 decoding length 图一致）。
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

# 顺序对齐四列：绿 / 蓝 / 橙 / 紫；图例只显示 model | nprobe，不写 Search-R1
FILES = [
    ("doc_lengths_qwen3_4b_search_r1_sync_nprobe128.csv", "Qwen3-4B", "nprobe128"),
    ("doc_lengths_llama3.1_8b_search_r1_sync_nprobe128.csv", "Llama-3.1-8B", "nprobe128"),
    ("doc_lengths_qwen3_4b_instruct_search_r1_sync_nprobe32.csv", "Qwen3-4B-instruct", "nprobe32"),
    ("doc_lengths_qwen3_32b_search_r1_sync_nprobe32_bs256.csv", "Qwen3-32B", "nprobe32"),
]


def load_df(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    # 期望列：length, turn, step（turn 可选）
    if "length" not in df.columns:
        raise ValueError("CSV 缺少 length 列")
    if "step" not in df.columns:
        raise ValueError("CSV 缺少 step 列")
    if "turn" not in df.columns:
        df["turn"] = 0
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Doc length first-step distribution: default histogram, --line for line plot."
    )
    parser.add_argument(
        "--line",
        action="store_true",
        help="Draw line plot instead of (default) fine-grained histogram.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Use this step only (default: 1).",
    )
    parser.add_argument(
        "--x-max",
        type=int,
        default=None,
        help="Optional x-axis right limit; if unset, 横轴由数据范围自动确定（与 draw_doc_length_plot 一致）.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50, 与 draw_doc_length_plot 一致).",
    )
    args = parser.parse_args()

    use_histogram = not args.line
    step = int(args.step)

    # 颜色与 Search-R1 四列保持一致：绿 / 蓝 / 橙 / 紫
    colors_solid = [
        (0.35, 0.65, 0.4),   # 绿
        (0.25, 0.55, 0.75),  # 蓝
        (0.85, 0.55, 0.2),   # 橙
        (0.6, 0.35, 0.65),   # 紫
    ]
    colors_alpha = [(c[0], c[1], c[2], 0.6) for c in colors_solid]
    # Qwen3-4B 两个配置为实线；instruct/32B 用虚线区分
    line_styles = ["-", "-", "--", "--"]

    all_lengths = []
    labels = []
    first_step_series = []

    for (fname, model, tag) in FILES:
        path = os.path.join(DIR, fname)
        if not os.path.isfile(path):
            print(f"跳过不存在的文件: {path}")
            continue
        df = load_df(path)
        df_step = df[df["step"] == step]
        if df_step.empty:
            print(f"跳过 step={step} 无数据: {path}")
            continue
        # 只取第一个训练 step 对应的一批（与 decoding length 图一致）
        n_take = min(chunk_size, len(df_step))
        chunk = df_step.iloc[:n_take]
        lengths = chunk["length"].astype(int).values
        first_step_series.append(lengths)
        all_lengths.extend(lengths.tolist())
        labels.append(f"{model} | {tag}")

    if not first_step_series:
        print("没有可用的数据。")
        return

    # 横轴与 draw_doc_length_plot 一致：由数据范围自动确定，不固定 0~x_max，避免左侧大片空白
    all_arr = np.array(all_lengths)
    x_min = float(all_arr.min())
    x_max = float(all_arr.max())
    if args.x_max is not None:
        x_max = min(x_max, float(args.x_max))
    n_bins = int(args.bins)
    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    if use_histogram:
        for lengths, label, color, ls in zip(first_step_series, labels, colors_alpha, line_styles):
            ax.hist(
                lengths,
                bins=bins,
                alpha=color[3],
                color=color[:3],
                edgecolor=color[:3],
                linestyle=ls,
                linewidth=1.8,
                label=label,
            )
    else:
        for lengths, label, color, ls in zip(
            first_step_series, labels, colors_solid[: len(first_step_series)], line_styles
        ):
            counts, _ = np.histogram(lengths, bins=bins)
            ax.plot(bin_centers, counts, color=color, linestyle=ls, label=label, linewidth=1.5)

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Doc Length (Tokens)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Doc Length (Step 1)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    out_name = "plot_four_doc_length_first_step.png" if use_histogram else "plot_four_doc_length_first_step_line.png"
    out = os.path.join(DIR, out_name)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"已生成: {out}")


if __name__ == "__main__":
    main()

