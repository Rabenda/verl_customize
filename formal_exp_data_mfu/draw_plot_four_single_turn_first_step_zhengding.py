import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_batch_size = 256
rollout_n = 4
chunk_size = train_batch_size * rollout_n  # 1024

# histogram 透明度，1.0 为不透明，0.5 为半透明（重叠可看清）
HIST_ALPHA = 0.9

DIR = os.path.dirname(os.path.abspath(__file__))

# 四个 single-turn CSV（自下而上：instruct|GSM8K, instruct|MATH, base|GSM8K, base|MATH）
FILES = [
    ("lengths_model_gsm8k_qwen3-4b-instruct.csv", "Qwen3-4B-instruct", "GSM8K"),
    ("lengths_model_math_qwen3-4b-instruct.csv", "Qwen3-4B-instruct", "MATH"),
    ("lengths_model_gsm8k_qwen3-4b.csv", "Qwen3-4B", "GSM8K"),
    ("lengths_model_math_qwen3-4b.csv", "Qwen3-4B", "MATH"),
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=HIST_ALPHA,
        help=f"Histogram transparency (0-1), default={HIST_ALPHA}. 1=opaque, 0.5=semi-transparent.",
    )
    args = parser.parse_args()
    use_histogram = not args.line

    # 改为四个独立的颜色（顺序与 FILES 一致：instruct|GSM8K, instruct|MATH, base|GSM8K, base|MATH）
    colors_solid = [
        (0.25, 0.55, 0.75),  # 蓝 (Qwen3-4B-instruct | GSM8K)
        (0.85, 0.55, 0.2),   # 橙 (Qwen3-4B-instruct | MATH)
        (0.35, 0.65, 0.4),   # 绿 (Qwen3-4B | GSM8K)
        (0.7, 0.35, 0.3),    # 红 (Qwen3-4B | MATH)
    ]
    # Base 用实线 '-', Instruct 用短虚线 '--'
    line_styles = ['--', '--', '-', '-']

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

    x_max_plot = 2048
    x_max_lim = 2120  # 略大于 2048，便于显示截断线
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
        for lengths, label, color, ls in zip(first_step_series, labels, colors_solid, line_styles):
            weights = 100.0 * np.ones_like(lengths) / len(lengths)
            ax.hist(
                lengths,
                bins=bins,
                weights=weights,
                color=color,
                alpha=args.alpha,
                edgecolor='none',     # 无外廓线
                label=label,
            )
    else:
        for lengths, label, color, ls in zip(first_step_series, labels, colors_solid[: len(first_step_series)], line_styles):
            counts, _ = np.histogram(lengths, bins=bins)
            pct = 100.0 * counts / len(lengths)
            ax.plot(
                bin_centers, 
                pct, 
                color=color, 
                linestyle=ls,         
                label=label, 
                linewidth=1.5       
            )

    ax.set_xlim(0, x_max_lim)
    ax.set_xlabel("Decoding Length (Tokens)", fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_title("Decoding Length (Single Turn)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # 用白色矩形覆盖 x > 2048 的区域，遮住超出部分（不处理数据）
    y_min, y_max = ax.get_ylim()
    ax.axvspan(2048, x_max_lim + 50, facecolor="white", edgecolor="none", zorder=5)

    # 右端 2048 处锯齿状截断线，左边标注 length > 2048 往右下角（深灰，对比度稍低）
    zig_color = (0.35, 0.35, 0.35)
    zig_amp = 25  # 锯齿幅度
    n_zig = 16
    y_zig = np.linspace(y_min, y_max, n_zig * 2 + 1)
    x_zig = np.array([2048 - (zig_amp if i % 2 else 0) for i in range(len(y_zig))])
    ax.plot(x_zig, y_zig, color=zig_color, linewidth=1.5, zorder=6)
    ax.annotate(
        "Length > 2048",
        xy=(2000, y_min + (y_max - y_min) * 0.15),
        xytext=(2048 - zig_amp - 100, y_min + (y_max - y_min) * 0.35),
        fontsize=9, color=zig_color, ha="right",
        arrowprops=dict(arrowstyle="->", color=zig_color, lw=1),
        fontweight="bold",
    )

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    out_name = "pic/plot_four_single_turn_first_step.pdf" if use_histogram else "pic/plot_four_single_turn_first_step_line.pdf"
    out = os.path.join(DIR, out_name)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"已生成: {out}")


if __name__ == "__main__":
    main()
