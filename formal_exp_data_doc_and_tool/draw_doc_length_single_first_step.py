"""
单独绘制一个 doc_lengths_*.csv 的 first step 分布（zhengding 风格）。

默认：
- 只取 step=1
- 只取该 step 的前 1024 条（train_batch_size=256, rollout_n=4）
- 细粒度直方图；加 --line 为折线图
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


train_batch_size = 256
rollout_n = 4
chunk_size = train_batch_size * rollout_n  # 1024


def load_df(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "length" not in df.columns:
        raise ValueError("CSV 缺少 length 列")
    if "step" not in df.columns:
        raise ValueError("CSV 缺少 step 列")
    if "turn" not in df.columns:
        df["turn"] = 0
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw single doc length first-step distribution.")
    parser.add_argument(
        "--file",
        type=str,
        default="doc_lengths_qwen3_32b_search_r1_sync_nprobe32_bs256.csv",
        help="CSV filename under this directory, or absolute path.",
    )
    parser.add_argument("--label", type=str, default="Qwen3-32B | nprobe32")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--x-max", type=int, default=None)
    parser.add_argument("--line", action="store_true", help="Draw line plot instead of histogram.")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output PNG path (default: plot_doc_length_single_first_step.png under this dir).",
    )
    args = parser.parse_args()

    use_histogram = not args.line
    step = int(args.step)
    n_bins = int(args.bins)

    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = args.file
    if not os.path.isabs(file_path):
        file_path = os.path.join(dir_path, file_path)

    df = load_df(file_path)
    df_step = df[df["step"] == step]
    if df_step.empty:
        print(f"step={step} 无数据: {file_path}")
        return

    n_take = min(chunk_size, len(df_step))
    chunk = df_step.iloc[:n_take]
    lengths = chunk["length"].astype(int).values

    x_min = float(np.min(lengths))
    x_max = float(np.max(lengths))
    if args.x_max is not None:
        x_max = min(x_max, float(args.x_max))
    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    color_solid = (0.25, 0.55, 0.75)  # 蓝
    color_alpha = (color_solid[0], color_solid[1], color_solid[2], 0.6)

    if use_histogram:
        ax.hist(
            lengths,
            bins=bins,
            alpha=color_alpha[3],
            color=color_alpha[:3],
            edgecolor=color_alpha[:3],
            linewidth=1.8,
            label=args.label,
        )
    else:
        counts, _ = np.histogram(lengths, bins=bins)
        ax.plot(bin_centers, counts, color=color_solid, linewidth=1.6, label=args.label)

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Doc Length (Tokens)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title(f"Doc Length (Step {step})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    out = args.save or os.path.join(dir_path, "plot_doc_length_single_first_step.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"已生成: {out}")


if __name__ == "__main__":
    main()

