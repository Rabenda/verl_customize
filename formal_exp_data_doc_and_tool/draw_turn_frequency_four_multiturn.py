"""
四个 multiturn decoding length CSV 的 turn 统计（仅 first step）：横轴 turn 数，纵轴 frequency。
数据来自 formal_exp_data_decoding_length 的 lengths_model_multiturn_*.csv（列 length,turn）。
First step 与 draw_length_zhengding_plot_four_single_turn_first_step 一致：取前 chunk_size 行。
绘图风格与 draw_doc_length_zhengding_plot_four_first_step.py 一致（绿/蓝/橙/紫）。
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 与 decoding length first-step 脚本一致
train_batch_size = 256
rollout_n = 4
chunk_size = train_batch_size * rollout_n  # 1024

# 脚本在 formal_exp_data_doc_and_tool，CSV 在 formal_exp_data_decoding_length
DIR = os.path.dirname(os.path.abspath(__file__))
DECODING_DIR = os.path.join(os.path.dirname(DIR), "formal_exp_data_decoding_length")

# 4 个：Qwen3-4B GSM8K, Qwen3-4B MATH, Qwen3-4B-instruct MATH, Qwen3-32B AIME (bs256)
FILES = [
    ("lengths_model_multiturn_gsm8k_qwen3-4b.csv", "Qwen3-4B", "GSM8K"),
    ("lengths_model_multiturn_math_qwen3-4b.csv", "Qwen3-4B", "MATH"),
    ("lengths_model_multiturn_math_qwen3-4b-instruct.csv", "Qwen3-4B-instruct", "MATH"),
    ("lengths_model_multiturn_aime_qwen3-32b-bs256.csv", "Qwen3-32B", "AIME (bs256)"),
]

# 与 doc length 四图一致：绿 / 蓝 / 橙 / 紫
COLORS_SOLID = [
    (0.35, 0.65, 0.4),   # 绿
    (0.25, 0.55, 0.75),  # 蓝
    (0.85, 0.55, 0.2),   # 橙
    (0.6, 0.35, 0.65),   # 紫
]


def main():
    # 先收集四组 (turns, freqs)，再统一横轴做分组柱状图
    all_turns_set = set()
    series = []  # list of (label, color, dict[turn -> freq])
    for idx, (fname, model, dataset) in enumerate(FILES):
        path = os.path.join(DECODING_DIR, fname)
        if not os.path.isfile(path):
            print(f"跳过不存在的文件: {path}")
            continue
        df = pd.read_csv(path)
        if "turn" not in df.columns:
            print(f"缺少 turn 列: {path}")
            continue
        n_take = min(chunk_size, len(df))
        df_first = df.iloc[:n_take]
        counts = df_first["turn"].value_counts().sort_index()
        turn_to_freq = counts.to_dict()
        all_turns_set.update(turn_to_freq.keys())
        label = f"{model} | {dataset}"
        color = COLORS_SOLID[idx % len(COLORS_SOLID)]
        series.append((label, color, turn_to_freq))

    if not series:
        print("没有可绘图数据")
        return
    turns_sorted = sorted(all_turns_set)
    n_series = len(series)
    bar_width = 0.8 / n_series
    x = np.arange(len(turns_sorted))

    # 两段式断轴：下半段看 0~16 细节，上半段显示大数（截断）
    low_ylim = (0, 16)
    all_freqs = []
    for _, _, turn_to_freq in series:
        all_freqs.extend([turn_to_freq.get(t, 0) for t in turns_sorted])
    max_freq = max(all_freqs) if all_freqs else 0
    high_bottom = max(low_ylim[1] + 1, int(max_freq * 0.35))
    high_ylim = (high_bottom, int(max_freq * 1.08) + 1)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6.2, 4.2),
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.05},
    )

    for i, (label, color, turn_to_freq) in enumerate(series):
        freqs = [turn_to_freq.get(t, 0) for t in turns_sorted]
        offset = (i - (n_series - 1) / 2) * bar_width

        bars_top = ax_top.bar(
            x + offset,
            freqs,
            width=bar_width,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.5,
        )
        bars_bottom = ax_bottom.bar(
            x + offset,
            freqs,
            width=bar_width,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

        # 数量标注：小值标在下半段，大值标在上半段
        for bar_t, bar_b, val in zip(bars_top, bars_bottom, freqs):
            if val <= 0:
                continue
            if val <= low_ylim[1]:
                ax_bottom.text(
                    bar_b.get_x() + bar_b.get_width() / 2,
                    bar_b.get_height(),
                    str(int(val)),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            else:
                ax_top.text(
                    bar_t.get_x() + bar_t.get_width() / 2,
                    bar_t.get_height(),
                    str(int(val)),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    # 轴样式与断轴
    ax_top.set_ylim(*high_ylim)
    ax_bottom.set_ylim(*low_ylim)

    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([int(t) for t in turns_sorted])
    ax_bottom.set_xlabel("Turn", fontsize=10)
    ax_top.set_ylabel("Frequency", fontsize=10)
    ax_bottom.set_ylabel("Frequency", fontsize=10)
    ax_top.set_title("Turn vs Frequency (Multi-turn, First Step)", fontsize=12, fontweight="bold")
    ax_top.legend(fontsize=8, ncol=2)

    for ax in (ax_top, ax_bottom):
        ax.set_facecolor("white")
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_top.spines["top"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)

    # 断轴的斜杠标记
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()

    d = 0.008  # 斜杠尺寸（轴坐标系）
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)  # 左下
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右下
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 左上
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右上

    fig.patch.set_facecolor("white")
    plt.tight_layout()

    out = os.path.join(DIR, "plot_four_multiturn_turn_frequency.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"已生成: {out}")


if __name__ == "__main__":
    main()
