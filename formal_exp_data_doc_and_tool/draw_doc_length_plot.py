"""
统计 doc_lengths_*.csv 中第一个 step 的 doc length 分布并绘图。
逻辑类似 formal_exp_data_decoding_length/draw_deocding_plot.py。
横坐标: doc length (tokens)
纵坐标: frequency
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _infer_label_from_path(file_path):
    """从文件名推断简短标签（如 nprobe128）。"""
    if not file_path:
        return "unknown"
    base = os.path.basename(file_path).lower()
    if "nprobe32" in base:
        return "nprobe32"
    if "nprobe128" in base:
        return "nprobe128"
    if "nprobe512" in base:
        return "nprobe512"
    return "doc_lengths"


def main():
    parser = argparse.ArgumentParser(
        description="Doc length distribution (first step): x=doc length, y=frequency."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default=os.path.join(
            os.path.dirname(__file__) or ".",
            "doc_lengths_qwen3_4b_instruct_search_r1_sync_nprobe128.csv",
        ),
        help="Path to doc_lengths CSV (columns: length,turn,step)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Use this step only (default: 1).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path (default: formal_plot/plot_doc_length_<label>_step<N>.png).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50).",
    )
    args = parser.parse_args()
    file_path = args.file_path
    step = args.step
    output_path = args.output
    bins = args.bins

    if not os.path.isfile(file_path):
        print(f"文件不存在: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    if "length" not in df.columns or "step" not in df.columns:
        print("CSV 需包含列: length, step")
        return

    df_step = df[df["step"] == step]
    if df_step.empty:
        print(f"没有 step={step} 的数据")
        return

    lengths = df_step["length"].astype(int)
    n = len(lengths)
    label = _infer_label_from_path(file_path)
    title = f"Doc Length Distribution (Step {step}) — {label} (n={n})"

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.histplot(lengths, bins=bins, kde=True, color="skyblue", edgecolor="black", ax=ax)
    avg_len = lengths.mean()
    ax.axvline(avg_len, color="green", linestyle="--", alpha=0.8, label=f"Avg: {avg_len:.1f}")
    ax.axvline(lengths.max(), color="red", linestyle=":", alpha=0.8, label=f"Max: {lengths.max()}")
    ax.set_xlabel("Doc Length (Tokens)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if output_path is None:
        repo_root = os.path.dirname(os.path.dirname(__file__)) or "."
        plot_dir = os.path.join(repo_root, "formal_plot")
        os.makedirs(plot_dir, exist_ok=True)
        output_path = os.path.join(plot_dir, f"plot_doc_length_{label}_step{step}.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"已生成: {output_path} (step={step}, n={n})")


if __name__ == "__main__":
    main()
