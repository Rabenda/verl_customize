import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ================= 1. 外部参数设置 =================
train_batch_size = 256
rollout_n = 4
chunk_size = train_batch_size * rollout_n  # 每个 step 的样本数 (256*4=1024)


def _infer_dataset_and_turn(file_path):
    """从文件路径推断数据集名和 single/multiturn。"""
    if not file_path:
        return "unknown", "single"
    path_lower = os.path.basename(file_path).lower()
    turn_type = "multiturn" if "multiturn" in path_lower else "single"
    if "gsm8k" in path_lower:
        dataset = "gsm8k"
    elif "math" in path_lower:
        dataset = "math"
    else:
        dataset = "unknown"
    return dataset, turn_type


def _infer_model_from_path(file_path):
    """从文件路径推断模型名。qwen3-4b-instruct 与 qwen3-4b 区分开。"""
    if not file_path:
        return None
    path_lower = os.path.basename(file_path).lower()
    if "llama3.1-8b" in path_lower or "llama-3.1-8b" in path_lower:
        return "Llama-3.1-8B"
    if "qwen3-0.6b" in path_lower:
        return "Qwen3-0.6B"
    if "qwen3-4b-instruct" in path_lower:
        return "Qwen3-4B-instruct"
    if "qwen3-4b" in path_lower:
        return "Qwen3-4B"
    if "qwen3-8b" in path_lower:
        return "Qwen3-8B"
    return None


def _default_output_path(dataset, turn_type, model):
    """默认输出名：plot_decoding_single/multiturn_数据集_模型.png"""
    model_safe = (model or "unknown").replace(" ", "_")
    return f"plot_decoding_{turn_type}_{dataset}_{model_safe}.png"


def load_df(file_path):
    """支持两种格式：无表头单列 length（single-turn）或 表头 length,turn（multi-turn）"""
    with open(file_path) as f:
        first_line = f.readline().strip()
    if first_line.lower().startswith("length") and "turn" in first_line.lower():
        df = pd.read_csv(file_path)
        if "turn" not in df.columns:
            df["turn"] = 0
    else:
        df = pd.read_csv(file_path, header=None, names=["length"])
    return df


def plot_single_histogram(chunk, ax, has_turn, title_suffix="", show_step_n=True):
    """在给定 ax 上绘制一个 chunk 的直方图。show_step_n=False 时不显示 Step x (n=...) 标题（用于 single-step 时）。"""
    current_chunk_len = len(chunk)
    if current_chunk_len > 1:
        if has_turn and "turn" in chunk.columns:
            chunk_plot = chunk.copy()
            chunk_plot["turn"] = chunk_plot["turn"].astype(int).astype(str)
            sns.histplot(
                data=chunk_plot, x="length", hue="turn", bins=50, kde=True,
                alpha=0.5, multiple="layer", ax=ax, legend=True
            )
        else:
            sns.histplot(chunk["length"], bins=50, kde=True, color="skyblue", edgecolor="black", ax=ax)
    else:
        ax.hist(chunk["length"], color="skyblue", edgecolor="black")
    avg_len = chunk["length"].mean()
    max_len = chunk["length"].max()
    ax.axvline(avg_len, color="green", linestyle="--", alpha=0.8, label=f"Avg: {avg_len:.1f}")
    ax.axvline(max_len, color="red", linestyle=":", alpha=0.8, label=f"Max: {max_len}")
    if show_step_n:
        ax.set_title(f"{title_suffix} (n={current_chunk_len})", fontsize=12)
    else:
        ax.set_title("", fontsize=12)
    ax.set_xlabel("Decoding Length (Tokens)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser(description="Decoding length distribution by step (multi or single step).")
    parser.add_argument(
        "file_path",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__) or ".", "lengths_model_gsm8k_workload_analysis.csv"),
        help="Path to lengths CSV",
    )
    parser.add_argument(
        "--single-step",
        action="store_true",
        help="Only use first 256*4 samples (first step), output one histogram.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output image path (default: auto by mode).",
    )
    args = parser.parse_args()
    file_path = args.file_path
    single_step = args.single_step
    output_path = args.output

    try:
        df = load_df(file_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    total_rows = len(df)
    has_turn = "turn" in df.columns
    dataset, turn_type = _infer_dataset_and_turn(file_path)
    model = _infer_model_from_path(file_path) or "unknown"
    title_main = f"Decoding Length Distribution — {dataset} | {model} | {turn_type}"

    print(f"成功读取文件: {file_path}")
    print(f"总记录数: {total_rows} | 每 Step 记录数: {chunk_size} | 按 turn 区分: {has_turn}")

    sns.set_theme(style="whitegrid")

    if single_step:
        # 只取前 chunk_size 行（第一个 step）
        n_take = min(chunk_size, total_rows)
        chunk = df.iloc[:n_take]
        print(f"[single-step] 使用前 {n_take} 条 (第一个 step)")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plot_single_histogram(chunk, ax, has_turn, title_suffix="", show_step_n=False)
        fig.suptitle(title_main, fontsize=14)
        plt.tight_layout()
        out = output_path or _default_output_path(dataset, turn_type, model)
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"已生成: {out}")
        return

    # 多 step 模式：按 chunk_size 切分
    chunks = []
    for i in range(0, total_rows, chunk_size):
        batch_idx = (i // chunk_size) + 1
        start_idx = i
        end_idx = min(i + chunk_size, total_rows)
        chunk = df.iloc[start_idx:end_idx]
        chunks.append((batch_idx, chunk))

    if not chunks:
        print("没有可绘制的数据块。")
        return

    n_steps = len(chunks)
    n_cols = min(4, math.ceil(math.sqrt(n_steps)))
    n_rows = math.ceil(n_steps / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, (batch_idx, chunk) in enumerate(chunks):
        ax = axes_flat[idx]
        plot_single_histogram(chunk, ax, has_turn, title_suffix=f"Step {batch_idx}")

    for idx in range(n_steps, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title_main, fontsize=14)
    plt.tight_layout()
    out = output_path or _default_output_path(dataset, turn_type, model)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"已生成一张图包含 {n_steps} 个 Step: {out}")


if __name__ == "__main__":
    main()
