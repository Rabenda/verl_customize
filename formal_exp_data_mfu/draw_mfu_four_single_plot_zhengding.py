import argparse
import csv
import os
from typing import List, Dict, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 直接复用单曲线脚本里的硬件/模型配置与 MFU 计算逻辑，确保数值一致
from draw_mfu_plot import (  # type: ignore
    HARDWARE_CONFIGS,
    MODEL_CONFIGS,
    MFUMonitor,
    find_global_step_boundaries,
)


SUBPLOT_SIZE = (4.5, 3)  # 单个子图宽 x 高

# 每个 model 固定一种颜色（跨列保持一致）
MODEL_COLORS_SOLID = {
    "Qwen3-4B": (0.35, 0.65, 0.4),  # 绿
    "Qwen3-4B-instruct": (0.85, 0.55, 0.2),  # 橙
    "Qwen3-32B": (0.6, 0.35, 0.65),  # 紫
}

# (csv 文件, 模型名, 数据集名)，顺序严格按照用户要求：
# 1) gsm8k | Qwen3-4B
# 2) gsm8k | Qwen3-4B-instruct
# 3) math  | Qwen3-4B-instruct
# 4) gsm8k | Qwen3-32B
FILES: List[Tuple[str, str, str]] = [
    ("inference_step_log_gsm8k_qwen3-4b.csv", "Qwen3-4B", "GSM8K"),
    ("inference_step_log_gsm8k_qwen3-4b-instruct.csv", "Qwen3-4B-instruct", "GSM8K"),
    ("inference_step_log_math_qwen3-4b-instruct.csv", "Qwen3-4B-instruct", "MATH"),
    ("inference_step_log_gsm8k_qwen3-32b.csv", "Qwen3-32B", "GSM8K"),
]


def _parse_first_step_metrics(csv_path: str, min_prefill_tokens: int = 5000) -> List[Dict]:
    """从 inference_step_log_*.csv 中解析出第一个 global step 的 metric 列表。

    与 draw_mfu_plot 中 main 的 CSV 分支保持逻辑一致：
    - 若存在 global_step 列：只取 global_step == "1"
    - 否则：用 find_global_step_boundaries 取第一个 step 的行
    - 忽略 pass_id == 1 的行
    """
    metrics: List[Dict] = []
    if not csv_path or not os.path.isfile(csv_path):
        return metrics

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if not all_rows:
        return metrics

    fieldnames = list(all_rows[0].keys())
    rows_to_use: List[Dict] = []

    if "global_step" in fieldnames:
        rows_to_use = [r for r in all_rows if str(r.get("global_step", "")).strip() == "1"]
    else:
        boundaries = find_global_step_boundaries(
            all_rows, csv_path=csv_path, min_prefill_tokens=min_prefill_tokens
        )
        end = boundaries[1] if len(boundaries) > 1 else len(all_rows)
        rows_to_use = all_rows[:end]

    for row in rows_to_use:
        if row.get("pass_id") == "1":
            continue
        try:
            metrics.append(
                {
                    "prefill_tokens": float(row["prefill_tokens"]),
                    "decoding_batch_size": int(row["batch_size"]),
                    "avg_seq_length": float(row["avg_seq_len"]),
                    "delta_t": float(row["forward_time_ms"]) / 1000.0,
                }
            )
        except (ValueError, TypeError, KeyError):
            continue

    return metrics


def _smooth_envelope(vals: np.ndarray, window: int = 40, sigma: int = 5) -> np.ndarray:
    """简单的 peak envelope 平滑：窗口内取最大值后再做一次高斯/卷积平滑。"""
    from scipy.ndimage import gaussian_filter1d  # type: ignore

    n = len(vals)
    env = np.zeros_like(vals)
    for i in range(n):
        s = max(0, i - window)
        m = np.max(vals[s : i + 1])
        env[i] = m if m > 0.01 else 0.0
    if sigma and sigma > 0:
        env = gaussian_filter1d(env, sigma=sigma)
    # 把原本为 0 的位置保持为 0，避免尾部抬起
    for i in range(n):
        if env[i] < 1e-3:
            env[i] = 0.0
    return env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="四个 single-turn 实验的 MFU 1x4 子图 (zhengding 版)"
    )
    parser.add_argument("-hw", "--hardware", default="H100_SXM")
    parser.add_argument(
        "--env-window", type=int, default=40, help="peak envelope 窗口长度（默认 40）"
    )
    parser.add_argument(
        "--sigma", type=int, default=5, help="peak envelope 高斯平滑 sigma（默认 5）"
    )
    parser.add_argument(
        "--min-prefill-tokens",
        type=int,
        default=5000,
        help="判定新 global step 的 EXTEND prefill 下限（默认 5000）",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="输出 PDF 路径（默认 pic/new1.pdf）",
    )
    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("需要 matplotlib")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))

    fig, axes = plt.subplots(1, 4, figsize=(SUBPLOT_SIZE[0] * 4, SUBPLOT_SIZE[1]), sharey=True)

    titles = [
        "Qwen3-4B | GSM8K",
        "Qwen3-4B-instruct | GSM8K",
        "Qwen3-4B-instruct | MATH",
        "Qwen3-32B | GSM8K",
    ]

    for idx, (fname, model_name, dataset) in enumerate(FILES):
        ax = axes[idx]
        csv_path = os.path.join(base_dir, fname)
        if not os.path.isfile(csv_path):
            ax.set_title(f"缺失: {fname}", fontsize=10, color="red")
            continue

        if model_name not in MODEL_CONFIGS:
            ax.set_title(f"未知模型: {model_name}", fontsize=10, color="red")
            continue

        metrics = _parse_first_step_metrics(
            csv_path, min_prefill_tokens=args.min_prefill_tokens
        )
        if not metrics:
            ax.set_title(f"无数据: {fname}", fontsize=10, color="red")
            continue

        monitor = MFUMonitor(args.hardware, model_name)

        times: List[float] = []
        mfus: List[float] = []
        curr_t = 0.0
        for m in metrics:
            curr_t += m["delta_t"]
            times.append(curr_t)
            mfus.append(monitor.calculate_step_mfu(**m) * 100.0)

        t_arr = np.array(times)
        v_arr = np.array(mfus)
        if len(t_arr) == 0:
            ax.set_title(f"无有效点: {fname}", fontsize=10, color="red")
            continue

        env = _smooth_envelope(v_arr, window=args.env_window, sigma=args.sigma)

        color = MODEL_COLORS_SOLID.get(model_name, (0.25, 0.55, 0.75))
        ax.plot(t_arr, env, color=color, linewidth=1.8)
        ax.set_title(titles[idx], fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("white")

    for ax in axes:
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("MFU (%)", fontsize=10)

    fig.patch.set_facecolor("white")
    plt.tight_layout()

    pic_dir = os.path.join(base_dir, "pic")
    os.makedirs(pic_dir, exist_ok=True)
    out_path = args.save or os.path.join(pic_dir, "new1.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 已生成: {out_path}")


if __name__ == "__main__":
    main()

