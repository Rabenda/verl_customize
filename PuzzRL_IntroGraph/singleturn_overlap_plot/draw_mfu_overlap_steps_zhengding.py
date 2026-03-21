import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 复用同一套 MFU 计算（与其它 zhengding 脚本一致）
from draw_mfu_plot import MFUMonitor  # type: ignore


def _smooth_envelope(vals_pct: np.ndarray, window: int = 40, sigma: int = 5) -> np.ndarray:
    """与 zhengding 系列一致的 peak envelope 平滑（因果窗口取 max，再平滑，零段抹平尾迹）。"""
    try:
        from scipy.ndimage import gaussian_filter1d  # type: ignore

        smooth_fn = lambda x, s: gaussian_filter1d(x, sigma=s)
    except ImportError:
        def smooth_fn(x, s):
            k = max(3, int(s * 2) | 1)
            return np.convolve(x, np.ones(k) / k, mode="same")

    env = np.zeros_like(vals_pct)
    for i in range(len(vals_pct)):
        s = max(0, i - window)
        m = float(np.max(vals_pct[s : i + 1]))
        env[i] = m if m > 0.01 else 0.0
    if sigma and sigma > 0:
        env = smooth_fn(env, sigma)
    for i in range(len(vals_pct)):
        if env[i] < 1e-3:
            env[i] = 0.0
    return env


def _read_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


def _infer_dataset_tag(csv_path: str) -> str:
    b = os.path.basename(csv_path).lower()
    if "aime" in b:
        return "AIME"
    if "gsm8k" in b or "gsm" in b:
        return "GSM8K"
    if "math" in b:
        return "MATH"
    return "multiturn"


def _split_by_global_step(rows: List[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    out: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        gs = r.get("global_step", "")
        if gs is None or str(gs).strip() == "":
            continue
        try:
            step = int(str(gs).strip())
        except ValueError:
            continue
        out.setdefault(step, []).append(r)
    return out


def _rows_to_series(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    """把一个 step 的行转成 (time_s, mfu_pct) 序列，time 从 0 累加。"""
    times: List[float] = []
    mfus: List[float] = []
    t = 0.0
    for r in rows:
        if str(r.get("pass_id", "")).strip() == "1":
            continue
        try:
            prefill_tokens = float(r.get("prefill_tokens", 0) or 0)
            batch_size = int(float(r.get("batch_size", 0) or 0))
            avg_seq_len = float(r.get("avg_seq_len", 0) or 0)
            delta_t = float(r.get("forward_time_ms", 0) or 0) / 1000.0
        except (ValueError, TypeError):
            continue
        if delta_t <= 0:
            continue
        t += delta_t
        times.append(t)
        mfus.append(
            float(
                monitor.calculate_step_mfu(
                    prefill_tokens=prefill_tokens,
                    decoding_batch_size=batch_size,
                    avg_seq_length=avg_seq_len,
                    delta_t=delta_t,
                )
                * 100.0
            )
        )
    return np.array(times, dtype=np.float32), np.array(mfus, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay MFU curves of multiple global_steps (zhengding style).")
    parser.add_argument(
        "--csv",
        type=str,
        default="inference_step_log_multiturn_gsm8k_qwen3-4b-instruct-10step.csv",
        help="CSV path (absolute or relative to this directory).",
    )
    parser.add_argument("--hardware", type=str, default="H100_SXM")
    parser.add_argument("--model", type=str, default="Qwen3-4B-instruct")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size (used in MFU calculation).")
    parser.add_argument("--max-steps", type=int, default=10, help="How many steps to plot (default: 10).")
    parser.add_argument("--env-window", type=int, default=40)
    parser.add_argument("--sigma", type=int, default=5)
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="输出路径 .png/.pdf（默认 pic/mfu_overlap_steps.png）。",
    )
    parser.add_argument(
        "--title-suffix",
        type=str,
        default="",
        help="标题中 dataset 后的补充说明。",
    )
    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("需要 matplotlib")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(base_dir, csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    global monitor
    monitor = MFUMonitor(args.hardware, args.model)

    rows = _read_rows(csv_path)
    step_rows = _split_by_global_step(rows)
    if not step_rows:
        raise ValueError("CSV 中没有可解析的 global_step。")

    steps_sorted = sorted(step_rows.keys())
    if args.max_steps is not None and args.max_steps > 0:
        steps_sorted = steps_sorted[: int(args.max_steps)]

    series = []
    max_t = 0.0
    for step in steps_sorted:
        t_arr, mfu_arr = _rows_to_series(step_rows[step])
        if len(t_arr) == 0:
            continue
        env = _smooth_envelope(mfu_arr, window=args.env_window, sigma=args.sigma)
        max_t = max(max_t, float(t_arr[-1]))
        series.append((step, t_arr, env))

    if not series:
        print("没有可绘图的 step 数据。")
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))

    # 同一色系叠加，看重叠与相似度；step 越大 alpha 越深
    base_color = (0.25, 0.55, 0.75)  # 蓝（对齐你之前的配色）
    alphas = np.linspace(0.25, 0.95, num=len(series))
    for (alpha, (step, t_arr, env)) in zip(alphas, series):
        ax.plot(t_arr, env, color=base_color, alpha=float(alpha), linewidth=1.6, label=f"step {step}")

    ax.set_xlim(0, max_t)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time within step (s)", fontsize=10)
    ax.set_ylabel("MFU (%)", fontsize=10)
    tag = _infer_dataset_tag(csv_path)
    title_mid = f"multiturn {tag}"
    if args.title_suffix.strip():
        title_mid = f"{title_mid} {args.title_suffix.strip()}"
    ax.set_title(f"MFU overlap across steps — {args.model} | {title_mid}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # legend 放右上角，避免挡曲线（可以按需关掉）
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
    plt.tight_layout()

    pic_dir = os.path.join(base_dir, "pic")
    os.makedirs(pic_dir, exist_ok=True)
    out = args.save or os.path.join(pic_dir, "mfu_overlap_steps.png")
    out = os.path.abspath(out)
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    save_kw = {"bbox_inches": "tight", "facecolor": "white"}
    if out.lower().endswith((".png", ".jpg", ".jpeg")):
        save_kw["dpi"] = 150
    plt.savefig(out, **save_kw)
    plt.close()
    print(f"✅ 已生成: {out}")


if __name__ == "__main__":
    main()

