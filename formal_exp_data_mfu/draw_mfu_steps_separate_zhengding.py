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
    """peak envelope 平滑：因果窗口取 max，再平滑，低值抹平尾迹。"""
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


def _rows_to_series(monitor: MFUMonitor, rows: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
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
        mfu_pct = (
            monitor.calculate_step_mfu(
                prefill_tokens=prefill_tokens,
                decoding_batch_size=batch_size,
                avg_seq_length=avg_seq_len,
                delta_t=delta_t,
            )
            * 100.0
        )
        times.append(t)
        mfus.append(float(mfu_pct))
    return np.array(times, dtype=np.float32), np.array(mfus, dtype=np.float32)


def _style_ax(ax):
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw MFU curve per step (separate figures).")
    parser.add_argument(
        "--csv",
        type=str,
        default="inference_step_log_multiturn_gsm8k_qwen3-4b-instruct-10step.csv",
        help="CSV path (absolute or relative to this directory).",
    )
    parser.add_argument("--hardware", type=str, default="H100_SXM")
    parser.add_argument("--model", type=str, default="Qwen3-4B-instruct")
    parser.add_argument("--env-window", type=int, default=40)
    parser.add_argument("--sigma", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--xlim-mode",
        type=str,
        default="max",
        choices=["max", "self"],
        help="'max': all plots share xlim = max step duration; 'self': each step uses its own duration.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory (default: pic/ under this directory).",
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

    out_dir = args.save_dir or os.path.join(base_dir, "pic")
    os.makedirs(out_dir, exist_ok=True)

    monitor = MFUMonitor(args.hardware, args.model)

    rows = _read_rows(csv_path)
    step_rows = _split_by_global_step(rows)
    steps = sorted(step_rows.keys())
    if args.max_steps and args.max_steps > 0:
        steps = steps[: int(args.max_steps)]

    # 先把所有 step 的时长算出来，便于统一 xlim
    step_series: List[Tuple[int, np.ndarray, np.ndarray]] = []
    max_t = 0.0
    max_y = 0.0
    for step in steps:
        t_arr, mfu_arr = _rows_to_series(monitor, step_rows[step])
        if len(t_arr) == 0:
            continue
        env = _smooth_envelope(mfu_arr, window=args.env_window, sigma=args.sigma)
        step_series.append((step, t_arr, env))
        max_t = max(max_t, float(t_arr[-1]))
        max_y = max(max_y, float(np.max(env)) if len(env) else 0.0)

    if not step_series:
        print("没有可绘图的 step 数据。")
        return

    base_color = (0.25, 0.55, 0.75)  # 蓝
    for step, t_arr, env in step_series:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
        ax.plot(t_arr, env, color=base_color, linewidth=1.7)
        if args.xlim_mode == "max":
            ax.set_xlim(0, max_t)
        else:
            ax.set_xlim(0, float(t_arr[-1]))
        ax.set_ylim(0, max(1.0, max_y * 1.05))
        ax.set_xlabel("Time within step (s)", fontsize=10)
        ax.set_ylabel("MFU (%)", fontsize=10)
        ax.set_title(f"MFU — step {step} | {args.model}", fontsize=12, fontweight="bold")
        _style_ax(ax)
        fig.patch.set_facecolor("white")
        plt.tight_layout()

        out = os.path.join(out_dir, f"mfu_step_{step:02d}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ 已生成: {out}")


if __name__ == "__main__":
    main()

