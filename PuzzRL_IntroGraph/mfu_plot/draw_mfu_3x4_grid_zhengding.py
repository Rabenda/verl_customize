import argparse
import os
import sys
from typing import List, Dict, Tuple

_MFU_PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
if _MFU_PLOT_DIR not in sys.path:
    sys.path.insert(0, _MFU_PLOT_DIR)

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# single / multiturn 复用这两个脚本的解析与 envelope 逻辑（保证数值一致）
from draw_mfu_four_single_plot_zhengding import (  # type: ignore
    _parse_first_step_metrics as _parse_first_step_metrics_single,
    _smooth_envelope as _smooth_envelope_single,
    FILES as _SINGLE_FILES_BASE,
)

# 3x4 图专用：Qwen3-4B-instruct MATH 换用 gpuuti0.85 版本
SINGLE_FILES = [
    (fname, model, dataset) if (model, dataset) != ("Qwen3-4B-instruct", "MATH")
    else ("inference_step_log_math_qwen3-4b-instruct_gpuuti0.85.csv", model, dataset)
    for fname, model, dataset in _SINGLE_FILES_BASE
]

from draw_mfu_four_multiturn_plot_zhengding import (  # type: ignore
    _parse_first_step_metrics as _parse_first_step_metrics_multiturn,
    _smooth_envelope as _smooth_envelope_multiturn,
    FILES as _MULTITURN_FILES_BASE,
)

# 3x4 图专用：32B 换用 GSM8K gpuuti0.85 版本
MULTITURN_FILES = [
    (fname, model, dataset) if model != "Qwen3-32B"
    else ("inference_step_log_multiturn_aime_qwen3-32b.csv", "Qwen3-32B", "AIME")
    for fname, model, dataset in _MULTITURN_FILES_BASE
]

# search-r1 复用现有逻辑（含 faiss 灰块与 turn 分段）
from draw_mfu_four_plot_zhengding import (  # type: ignore
    DOC_AND_TOOL_DIR,
    FAISS_RAMP_TIME,
    MFU_ZERO_OFFSET_PCT,
    FILES as SEARCH_R1_FILES,
    MFUMonitor,
    get_first_step_rows,
    find_turn_boundaries,
    parse_csv_to_metrics,
    _read_tool_times_by_turn,
)


# 与 draw_3x1_combined_zhengding.py 对齐的 (model, dataset) → color 表
# 不在表中的组合用黑色
_COLOR_MAP = {
    ("Qwen3-4B",          "GSM8K"):            (0.35, 0.65, 0.4),   # 绿
    ("Qwen3-4B",          "MATH"):             (0.7,  0.35, 0.3),   # 红
    ("Qwen3-4B-instruct", "GSM8K"):            (0.25, 0.55, 0.75),  # 蓝
    ("Qwen3-4B-instruct", "MATH"):             (0.85, 0.55, 0.2),   # 橙
    ("Qwen3-32B",         "AIME"):              (0.6,  0.35, 0.65),  # 紫（对齐 3x1 Row1）
    ("Qwen3-32B",         "GSM8K"):             (0.6,  0.35, 0.65),  # 紫
    ("Qwen3-32B",         "Search-R1 faiss32"): (0.3,  0.65, 0.75), # 青（对齐 3x1 Row2）
}
_GRAY = (0.65, 0.65, 0.65)


def _get_color(model: str, dataset: str) -> tuple:
    return _COLOR_MAP.get((model, dataset), _GRAY)


def _setup_ax(ax):
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")


def _compute_peak_envelope(vals: np.ndarray, env_window: int, sigma: int) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter1d  # type: ignore

        smooth_fn = lambda x, s: gaussian_filter1d(x, sigma=s)
    except ImportError:
        def smooth_fn(x, s):
            k = max(3, int(s * 2) | 1)
            return np.convolve(x, np.ones(k) / k, mode="same")

    peak = np.zeros_like(vals)
    for i in range(len(vals)):
        start = max(0, i - env_window)
        m = float(np.max(vals[start : i + 1]))
        peak[i] = m if m > 0.01 else 0.0
    if sigma and sigma > 0:
        smooth = smooth_fn(peak, sigma)
        for i in range(len(vals)):
            if peak[i] == 0:
                smooth[i] = 0
    else:
        smooth = peak
    return smooth


def _smooth_direct(vals: np.ndarray, sigma: int) -> np.ndarray:
    """直接 Gaussian 平滑，保留 dip，不做 peak envelope。"""
    if sigma and sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter1d  # type: ignore
            return gaussian_filter1d(vals.astype(float), sigma=sigma)
        except ImportError:
            k = max(3, int(sigma * 2) | 1)
            return np.convolve(vals, np.ones(k) / k, mode="same")
    return vals.copy()


def _plot_single_row(ax_row, base_dir: str, hardware: str, env_window: int, sigma: int,
                     min_prefill_tokens: int, metric: str):
    titles = [
        "Qwen3-4B | GSM8K",
        "Qwen3-4B-Instruct | GSM8K",
        "Qwen3-4B-Instruct | MATH",
        "Qwen3-32B | MATH",
    ]
    for col, (fname, model_name, dataset) in enumerate(SINGLE_FILES):
        ax = ax_row[col]
        csv_path = os.path.join(base_dir, fname)
        if not os.path.isfile(csv_path):
            ax.set_title(f"缺失: {fname}", fontsize=10, color="red")
            _setup_ax(ax)
            continue
        metrics = _parse_first_step_metrics_single(csv_path, min_prefill_tokens=min_prefill_tokens)
        if not metrics:
            ax.set_title(f"无数据: {fname}", fontsize=10, color="red")
            _setup_ax(ax)
            continue
        monitor = MFUMonitor(hardware, model_name)
        times: List[float] = []
        vals: List[float] = []
        t = 0.0
        for m in metrics:
            t += m["delta_t"]
            times.append(t)
            if metric == "mfu":
                vals.append(monitor.calculate_step_mfu(**m) * 100.0)
            else:
                vals.append(float(m["decoding_batch_size"]))
        t_arr = np.array(times)
        v_arr = np.array(vals)
        env = _smooth_envelope_single(v_arr, window=env_window, sigma=sigma)
        color = _get_color(model_name, dataset)
        lw = 3.2 if color != _GRAY else 1.6
        ax.plot(t_arr, env, color=color, linewidth=lw)
        ax.set_title(titles[col], fontsize=11)
        _setup_ax(ax)


def _plot_multiturn_row(ax_row, base_dir: str, hardware: str, env_window: int, sigma: int,
                        min_prefill_tokens: int, metric: str):
    titles = [
        "Qwen3-4B | GSM8K",
        "Qwen3-4B | MATH",
        "Qwen3-4B-Instruct | AIME",
        "Qwen3-32B | AIME",
    ]
    for col, (fname, model_name, dataset) in enumerate(MULTITURN_FILES):
        ax = ax_row[col]
        csv_path = os.path.join(base_dir, fname)
        if not os.path.isfile(csv_path):
            ax.set_title(f"缺失: {fname}", fontsize=10, color="red")
            _setup_ax(ax)
            continue
        metrics = _parse_first_step_metrics_multiturn(csv_path, min_prefill_tokens=min_prefill_tokens)
        if not metrics:
            ax.set_title(f"无数据: {fname}", fontsize=10, color="red")
            _setup_ax(ax)
            continue
        monitor = MFUMonitor(hardware, model_name)
        times: List[float] = []
        vals: List[float] = []
        t = 0.0
        for m in metrics:
            t += m["delta_t"]
            times.append(t)
            if metric == "mfu":
                vals.append(monitor.calculate_step_mfu(**m) * 100.0)
            else:
                vals.append(float(m["decoding_batch_size"]))
        t_arr = np.array(times)
        v_arr = np.array(vals)
        env = _smooth_envelope_multiturn(v_arr, window=env_window, sigma=sigma)
        color = _get_color(model_name, dataset)
        lw = 3.2 if color != _GRAY else 1.6
        ax.plot(t_arr, env, color=color, linewidth=lw)
        ax.set_title(titles[col], fontsize=11)
        _setup_ax(ax)


def _plot_search_r1_row(ax_row, base_dir: str, hardware: str, env_window: int, sigma: int,
                        min_prefill_tokens: int, metric: str):
    titles = [
        "Qwen3-4B | Faiss-128",
        "Qwen3-4B | Faiss-32",
        "Qwen3-4B-Instruct | Faiss-32",
        "Qwen3-32B | Faiss-32",
    ]
    for col, file_entry in enumerate(SEARCH_R1_FILES):
        ax = ax_row[col]
        fname = file_entry[0]
        model, dataset = file_entry[1], file_entry[2]
        tool_times_fname = file_entry[3] if len(file_entry) >= 4 else None
        path = os.path.join(base_dir, fname)
        if not os.path.isfile(path):
            ax.set_title(f"缺失: {fname}", fontsize=10, color="red")
            _setup_ax(ax)
            continue

        monitor = MFUMonitor(hardware, model)
        metrics = parse_csv_to_metrics(path, first_step_only=True, min_prefill_tokens=min_prefill_tokens)
        if not metrics:
            ax.set_title(f"无数据: {fname}", fontsize=10, color="red")
            _setup_ax(ax)
            continue

        times, vals = [], []
        t = 0.0
        for d in metrics:
            t += d["delta_t"]
            times.append(t)
            if metric == "mfu":
                vals.append(monitor.calculate_step_mfu(**d) * 100.0)
            else:
                vals.append(float(d["decoding_batch_size"]))
        times = np.array(times)
        vals = np.array(vals)
        smooth = _compute_peak_envelope(vals, env_window=env_window, sigma=sigma)

        tool_times_by_turn = {}
        boundary_metric_idxs = []
        if tool_times_fname:
            tool_path = os.path.join(DOC_AND_TOOL_DIR, tool_times_fname)
            tool_times_by_turn = _read_tool_times_by_turn(tool_path, step=1)
            first_step_rows = get_first_step_rows(path, min_prefill_tokens=min_prefill_tokens)
            if first_step_rows:
                turn_row_boundaries = find_turn_boundaries(first_step_rows, min_prefill_tokens=2000, max_turns=4)
                if len(tool_times_by_turn) >= 2 and len(turn_row_boundaries) < 2:
                    turn_row_boundaries = find_turn_boundaries(
                        first_step_rows, min_prefill_tokens=1500, lookback=100, max_turns=4
                    )
                row_to_metric = {}
                mi = 0
                for ri, row in enumerate(first_step_rows):
                    if row.get("pass_id") == "1":
                        continue
                    row_to_metric[ri] = mi
                    mi += 1
                for b in turn_row_boundaries[1:]:
                    if b in row_to_metric:
                        boundary_metric_idxs.append(row_to_metric[b])

        color = _get_color(model, dataset)
        zero_offset = MFU_ZERO_OFFSET_PCT if metric == "mfu" else 0.0

        # 有 turn 边界时：按段画，并在段间插 faiss 灰块
        if tool_times_by_turn and boundary_metric_idxs and len(times) > 0:
            splits = [0] + sorted(set(boundary_metric_idxs)) + [len(times)]
            splits = [x for x in splits if 0 <= x <= len(times)]
            if len(splits) >= 2:
                shift = 0.0
                prev_shift = 0.0
                prev_had_tool = False
                prev_ramp_t = 0.0
                x_full = []
                y_full = []
                for seg_i in range(len(splits) - 1):
                    s, e = splits[seg_i], splits[seg_i + 1]
                    if e <= s:
                        continue
                    turn_id = seg_i + 1
                    x0 = float(times[s])
                    x_seg = (times[s:e] - x0) + shift
                    y_seg = smooth[s:e]
                    t_tool = float(tool_times_by_turn.get(turn_id, 0.0) or 0.0)
                    x_end = float(x_seg[-1])
                    ramp_t = min(FAISS_RAMP_TIME, t_tool / 3.0) if t_tool > 0 else 0.0

                    if seg_i > 0 and prev_had_tool:
                        x_full.extend([prev_shift - prev_ramp_t, prev_shift])
                        y_full.extend([zero_offset, float(y_seg[0])])
                        x_full.extend(x_seg[1:].tolist())
                        y_full.extend(y_seg[1:].tolist())
                    else:
                        x_full.extend(x_seg.tolist())
                        y_full.extend(y_seg.tolist())

                    if t_tool > 0:
                        x_full.extend([x_end + ramp_t, x_end + t_tool - ramp_t])
                        y_full.extend([zero_offset, zero_offset])
                        ax.axvspan(x_end, x_end + t_tool, color="#fff176", alpha=0.2, linewidth=0, zorder=0.2)
                        trans = blended_transform_factory(ax.transData, ax.transAxes)
                        ax.text(
                            x_end + t_tool / 2.0,
                            0.92,
                            f"Search Time\n({t_tool:.2f}s)",
                            fontsize=9,
                            ha="center",
                            va="top",
                            color="#2c3e50",
                            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2.0),
                            transform=trans,
                        )
                        prev_ramp_t = ramp_t
                    else:
                        prev_ramp_t = 0.0
                    prev_shift = x_end + t_tool
                    prev_had_tool = (t_tool > 0)
                    shift = prev_shift

                lw = 2.8 if color != _GRAY else 1.4
                ax.plot(x_full, y_full, color=color, linewidth=lw)
            else:
                lw = 2.8 if color != _GRAY else 1.4
                ax.plot(times, smooth, color=color, linewidth=lw)
        else:
            lw = 2.8 if color != _GRAY else 1.4
            ax.plot(times, smooth, color=color, linewidth=lw)

        ax.set_title(titles[col], fontsize=11)
        _setup_ax(ax)


def main():
    parser = argparse.ArgumentParser(description="拼成 3x4：single / multiturn / search-r1 图")
    parser.add_argument("-hw", "--hardware", default="H100_SXM")
    parser.add_argument("--metric", choices=["mfu", "batch_size"], default="mfu",
                        help="画 MFU(%%) 还是 Batch size（默认 mfu）")
    parser.add_argument("--single-env-window", type=int, default=40)
    parser.add_argument("--single-sigma", type=int, default=5)
    parser.add_argument("--multiturn-env-window", type=int, default=500)
    parser.add_argument("--multiturn-sigma", type=int, default=10)
    parser.add_argument("--search-env-window", type=int, default=200)
    parser.add_argument("--search-sigma", type=int, default=20)
    parser.add_argument("--min-prefill-tokens", type=int, default=5000)
    parser.add_argument("--save", type=str, default=None, help="输出 PDF 路径（默认 pic/mfu_3x4.pdf）")
    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("需要 matplotlib")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(3, 4, figsize=(2.5 * 4, 2 * 3), sharey="row")

    # row 0: single
    _plot_single_row(
        axes[0],
        base_dir=base_dir,
        hardware=args.hardware,
        env_window=args.single_env_window,
        sigma=args.single_sigma,
        min_prefill_tokens=args.min_prefill_tokens,
        metric=args.metric,
    )
    # row 1: multiturn
    _plot_multiturn_row(
        axes[1],
        base_dir=base_dir,
        hardware=args.hardware,
        env_window=args.multiturn_env_window,
        sigma=args.multiturn_sigma,
        min_prefill_tokens=args.min_prefill_tokens,
        metric=args.metric,
    )
    # row 2: search-r1
    _plot_search_r1_row(
        axes[2],
        base_dir=base_dir,
        hardware=args.hardware,
        env_window=args.search_env_window,
        sigma=args.search_sigma,
        min_prefill_tokens=args.min_prefill_tokens,
        metric=args.metric,
    )

    ylabel = "MFU (%)" if args.metric == "mfu" else "Batch size"
    axes[0][0].set_ylabel(ylabel, fontsize=10)
    axes[1][0].set_ylabel(ylabel, fontsize=10)
    axes[2][0].set_ylabel(ylabel, fontsize=10)
    for ax in axes[2]:
        ax.set_xlabel("Time (s)", fontsize=10)

    for r in range(3):
        for c in range(4):
            axes[r][c].set_ylim(bottom=0)

    # 所有子图黑框闭合
    for r in range(3):
        for c in range(4):
            for spine in axes[r][c].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("black")
                spine.set_linewidth(0.8)

    fig.patch.set_facecolor("white")

    # ── 间距调节 ──────────────────────────────────────────────────────────────
    WSPACE = 0.15   # 列间距（figure 坐标，tight_layout 后覆盖）
    HSPACE = 0.65   # 行间距
    # ──────────────────────────────────────────────────────────────────────────
    plt.tight_layout(pad=0.8)
    fig.subplots_adjust(wspace=WSPACE, hspace=HSPACE)

    # 每行加独立灰色背景框，左上角偏白色字体标注 workload 类型
    from matplotlib.patches import FancyBboxPatch
    row_labels = ["Single-Turn", "Multi-Turn", "Search-R1"]

    # ── Patch 大小调节 ────────────────────────────────────────────────────────
    ROW_PAD_X = 0.018   # 灰框左右外扩（figure 坐标）
    ROW_PAD_Y = 0.022   # 灰框上下外扩
    # ──────────────────────────────────────────────────────────────────────────
    for row_idx, label in enumerate(row_labels):
        bboxes = [ax.get_position() for ax in axes[row_idx]]
        x0 = min(b.x0 for b in bboxes) - ROW_PAD_X
        x1 = max(b.x1 for b in bboxes) + ROW_PAD_X
        y0 = min(b.y0 for b in bboxes) - ROW_PAD_Y
        y1 = max(b.y1 for b in bboxes) + ROW_PAD_Y
        rect = FancyBboxPatch(
            (x0 - 0.03, y0 - 0.02), x1 - x0 + 0.03, y1 - y0 + 0.04,
            boxstyle="round,pad=0.002",
            transform=fig.transFigure,
            facecolor=(0.93, 0.93, 0.93),
            edgecolor="none",
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(rect)
        fig.text(
            x0 - 0.03, y1 - 0.008 + 0.06,
            label,
            fontsize=11,
            color=(0.72, 0.72, 0.72),
            va="top",
            ha="left",
            fontweight="bold",
            zorder=2,
        )

    pic_dir = os.path.join(base_dir, "pic")
    os.makedirs(pic_dir, exist_ok=True)
    out = args.save or os.path.join(pic_dir, "mfu_3x4.pdf")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 已生成: {out}")


if __name__ == "__main__":
    main()
