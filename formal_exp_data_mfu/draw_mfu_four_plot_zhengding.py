"""
绘制四个配置的 MFU 线图，1x4 子图：Qwen3-4B|GSM8K, Qwen3-4B|MATH, Qwen3-4B-instruct|AIME, Qwen3-30B|AIME (均为 multiturn)
子图尺寸 (4.5 x 3 per subplot)
"""
import csv
import os
import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.transforms import blended_transform_factory
    matplotlib.use("Agg")
    try:
        from scipy.ndimage import gaussian_filter1d
        _smooth_fn = lambda x, s: gaussian_filter1d(x, sigma=s)
    except ImportError:
        def _smooth_fn(x, sigma):
            k = max(3, int(sigma * 2) | 1)
            return np.convolve(x, np.ones(k) / k, mode='same')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 与 draw_plot_four_single_turn_first_step.py 一致的子图尺寸
SUBPLOT_SIZE = (4.5, 3)  # 单个子图宽 x 高
# faiss 灰块内 MFU“0”线画在横轴上方多少（%），避免与轴重合；斜坡过渡时长（秒）
MFU_ZERO_OFFSET_PCT = 0.4
FAISS_RAMP_TIME = 0.4

HARDWARE_CONFIGS = {
    "H100_SXM": {"peak_tflops": 989.0, "mem_bandwidth_tb": 3.35},
    "H100_PCIe": {"peak_tflops": 756.0, "mem_bandwidth_tb": 2.0},
}

MODEL_CONFIGS = {
    "Qwen3-0.6B":        {"params_b": 0.46, "num_layers": 24, "hidden_size": 1024},
    "Qwen3-4B":         {"params_b": 3.96, "num_layers": 40, "hidden_size": 2560},
    "Qwen3-4B-instruct": {"params_b": 3.96, "num_layers": 40, "hidden_size": 2560},
    "Llama-3.1-8B":     {"params_b": 8.03, "num_layers": 32, "hidden_size": 4096},
    "Qwen3-8B":         {"params_b": 7.61, "num_layers": 28, "hidden_size": 3584},
    "DeepSeek-R1-Distill-Qwen-14B": {"params_b": 14.0, "num_layers": 48, "hidden_size": 5120},  # HF config.json
    "Qwen3-30B-A3B-Thinking-2507": {"params_b": 30.5, "num_layers": 48, "hidden_size": 2048},  # HF Qwen3 MoE 30.5B total, 8 active
    # 稠密 Qwen3-32B（config.json: num_hidden_layers=64, hidden_size=5120, ~32B params）
    "Qwen3-32B": {"params_b": 32.0, "num_layers": 64, "hidden_size": 5120},
}

# (文件名, 模型名, 数据集名[, tool_times文件名或None]) — Search-R1 四配置，faiss 来自 formal_exp_data_doc_and_tool
FILES = [
    ("inference_step_log_search_r1_qwen3-4b_faiss128.csv", "Qwen3-4B", "Search-R1 faiss128", "tool_times_qwen3_4b_search_r1_sync_nprobe128.csv"),
    ("inference_step_log_search_r1_qwen-4b_faiss32.csv", "Qwen3-4B", "Search-R1 faiss32", "tool_times_qwen3_4b_search_r1_sync_nprobe32.csv"),
    ("inference_step_log_search_r1_qwen3-4b-instruct_faiss32.csv", "Qwen3-4B-instruct", "Search-R1 faiss32", "tool_times_qwen3_4b_instruct_search_r1_sync_nprobe32.csv"),
    ("inference_step_log_search_r1_qwen3-32b_faiss32.csv", "Qwen3-32B", "Search-R1 faiss32", "tool_times_qwen3_32b_search_r1_sync_nprobe32.csv"),
]

# 原始 Search-R1 配色（这是你之前定好的，也是“对的”）：
# Qwen3-4B           -> 绿（按 model 固定，不再按列区分）
# Qwen3-4B-instruct  -> 橙
# Qwen3-32B          -> 紫
MODEL_COLORS_SOLID = {
    "Qwen3-4B": (0.35, 0.65, 0.4),  # 绿
    "Qwen3-4B-instruct": (0.85, 0.55, 0.2),  # 橙
    "Qwen3-32B": (0.6, 0.35, 0.65),  # 紫
}
LINE_STYLES = ['-', '-', '-', '-']

# FAISS/tool 时间来自 formal_exp_data_doc_and_tool
DOC_AND_TOOL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "formal_exp_data_doc_and_tool")


def _read_faiss_total_sec(csv_path, step=1):
    """读取 tool_times_*.csv，返回指定 step 的 time_s 总和（该 step 下所有 turn 的 faiss-gpu 时间）。"""
    if not csv_path or not os.path.isfile(csv_path):
        return None
    try:
        with open(csv_path, "r") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    total = 0.0
    for r in rows:
        try:
            if int(r.get("step", 0) or 0) != int(step):
                continue
            total += float(r.get("time_s", 0) or 0)
        except (ValueError, TypeError):
            continue
    return total if total > 0 else None


def _read_tool_times_by_turn(csv_path, step=1):
    """读取 tool_times_*.csv，返回指定 step 的 {turn_id: time_s}（用于在 turn 之间插 faiss 块）。"""
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    try:
        with open(csv_path, "r") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}
    out = {}
    for r in rows:
        try:
            if int(r.get("step", 0) or 0) != int(step):
                continue
            turn = int(r.get("turn", 0) or 0)
            if turn > 0:
                out[turn] = float(r.get("time_s", 0) or 0)
        except (ValueError, TypeError):
            continue
    return out


def find_turn_boundaries(rows, min_prefill_tokens=2000, tail_decode_max_batch=5, lookback=80, max_turns=4):
    """同一 global_step 内 turn 边界：尾部小 batch DECODE 后再次出现大 EXTEND prefill 视为下一 turn 开始。"""
    if len(rows) < 2:
        return [0]
    boundaries = [0]
    for i in range(1, len(rows)):
        try:
            mode_cur = (rows[i].get("mode") or "").strip()
            if mode_cur != "EXTEND":
                continue
            prefill = float(rows[i].get("prefill_tokens", 0) or 0)
            if prefill < min_prefill_tokens:
                continue
            start = max(0, i - lookback)
            seen_tail_decode = False
            for j in range(start, i):
                if (rows[j].get("mode") or "").strip() != "DECODE":
                    continue
                try:
                    if int(rows[j].get("batch_size", 0) or 0) <= tail_decode_max_batch:
                        seen_tail_decode = True
                        break
                except (ValueError, TypeError):
                    continue
            if seen_tail_decode:
                boundaries.append(i)
                if max_turns and len(boundaries) >= int(max_turns):
                    break
        except Exception:
            continue
    return boundaries


def get_first_step_rows(csv_path, min_prefill_tokens=5000):
    """返回与 parse_csv_to_metrics first_step_only 一致的首 step 行（用于 turn 边界检测）。"""
    if not csv_path or not os.path.isfile(csv_path):
        return []
    with open(csv_path, "r") as f:
        all_rows = list(csv.DictReader(f))
    if not all_rows:
        return []
    fieldnames = list(all_rows[0].keys())
    if "global_step" in fieldnames:
        return [r for r in all_rows if r.get("global_step") == "1"]
    boundaries = find_global_step_boundaries(all_rows, min_prefill_tokens=min_prefill_tokens)
    end = boundaries[1] if len(boundaries) > 1 else len(all_rows)
    return all_rows[:end]


class MFUMonitor:
    def __init__(self, hw_name, model_name):
        self.hw = HARDWARE_CONFIGS[hw_name]
        self.model = MODEL_CONFIGS[model_name]

    def calculate_step_mfu(self, prefill_tokens, decoding_batch_size, avg_seq_length, delta_t, tp_size=4):
        N, L, h = self.model["params_b"] * 1e9, self.model["num_layers"], self.model["hidden_size"]
        total_tokens = prefill_tokens + decoding_batch_size
        flops_linear = 2.0 * N * total_tokens
        flops_attn = 2.0 * L * h * total_tokens * avg_seq_length
        flops_total = flops_linear + flops_attn
        theoretical_peak = (self.hw["peak_tflops"] * 1e12) * delta_t * tp_size
        return flops_total / theoretical_peak if theoretical_peak > 0 else 0


def find_global_step_boundaries(rows, min_prefill_tokens=5000, tail_decode_max_batch=5, lookback=20):
    if len(rows) < 2:
        return [0]
    boundaries = [0]
    for i in range(1, len(rows)):
        try:
            mode_cur = (rows[i].get("mode") or "").strip()
            if mode_cur != "EXTEND":
                continue
            prefill = float(rows[i].get("prefill_tokens", 0))
            if prefill < min_prefill_tokens:
                continue
            start = max(0, i - lookback)
            seen_tail_decode = False
            for j in range(start, i):
                if (rows[j].get("mode") or "").strip() == "DECODE":
                    try:
                        if int(rows[j].get("batch_size", 0)) <= tail_decode_max_batch:
                            seen_tail_decode = True
                            break
                    except (ValueError, TypeError):
                        pass
            if seen_tail_decode:
                boundaries.append(i)
        except (ValueError, TypeError):
            continue
    return boundaries


def parse_csv_to_metrics(csv_path, first_step_only=True, min_prefill_tokens=5000):
    metrics = []
    if not csv_path or not os.path.isfile(csv_path):
        return metrics
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    if not all_rows:
        return metrics

    if first_step_only:
        fieldnames = list(all_rows[0].keys()) if all_rows else []
        if "global_step" in fieldnames:
            rows_to_use = [r for r in all_rows if r.get("global_step") == "1"]
        else:
            boundaries = find_global_step_boundaries(all_rows, min_prefill_tokens=min_prefill_tokens)
            end = boundaries[1] if len(boundaries) > 1 else len(all_rows)
            rows_to_use = all_rows[:end]
    else:
        rows_to_use = all_rows

    for row in rows_to_use:
        if row.get("pass_id") == "1":
            continue
        try:
            metrics.append({
                "prefill_tokens": float(row["prefill_tokens"]),
                "decoding_batch_size": int(row["batch_size"]),
                "avg_seq_length": float(row["avg_seq_len"]),
                "delta_t": float(row["forward_time_ms"]) / 1000.0,
            })
        except (ValueError, TypeError, KeyError):
            continue
    return metrics


def compute_peak_envelope(vals, env_window=30, sigma=5):
    peak_envelope = np.zeros_like(vals)
    for i in range(len(vals)):
        start = max(0, i - env_window)
        window_data = vals[start : i + 1]
        m = np.max(window_data)
        peak_envelope[i] = m if m > 0.01 else 0.0
    smooth_envelope = _smooth_fn(peak_envelope, sigma)
    for i in range(len(vals)):
        if peak_envelope[i] == 0:
            smooth_envelope[i] = 0
    return smooth_envelope


def main():
    parser = argparse.ArgumentParser(description="Draw MFU 1x3: base|GSM8K, instruct|GSM8K, instruct|MATH")
    parser.add_argument("-hw", "--hardware", default="H100_SXM")
    parser.add_argument("--env-window", type=int, default=80)
    parser.add_argument("--sigma", type=int, default=20)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--first-step-only", action="store_true", default=True)
    parser.add_argument("--no-first-step-only", action="store_true", dest="no_first_step")
    parser.add_argument("--min-prefill-tokens", type=int, default=5000)
    parser.add_argument("--show-instant", action="store_true")
    args = parser.parse_args()

    DIR = os.path.dirname(os.path.abspath(__file__))

    if not HAS_MATPLOTLIB:
        print("需要 matplotlib")
        return

    # 1x4 子图
    fig, axes = plt.subplots(1, 4, figsize=(SUBPLOT_SIZE[0] * 4, SUBPLOT_SIZE[1]), sharey=True)

    all_data = []
    titles = [
        "Qwen3-4B | Search-R1 faiss128",
        "Qwen3-4B | Search-R1 faiss32",
        "Qwen3-4B-instruct | Search-R1 faiss32",
        "Qwen3-32B | Search-R1 faiss32",
    ]
    for idx, file_entry in enumerate(FILES):
        fname = file_entry[0]
        model, dataset = file_entry[1], file_entry[2]
        tool_times_fname = file_entry[3] if len(file_entry) >= 4 else None
        path = os.path.join(DIR, fname)
        if not os.path.isfile(path):
            print(f"跳过不存在的文件: {path}")
            continue

        monitor = MFUMonitor(args.hardware, model)
        first_step = not getattr(args, "no_first_step", False)
        metrics = parse_csv_to_metrics(path, first_step_only=first_step, min_prefill_tokens=args.min_prefill_tokens)
        if not metrics:
            print(f"无有效数据: {path}")
            continue

        times, mfus = [], []
        curr_time = 0
        for d in metrics:
            curr_time += d["delta_t"]
            times.append(curr_time)
            mfus.append(monitor.calculate_step_mfu(**d))

        times = np.array(times)
        vals = np.array(mfus) * 100
        smooth = compute_peak_envelope(vals, env_window=args.env_window, sigma=args.sigma)
        tool_times_by_turn = {}
        boundary_metric_idxs = []
        if tool_times_fname:
            tool_path = os.path.join(DOC_AND_TOOL_DIR, tool_times_fname)
            tool_times_by_turn = _read_tool_times_by_turn(tool_path, step=1)
            first_step_rows = get_first_step_rows(path, min_prefill_tokens=args.min_prefill_tokens)
            if first_step_rows:
                turn_row_boundaries = find_turn_boundaries(first_step_rows, min_prefill_tokens=2000, max_turns=4)
                # 若 tool_times 标明有 2+ turn 但启发式只找到 1 段，用更松参数再试一次（避免第一子图等只画出一段）
                if len(tool_times_by_turn) >= 2 and len(turn_row_boundaries) < 2:
                    turn_row_boundaries = find_turn_boundaries(
                        first_step_rows, min_prefill_tokens=1500, lookback=100, max_turns=4
                    )
                # 将 row 边界映射到 metric 下标（metrics 跳过 pass_id==1）
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
        color = MODEL_COLORS_SOLID.get(model, (0.25, 0.55, 0.75))
        all_data.append((times, smooth, vals, color, LINE_STYLES[idx], f"{model} | {dataset}", tool_times_by_turn, boundary_metric_idxs))

    for idx, item in enumerate(all_data):
        times, smooth, vals, color, ls, label = item[0], item[1], item[2], item[3], item[4], item[5]
        tool_times_by_turn = item[6] if len(item) >= 7 else {}
        boundary_metric_idxs = item[7] if len(item) >= 8 else []
        ax = axes[idx]
        # Search-R1：有 turn 边界时，按段画 MFU，并在每两段之间插 faiss_gpu 块（turn1 后、turn2 后…）
        if tool_times_by_turn and boundary_metric_idxs and len(times) > 0:
            splits = [0] + sorted(set(boundary_metric_idxs)) + [len(times)]
            splits = [x for x in splits if 0 <= x <= len(times)]
            if len(splits) >= 2:
                zero_offset = MFU_ZERO_OFFSET_PCT
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
                    # 接上一段末的“上升”到本段起点（与上一段末的“水平”无缝接）
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
                        ax.axvspan(x_end, x_end + t_tool, color="#7f8c8d", alpha=0.35, linewidth=0, zorder=0.2)
                        # 用 axes 坐标固定 y，使四个子图的 faiss 标注竖直对齐（与最后一子图位置一致）
                        trans = blended_transform_factory(ax.transData, ax.transAxes)
                        ax.text(x_end + t_tool / 2.0, 0.92, f"faiss t{turn_id}={t_tool:.2f}s", fontsize=9, ha="center", va="top", color="#2c3e50", bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2.0), transform=trans)
                        prev_ramp_t = ramp_t
                    else:
                        prev_ramp_t = 0.0
                    prev_shift = x_end + t_tool
                    prev_had_tool = (t_tool > 0)
                    shift = prev_shift
                ax.plot(x_full, y_full, color=color, linestyle=ls, linewidth=1.5, label=label)
                ax.set_title(titles[idx], fontsize=12, fontweight="bold")
            else:
                ax.plot(times, smooth, color=color, linestyle=ls, linewidth=1.5, label=label)
                ax.set_title(titles[idx], fontsize=12, fontweight="bold")
        else:
            if args.show_instant:
                ax.plot(times, vals, color=color, alpha=0.15, linewidth=0.5)
            ax.plot(times, smooth, color=color, linestyle=ls, linewidth=1.5, label=label)
            ax.set_title(titles[idx], fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("MFU (%)", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    pic_dir = os.path.join(DIR, "pic")
    os.makedirs(pic_dir, exist_ok=True)
    out = args.save if args.save else os.path.join(pic_dir, "new.pdf")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已生成: {out}")


if __name__ == "__main__":
    main()
