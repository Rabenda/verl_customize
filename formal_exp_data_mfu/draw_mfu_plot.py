import csv
import json
import os
import re
import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    from scipy.ndimage import gaussian_filter1d
    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ==========================================
# 1. 硬件 & 模型配置 (保持不变)
# ==========================================
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
}

class MFUMonitor:
    def __init__(self, hw_name, model_name):
        self.hw = HARDWARE_CONFIGS[hw_name]
        self.model = MODEL_CONFIGS[model_name]

    def calculate_step_mfu(self, prefill_tokens, decoding_batch_size, avg_seq_length, delta_t, tp_size=4):
        N, L, h = self.model["params_b"] * 1e9, self.model["num_layers"], self.model["hidden_size"]

        # 1. 线性层 FLOPs
        total_tokens = prefill_tokens + decoding_batch_size
        flops_linear = 2.0 * N * total_tokens

        # 2. 注意力层 FLOPs
        flops_attn = 2.0 * L * h * total_tokens * avg_seq_length

        flops_total = flops_linear + flops_attn

        # 如果你启用了 FP8，理论上还要再乘以 2
        theoretical_peak = (self.hw["peak_tflops"] * 1e12) * delta_t * tp_size

        return flops_total / theoretical_peak if theoretical_peak > 0 else 0

# ==========================================
# 2. 优化后的绘图逻辑 (解决 0 位悬浮问题)
# ==========================================
def _infer_dataset_and_turn(csv_path):
    """从 CSV 路径推断数据集名和 single/multi_turn。"""
    if not csv_path:
        return "unknown", "single"
    path_lower = csv_path.lower()
    turn_type = "multi_turn" if "multiturn" in path_lower else "single"
    if "gsm8k" in path_lower:
        dataset = "gsm8k"
    elif "math" in path_lower:
        dataset = "math"
    else:
        dataset = "unknown"
    return dataset, turn_type


def _infer_model_from_csv_path(csv_path):
    """从 CSV 路径/文件名推断模型名（与 MODEL_CONFIGS 的 key 一致）。instruct 优先于 4b 单独识别。"""
    if not csv_path:
        return None
    path_lower = os.path.basename(csv_path).lower()
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
    # 旧文件名兼容：qwen-* 映射到现有 Qwen3-* key
    if "qwen-0.6b" in path_lower:
        return "Qwen3-0.6B"
    if "qwen-4b" in path_lower:
        return "Qwen3-4B"
    if "qwen-8b" in path_lower:
        return "Qwen3-8B"
    return None


def _default_save_path(dataset, turn_type, model):
    """生成默认保存名：plot_mfu_single/multiturn_数据集_模型.png"""
    turn = "multiturn" if turn_type == "multi_turn" else "single"
    model_safe = (model or "unknown").replace(" ", "_")
    return f"plot_mfu_{turn}_{dataset}_{model_safe}.png"


def plot_mfu_fixed(
    times,
    instant_mfu,
    save_path=None,
    env_window=30,
    sigma=5,
    first_step_only=False,
    dataset=None,
    model=None,
    turn_type=None,
    *,
    turn_boundaries=None,
    tool_calls_sec=None,
    turns_count=None,
):
    if not HAS_MATPLOTLIB: return
    
    vals = np.array(instant_mfu) * 100
    times = np.array(times)
    
    # 步骤 A: 提取波峰，但增加“全零检测”
    peak_envelope = np.zeros_like(vals)
    for i in range(len(vals)):
        # 使用因果窗口 (只看当前和过去)，防止波峰提前出现
        start = max(0, i - env_window)
        window_data = vals[start : i + 1]
        
        # 只有当窗口内最大值超过极小阈值时才记录，否则强制为 0
        m = np.max(window_data)
        peak_envelope[i] = m if m > 0.01 else 0.0

    # 步骤 B: 丝滑化平滑
    smooth_envelope = gaussian_filter1d(peak_envelope, sigma=sigma)
    
    # 步骤 C: 最终校准 - 如果原始数据在某段时间内全是 0，强制抹平平滑后的余温
    # 解决高斯平滑带来的“尾迹”
    for i in range(len(vals)):
        if peak_envelope[i] == 0:
            smooth_envelope[i] = 0

    plt.figure(figsize=(15, 7), dpi=150)
    
    # 1. 原始数据 (极浅蓝色)
    plt.plot(times, vals, color='steelblue', alpha=0.15, linewidth=0.5, label="Instant MFU")
    
    # 2. 修正后的红线
    plt.plot(times, smooth_envelope, color='#e74c3c', linewidth=2.5, label="Peak Trend")
    
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("MFU (%)", fontsize=12)
    ds = dataset or "unknown"
    md = model or "unknown"
    tt = turn_type or "single"
    title = f"Rollout MFU — {ds} | {md} | {tt}"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.ylim(bottom=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # ---- optional overlays ----
    # If tool_calls_sec is provided, show a small fixed-width "tool time" block right after MFU curve ends.
    # Note: inference_step_log only contains model forward; tool time has no MFU samples and is NOT to scale.
    if tool_calls_sec is not None and len(times) > 0:
        try:
            tc = float(tool_calls_sec)
            if tc > 0:
                ax = plt.gca()
                t_end = float(times[-1])
                # Add a small tail window so the block is visible (default: +15% of MFU x-range, at least 0.8s).
                tail_w = max(0.8, 0.15 * max(t_end, 1.0))
                ax.set_xlim(0, t_end + tail_w)

                # Shade the tail block.
                ax.axvspan(t_end, t_end + tail_w, color="#7f8c8d", alpha=0.12, linewidth=0, zorder=0.5)

                # Axis-break marker at the boundary.
                ax.text(
                    t_end,
                    ax.get_ylim()[1] * 0.035,
                    "//",
                    fontsize=14,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    color="#2c3e50",
                    alpha=0.8,
                )

                # Label inside the block (near bottom center).
                ax.text(
                    t_end + tail_w * 0.5,
                    ax.get_ylim()[1] * 0.06,
                    f"tool_calls={tc:.2f}s (not to scale)",
                    fontsize=10,
                    ha="center",
                    va="bottom",
                    color="#2c3e50",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2.5),
                )
        except Exception:
            pass

    if tool_calls_sec is not None or turns_count is not None:
        pieces = []
        if turns_count is not None:
            pieces.append(f"turns: {int(turns_count)}")
        if tool_calls_sec is not None:
            pieces.append(f"tool_calls: {float(tool_calls_sec):.2f}s")
        if pieces:
            plt.text(
                0.01,
                0.02,
                " | ".join(pieces),
                transform=plt.gca().transAxes,
                fontsize=10,
                color="#2c3e50",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3),
            )

    if isinstance(turn_boundaries, list) and len(turn_boundaries) > 0:
        ymax = plt.ylim()[1]
        for b in turn_boundaries:
            try:
                t = float(b.get("t"))
            except Exception:
                continue
            label = str(b.get("label", "")).strip()
            plt.axvline(t, color="#34495e", linestyle="--", linewidth=1.2, alpha=0.85)
            if label:
                plt.text(
                    t,
                    ymax * 0.98,
                    label,
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=9,
                    color="#34495e",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
                )

        # Background partition for Turn1/Turn2/... (forward-time axis only)
        try:
            # Assume boundaries are sorted by time and represent Turn k start for k>=2.
            t_end_forward = float(times[-1]) if len(times) > 0 else None
            if t_end_forward is not None and t_end_forward > 0:
                # segments: [0, t2), [t2, t3), ..., [t_last, t_end_forward]
                starts = [0.0] + [float(b.get("t")) for b in turn_boundaries if b.get("t") is not None]
                starts = sorted([s for s in starts if s is not None])
                # remove any >= end
                starts = [s for s in starts if s < t_end_forward]
                # Stronger but still subtle background colors for turn partitions
                colors = ["#a9cce3", "#a9dfbf", "#f5cba7", "#d7bde2"]  # blue/green/orange/purple
                for i, s in enumerate(starts):
                    e = starts[i + 1] if i + 1 < len(starts) else t_end_forward
                    if e <= s:
                        continue
                    plt.axvspan(s, e, color=colors[i % len(colors)], alpha=0.22, linewidth=0, zorder=0.1)
        except Exception:
            pass
    
    path = save_path or "rollout_mfu_fixed.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"✅ 修正版图表已生成: {path}")
    plt.close()


def _read_tool_times_csv(path: str, *, step: int | None = None) -> dict[int, float]:
    """读取 tool_times_{exp}.csv，返回指定 step 的 {turn: time_s}（step=None 时取最小 step）。"""
    if not path:
        return {}
    try:
        with open(path, "r") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}
    if not rows:
        return {}
    if step is None:
        try:
            step = min(int(r.get("step", 0) or 0) for r in rows if str(r.get("step", "")).strip())
        except Exception:
            step = None
    out: dict[int, float] = {}
    for r in rows:
        try:
            if step is not None and int(r.get("step", 0) or 0) != int(step):
                continue
            turn = int(r.get("turn", 0) or 0)
            t = float(r.get("time_s", 0) or 0)
            if turn > 0:
                out[turn] = t
        except Exception:
            continue
    return out


def plot_mfu_segmented_with_tool_blocks(
    times,
    instant_mfu,
    *,
    boundary_metric_idxs=None,
    tool_times_by_turn=None,
    save_path=None,
    sigma=5,
    dataset=None,
    model=None,
    turn_type=None,
):
    """按 turn 切分 MFU 曲线，并插入“按真实 wall time 缩放”的 tool block（灰块）。

    注意：横轴将变成近似墙钟轴 = forward_time(模型侧) + tool_wall(检索侧)。
    """
    if not HAS_MATPLOTLIB:
        return
    times = np.asarray(times, dtype=float)
    vals = np.asarray(instant_mfu, dtype=float) * 100.0
    if len(times) == 0:
        return

    # 复用原逻辑的平滑（不再做 peak envelope，保持简单直观）
    try:
        y = gaussian_filter1d(vals, sigma=sigma) if sigma and sigma > 0 else vals
    except Exception:
        y = vals

    # build split points in metric index space
    splits = [0]
    if isinstance(boundary_metric_idxs, list):
        for x in boundary_metric_idxs:
            try:
                xi = int(x)
                if 0 < xi < len(times):
                    splits.append(xi)
            except Exception:
                pass
    splits = sorted(set(splits))
    splits.append(len(times))

    tool_times_by_turn = tool_times_by_turn or {}
    bg_colors = ["#a9cce3", "#a9dfbf", "#f5cba7", "#d7bde2"]

    plt.figure(figsize=(15, 7), dpi=150)
    ax = plt.gca()
    shift = 0.0

    for seg_i in range(len(splits) - 1):
        s = splits[seg_i]
        e = splits[seg_i + 1]
        if e <= s:
            continue
        turn_id = seg_i + 1

        # normalize this segment to start at 0, then shift by inserted tool time before it
        x0 = float(times[s])
        x_seg = (times[s:e] - x0) + shift
        y_seg = y[s:e]

        # forward segment background
        ax.axvspan(float(x_seg[0]), float(x_seg[-1]), color=bg_colors[(turn_id - 1) % len(bg_colors)], alpha=0.18, linewidth=0, zorder=0.1)

        # plot MFU segment (red)
        ax.plot(x_seg, y_seg, color="#e74c3c", linewidth=2.5, label="MFU" if seg_i == 0 else None)

        # tool block after this turn
        t_tool = float(tool_times_by_turn.get(turn_id, 0.0) or 0.0)
        if t_tool > 0:
            x_end = float(x_seg[-1])
            ax.axvspan(x_end, x_end + t_tool, color="#7f8c8d", alpha=0.35, linewidth=0, zorder=0.2)
            ax.text(
                x_end + t_tool / 2.0,
                max(5.0, float(np.max(y)) * 0.06),
                f"tool t{turn_id}={t_tool:.2f}s",
                fontsize=10,
                ha="center",
                va="bottom",
                color="#2c3e50",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2.0),
            )
            shift += t_tool

    ds = dataset or "unknown"
    md = model or "unknown"
    tt = turn_type or "single"
    plt.title(f"Rollout MFU — {ds} | {md} | {tt} (segmented + tool blocks)", fontsize=14, fontweight="bold")
    plt.xlabel("Approx wall time (forward_time + tool_wall) [s]", fontsize=12)
    plt.ylabel("MFU (%)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ax.get_legend_handles_labels()[1]:
        plt.legend(loc="upper right")

    path = save_path or "rollout_mfu_segmented_tool.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"✅ 分段+tool block 图表已生成: {path}")
    plt.close()

# ==========================================
# 3. Global step 边界检测（一个 global step = rollout 对应的一段，图上一个尖峰+回落）
# ==========================================
#
# 以下 CSV 均已用「一份通用逻辑」验证，参数一致，故只保留一份备份，其余在注释说明即可。
# 已验证 CSV：inference_step_log_gsm8k_qwen3-8b.csv, inference_step_log_gsm8k_qwen3-4b.csv,
#             inference_step_log_math_qwen3-8b.csv, inference_step_log_math_qwen3-8b_2step.csv,
#             inference_step_log_math_qwen3-4b.csv
# 若将来某 CSV 用此逻辑不准，可据此备份新增专用函数并在 find_global_step_boundaries 里分支。


def find_global_step_boundaries_reference(rows):
    """【备份】通用边界逻辑的一份固定实现，参数写死，便于以后还原所有已验证 CSV 的精准图。
    不要改此函数。"""
    return _find_global_step_boundaries_impl(
        rows,
        min_prefill_tokens=5000,
        tail_decode_max_batch=5,
        lookback=20,
    )


def find_global_step_boundaries_generic(
    rows, min_prefill_tokens=5000, tail_decode_max_batch=5, lookback=20
):
    """通用边界逻辑，参数可调。若某 CSV 用此逻辑不准，可为该 CSV 新增专用函数并走 dispatch。"""
    return _find_global_step_boundaries_impl(
        rows,
        min_prefill_tokens=min_prefill_tokens,
        tail_decode_max_batch=tail_decode_max_batch,
        lookback=lookback,
    )


def _find_global_step_boundaries_impl(
    rows, min_prefill_tokens=5000, tail_decode_max_batch=5, lookback=20
):
    """内部实现：新 step 开始 = 近期有小 batch DECODE 之后出现的大 prefill EXTEND。"""
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


def find_global_step_boundaries(rows, csv_path=None, **kwargs):
    """有 csv_path 时用一份备份逻辑（reference），否则用通用逻辑（可传参）。"""
    if csv_path:
        return find_global_step_boundaries_reference(rows)
    return find_global_step_boundaries_generic(rows, **kwargs)

def find_turn_boundaries_generic(
    rows,
    *,
    min_prefill_tokens=2000,
    tail_decode_max_batch=5,
    lookback=80,
    max_turns=2,
):
    """启发式 turn 边界检测（同一个 global_step 内的多轮 assistant 生成分段）。

    逻辑：在出现“尾部 decode 小 batch”（活跃序列很少）之后，若再次出现较大的 EXTEND prefill，
    视为下一轮 assistant generation 的开始。
    """
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

def _parse_search_r1_turns_and_tool_time_from_log(log_path: str):
    """Parse Search-R1 sync summary from log.

    Returns:
        (turns_count, tool_calls_sec) where tool_calls_sec is retrieval/tool_calls total wall time.
    """
    if not log_path:
        return None, None
    turns_count = None
    tool_calls_sec = None
    try:
        # Example:
        # [Search-R1 Sync] Done: 2 turns, gen=376.844s, retrieval=352.617s, wall=377.213s
        pat_done = re.compile(
            r"\[Search-R1 Sync\]\s*Done:\s*(\d+)\s*turns,.*?retrieval=([0-9]+(?:\.[0-9]+)?)s"
        )
        # Fallback: agent_loop/tool_calls/mean: 352.62s
        pat_tool_mean = re.compile(r"agent_loop/tool_calls/mean:\s*([0-9]+(?:\.[0-9]+)?)s")
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                if turns_count is None or tool_calls_sec is None:
                    m = pat_done.search(line)
                    if m:
                        try:
                            turns_count = int(m.group(1))
                        except Exception:
                            turns_count = turns_count
                        try:
                            tool_calls_sec = float(m.group(2))
                        except Exception:
                            tool_calls_sec = tool_calls_sec
                        if turns_count is not None and tool_calls_sec is not None:
                            break
                if tool_calls_sec is None:
                    m2 = pat_tool_mean.search(line)
                    if m2:
                        try:
                            tool_calls_sec = float(m2.group(1))
                        except Exception:
                            pass
    except Exception:
        return None, None
    return turns_count, tool_calls_sec


# ==========================================
# 4. 数据解析 & 主程序
# ==========================================
def parse_data(args):
    metrics = []
    rows_to_use = None
    if args.csv:
        with open(args.csv, "r") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
        if not all_rows:
            return metrics, []
        # 可选：只保留某个 global_step（与 inference_step_log 的 global_step 列对齐）
        if getattr(args, "global_step", None) is not None:
            gs = str(getattr(args, "global_step"))
            rows_to_use = [r for r in all_rows if str(r.get("global_step", "")).strip() == gs]
            print(f"[global_step] 按 global_step=={gs} 筛选: 共 {len(rows_to_use)} 行")
        # 若只画第一个 global step，先根据边界或 global_step 列截取
        if rows_to_use is None and getattr(args, "first_step_only", False):
            fieldnames = list(all_rows[0].keys()) if all_rows else []
            if "global_step" in fieldnames:
                # 有 global_step 列时只保留 step 1（如 multiturn 的 qwen3-4b 有多个 step，只取第一个 step1）
                rows_to_use = [r for r in all_rows if r.get("global_step") == "1"]
                print(f"[first_step_only] 按 global_step==1 筛选: 共 {len(rows_to_use)} 行")
            else:
                min_prefill = getattr(args, "min_prefill_tokens", 5000)
                boundaries = find_global_step_boundaries(
                    all_rows,
                    csv_path=args.csv,
                    min_prefill_tokens=min_prefill,
                )
                end = boundaries[1] if len(boundaries) > 1 else len(all_rows)
                rows_to_use = all_rows[:end]
                print(f"[first_step_only] 使用第 1 个 global step: 行 0 到 {end - 1} (共 {end} 行), 检测到 {len(boundaries)} 个 global step 边界")
        elif rows_to_use is None:
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
    elif args.log:
        pattern = re.compile(r'\[rollout_mfu\].*?(\{[^}]+\})')
        with open(args.log, "r", errors="ignore") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    d = json.loads(m.group(1))
                    metrics.append({
                        "prefill_tokens": float(d.get("prefill_tokens", 0)),
                        "decoding_batch_size": int(d.get("decoding_batch_size", 0)),
                        "avg_seq_length": float(d.get("avg_seq_length", 0)),
                        "delta_t": float(d.get("delta_t", 0)),
                    })
    return metrics, (rows_to_use or [])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv", default=None)
    parser.add_argument("-l", "--log", default=None)
    parser.add_argument(
        "--profile-log",
        default=None,
        help="Optional training log path to annotate turns/tool_calls time (e.g., formal_log/formal_search_r1_*.log).",
    )
    parser.add_argument("-hw", "--hardware", default="H100_SXM")
    parser.add_argument("-m", "--model", default="Qwen3-0.6B")
    parser.add_argument("--env-window", type=int, default=40)
    parser.add_argument("--sigma", type=int, default=5)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--first-step-only", action="store_true", help="只使用第 1 个 global step（训练步）的数据绘图")
    parser.add_argument("--min-prefill-tokens", type=int, default=5000, help="判定新 global step 的 EXTEND prefill 下限（默认 5000，避免 step 内 chunk 被误判）")
    parser.add_argument("--annotate-turns", action="store_true", help="启发式推断 turn 分界并在图中标注（仅 CSV 模式）")
    parser.add_argument("--turn-min-prefill", type=int, default=2000, help="turn 分界判定的 EXTEND prefill 下限（默认 2000）")
    parser.add_argument("--turn-lookback", type=int, default=80, help="turn 分界 lookback（默认 80）")
    parser.add_argument("--turn-tail-decode-max-batch", type=int, default=5, help="turn 分界 tail decode 的 batch_size 上限（默认 5）")
    parser.add_argument("--global-step", type=int, default=None, help="只使用指定的 global_step（与 inference_step_log 的 global_step 列对齐）")
    parser.add_argument("--tool-times-csv", type=str, default=None, help="tool_times_{exp}.csv：用于插入每个 turn 的 tool wall time 灰块")
    args = parser.parse_args()

    # 若提供了 CSV 且未指定模型，从 CSV 路径推断模型
    if args.csv and args.model == "Qwen3-0.6B":
        inferred = _infer_model_from_csv_path(args.csv)
        if inferred:
            args.model = inferred

    monitor = MFUMonitor(args.hardware, args.model)
    metrics, rows_used = parse_data(args)

    if metrics:
        dataset, turn_type = _infer_dataset_and_turn(args.csv)
        if not args.save:
            args.save = _default_save_path(dataset, turn_type, args.model)
        times, mfus = [], []
        curr_time = 0
        for d in metrics:
            curr_time += d["delta_t"]
            times.append(curr_time)
            mfus.append(monitor.calculate_step_mfu(**d))

        turns_count = None
        tool_calls_sec = None
        turn_boundaries = None
        if args.annotate_turns and args.csv:
            if args.profile_log:
                turns_count, tool_calls_sec = _parse_search_r1_turns_and_tool_time_from_log(args.profile_log)

            max_turns = int(turns_count or 2)
            idxs = find_turn_boundaries_generic(
                rows_used,
                min_prefill_tokens=int(args.turn_min_prefill),
                tail_decode_max_batch=int(args.turn_tail_decode_max_batch),
                lookback=int(args.turn_lookback),
                max_turns=max_turns,
            )

            if len(idxs) > 1:
                # Convert boundary indices (in rows_used space) to times on MFU axis.
                # MFU axis uses `metrics`, which excludes pass_id==1. So we should accumulate
                # forward_time_ms excluding pass_id==1 as well.
                idx_set = set(idxs[1:])
                cum = 0.0
                marks = []
                for i, r in enumerate(rows_used):
                    if str(r.get("pass_id")) == "1":
                        continue
                    try:
                        cum += float(r.get("forward_time_ms", 0) or 0) / 1000.0
                    except Exception:
                        pass
                    if i in idx_set:
                        marks.append(cum)
                if marks:
                    turn_boundaries = [{"t": t, "label": f"Turn {k} start"} for k, t in enumerate(marks, start=2)]
        # 若提供 tool_times_csv，则画“分段 + tool block”的近似墙钟轴图
        if args.tool_times_csv and args.csv:
            tool_by_turn = _read_tool_times_csv(args.tool_times_csv, step=getattr(args, "global_step", None))

            boundary_metric_idxs = []
            if args.annotate_turns:
                max_turns = int(turns_count or 2)
                idxs = find_turn_boundaries_generic(
                    rows_used,
                    min_prefill_tokens=int(args.turn_min_prefill),
                    tail_decode_max_batch=int(args.turn_tail_decode_max_batch),
                    lookback=int(args.turn_lookback),
                    max_turns=max_turns,
                )
                # 将 rows_used index 边界映射到 metrics index（metrics 跳过 pass_id==1）
                if len(idxs) > 1:
                    bset = set(idxs[1:])
                    mi = 0
                    for i, r in enumerate(rows_used):
                        if str(r.get("pass_id")) == "1":
                            continue
                        if i in bset:
                            boundary_metric_idxs.append(mi)
                        mi += 1

            plot_mfu_segmented_with_tool_blocks(
                times,
                mfus,
                boundary_metric_idxs=boundary_metric_idxs,
                tool_times_by_turn=tool_by_turn,
                save_path=args.save,
                sigma=args.sigma,
                dataset=dataset,
                model=args.model,
                turn_type=turn_type,
            )
        else:
            plot_mfu_fixed(
                times, mfus,
                save_path=args.save,
                env_window=args.env_window,
                sigma=args.sigma,
                first_step_only=getattr(args, "first_step_only", False),
                dataset=dataset,
                model=args.model,
                turn_type=turn_type,
                turn_boundaries=turn_boundaries,
                tool_calls_sec=tool_calls_sec,
                turns_count=turns_count,
            )