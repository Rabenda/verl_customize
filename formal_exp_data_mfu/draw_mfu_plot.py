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


def plot_mfu_fixed(times, instant_mfu, save_path=None, env_window=30, sigma=5, first_step_only=False, dataset=None, model=None, turn_type=None):
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
    
    path = save_path or "rollout_mfu_fixed.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"✅ 修正版图表已生成: {path}")
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


# ==========================================
# 4. 数据解析 & 主程序
# ==========================================
def parse_data(args):
    metrics = []
    if args.csv:
        with open(args.csv, "r") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
        if not all_rows:
            return metrics
        # 若只画第一个 global step，先根据边界或 global_step 列截取
        if getattr(args, "first_step_only", False):
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
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv", default=None)
    parser.add_argument("-l", "--log", default=None)
    parser.add_argument("-hw", "--hardware", default="H100_SXM")
    parser.add_argument("-m", "--model", default="Qwen3-0.6B")
    parser.add_argument("--env-window", type=int, default=40)
    parser.add_argument("--sigma", type=int, default=5)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--first-step-only", action="store_true", help="只使用第 1 个 global step（训练步）的数据绘图")
    parser.add_argument("--min-prefill-tokens", type=int, default=5000, help="判定新 global step 的 EXTEND prefill 下限（默认 5000，避免 step 内 chunk 被误判）")
    args = parser.parse_args()

    # 若提供了 CSV 且未指定模型，从 CSV 路径推断模型
    if args.csv and args.model == "Qwen3-0.6B":
        inferred = _infer_model_from_csv_path(args.csv)
        if inferred:
            args.model = inferred

    monitor = MFUMonitor(args.hardware, args.model)
    data = parse_data(args)

    if data:
        dataset, turn_type = _infer_dataset_and_turn(args.csv)
        if not args.save:
            args.save = _default_save_path(dataset, turn_type, args.model)
        times, mfus = [], []
        curr_time = 0
        for d in data:
            curr_time += d["delta_t"]
            times.append(curr_time)
            mfus.append(monitor.calculate_step_mfu(**d))
        plot_mfu_fixed(
            times, mfus,
            save_path=args.save,
            env_window=args.env_window,
            sigma=args.sigma,
            first_step_only=getattr(args, "first_step_only", False),
            dataset=dataset,
            model=args.model,
            turn_type=turn_type,
        )