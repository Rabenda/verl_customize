from collections import deque
import csv
import json
import re
import argparse

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")  # 无界面环境
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ==========================================
# 1. 硬件配置字典 (以 BF16/FP16 稠密算力为准)
# ==========================================
HARDWARE_CONFIGS = {
    # H100 SXM: 稠密 BF16/FP16 峰值算力约为 989 TFLOPS, 显存带宽 3.35 TB/s
    "H100_SXM": {"peak_tflops": 989.0, "mem_bandwidth_tb": 3.35},
    # H100 PCIe: 稠密 BF16/FP16 峰值算力约为 756 TFLOPS, 显存带宽 2.0 TB/s
    "H100_PCIe": {"peak_tflops": 756.0, "mem_bandwidth_tb": 2.0},
}

# ==========================================
# 2. 模型配置字典 
# (注: Qwen 0.6B 对应官方 0.5B 架构, Qwen 8B 对应官方 7B/8B 级架构)
# ==========================================
MODEL_CONFIGS = {
    "Qwen-0.6B":    {"params_b": 0.46, "num_layers": 24, "hidden_size": 1024}, # 参考 Qwen1.5-0.5B
    "Qwen-4B":      {"params_b": 3.96, "num_layers": 40, "hidden_size": 2560}, # 参考 Qwen1.5-4B
    "Llama-3.1-8B": {"params_b": 8.03, "num_layers": 32, "hidden_size": 4096}, # 官方 Llama-3.1-8B
    "Qwen-8B":      {"params_b": 7.61, "num_layers": 28, "hidden_size": 3584}, # 参考 Qwen2.5-7B
}


# ==========================================
# 3. 核心计算类
# ==========================================
class MFUMonitor:
    def __init__(self, hw_name, model_name, window_size=10):
        # 从字典加载配置
        if hw_name not in HARDWARE_CONFIGS:
            raise ValueError(f"不支持的硬件: {hw_name}")
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"不支持的模型: {model_name}")
            
        self.hw = HARDWARE_CONFIGS[hw_name]
        self.model = MODEL_CONFIGS[model_name]
        
        self.history = deque(maxlen=window_size)
        
        print(f"✅ 成功初始化 MFU 监控: [硬件] {hw_name} | [模型] {model_name}")
        print(f"   峰值算力: {self.hw['peak_tflops']} TFLOPS | 参数量: {self.model['params_b']}B")
        print("-" * 50)
    
    def calculate_step_mfu(self, prefill_tokens, decoding_batch_size, avg_seq_length, delta_t):
        N = self.model["params_b"] * 1e9
        L = self.model["num_layers"]
        h = self.model["hidden_size"]
        
        # 1. 权重 FLOPs (2 * N * Tokens)
        flops_weights = 2.0 * N * (prefill_tokens + decoding_batch_size)
        
        # 2. Attention FLOPs (4 * L * h * 序列长度)
        flops_attn_decode = 4.0 * L * h * (decoding_batch_size * avg_seq_length)
        flops_attn_prefill = 4.0 * L * h * (prefill_tokens ** 2) 
        
        flops_total = flops_weights + flops_attn_decode + flops_attn_prefill
        
        # 3. 计算 MFU
        theoretical_peak_flops = self.hw["peak_tflops"] * 1e12 * delta_t
        instant_mfu = flops_total / theoretical_peak_flops if theoretical_peak_flops > 0 else 0
        
        self.history.append((flops_total, delta_t))
        return flops_total, instant_mfu
    
    def get_smoothed_mfu(self):
        if not self.history:
            return 0.0
        
        total_flops = sum(item[0] for item in self.history)
        total_time = sum(item[1] for item in self.history)
        
        if total_time == 0:
            return 0.0
            
        smoothed_peak = self.hw["peak_tflops"] * 1e12 * total_time
        return total_flops / smoothed_peak


# ==========================================
# 4. 从 verl 日志解析 rollout_mfu 数据
# 日志格式: [rollout_mfu] gstep=N {"prefill_tokens": ..., "decoding_batch_size": ..., "avg_seq_length": ..., "delta_t": ...}
# ==========================================
def parse_rollout_mfu_from_log(log_path: str) -> list[dict]:
    """从 verl 训练日志中解析 rollout_mfu 记录，返回与 mock_metrics_stream 格式一致的列表。"""
    metrics_stream = []
    # 匹配 [rollout_mfu] 后的 JSON 对象（支持 Ray 前缀等噪音）
    pattern = re.compile(r'\[rollout_mfu\].*?(\{[^}]+\})')
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                try:
                    d = json.loads(m.group(1))
                    # 只保留 MFU 计算需要的 4 个字段
                    metrics_stream.append({
                        "prefill_tokens": float(d.get("prefill_tokens", 0)),
                        "decoding_batch_size": int(d.get("decoding_batch_size", 0)),
                        "avg_seq_length": float(d.get("avg_seq_length", 0)),
                        "delta_t": float(d.get("delta_t", 0)),
                    })
                except (json.JSONDecodeError, ValueError):
                    continue
    return metrics_stream


# ==========================================
# 4b. 从 inference step CSV 解析 rollout_mfu 数据
# CSV 列: pass_id, mode, batch_size, prefill_tokens, decode_tokens, avg_seq_len, forward_time_ms
# ==========================================
def parse_rollout_mfu_from_csv(csv_path: str, skip_pass_id_one: bool = True) -> list[dict]:
    """从 inference step CSV 中解析记录，返回与 mock_metrics_stream 格式一致的列表。"""
    metrics_stream = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if skip_pass_id_one and row.get("pass_id") == "1":
                continue
            metrics_stream.append({
                "prefill_tokens": float(row["prefill_tokens"]),
                "decoding_batch_size": int(row["batch_size"]),
                "avg_seq_length": float(row["avg_seq_len"]),
                "delta_t": float(row["forward_time_ms"]) / 1000.0,
            })
    return metrics_stream


# ==========================================
# 5. 画图：横坐标时间，纵坐标 MFU
# ==========================================
def plot_mfu_vs_time(
    times: list[float],
    instant_mfu: list[float],
    smoothed_mfu: list[float] | None = None,
    save_path: str | None = None,
):
    """绘制 MFU vs 时间曲线。times 为累计时间(秒)，MFU 为 0~1 的小数。"""
    if not HAS_MATPLOTLIB:
        print("⚠ matplotlib 未安装，跳过画图。可运行: pip install matplotlib")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, [m * 100 for m in instant_mfu], label="Instant MFU", alpha=0.7, linewidth=0.8)
    if smoothed_mfu:
        ax.plot(times, [m * 100 for m in smoothed_mfu], label="Smoothed MFU", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFU (%)")
    ax.set_title("Rollout MFU vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = save_path or "rollout_mfu_plot.png"
    plt.savefig(path, dpi=150)
    print(f"📈 图已保存: {path}")
    plt.close()


# ==========================================
# 6. 主程序：可从日志读入或使用 mock 数据
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 verl 日志计算 rollout MFU，或使用 mock 数据")
    parser.add_argument(
        "-c", "--csv",
        default=None,
        help="inference step 的 CSV 路径（如 inference_step_log_a.csv），与 -l 二选一，优先 CSV",
    )
    parser.add_argument(
        "-l", "--log",
        default=None,
        help="verl 训练日志路径（如 test0223.log），不填则使用 mock 数据",
    )
    parser.add_argument("-hw", "--hardware", default="H100_SXM", help="硬件配置名")
    parser.add_argument("-m", "--model", default="Qwen-0.6B", help="模型配置名")
    parser.add_argument("-w", "--window", type=int, default=10, help="平滑窗口大小")
    parser.add_argument("--no-print-steps", action="store_true", help="不逐条打印每个 step")
    parser.add_argument("--plot", action="store_true", help="绘制 MFU vs 时间曲线")
    parser.add_argument("--save", type=str, default=None, metavar="PATH", help="将图保存到文件（需配合 --plot）")
    parser.add_argument("--decode-only", action="store_true", help="排除 prefill 记录（仅保留 prefill_tokens==0 的 decode 窗口）")
    args = parser.parse_args()

    monitor = MFUMonitor(hw_name=args.hardware, model_name=args.model, window_size=args.window)

    if args.csv:
        metrics_stream = parse_rollout_mfu_from_csv(args.csv)
        n_before = len(metrics_stream)
        if args.decode_only:
            metrics_stream = [m for m in metrics_stream if m["prefill_tokens"] == 0]
            print(f"📂 从 {args.csv} 解析到 {n_before} 条，排除 prefill 后保留 {len(metrics_stream)} 条 decode 记录\n")
        else:
            print(f"📂 从 {args.csv} 解析到 {len(metrics_stream)} 条记录\n")
    elif args.log:
        metrics_stream = parse_rollout_mfu_from_log(args.log)
        n_before = len(metrics_stream)
        if args.decode_only:
            metrics_stream = [m for m in metrics_stream if m["prefill_tokens"] == 0]
            print(f"📂 从 {args.log} 解析到 {n_before} 条，排除 prefill 后保留 {len(metrics_stream)} 条 decode 记录\n")
        else:
            print(f"📂 从 {args.log} 解析到 {len(metrics_stream)} 条 rollout_mfu 记录\n")
    else:
        metrics_stream = [
            {"prefill_tokens": 0, "decoding_batch_size": 64, "avg_seq_length": 1024, "delta_t": 0.012},
            {"prefill_tokens": 512, "decoding_batch_size": 64, "avg_seq_length": 1024, "delta_t": 0.015},
        ]
        print("📋 使用 mock 数据\n")

    total_flops = 0.0
    total_time = 0.0
    plot_times, plot_instant, plot_smoothed = [], [], []
    for i, metrics in enumerate(metrics_stream):
        flops, instant_mfu = monitor.calculate_step_mfu(**metrics)
        smoothed_mfu = monitor.get_smoothed_mfu()
        total_flops += flops
        t_start = total_time
        total_time += metrics["delta_t"]
        t_end = total_time
        if args.plot:
            if metrics["prefill_tokens"] > 0 and metrics["delta_t"] > 0.01:
                # prefill：用水平线段表示时长，而非单点
                n_pts = max(2, min(50, int(metrics["delta_t"] / 0.5) + 1))
                for j in range(n_pts):
                    frac = j / (n_pts - 1) if n_pts > 1 else 1.0
                    plot_times.append(t_start + (t_end - t_start) * frac)
                    plot_instant.append(instant_mfu)
                    plot_smoothed.append(smoothed_mfu)
            else:
                # decode：单点（delta_t 很短）
                plot_times.append(t_end)
                plot_instant.append(instant_mfu)
                plot_smoothed.append(smoothed_mfu)
        if not args.no_print_steps:
            print(f"Step {i+1} | Prefill: {metrics['prefill_tokens']} | Decode BS: {metrics['decoding_batch_size']} | "
                  f"AvgSeq: {metrics['avg_seq_length']:.1f} | δt: {metrics['delta_t']:.4f}s | "
                  f"瞬时 MFU: {instant_mfu*100:.2f}% | 平滑 MFU: {smoothed_mfu*100:.2f}%")

    if metrics_stream:
        overall_mfu = total_flops / (monitor.hw["peak_tflops"] * 1e12 * total_time) if total_time > 0 else 0.0
        print("-" * 50)
        print(f"📊 总计: {len(metrics_stream)} steps | 总时间: {total_time:.2f}s | 整体 MFU: {overall_mfu*100:.2f}%")

    if args.plot and plot_times:
        save_path = args.save or None
        plot_mfu_vs_time(plot_times, plot_instant, plot_smoothed, save_path)