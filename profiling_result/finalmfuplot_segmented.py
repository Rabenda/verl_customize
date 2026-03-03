import csv
import json
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
    "Qwen-0.6B":    {"params_b": 0.46, "num_layers": 24, "hidden_size": 1024},
    "Qwen-4B":      {"params_b": 3.96, "num_layers": 40, "hidden_size": 2560},
    "Llama-3.1-8B": {"params_b": 8.03, "num_layers": 32, "hidden_size": 4096},
    "Qwen-8B":      {"params_b": 7.61, "num_layers": 28, "hidden_size": 3584},
}

class MFUMonitor:
    def __init__(self, hw_name, model_name):
        self.hw = HARDWARE_CONFIGS[hw_name]
        self.model = MODEL_CONFIGS[model_name]
    
    def calculate_step_mfu(self, prefill_tokens, decoding_batch_size, avg_seq_length, delta_t):
        N, L, h = self.model["params_b"] * 1e9, self.model["num_layers"], self.model["hidden_size"]
        flops_total = (2.0 * N * (prefill_tokens + decoding_batch_size) + 
                       4.0 * L * h * (decoding_batch_size * avg_seq_length) + 
                       4.0 * L * h * (prefill_tokens ** 2))
        theoretical_peak = self.hw["peak_tflops"] * 1e12 * delta_t
        return flops_total / theoretical_peak if theoretical_peak > 0 else 0

# ==========================================
# 2. 优化后的绘图逻辑 (解决 0 位悬浮问题)
# ==========================================
def plot_mfu_fixed(times, instant_mfu, save_path=None, env_window=30, sigma=5):
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
    plt.plot(times, vals, color='steelblue', alpha=0.15, linewidth=0.5, label="Instant MFU (Raw)")
    
    # 2. 修正后的红线
    plt.plot(times, smooth_envelope, color='#e74c3c', linewidth=2.5, label="Peak Trend (Fixed)")
    
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("MFU (%)", fontsize=12)
    plt.title("Rollout MFU Analysis (Fixed Zero-Gaps)", fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.ylim(bottom=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    path = save_path or "rollout_mfu_fixed.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"✅ 修正版图表已生成: {path}")
    plt.close()

# ==========================================
# 3. 数据解析 & 主程序
# ==========================================
def parse_data(args):
    metrics = []
    if args.csv:
        with open(args.csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
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
                    continue  # 跳过列错位或非法行（如 forward_time_ms 读到 "EXTEND"）
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
    parser.add_argument("-m", "--model", default="Qwen-0.6B")
    parser.add_argument("--env-window", type=int, default=40)
    parser.add_argument("--sigma", type=int, default=5)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    monitor = MFUMonitor(args.hardware, args.model)
    data = parse_data(args)

    if data:
        times, mfus = [], []
        curr_time = 0
        for d in data:
            curr_time += d["delta_t"]
            times.append(curr_time)
            mfus.append(monitor.calculate_step_mfu(**d))
        
        plot_mfu_fixed(times, mfus, save_path=args.save, env_window=args.env_window, sigma=args.sigma)