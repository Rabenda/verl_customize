import matplotlib.pyplot as plt
import re
import numpy as np

def plot_dual_axis_with_steps(log_path):
    # 数据存储
    time_util = []
    util_values = []
    time_tokens = []
    token_values = []
    step_boundaries = [] # 存储 Step 切换点
    
    # 正则表达式
    pattern_util = re.compile(r"step:(\d+)\s+-\s+profiler/logical_utilization_pct:([\d.]+)")
    pattern_tokens = re.compile(r"step:(\d+)\s+-\s+profiler/total_active_tokens:([\d.]+)")
    pattern_step = re.compile(r"training/global_step:(\d+)")

    current_last_time = 0
    with open(log_path, 'r') as f:
        for line in f:
            # 1. 匹配利用率
            m_u = pattern_util.search(line)
            if m_u:
                t = int(m_u.group(1)) / 1000.0
                time_util.append(t)
                util_values.append(float(m_u.group(2)))
                current_last_time = t
            
            # 2. 匹配 Token 数
            m_t = pattern_tokens.search(line)
            if m_t:
                t = int(m_t.group(1)) / 1000.0
                time_tokens.append(t)
                token_values.append(float(m_t.group(2)))
                current_last_time = t

            # 3. 匹配 Step 结束标志
            m_s = pattern_step.search(line)
            if m_s:
                step_num = m_s.group(1)
                # 记录该 Step 结束时的最近时间点
                step_boundaries.append((current_last_time, step_num))

    if not time_util or not time_tokens:
        print("未发现匹配数据，请检查日志格式。")
        return

    # 时间对齐：以第一个数据点为 0 秒
    start_t = min(time_util[0], time_tokens[0])
    time_util = np.array(time_util) - start_t
    time_tokens = np.array(time_tokens) - start_t

    # --- 开始绘图 ---
    fig, ax1 = plt.subplots(figsize=(16, 8), dpi=150)

    # 左轴：逻辑利用率 (蓝色实线)
    color_util = '#1f77b4'
    ax1.set_xlabel('Time (Seconds)', fontsize=12)
    ax1.set_ylabel('Logical Utilization (%)', color=color_util, fontsize=12)
    ax1.plot(time_util, util_values, color=color_util, label='Batch Utilization', drawstyle='steps-post', alpha=0.8)
    ax1.fill_between(time_util, util_values, color=color_util, alpha=0.1)
    ax1.tick_params(axis='y', labelcolor=color_util)
    ax1.set_ylim(-5, 120)

    # 右轴：总活跃 Token 数 (橙色虚线)
    ax2 = ax1.twinx() 
    color_tokens = '#ff7f0e'
    ax2.set_ylabel('Total Active Tokens', color=color_tokens, fontsize=12)
    ax2.plot(time_tokens, token_values, color=color_tokens, label='Active Tokens', drawstyle='steps-post', linestyle='--', alpha=0.9)
    ax2.tick_params(axis='y', labelcolor=color_tokens)

    # --- 关键：绘制红色 Step 分界线 ---
    for b_time, b_num in step_boundaries:
        adj_time = b_time - start_t
        ax1.axvline(x=adj_time, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
        # 标注 Step 编号
        ax1.text(adj_time, 115, f'Step {b_num}', color='red', fontsize=10, ha='right', fontweight='bold')

    # 装饰
    plt.title('Workload Analysis: Batch vs. Token Decay (with Step Divisions)', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.4)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.tight_layout()
    plt.savefig('workload_dual_with_steps.png')
    print("分析图已保存为 workload_dual_with_steps.png")

# 执行
plot_dual_axis_with_steps("/workspace/repo/verl/aime_qwen3-0.6b_grpo_rollout4.log")