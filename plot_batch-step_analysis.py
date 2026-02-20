import matplotlib.pyplot as plt
import re
import numpy as np

def plot_with_step_division(log_path):
    # 存储轨迹数据
    times = []
    utils = []
    # 存储 Step 切换的时间点
    step_boundaries = []
    
    pattern_util = re.compile(r"step:(\d+)\s+-\s+profiler/logical_utilization_pct:([\d.]+)")
    pattern_step = re.compile(r"training/global_step:(\d+)")

    current_step_time = 0
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # 提取利用率点
            match_util = pattern_util.search(line)
            if match_util:
                t = int(match_util.group(1)) / 1000.0
                times.append(t)
                utils.append(float(match_util.group(2)))
                current_step_time = t # 记录当前最后的时间点
            
            # 提取 Step 结束的标志
            match_step = pattern_step.search(line)
            if match_step:
                step_num = match_step.group(1)
                # 记录该 Step 结束的时间位置
                step_boundaries.append((current_step_time, step_num))

    if not times:
        print("未找到数据。")
        return

    # 时间轴归零
    start_t = times[0]
    times = np.array(times) - start_t
    
    plt.figure(figsize=(16, 6), dpi=200)
    plt.plot(times, utils, label='Logical GPU Utilization', color='#1f77b4', drawstyle='steps-post', alpha=0.8)
    plt.fill_between(times, utils, color='#1f77b4', alpha=0.1)

    # --- 关键：绘制 Step 分界线 ---
    for b_time, b_num in step_boundaries:
        adj_time = b_time - start_t
        # 画垂直虚线
        plt.axvline(x=adj_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
        # 在线旁边标注 Step 编号
        plt.text(adj_time, 105, f'Step {b_num}', color='red', fontsize=10, ha='right', rotation=45)

    plt.title("Workload Analysis with Step Division", fontsize=14)
    plt.xlabel("Time (Seconds)", fontsize=12)
    plt.ylabel("Utilization (%)", fontsize=12)
    plt.ylim(-5, 120) 
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('utilization_with_steps.png')
    print("带有 Step 分区的图片已保存。")

plot_with_step_division("/workspace/repo/verl/aime_qwen3-0.6b_grpo_rollout4.log")