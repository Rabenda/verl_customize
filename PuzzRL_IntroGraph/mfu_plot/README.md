# MFU 绘图（自包含目录）

本目录包含绘图脚本与所需 CSV（`inference_step_log_*.csv`、`tool_times_*.csv`），**不依赖仓库其他路径**。解压或复制本文件夹后即可单独使用。

## 环境

```bash
pip install -r requirements.txt
```

- `scipy` 可选；未安装时曲线平滑会退化为卷积实现。

## 生成图片

在**本目录**下执行（推荐）：

```bash
cd /path/to/mfu_plot

# 3×4 总图（single / multiturn / search-r1 各一行）
python draw_mfu_3x4_grid_zhengding.py --save pic/mfu_3x4.pdf

# 单行四子图（可选）
python draw_mfu_four_single_plot_zhengding.py --save pic/mfu_four_single.pdf
python draw_mfu_four_multiturn_plot_zhengding.py --save pic/mfu_four_multiturn.pdf
python draw_mfu_four_plot_zhengding.py --save pic/mfu_four_search_r1.pdf
```

也可从其他目录用绝对路径运行脚本；脚本会把自身所在目录加入 `sys.path`，`from draw_mfu_plot import ...` 仍指向本目录下的模块。

输出目录默认为 `pic/`，若不存在请先 `mkdir -p pic`。

## 文件说明

| 脚本 | 说明 |
|------|------|
| `draw_mfu_plot.py` | 公共函数（被 single/multiturn 脚本 import） |
| `draw_mfu_four_single_plot_zhengding.py` | Single-turn 四配置 |
| `draw_mfu_four_multiturn_plot_zhengding.py` | Multiturn 四配置 |
| `draw_mfu_four_plot_zhengding.py` | Search-R1 四配置（faiss 段 + tool_times） |
| `draw_mfu_3x4_grid_zhengding.py` | 组合上述三行成 3×4 图 |

CSV 文件名与各脚本内 `FILES` / `DOC_AND_TOOL_DIR` 一致，勿随意改名。
