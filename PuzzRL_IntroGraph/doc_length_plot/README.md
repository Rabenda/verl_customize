# Doc length 绘图（自包含）

脚本与 `doc_lengths_*.csv` 放在同一目录即可离线出图（`--file` 可为文件名或绝对路径）。

## 环境

```bash
pip install -r requirements.txt
```

## 生成图（默认矢量 PDF）

```bash
cd /path/to/doc_length_plot
python draw_doc_length_single_first_step.py
# 默认读: doc_lengths_qwen3_32b_search_r1_sync_nprobe32_bs256.csv
# 输出: pic/plot_doc_length_single_first_step.pdf
```

常用参数：

```bash
python draw_doc_length_single_first_step.py --file doc_lengths_xxx.csv --label "Model | Setup"
python draw_doc_length_single_first_step.py --save pic/custom.pdf
python draw_doc_length_single_first_step.py --line --save pic/custom_line.pdf
```

CSV 需含列：`length`、`step`；若有 `turn` 列会保留（否则补 0）。
