# Turn frequency 绘图（自包含）

本目录含脚本与 `lengths_model_multiturn_*.csv`，**不依赖** `formal_exp_data_decoding_length` 等仓库其他路径。

## 环境

```bash
pip install -r requirements.txt
```

## 生成图（默认矢量 PDF）

```bash
cd /path/to/turn_plot
python draw_turn_frequency_four_multiturn.py
# 输出: pic/turn_frequency_four_multiturn.pdf
```

指定输出路径：

```bash
python draw_turn_frequency_four_multiturn.py --save pic/my_turn_plot.pdf
python draw_turn_frequency_four_multiturn.py --save pic/my_turn_plot.png
```

## 文件

| 文件 | 说明 |
|------|------|
| `draw_turn_frequency_four_multiturn.py` | 四配置分组柱状图（断轴） |
| `lengths_model_multiturn_*.csv` | 列需含 `turn`（及 `length`） |
