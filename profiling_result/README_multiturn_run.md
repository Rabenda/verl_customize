# 多轮训练怎么跑

## 一、通用前提

- **工作目录**：所有命令都在**项目根目录**执行（例如 `verl_customize`），除非写明了要 `cd` 到子目录。
- **图里那种「中间有 Tool Calling」**：需要 **多轮开启** + **配置 tool_config_path**；GSM8K 和 Search 都满足。

---

## 二、跑 GSM8K 多轮（数学题 + calc_gsm8k_reward）

### 1. 准备数据

```bash
cd examples/data_preprocess
python3 gsm8k_multiturn_w_tool.py --local_save_dir ~/data/gsm8k
cd ../..
```

- 会从 HuggingFace 拉 `openai/gsm8k`，生成多轮+工具格式的 `train.parquet`、`test.parquet`，写到 `~/data/gsm8k/`（可改 `--local_save_dir`）。

### 2. 跑训练（二选一）

**方式 A：进程内 SGLang（不单独起服务）**

```bash
bash examples/sglang_multiturn/run_qwen3-4b_gsm8k_multiturn.sh
```

- 配置：`gsm8k_multiturn_grpo`，数据默认 `$HOME/data/gsm8k/`，工具：`gsm8k_tool_config.yaml`（脚本里已写 `multi_turn.tool_config_path`）。

**方式 B：SGLang Server 模式（需先起推理服务）**

```bash
# 1）先在本机/远程起好 SGLang 推理服务（同模型、端口与配置一致）
# 2）再跑：
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn_server.sh
```

- 配置：`gsm8k_multiturn_grpo_server`，同样用 `~/data/gsm8k/` 和 `gsm8k_tool_config.yaml`。

### 3. 可选修改

- 数据路径：改脚本里的 `data.train_files`、`data.val_files`（和 `--local_save_dir` 一致）。
- 模型：改 `actor_rollout_ref.model.path`。
- 卡数：默认 8 卡，可覆盖，例如 `trainer.n_gpus_per_node=4`。

---

## 三、跑 Search 多轮（检索 + search 工具）

### 1. 准备数据

```bash
cd examples/data_preprocess
python3 preprocess_search_r1_dataset.py --local_dir ~/data/searchR1_processed_direct
cd ../..
```

- 从 HuggingFace 下 Search-R1 类数据，生成带 search 工具用的 `tools_kwargs` 的 parquet，默认输出到 `~/data/searchR1_processed_direct/`（可改 `--local_dir`）。

### 2. 起检索服务（必须）

- 工具配置里请求的是：`http://127.0.0.1:8000/retrieve`。
- 在 `examples/sglang_multiturn/search_r1_like/local_dense_retriever/` 下按说明准备 corpus、模型，起检索服务并保证 **8000 端口** 可访问。

### 3. 跑训练

```bash
bash examples/sglang_multiturn/search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh
```

- 配置：`search_multiturn_grpo`，数据默认 `$HOME/data/searchR1_processed_direct/`，工具：`search_tool_config.yaml`（脚本里已写 `tool_config_path`）。
- 若要用 **search_multiturn_grpo_one_step_off**：把脚本里的 `--config-name='search_multiturn_grpo'` 改成 `--config-name='search_multiturn_grpo_one_step_off'`。

### 4. 可选修改

- 数据路径：改脚本里的 `TRAIN_DATA`、`VAL_DATA`（或对应的 `data.train_files`、`data.val_files`）。
- 模型、卡数：同 GSM8K，按需改 `actor_rollout_ref.model.path`、`trainer.n_gpus_per_node`。
