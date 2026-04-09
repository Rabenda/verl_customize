#!/usr/bin/env bash
# ============================================================
# Benchmark: naive (sequential) vs overlap_decode
#
# Usage:
#   ./aaa_bench_naive_vs_overlap.sh              # run both modes
#   ./aaa_bench_naive_vs_overlap.sh naive        # only naive
#   ./aaa_bench_naive_vs_overlap.sh overlap      # only overlap_decode
#
# Env overrides (all have defaults):
#   MODEL_PATH, GPUS, TRAIN_STEPS, BATCH_SIZE, N_ROLLOUT
#   TOOL_CONFIG, TRAIN_DATA, VAL_DATA, LOG_DIR
# ============================================================
set -euo pipefail

# ── defaults ────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
GPUS=${GPUS:-4}
TRAIN_STEPS=${TRAIN_STEPS:-1}
BATCH_SIZE=${BATCH_SIZE:-64}
N_ROLLOUT=${N_ROLLOUT:-4}

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TOOL_CONFIG=${TOOL_CONFIG:-"$CONFIG_PATH/tool_config/search_tool_config.yaml"}
TRAIN_DATA=${TRAIN_DATA:-"/data/search_r1_data/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/search_r1_data/test.parquet"}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${LOG_DIR:-"$PROJECT_DIR/bench_logs/search_r1_naive_vs_overlap_${TIMESTAMP}"}
mkdir -p "$LOG_DIR"

RUN_MODE=${1:-both}   # naive | overlap | both

TRAINER_CLASS="verl.trainer.ppo.ray_search_r1_like_sync_trainer.SearchR1LikeSyncRayPPOTrainer"

# ── shared python args ───────────────────────────────────────
common_args() {
    echo \
        --config-path="$CONFIG_PATH" \
        --config-name='search_multiturn_grpo' \
        algorithm.adv_estimator=grpo \
        data.train_files="$TRAIN_DATA" \
        data.val_files="$VAL_DATA" \
        data.train_max_samples=64 \
        data.train_batch_size=$BATCH_SIZE \
        data.val_batch_size=256 \
        data.max_prompt_length=4096 \
        data.max_response_length=3000 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.max_model_len=15000 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$GPUS \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=$N_ROLLOUT \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
        actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
        actor_rollout_ref.rollout.multi_turn.format=hermes \
        actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.val_before_train=False \
        trainer.logger='["console"]' \
        trainer.trainer_class=$TRAINER_CLASS \
        trainer.project_name='search_r1_bench' \
        trainer.n_gpus_per_node=$GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=0 \
        trainer.test_freq=0 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=$TRAIN_STEPS \
        trainer.balance_batch=false \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
}

# ── run helper ───────────────────────────────────────────────
run_mode() {
    local mode="$1"          # naive | overlap
    local fit_method="$2"    # fit   | fit_overlap_decode
    local exp_name="search_r1_${mode}_${TIMESTAMP}"
    local log_file="$LOG_DIR/${mode}.log"

    echo ""
    echo "============================================================"
    echo "  MODE : $mode"
    echo "  FIT  : $fit_method"
    echo "  LOG  : $log_file"
    echo "============================================================"

    set -x
    python3 -m verl.trainer.main_ppo \
        $(common_args) \
        trainer.experiment_name="$exp_name" \
        trainer.fit_method="$fit_method" \
        2>&1 | tee "$log_file"
    set +x

    echo "[bench] $mode finished → $log_file"
}

# ── main ─────────────────────────────────────────────────────
echo "LOG_DIR : $LOG_DIR"
echo "MODEL   : $MODEL_PATH"
echo "GPUS    : $GPUS"
echo "STEPS   : $TRAIN_STEPS"
echo "BATCH   : $BATCH_SIZE  (n_rollout=$N_ROLLOUT)"
echo ""

case "$RUN_MODE" in
    naive)
        run_mode naive   fit
        ;;
    overlap)
        run_mode overlap fit_overlap_decode
        ;;
    both)
        run_mode naive   fit
        run_mode overlap fit_overlap_decode
        ;;
    *)
        echo "Unknown mode: $RUN_MODE  (use naive | overlap | both)"
        exit 1
        ;;
esac

# ── summary ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  BENCHMARK COMPLETE"
echo "  logs in: $LOG_DIR"
echo "============================================================"

for f in "$LOG_DIR"/*.log; do
    label=$(basename "$f" .log)

    # naive:   "[Search-R1 Sync] Done: ... wall=12.345s"
    # overlap: "[OVERLAP_DECODE] rollout=12.345s train=8.234s overlap=8.234s ..."
    if [[ "$label" == *naive* ]]; then
        # grab the total per-rollout wall time (one per step)
        vals=$(grep -oP '(?<=\bwall=)\d+\.\d+' "$f" 2>/dev/null || true)
    else
        # grab rollout wall (first timing field per OVERLAP_DECODE summary line)
        vals=$(grep '\[OVERLAP_DECODE\].*rollout=' "$f" \
               | grep -oP '(?<=rollout=)\d+\.\d+' 2>/dev/null || true)
    fi

    if [ -n "$vals" ]; then
        mean=$(echo "$vals" | awk '{s+=$1; n++} END {printf "%.2f", s/n}')
        count=$(echo "$vals" | wc -l | tr -d ' ')
        echo "  [$label]  mean rollout wall = ${mean}s  (over $count steps)"
    else
        echo "  [$label]  no timing data found"
    fi
done
echo "============================================================"
