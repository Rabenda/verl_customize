#!/bin/bash

set -euo pipefail

# 可配置参数：
#   DATASET=math MODEL=Qwen/Qwen3-7B bash run_areal_single_7b_gsm8k.sh
DATASET=${DATASET:-gsm8k}
MODEL=${MODEL:-Qwen/Qwen3-4B}

# GPU-R (rollout pool) / GPU-T (trainer pool)
NNODES=${NNODES:-1}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-2}
N_GPUS_TRAINER=${N_GPUS_TRAINER:-2}

TRAIN_FILE="/data/${DATASET}/train.parquet"
VAL_FILE="/data/${DATASET}/test.parquet"

# staleness 三档：sync / light_async / one_step_off
STALENESS_CASES=(one_step_off)

MODEL_SHORT="${MODEL//\//_}"
LOG_DIR=areal_single_log
mkdir -p "${LOG_DIR}"

set -x

for staleness_case in "${STALENESS_CASES[@]}"; do
  CASE_NAME="${DATASET}_${MODEL_SHORT}_areal_single_${staleness_case}"
  LOG_FILE="${LOG_DIR}/${CASE_NAME}.log"

  if [ "$staleness_case" = "sync" ]; then
    STALENESS_THRESHOLD=0
    TRIGGER_SYNC_STEP=4
    PARTIAL_ROLLOUT=False
  elif [ "$staleness_case" = "light_async" ]; then
    STALENESS_THRESHOLD=0.1
    TRIGGER_SYNC_STEP=4
    PARTIAL_ROLLOUT=True
  elif [ "$staleness_case" = "one_step_off" ]; then
    STALENESS_THRESHOLD=1.0
    TRIGGER_SYNC_STEP=4
    PARTIAL_ROLLOUT=True
  else
    echo "Invalid staleness_case: $staleness_case"
    exit 1
  fi

  echo "============================================================"
  echo "Running case: ${CASE_NAME}"
  echo "  DATASET             = ${DATASET}"
  echo "  MODEL               = ${MODEL}"
  echo "  STALENESS_THRESHOLD = ${STALENESS_THRESHOLD}"
  echo "  LOG_FILE            = ${LOG_FILE}"
  echo "============================================================"

  python3 -m verl.experimental.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=True \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    data.truncation=error \
    data.filter_overlong_prompts=True \
    \
    actor_rollout_ref.model.path="${MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.hybrid_engine=False \
    \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS_ROLLOUT} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.max_model_len=8704 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    \
    critic.strategy=fsdp2 \
    \
    trainer.logger='["console"]' \
    trainer.project_name='areal_single' \
    trainer.experiment_name="${CASE_NAME}" \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${N_GPUS_TRAINER} \
    trainer.total_epochs=1 \
    trainer.total_training_steps=10 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.balance_batch=false \
    trainer.critic_warmup=0 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="ckpts/${CASE_NAME}" \
    \
    rollout.nnodes=${NNODES} \
    rollout.n_gpus_per_node=${N_GPUS_ROLLOUT} \
    rollout.total_rollout_steps=640 \
    rollout.total_epochs=10 \
    rollout.test_freq=-1 \
    \
    async_training.staleness_threshold=${STALENESS_THRESHOLD} \
    async_training.trigger_parameter_sync_step=${TRIGGER_SYNC_STEP} \
    async_training.require_batches=4 \
    async_training.partial_rollout=${PARTIAL_ROLLOUT} \
    async_training.use_rollout_log_probs=True \
    async_training.use_trainer_do_validate=False \
    async_training.checkpoint_engine.enable=True \
    async_training.checkpoint_engine.device_buffer_size_M=4096 \
    "$@" \
    > "${LOG_FILE}" 2>&1
done
