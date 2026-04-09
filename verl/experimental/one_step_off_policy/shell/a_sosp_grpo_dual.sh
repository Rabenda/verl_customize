#!/usr/bin/env bash
# Dual async one-step-off-policy training with two 0.6B models on GSM8K.
#
# GPU layout (4 GPUs total, 2 per pool):
#   pool_a (GPU 0,1): Model-A SGLang rollout  +  Model-B FSDP actor
#   pool_b (GPU 2,3): Model-B SGLang rollout  +  Model-A FSDP actor
#
# Usage:
#   MODEL_A=Qwen/Qwen3-0.6B MODEL_B=Qwen/Qwen3-0.6B bash grpo_dual_0.6b_gsm8k_sglang_2_2.sh
#
set -x

project_name='GRPO_DUAL'
exp_name='grpo-dual-qwen3-0.6b-gsm8k-sglang-2-2'

RAY_DATA_HOME=${RAY_DATA_HOME:-"/data/verl_home"}
MODEL_A=${MODEL_A:-"Qwen/Qwen3-0.6B"}
MODEL_B=${MODEL_B:-"Qwen/Qwen3-0.6B"}
DATASET=${DATASET:-"gsm8k"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"/data/${DATASET}/train.parquet"}
TEST_FILE=${TEST_FILE:-"/data/${DATASET}/test.parquet"}

# n GPUs per pool (pool_a = pool_b = N_GPUS_PER_POOL)
N_GPUS_PER_POOL=${N_GPUS_PER_POOL:-4}
NNODES=${NNODES:-1}

python3 -m verl.experimental.one_step_off_policy.dual_main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    +actor_rollout_ref_a.model.path="${MODEL_A}" \
    +actor_rollout_ref_b.model.path="${MODEL_B}" \
    \
    +actor_rollout_ref_a.actor.strategy=fsdp2 \
    +actor_rollout_ref_b.actor.strategy=fsdp2 \
    \
    +actor_rollout_ref_a.actor.optim.lr=1e-6 \
    +actor_rollout_ref_b.actor.optim.lr=1e-6 \
    \
    +actor_rollout_ref_a.hybrid_engine=False \
    +actor_rollout_ref_b.hybrid_engine=False \
    \
    +actor_rollout_ref_a.model.use_remove_padding=True \
    +actor_rollout_ref_b.model.use_remove_padding=True \
    \
    +actor_rollout_ref_a.actor.ppo_mini_batch_size=64 \
    +actor_rollout_ref_b.actor.ppo_mini_batch_size=64 \
    \
    +actor_rollout_ref_a.actor.ppo_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref_b.actor.ppo_micro_batch_size_per_gpu=1 \
    \
    +actor_rollout_ref_a.actor.use_kl_loss=True \
    +actor_rollout_ref_b.actor.use_kl_loss=True \
    \
    +actor_rollout_ref_a.actor.kl_loss_coef=0.001 \
    +actor_rollout_ref_b.actor.kl_loss_coef=0.001 \
    \
    +actor_rollout_ref_a.actor.kl_loss_type=low_var_kl \
    +actor_rollout_ref_b.actor.kl_loss_type=low_var_kl \
    \
    +actor_rollout_ref_a.actor.entropy_coeff=0 \
    +actor_rollout_ref_b.actor.entropy_coeff=0 \
    \
    +actor_rollout_ref_a.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref_b.model.enable_gradient_checkpointing=True \
    \
    +actor_rollout_ref_a.actor.fsdp_config.param_offload=False \
    +actor_rollout_ref_b.actor.fsdp_config.param_offload=False \
    \
    +actor_rollout_ref_a.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref_b.actor.fsdp_config.optimizer_offload=False \
    \
    +actor_rollout_ref_a.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref_b.actor.fsdp_config.model_dtype=bfloat16 \
    \
    +actor_rollout_ref_a.rollout.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref_b.rollout.log_prob_micro_batch_size_per_gpu=1 \
    \
    +actor_rollout_ref_a.rollout.tensor_model_parallel_size=${N_GPUS_PER_POOL} \
    +actor_rollout_ref_b.rollout.tensor_model_parallel_size=${N_GPUS_PER_POOL} \
    \
    +actor_rollout_ref_a.rollout.name=sglang \
    +actor_rollout_ref_b.rollout.name=sglang \
    \
    +actor_rollout_ref_a.rollout.gpu_memory_utilization=0.4 \
    +actor_rollout_ref_b.rollout.gpu_memory_utilization=0.4 \
    \
    +actor_rollout_ref_a.rollout.mps_active_thread_percentage=50 \
    +actor_rollout_ref_b.rollout.mps_active_thread_percentage=50 \
    \
    +actor_rollout_ref_a.rollout.max_model_len=8704 \
    +actor_rollout_ref_b.rollout.max_model_len=8704 \
    \
    +actor_rollout_ref_a.rollout.n=4 \
    +actor_rollout_ref_b.rollout.n=4 \
    \
    +actor_rollout_ref_a.rollout.load_format=safetensors \
    +actor_rollout_ref_b.rollout.load_format=safetensors \
    \
    +actor_rollout_ref_a.rollout.layered_summon=True \
    +actor_rollout_ref_b.rollout.layered_summon=True \
    \
    +actor_rollout_ref_a.rollout.server_name_suffix=a \
    +actor_rollout_ref_b.rollout.server_name_suffix=b \
    \
    +actor_rollout_ref_a.ref.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref_b.ref.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref_a.ref.fsdp_config.param_offload=False \
    +actor_rollout_ref_b.ref.fsdp_config.param_offload=False \
    +actor_rollout_ref_a.ref.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref_b.ref.fsdp_config.model_dtype=bfloat16 \
    \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=5 \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_POOL}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${N_GPUS_PER_POOL}" \
    trainer.balance_batch=false \
    "$@"
