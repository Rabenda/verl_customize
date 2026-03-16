GPUS=4

# 环境变量: FIT_METHOD, START_B_WHEN_ACTIVE_A_LEQ
# 用法: FIT_METHOD=naive ./aaa_run_dual_tiny_model_sglang.sh
#       START_B_WHEN_ACTIVE_A_LEQ=256 ./aaa_run_dual_tiny_model_sglang.sh
FIT_METHOD=${FIT_METHOD:-overlap_decode}
START_B_WHEN_ACTIVE_A_LEQ=${START_B_WHEN_ACTIVE_A_LEQ:-1024}
MODEL_PATH=${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}
DATASET=${DATASET:-gsm8k}

echo "FIT_METHOD: $FIT_METHOD"
echo "START_B_WHEN_ACTIVE_A_LEQ: $START_B_WHEN_ACTIVE_A_LEQ"
echo "MODEL_PATH: $MODEL_PATH"
echo "DATASET: $DATASET"

if [ "$FIT_METHOD" = "overlap_decode" ]; then
  ROLLOUT_MPS=50
elif [ "$FIT_METHOD" = "overlap_b_rollout_a_train" ]; then
  ROLLOUT_MPS=100
elif [ "$FIT_METHOD" = "naive" ]; then
  FIT_METHOD=overlap_decode
  START_B_WHEN_ACTIVE_A_LEQ=0
  ROLLOUT_MPS=100
else
  echo "Invalid FIT_METHOD: $FIT_METHOD"
  exit 1
fi

set -x

python3 -m verl.trainer.main_ppo_stone \
  trainer.dual_model=true \
  algorithm.adv_estimator=grpo \
  data.train_files=/data/$DATASET/train.parquet \
  data.val_files=/data/$DATASET/test.parquet \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  \
  +actor_rollout_ref_a.model.path=$MODEL_PATH \
  +actor_rollout_ref_b.model.path=$MODEL_PATH \
  \
  +actor_rollout_ref_a.actor.optim.lr=1e-6 \
  +actor_rollout_ref_b.actor.optim.lr=1e-6 \
  \
  +actor_rollout_ref_a.model.use_remove_padding=True \
  +actor_rollout_ref_b.model.use_remove_padding=True \
  \
  +actor_rollout_ref_a.actor.ppo_mini_batch_size=256 \
  +actor_rollout_ref_b.actor.ppo_mini_batch_size=256 \
  \
  +actor_rollout_ref_a.actor.ppo_micro_batch_size_per_gpu=8 \
  +actor_rollout_ref_b.actor.ppo_micro_batch_size_per_gpu=8 \
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
  +actor_rollout_ref_a.actor.fsdp_config.param_offload=True \
  +actor_rollout_ref_b.actor.fsdp_config.param_offload=True \
  \
  +actor_rollout_ref_a.actor.fsdp_config.optimizer_offload=True \
  +actor_rollout_ref_b.actor.fsdp_config.optimizer_offload=True \
  \
  +actor_rollout_ref_a.rollout.log_prob_micro_batch_size_per_gpu=8 \
  +actor_rollout_ref_b.rollout.log_prob_micro_batch_size_per_gpu=8 \
  \
  +actor_rollout_ref_a.rollout.tensor_model_parallel_size=$GPUS \
  +actor_rollout_ref_b.rollout.tensor_model_parallel_size=$GPUS \
  \
  +actor_rollout_ref_a.rollout.name=sglang \
  +actor_rollout_ref_b.rollout.name=sglang \
  \
  +actor_rollout_ref_a.rollout.gpu_memory_utilization=0.3 \
  +actor_rollout_ref_b.rollout.gpu_memory_utilization=0.3 \
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
  +actor_rollout_ref_a.ref.log_prob_micro_batch_size_per_gpu=8 \
  +actor_rollout_ref_b.ref.log_prob_micro_batch_size_per_gpu=8 \
  \
  +actor_rollout_ref_a.ref.fsdp_config.param_offload=True \
  +actor_rollout_ref_b.ref.fsdp_config.param_offload=True \
  \
  +actor_rollout_ref_a.actor.fsdp_config.model_dtype=bfloat16 \
  +actor_rollout_ref_b.actor.fsdp_config.model_dtype=bfloat16 \
  +actor_rollout_ref_a.ref.fsdp_config.model_dtype=bfloat16 \
  +actor_rollout_ref_b.ref.fsdp_config.model_dtype=bfloat16 \
  \
  +actor_rollout_ref_a.rollout.server_name_suffix=a \
  +actor_rollout_ref_b.rollout.server_name_suffix=b \
  \
  algorithm.use_kl_in_reward=True \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name='verl_grpo_example_gsm8k' \
  trainer.experiment_name='qwen3_8b_function_rm' \
  trainer.n_gpus_per_node=$GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=5 \
  trainer.test_freq=5 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.balance_batch=false \
  trainer.profile.print_every=1 \
  trainer.fit_method=$FIT_METHOD \
  trainer.start_b_when_active_a_leq=$START_B_WHEN_ACTIVE_A_LEQ \
  trainer.mps_active_thread_percentage=100 \
  "$@"

  # \
  # +actor_rollout_ref_a.rollout.enforce_eager=True \
  # +actor_rollout_ref_b.rollout.enforce_eager=True \