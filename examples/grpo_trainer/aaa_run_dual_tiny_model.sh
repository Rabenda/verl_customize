# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.

set -x

python3 -m verl.trainer.main_ppo_stone \
  trainer.dual_model=true \
  algorithm.adv_estimator=grpo \
  data.train_files=/data/gsm8k/train.parquet \
  data.val_files=/data/gsm8k/test.parquet \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  \
  +actor_rollout_ref_a.model.path=Qwen/Qwen3-0.6B \
  +actor_rollout_ref_b.model.path=Qwen/Qwen3-0.6B \
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
  +actor_rollout_ref_a.actor.ppo_micro_batch_size_per_gpu=32 \
  +actor_rollout_ref_b.actor.ppo_micro_batch_size_per_gpu=32 \
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
  +actor_rollout_ref_a.rollout.log_prob_micro_batch_size_per_gpu=32 \
  +actor_rollout_ref_b.rollout.log_prob_micro_batch_size_per_gpu=32 \
  \
  +actor_rollout_ref_a.rollout.tensor_model_parallel_size=4 \
  +actor_rollout_ref_b.rollout.tensor_model_parallel_size=4 \
  \
  +actor_rollout_ref_a.rollout.name=vllm \
  +actor_rollout_ref_b.rollout.name=vllm \
  \
  +actor_rollout_ref_a.rollout.gpu_memory_utilization=0.3 \
  +actor_rollout_ref_b.rollout.gpu_memory_utilization=0.3 \
  \
  +actor_rollout_ref_a.rollout.n=5 \
  +actor_rollout_ref_b.rollout.n=5 \
  \
  +actor_rollout_ref_a.ref.log_prob_micro_batch_size_per_gpu=32 \
  +actor_rollout_ref_b.ref.log_prob_micro_batch_size_per_gpu=32 \
  \
  +actor_rollout_ref_a.ref.fsdp_config.param_offload=True \
  +actor_rollout_ref_b.ref.fsdp_config.param_offload=True \
  \
  +actor_rollout_ref_a.rollout.server_name_suffix=a \
  +actor_rollout_ref_b.rollout.server_name_suffix=b \
  \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name='verl_grpo_example_gsm8k' \
  trainer.experiment_name='qwen3_8b_function_rm' \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=5 \
  trainer.test_freq=5 \
  trainer.total_epochs=1 \
  trainer.balance_batch=false \
  "$@"