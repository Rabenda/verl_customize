GPUS=4

set -x

export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_HOME="/data/huggingface_cache"
# model_runner 的 inference step CSV 输出到此目录
export SGLANG_INFERENCE_LOG_DIR="/workspace/repo/verl_sglang/verl_customize/formal_exp_data_mfu"
gsm8k_train_path=/data/verl/gsm8k/train.parquet
gsm8k_test_path=/data/verl/gsm8k/test.parquet

math_train_path=/data/verl/math/train.parquet
math_test_path=/data/verl/math/test.parquet

aime_train_path=/data/verl/aime/train.parquet

# train_files="['$gsm8k_train_path']"
# test_files="['$gsm8k_test_path']"
train_files="['$math_train_path']"
test_files="['$math_test_path']"
# train_files="['$aime_train_path']"
# test_files="['$aime_train_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-32B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GPUS \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_model_math' \
    trainer.experiment_name='model_math_workload_analysis' \
    trainer.profile.print_every=1 \
    trainer.decoding_length_csv_dir=/workspace/repo/verl_sglang/verl_customize/formal_exp_data_decoding_length \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=1000 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 $@