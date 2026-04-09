MODEL=${MODEL:-"Qwen/Qwen3-0.6B"}
DATASET=${DATASET:-"math"}

SAFE_MODEL="${MODEL//\//_}"
LOG_NAME="${SAFE_MODEL}_${DATASET}"

DATASET=${DATASET} MODEL_A=${MODEL} MODEL_B=${MODEL} bash verl/experimental/one_step_off_policy/shell/a_sosp_grpo_dual.sh >> async_logs/${LOG_NAME}_dual_8gpus
DATASET=${DATASET} MODEL_PATH=${MODEL} bash verl/experimental/one_step_off_policy/shell/a_sosp_grpo.sh >> async_logs/${LOG_NAME}_8gpus

