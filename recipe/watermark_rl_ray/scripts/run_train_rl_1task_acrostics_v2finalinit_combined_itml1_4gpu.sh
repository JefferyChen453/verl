#!/bin/bash
# Stage 2 RL — single-task acrostic, GRPO, LCS-z reward.
# Init from V2 KD+RL FINAL (round2 epoch 2 = continue grpo step_500/hf_model).
# Train data: combined acrostic 2000 (round1 1000 + round2 1000, prefix-disjoint).
# 1 epoch fine-tune. itml-1 4-GPU layout (matches V3 RL continue).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_LOG_LEVEL=${VLLM_LOG_LEVEL:-ERROR}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export WANDB_PROJECT=${WANDB_PROJECT:-watermark-rl-ray}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONWARNINGS=${PYTHONWARNINGS:-"ignore::FutureWarning:verl.utils.device"}

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"
PY="${PY:-${REPO_ROOT}/../.venv/bin/python}"

INIT_HF_PATH="${REPO_ROOT}/checkpoints/watermark-rl-ray/rl_1task_acrostics_kdv2s8init_continue_grpo_202605030503/global_step_500/hf_model"
if [ ! -f "${INIT_HF_PATH}/config.json" ]; then
    echo "[fatal] init hf_model not found: ${INIT_HF_PATH}" >&2; exit 1
fi

GMU=${GMU:-0.5}
SUFFIX=${EXP_SUFFIX:-}

TS=$(date +%Y%m%d%H%M)
EXP_NAME="rl_1task_acrostics_v2finalinit_combined_grpo_${TS}_itml1_4gpu${SUFFIX}"

TRAIN_PARQUET="${REPO_ROOT}/data/one_task_train/rl/rl_1task_acrostics_combined_2000.parquet"
VAL_PARQUET="${REPO_ROOT}/data/initials_icw/validation_acrostic177_neg177.parquet"

echo "[$(date)] launching RL: ${EXP_NAME}"
echo "  init = ${INIT_HF_PATH}"
echo "  train = ${TRAIN_PARQUET}"
echo "  gmu = ${GMU}"

mkdir -p logs

"${PY}" -m recipe.watermark_rl_ray.main \
  actor_rollout_ref.model.path="${INIT_HF_PATH}" \
  actor_rollout_ref.model.use_liger=false \
  actor_rollout_ref.model.use_fused_kernels=true \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GMU} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.rollout.n=8 \
  reward.active_tasks=[acrostics] \
  reward.acrostics_n_resample=1000 \
  reward.acrostics_detector_kind=lcs \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.project_name=watermark-rl-ray \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.total_epochs=1 \
  trainer.save_freq=after_each_epoch \
  trainer.test_freq=30 \
  trainer.val_before_train=true \
  trainer.logger=["console","wandb"] \
  2>&1 | tee "logs/${EXP_NAME}.log"
