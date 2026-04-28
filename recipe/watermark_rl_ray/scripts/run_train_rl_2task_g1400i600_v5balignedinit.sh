#!/bin/bash
# Stage 2 RL — green + initials, 2 tasks, GRPO, sequence z-score reward.
# Init from v5b_aligned_eval_202604240212 epoch-3 (global_step_1965).
# Train data: 1400 green + 600 initials = 2000 prompts (delta over 0816 run).
# Training settings IDENTICAL to rl_2task_discard_v5binit_grpo_202604210816.
set -euo pipefail

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

TS=$(date +%Y%m%d%H%M)
EXP_NAME="rl_2task_g1400i600_v5balignedinit_grpo_${TS}"

CKPT_BASE="${REPO_ROOT}/checkpoints/watermark-kd-ray/v5b_aligned_eval_202604240212/global_step_1965/hf_model"
TRAIN_PARQUET="${REPO_ROOT}/data/rl_stage2/train_rl_green1400_initials600.parquet"
VAL_PARQUET="${REPO_ROOT}/data/initials_icw/validation_mixed_green177_initials177_neg177.parquet"

"${PY}" -m recipe.watermark_rl_ray.main \
  actor_rollout_ref.model.path="${CKPT_BASE}" \
  actor_rollout_ref.model.use_liger=false \
  actor_rollout_ref.model.use_fused_kernels=true \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.rollout.n=8 \
  trainer.project_name=watermark-rl-ray \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.total_epochs=2 \
  trainer.save_freq=after_each_epoch \
  trainer.test_freq=30 \
  trainer.val_before_train=true \
  trainer.logger=["console","wandb"] \
  2>&1 | tee "logs/${EXP_NAME}.log"
