#!/bin/bash
# Stage 2 RL — green + initials + acrostics (3 tasks), GRPO, sequence z-score reward.
# Initialized from a v5b_aligned (or v5b old) ckpt — pick the best after the day-4 eval.
#
# Pre-reqs:
#   1. tools/build_train_parquet_3task.py has been run (creates train_rl_3task_*.parquet)
#   2. tools/build_val_parquet_3task.py has been run (creates validation_3task_*.parquet)
#   3. recipe/watermark_rl_ray/{reward,dataset}.py have been updated to support
#      per-sample acrostic_target (already done in icw-day4 session)

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

TS=$(date +%Y%m%d%H%M)
EXP_NAME="rl_3task_v5baligned_grpo_${TS}"

# Default init: v5b_aligned epoch 3 (filled in by autopilot after eval). Override
# via INIT_CKPT env to pick a different ckpt.
INIT_CKPT="${INIT_CKPT:-${REPO_ROOT}/checkpoints/watermark-kd-ray/v5b_aligned_eval_202604240212/global_step_1965/hf_model}"

TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/data/rl_stage2/train_rl_3task_green800_initials400_acrostics800.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${REPO_ROOT}/data/initials_icw/validation_3task_green177_initials177_neg177_acrostics177.parquet}"

# Acrostics val target (must match build_val_parquet_3task.py)
ACROSTICS_VAL_TARGET="${ACROSTICS_VAL_TARGET:-DETECTOR}"

mkdir -p logs

# Use full venv python path — non-interactive shells may not have python on PATH
PY=${PY:-/home/tianyichen/llm_watermark/.venv/bin/python}

"${PY}" -m recipe.watermark_rl_ray.main \
  actor_rollout_ref.model.path="${INIT_CKPT}" \
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
  reward.active_tasks='[green,initials,acrostics]' \
  reward.acrostics_target="${ACROSTICS_VAL_TARGET}" \
  reward.acrostics_n_resample=200 \
  trainer.project_name=watermark-rl-ray \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.total_epochs=2 \
  trainer.save_freq=after_each_epoch \
  trainer.test_freq=30 \
  trainer.val_before_train=true \
  trainer.logger=["console","wandb"] \
  2>&1 | tee "logs/${EXP_NAME}.log"
