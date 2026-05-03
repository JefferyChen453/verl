#!/bin/bash
# Stage 2 RL — single-task acrostic CONTINUE (round 2), GRPO, LCS-z reward.
# Init from RL round-1 epoch-2 step_500/hf_model (warm restart, no optimizer
# state — Adam moments reset, may dip ~50 steps).
# Train data: 1000 fresh acrostic prompts (rl_1task_acrostics_1000_round2.parquet,
# disjoint with round1; per-sample varying targets length 18-20).
# Val data: validation_acrostic177_neg177.parquet (same as round1 — apples-to-apples).
#
# Required env: V2_INIT_HF_PATH (RL round-1 step_500/hf_model dir).
# Optional env: V2_GMU (default 0.6 — bumped from 0.5 per user 2026-05-03;
#                ~58GB vLLM + 28GB train ≈ 86GB / 96GB, ~10GB headroom),
#               V2_PARAM_OFFLOAD (default false), V2_OPTIM_OFFLOAD (default false),
#               V2_REF_OFFLOAD (default false), V2_EXP_SUFFIX (appended to exp name).
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

if [ -z "${V2_INIT_HF_PATH:-}" ]; then
    echo "[fatal] V2_INIT_HF_PATH not set" >&2; exit 1
fi
if [ ! -f "${V2_INIT_HF_PATH}/config.json" ]; then
    echo "[fatal] init hf_model not found: ${V2_INIT_HF_PATH}" >&2; exit 1
fi

GMU=${V2_GMU:-0.6}
PARAM_OFF=${V2_PARAM_OFFLOAD:-false}
OPTIM_OFF=${V2_OPTIM_OFFLOAD:-false}
REF_OFF=${V2_REF_OFFLOAD:-false}
USE_FUSED=${V2_USE_FUSED_KERNELS:-true}
ATTN_IMPL=${V2_ATTN_IMPL:-}
SUFFIX=${V2_EXP_SUFFIX:-}

TS=$(date +%Y%m%d%H%M)
EXP_NAME="rl_1task_acrostics_kdv2s8init_continue_grpo_${TS}${SUFFIX}"

TRAIN_PARQUET="${REPO_ROOT}/data/one_task_train/rl/rl_1task_acrostics_1000_round2.parquet"
VAL_PARQUET="${REPO_ROOT}/data/initials_icw/validation_acrostic177_neg177.parquet"

echo "[$(date)] launching RL CONTINUE: ${EXP_NAME}"
echo "  init = ${V2_INIT_HF_PATH}"
echo "  train = ${TRAIN_PARQUET}"
echo "  gmu = ${GMU}"
echo "  param_offload=${PARAM_OFF}, optimizer_offload=${OPTIM_OFF}, ref_offload=${REF_OFF}"

mkdir -p logs

FUSED_FLAGS=(actor_rollout_ref.model.use_fused_kernels=${USE_FUSED})
if [ "${USE_FUSED}" = "true" ]; then
  FUSED_FLAGS+=(actor_rollout_ref.model.fused_kernel_options.impl_backend=triton)
fi

ATTN_FLAGS=()
if [ -n "${ATTN_IMPL}" ]; then
  ATTN_FLAGS=(+actor_rollout_ref.model.attn_implementation=${ATTN_IMPL})
fi

"${PY}" -m recipe.watermark_rl_ray.main \
  actor_rollout_ref.model.path="${V2_INIT_HF_PATH}" \
  actor_rollout_ref.model.use_liger=false \
  "${FUSED_FLAGS[@]}" \
  "${ATTN_FLAGS[@]}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFF} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIM_OFF} \
  actor_rollout_ref.ref.fsdp_config.param_offload=${REF_OFF} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GMU} \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.rollout.n=8 \
  reward.active_tasks=[acrostics] \
  reward.acrostics_n_resample=1000 \
  reward.acrostics_detector_kind=lcs \
  trainer.project_name=watermark-rl-ray \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.total_epochs=2 \
  trainer.save_freq=after_each_epoch \
  trainer.test_freq=50 \
  trainer.val_before_train=true \
  trainer.logger=["console","wandb"] \
  2>&1 | tee "logs/${EXP_NAME}.log"
