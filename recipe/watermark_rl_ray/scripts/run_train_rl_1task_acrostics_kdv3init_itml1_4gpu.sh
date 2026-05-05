#!/bin/bash
# Stage 2 RL — single-task acrostic, GRPO, LCS-z reward.
# itml-1 4-GPU variant. Init from V3 KD last-epoch ckpt.
#
# Diff from run_train_rl_1task_acrostics_kdv3init.sh (8-GPU itml-2 variant):
#   1. trainer.n_gpus_per_node=4 (itml-1 user authorized GPUs 0-3)
#   2. actor_rollout_ref.rollout.tensor_model_parallel_size=4 (single TP group)
#   3. CUDA_VISIBLE_DEVICES=0,1,2,3 export at top
#   4. experiment_name suffix _itml1_4gpu
#
# All other hyperparams identical (lr=1e-6, GRPO bsz=4 mini=16 n=8, epochs=2).
#
# Required env: V3_INIT_HF_PATH (KD V3 last-epoch hf_model dir).
# Optional env: V3_GMU (default 0.5), V3_PARAM_OFFLOAD (default false),
#               V3_OPTIM_OFFLOAD (default false), V3_REF_OFFLOAD (default false),
#               V3_EXP_SUFFIX (appended to experiment_name).
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

if [ -z "${V3_INIT_HF_PATH:-}" ]; then
    echo "[fatal] V3_INIT_HF_PATH not set" >&2; exit 1
fi
if [ ! -f "${V3_INIT_HF_PATH}/config.json" ]; then
    echo "[fatal] init hf_model not found: ${V3_INIT_HF_PATH}" >&2; exit 1
fi

GMU=${V3_GMU:-0.5}
PARAM_OFF=${V3_PARAM_OFFLOAD:-false}
OPTIM_OFF=${V3_OPTIM_OFFLOAD:-false}
REF_OFF=${V3_REF_OFFLOAD:-false}
USE_FUSED=${V3_USE_FUSED_KERNELS:-true}
ATTN_IMPL=${V3_ATTN_IMPL:-}
SUFFIX=${V3_EXP_SUFFIX:-}

TS=$(date +%Y%m%d%H%M)
EXP_NAME="rl_1task_acrostics_kdv3init_grpo_${TS}_itml1_4gpu${SUFFIX}"

TRAIN_PARQUET="${REPO_ROOT}/data/one_task_train/rl/rl_1task_acrostics_1000.parquet"
VAL_PARQUET="${REPO_ROOT}/data/initials_icw/validation_acrostic177_neg177.parquet"

echo "[$(date)] launching RL: ${EXP_NAME}"
echo "  init = ${V3_INIT_HF_PATH}"
echo "  gmu  = ${GMU}"
echo "  param_offload=${PARAM_OFF}, optimizer_offload=${OPTIM_OFF}, ref_offload=${REF_OFF}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

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
  actor_rollout_ref.model.path="${V3_INIT_HF_PATH}" \
  actor_rollout_ref.model.use_liger=false \
  "${FUSED_FLAGS[@]}" \
  "${ATTN_FLAGS[@]}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFF} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIM_OFF} \
  actor_rollout_ref.ref.fsdp_config.param_offload=${REF_OFF} \
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
  trainer.total_epochs=2 \
  trainer.save_freq=after_each_epoch \
  trainer.test_freq=30 \
  trainer.val_before_train=true \
  trainer.logger=["console","wandb"] \
  2>&1 | tee "logs/${EXP_NAME}.log"
