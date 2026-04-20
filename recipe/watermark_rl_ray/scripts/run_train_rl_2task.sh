#!/bin/bash
# Stage 2 RL — green + initials, 2 tasks, GRPO, sequence z-score reward.
# Initialized from v5b checkpoint.
set -euo pipefail

export VLLM_LOG_LEVEL=${VLLM_LOG_LEVEL:-ERROR}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export WANDB_PROJECT=${WANDB_PROJECT:-watermark-rl-ray}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
# Silence transformers tokenizer warnings (fix_mistral_regex false-positive on Qwen3 +
# Qwen2TokenizerFast "use __call__" advisory from agent_loop)
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
# Silence torch.cuda._set_allocator_settings FutureWarning from verl/utils/device.py:94
export PYTHONWARNINGS=${PYTHONWARNINGS:-"ignore::FutureWarning:verl.utils.device"}

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"

TS=$(date +%Y%m%d%H%M)
EXP_NAME="rl_2task_green1000_initials1000_v5binit_grpo_${TS}"

CKPT_V5B="${REPO_ROOT}/checkpoints/watermark-kd-ray/v5b_green3379+initials865+neg1000_dualKL_biasedRefTopK1000_202604180857/global_step_655/hf_model"
TRAIN_PARQUET="${REPO_ROOT}/data/rl_stage2/train_rl_green1000_initials1000.parquet"
VAL_PARQUET="${REPO_ROOT}/data/initials_icw/validation_mixed_green177_initials177_neg177.parquet"

python -m recipe.watermark_rl_ray.main \
  actor_rollout_ref.model.path="${CKPT_V5B}" \
  actor_rollout_ref.model.use_liger=false \
  actor_rollout_ref.model.use_fused_kernels=true \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.rollout.n=8 \
  trainer.project_name=watermark-rl-ray \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.total_epochs=2 \
  trainer.save_freq=after_each_epoch \
  trainer.test_freq=30 \
  trainer.val_before_train=false \
  trainer.logger=["console","wandb"] \
  2>&1 | tee "logs/${EXP_NAME}.log"
