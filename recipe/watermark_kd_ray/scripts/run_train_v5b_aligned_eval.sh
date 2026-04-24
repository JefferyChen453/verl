#!/usr/bin/env bash
# V5b re-run with eval-side config ALIGNED to RL recipe (gznyxqkj).
# Purpose: isolate which config(s) caused the KD val green_auc 0.845 vs
# RL val_before_train 0.970 gap on the same hf_model.
#
# Training loss / data / KD config identical to run_train_v5b_biased_ref_topk1000.sh.
# Only val-path-relevant fields are changed to match RL:
#   - data.val_batch_size: 128 -> None (=531, one batch)  ← prime suspect
#   - data.max_prompt_length: 130000 -> 60000
#   - data.truncation: error -> left
#   - data.max_response_length: 1072 -> 600
#   - rollout.prompt_length: 81920 -> 60000
#   - rollout.gpu_memory_utilization: 0.95 -> 0.9
#
# NOT aligned (these only affect actor train forward, not vLLM val rollout):
#   - actor_rollout_ref.model.use_fused_kernels: kept False
#     (RL=True+triton; KD's update_actor_kd doesn't pass return_dict=True so
#      triton backend crashes on forward — bug in the custom KD RPC)
#   - actor_rollout_ref.model.use_liger: kept True (RL=False)
#
# Expected outcome: if val_batch_size is the (main) cause, we'll see
# val/green_auc close to 0.97 at the end instead of 0.85.
set -euo pipefail

DATE=$(date +%Y%m%d%H%M)
PROJECT_ROOT="/home/tianyichen/llm_watermark"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/train_mixed_green3379_initials865_neg1000.parquet}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/validation_mixed_green177_initials177_neg177.parquet}"
STATS_FILE="${STATS_FILE:-${PROJECT_ROOT}/data/initials_icw/leading_space_first_letter_stats.json}"
PY="${PY:-${PROJECT_ROOT}/.venv/bin/python}"

cd "${PROJECT_ROOT}/verl"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"

"${PY}" -m recipe.watermark_kd_ray.main \
    actor_rollout_ref.model.path=Qwen/Qwen3-14B \
    +actor_rollout_ref.model.override_config.rope_scaling.rope_type=yarn \
    +actor_rollout_ref.model.override_config.rope_scaling.factor=4.0 \
    +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=40960 \
    +actor_rollout_ref.model.override_config.max_position_embeddings=131072 \
    actor_rollout_ref.actor.optim.lr=7e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.rollout.prompt_length=60000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=8 \
    data.val_max_samples=-1 \
    data.max_prompt_length=60000 \
    data.max_response_length=600 \
    data.truncation=left \
    watermark.mode=green \
    watermark.stats_file="${STATS_FILE}" \
    watermark.strength=5.0 \
    +watermark.task_strength.green=5.0 \
    +watermark.task_strength.initials=3.0 \
    watermark.ce_loss_weight=0.0 \
    watermark.green_loss_weight=0.0 \
    watermark.kl_biased_ref_actor_weight=1.0 \
    watermark.kl_ref_actor_weight=1.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.distill_topk_biased_ref=1000 \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_tasks=[green,initials] \
    watermark.eval_green_seed=0 \
    watermark.eval_green_fraction=0.25 \
    watermark.eval_initials_seed=0 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=3 \
    trainer.test_freq=50 \
    trainer.save_freq=after_each_epoch \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name=v5b_aligned_eval_${DATE} \
    2>&1 | tee "logs/v5b_aligned_eval_${DATE}.log"
