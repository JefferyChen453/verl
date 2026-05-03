#!/usr/bin/env bash
# V5a-lcs + 500 negative anchor samples.
#
# Difference from run_train_acrostics_only_2906_v5a_lcs_detector.sh:
#   1. TRAIN_FILE = train_acrostics2906_neg500.parquet (acrostic 2906 + neg 500
#      randomly downsampled from train_mixed_*'s neg=1000, seed=42).
#   2. kl_ref_actor_weight = 1.0  (was 0.0). Without this, neg samples don't
#      contribute to loss at all — clean-ref KL bucket is gated by weight > 0.
#      Now KL(D_ref ‖ D_actor) on neg responses provides a quality anchor:
#      "when there is no in-context watermark, stay close to base model".
#   3. EXP_NAME bumped.
#
# Loss routing reminder (per_task mode):
#   - acrostic samples (~85% of batch) → forward biased-ref KL on letter
#     positions, normalized by (n_pos + n_acr_active).
#   - neg samples (~15% of batch) → forward clean-ref KL KL(D_ref ‖ D_actor)
#     on full response, normalized by n_neg.
# Two terms add to the total loss as a weighted sum (single mean per term),
# NOT a per-task average of per-task means. See loss.py:449-507.
set -euo pipefail

DATE=$(date +%Y%m%d%H%M)
PROJECT_ROOT="/home/tianyichen/llm_watermark"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/train_acrostics2906_neg500.parquet}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/validation_4task_green177_initials177_neg177_acrostics177.parquet}"
STATS_FILE="${STATS_FILE:-${PROJECT_ROOT}/data/initials_icw/leading_space_first_letter_stats.json}"
LETTER_MAP="${LETTER_MAP:-${PROJECT_ROOT}/data/acrostic/letter_to_token_ids_qwen3_14b.json}"
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
    watermark.mode=acrostics \
    watermark.stats_file="${STATS_FILE}" \
    +watermark.acrostic_letter_token_ids="${LETTER_MAP}" \
    watermark.strength=10.0 \
    +watermark.task_strength.acrostics=10.0 \
    watermark.ce_loss_weight=0.0 \
    watermark.green_loss_weight=0.0 \
    watermark.kl_biased_ref_actor_weight=1.0 \
    watermark.kl_ref_actor_weight=1.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.distill_topk_biased_ref=1000 \
    +watermark.loss_normalization_mode=per_task \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_tasks=[acrostics] \
    +watermark.eval_acrostic_seed=0 \
    +watermark.eval_acrostic_length=18 \
    +watermark.acrostics_n_resample=1000 \
    +watermark.acrostics_detector_kind=lcs \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs="${EPOCHS:-2}" \
    trainer.test_freq=30 \
    trainer.save_freq=after_each_epoch \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name=acrostics2906_neg500_${DATE} \
    2>&1 | tee "logs/acrostics2906_neg500_${DATE}.log"
