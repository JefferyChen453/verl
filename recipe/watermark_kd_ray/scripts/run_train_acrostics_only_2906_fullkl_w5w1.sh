#!/usr/bin/env bash
# Acrostic full-response KL — weighted variant (active=5×, inactive=1×).
#
# Background: the unweighted fullkl run (active_weight=1, inactive_weight=1)
# plateau'd at chance AUC because the active letter-bias signal (~18 tokens/
# sample) was diluted ~30× by the inactive ref-anchor (~512 tokens/sample)
# under the single-bucket weighted mean.
#
# This variant amplifies active 5× under a proper weighted mean:
#   numerator = 5 * Σ_active KL(ref+bias‖actor) + 1 * Σ_inactive KL(ref‖actor)
#   denom    = n_pos + 5 * n_acr_active + 1 * n_acr_inactive
# Effective active per-token gradient grows ~5× relative to the inactive
# anchor, giving the letter-bias signal more headroom while keeping the
# inactive anchor as a (weaker) regularizer.
#
# Settings: identical to V5a-lcs-detector except for fullkl + 5/1 weights.
set -euo pipefail

DATE=$(date +%Y%m%d%H%M)
PROJECT_ROOT="/home/tianyichen/llm_watermark"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/train_acrostics_only_2906_sparse.parquet}"
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
    watermark.kl_ref_actor_weight=0.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.distill_topk_biased_ref=1000 \
    +watermark.loss_normalization_mode=per_task \
    watermark.acrostic_kl_full_response=true \
    watermark.acrostic_active_weight=5.0 \
    watermark.acrostic_inactive_weight=1.0 \
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
    trainer.experiment_name=acrostics_only_2906_fullkl_w5w1_${DATE} \
    2>&1 | tee "logs/acrostics_only_2906_fullkl_w5w1_${DATE}.log"
