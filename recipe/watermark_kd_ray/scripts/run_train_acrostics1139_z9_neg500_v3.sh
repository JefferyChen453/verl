#!/usr/bin/env bash
# V3: high-quality acrostic dataset (z>=9 only) + 500 neg.
# Diff from V2 (acrostics2906_neg500_v2_s8):
#   1. TRAIN_FILE → train_acrostics1139_z9_neg500.parquet
#      - 1139 acrostic samples filtered at hits_z >= 9.0
#        (round1 623 from 6334-prompt pool + round2 516 from new 5000-prompt
#         pool synthesized 2026-05-04 on itml-1; prefixes disjoint from each
#         other, from the previous 2906 acrostic, and from the 1000 neg)
#      - Quality vs V2's 2906 z>=6: hits_z mean 10.14 (vs 7.91), recovery
#        15.26/18 (vs 12.42/18), full-consume 70.7% (vs 19%)
#      - 500 neg sampled (seed=20260504) from the original 1000 neg pool
#   2. Total rows: 1639 (V2: 3406) → ~50% volume; epoch length will halve
#   3. experiment_name → acrostics1139_z9_neg500_v3_s8_<DATE>
#
# All other hyperparams identical to V2 (lr=7e-6, strength=8, kl weights
# 1.0/1.0, per_task normalization, lcs detector, 8 GPU, epochs=3 default).
set -euo pipefail

DATE=$(date +%Y%m%d%H%M)
PROJECT_ROOT="/home/tianyichen/llm_watermark"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/train_acrostics1139_z9_neg500.parquet}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/validation_acrostic177_neg177.parquet}"
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
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.rollout.prompt_length=60000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
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
    watermark.strength=8.0 \
    +watermark.task_strength.acrostics=8.0 \
    watermark.ce_loss_weight=0.0 \
    watermark.green_loss_weight=0.0 \
    watermark.kl_biased_ref_actor_weight=1.0 \
    watermark.kl_ref_actor_weight=1.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.distill_topk_biased_ref=1000 \
    +watermark.loss_normalization_mode=per_task \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_tasks=[acrostics] \
    +watermark.acrostics_n_resample=1000 \
    +watermark.acrostics_detector_kind=lcs \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs="${EPOCHS:-3}" \
    trainer.test_freq=30 \
    trainer.save_freq=after_each_epoch \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name=acrostics1139_z9_neg500_v3_s8_${DATE} \
    2>&1 | tee "logs/acrostics1139_z9_neg500_v3_s8_${DATE}.log"
