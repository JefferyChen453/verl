#!/usr/bin/env bash
# V5b: V5a config + LCS detector + n_resample=1000 (vs V5a's hits-z, 200).
#
# Same training side as V5a (sparse parquet, strength=10, per_task norm) —
# the ONLY change is the val/reward acrostic detector:
#   acrostics_detector_kind: hits → lcs
#   acrostics_n_resample:    200  → 1000
#
# Why LCS:
#   - hits-z's fail_streak=3 skip mechanism is brittle: 3 noise letters in
#     a row trigger skip, dropping subsequent real matches. Empirical 12-
#     char insertion attack on filtered KD pilot: hits-z AUC drops 6.5pp,
#     LCS-z drops only 1.0pp.
#   - SW (Smith-Waterman) has gap penalty that drowns spread-out hits.
#   - LCS = pure subsequence DP, no skip/penalty. Mentor-recommended.
#
# Why n_resample=1000:
#   - LCS null mean is higher than hits null (richer multiset → more
#     ordered subsequences). Tighter sigma needed for sensitive z.
#   - Already used at filter time; matches.
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
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_tasks=[acrostics] \
    +watermark.eval_acrostic_seed=0 \
    +watermark.eval_acrostic_length=18 \
    +watermark.acrostics_n_resample=1000 \
    +watermark.acrostics_detector_kind=lcs \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=4 \
    trainer.test_freq=30 \
    trainer.save_freq=after_each_epoch \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name=acrostics_only_2906_v5b_sparse_s10_lcs1000_${DATE} \
    2>&1 | tee "logs/acrostics_only_2906_v5b_sparse_s10_lcs1000_${DATE}.log"
