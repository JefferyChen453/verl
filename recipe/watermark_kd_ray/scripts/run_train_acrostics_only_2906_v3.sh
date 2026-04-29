#!/usr/bin/env bash
# V3 retry: 2906 acrostic rows from user_only synthesis (s=8, top_p=0.9,
# max_tokens=700) + filter at hits_z >= 6.0. PURE biased-ref KL (no CE).
#
# History:
#   V1 (1000 rows, clean_v3_noex synth, biased KL only): AUC 0.78 → 0.56 collapse
#   V2 (1000 rows, clean_v3_noex synth, CE + biased KL): AUC 0.78 → 0.78 flat
#   V3a (3165 rows, clean_v3_noex synth, biased KL only): aborted — wrong synth prompt
#   V3 (this): 2906 rows from CORRECT user_only synthesis, biased KL only.
#
# user_only synthesis = pure logit bias on base 14B (no instruction context),
# matching md-pipeline pilot 04-29. Filter at z>=6 yields strong supervision:
# mean hits 12.4/19, recovery 65.5%, full-consume 20.4%, mean n_sent 30.
#
# Per-active-position biased KL: at each sentence-start position in the syn
# response, teacher = softmax(clean_ref + 8.0 * letter_X_mask) where letter X
# is the controller's target letter at that step. Top-K=1000 on biased
# teacher, restricted to english vocab.
set -euo pipefail

DATE=$(date +%Y%m%d%H%M)
PROJECT_ROOT="/home/tianyichen/llm_watermark"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/train_acrostics_only_2906.parquet}"
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
    watermark.strength=8.0 \
    +watermark.task_strength.acrostics=8.0 \
    watermark.ce_loss_weight=0.0 \
    watermark.green_loss_weight=0.0 \
    watermark.kl_biased_ref_actor_weight=1.0 \
    watermark.kl_ref_actor_weight=0.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.distill_topk_biased_ref=1000 \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_tasks=[acrostics] \
    +watermark.eval_acrostic_seed=0 \
    +watermark.eval_acrostic_length=18 \
    +watermark.acrostics_n_resample=200 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=4 \
    trainer.test_freq=30 \
    trainer.save_freq=after_each_epoch \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name=acrostics_only_2906_v3_userOnly_synth_${DATE} \
    2>&1 | tee "logs/acrostics_full3165_v3_biased_ref_${DATE}.log"
