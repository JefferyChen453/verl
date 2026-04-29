#!/usr/bin/env bash
# V4: per_task normalization (Option 2 fix for V3 dilution). 2906 acrostic
# rows from user_only synthesis (s=8, top_p=0.9, max_tokens=700) + filter at
# hits_z >= 6.0. PURE biased-ref KL (no CE).
#
# History:
#   V1 (1000 rows, clean_v3_noex synth, biased KL only): AUC 0.78 → 0.56 collapse
#   V2 (1000 rows, clean_v3_noex synth, CE + biased KL): AUC 0.78 → 0.78 flat
#   V3 (2906 rows user_only synth, global mode, biased KL only):
#       AUC 0.78 → 0.54 at step 150. Better data still couldn't overcome the
#       16× gradient dilution from dividing by batch_num_tokens.
#   V4 (this): same data as V3 + per_task normalization. Each loss term
#       divides by its applicable token count (= n_pos + n_acr_active for
#       biased-ref). For acrostic-only batch: divisor = total active acr
#       tokens (~336/batch instead of ~5600). Effective gradient strength
#       per active position should be 16× higher than V3 → matches what
#       green/initials get per response token.
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
    +watermark.loss_normalization_mode=per_task \
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
    trainer.experiment_name=acrostics_only_2906_v4_perTask_${DATE} \
    2>&1 | tee "logs/acrostics_only_2906_v4_perTask_${DATE}.log"
