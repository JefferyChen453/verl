#!/usr/bin/env bash
# V2 of acrostics+neg run. Diffs from v1 (acrostics2906_neg500_202605012052):
#   1. strength: 10 → 8 (match training-data synthesis strength s=8 from
#      04-29 markdown-aware pipeline; teacher logits should mirror data dist)
#   2. epochs: 2 → 3
#   3. val parquet → validation_acrostic177_neg177.parquet (drop green/initials
#      since eval_tasks only uses [acrostics] anyway; saves val rollout time)
#   4. param_offload, optimizer_offload OFF for both actor + ref (was true);
#      keep optimizer states + weights on GPU for speed
#   5. rollout.gpu_memory_utilization 0.9 → 0.4 (bumped 2026-05-03 from
#      original 0.2 after wandb full_v1 analysis showed memory was severely
#      underutilized at 0.2 and 0.25; for KD val rollout alone, 0.4 gives
#      vLLM 38GB which is more than enough for 354 short val prompts and
#      leaves ~26GB headroom on 96GB GPU after actor+optim ~30GB).
#
# Loss routing (unchanged):
#   - acrostic ~85% → forward biased-ref KL on letter positions, denom n_acr_active
#   - neg ~15% → KL(D_ref ‖ D_actor) full response, denom n_neg
#   - per_task mode, single mean per term, weighted sum.
set -euo pipefail

DATE=$(date +%Y%m%d%H%M)
PROJECT_ROOT="/home/tianyichen/llm_watermark"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/train_acrostics2906_neg500.parquet}"
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
    # eval_acrostic_seed/length removed 2026-05-02 — per-sample target now read from
    # data.non_tensor_batch['acrostic_target'] in val parquet (no code-level fallback).
    +watermark.acrostics_n_resample=1000 \
    +watermark.acrostics_detector_kind=lcs \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs="${EPOCHS:-3}" \
    trainer.test_freq=30 \
    trainer.save_freq=after_each_epoch \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name=acrostics2906_neg500_v2_s8_${DATE} \
    2>&1 | tee "logs/acrostics2906_neg500_v2_s8_${DATE}.log"
