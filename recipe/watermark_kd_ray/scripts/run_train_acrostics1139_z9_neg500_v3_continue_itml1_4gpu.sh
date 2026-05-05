#!/usr/bin/env bash
# V3 4-GPU itml-1 CONTINUATION fork: resume from V3 KD epoch 3 ckpt for +3 more
# epochs (total = 6 KD epochs of acrostics1139_z9_neg500). User reasoning:
# V3 has ~50% of V2's data so 6 epochs of V3 ≈ 3 epochs of V2 in gradient steps.
#
# Init checkpoint (already converted hf_model on itml-1):
#   /home/tianyichen/llm_watermark/verl/checkpoints/watermark-kd-ray/
#     acrostics1139_z9_neg500_v3_s8_202605041115_itml1_4gpu/global_step_612/hf_model
#
# Diffs from run_train_acrostics1139_z9_neg500_v3_itml1_4gpu.sh:
#   1. actor_rollout_ref.model.path: Qwen/Qwen3-14B → V3 KD ep3 hf_model path
#   2. experiment_name suffix: v3_s8_${DATE} → v3_continue_${DATE}
#   All other hyperparams identical (lr=7e-6, strength=8, kl weights 1.0/1.0,
#   per_task normalization, lcs detector, max_actor_ckpt_to_keep=1, epochs=3,
#   yarn rope_scaling overrides — yarn already baked into ckpt config but
#   passing flags is harmless and matches base run).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATE=$(date +%Y%m%d%H%M)
PROJECT_ROOT="/home/tianyichen/llm_watermark"
INIT_CKPT="${PROJECT_ROOT}/verl/checkpoints/watermark-kd-ray/acrostics1139_z9_neg500_v3_s8_202605041115_itml1_4gpu/global_step_612/hf_model"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/train_acrostics1139_z9_neg500.parquet}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/validation_acrostic177_neg177.parquet}"
STATS_FILE="${STATS_FILE:-${PROJECT_ROOT}/data/initials_icw/leading_space_first_letter_stats.json}"
LETTER_MAP="${LETTER_MAP:-${PROJECT_ROOT}/data/acrostic/letter_to_token_ids_qwen3_14b.json}"
PY="${PY:-${PROJECT_ROOT}/.venv/bin/python}"

cd "${PROJECT_ROOT}/verl"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"

"${PY}" -m recipe.watermark_kd_ray.main \
    actor_rollout_ref.model.path="${INIT_CKPT}" \
    actor_rollout_ref.actor.optim.lr=7e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.rollout.prompt_length=60000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
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
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs="${EPOCHS:-3}" \
    trainer.test_freq=30 \
    trainer.save_freq=after_each_epoch \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name=acrostics1139_z9_neg500_v3_continue_${DATE}_itml1_4gpu \
    2>&1 | tee "logs/acrostics1139_z9_neg500_v3_continue_${DATE}_itml1_4gpu.log"
