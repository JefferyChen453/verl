#!/bin/bash
# V1 + privileged-GT distillation paradigm.
#
# Loss: clean forward KL only — KL(D_ref ‖ D_actor) on refresponse tokens.
# Ref input  = chat_template(wm_system, <example>{gt}</example> + query)
# Actor input = chat_template(wm_system, query)        # actor never sees GT
#
# refresponse is generated offline by the ref model under the privileged
# (wm + GT example) context, so it carries an effective watermark signal
# that the actor learns to imitate via distribution matching.
#
# Training data is produced by:
#   data_process/prepare_v1_paradigm_data.py     -> pair samples with GT
#   run_generate_v1_refresponse_vllm.py          -> generate refresponse
#   data_process/v1_jsonl_to_parquet.py          -> pack into parquet
#
# Validation reuses the existing offline-format parquet (whose system
# prompt template matches the V1 actor input exactly).

set -e
DATE=$(date +%Y%m%d%H%M)

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
export PYTHONWARNINGS="ignore::UserWarning:hydra._internal"

# >>> Path to V1 paradigm parquet — update after data pipeline runs <<<
TRAIN_PARQUET="${TRAIN_PARQUET:-/home/tianyichen/llm_watermark/verl/data/v1_paradigm/Qwen-Qwen3-14B_v1_strength_3.0.parquet}"
VAL_PARQUET="${VAL_PARQUET:-/home/tianyichen/llm_watermark/verl/data/sft_modified_loss/vblagoje_lfqa/validation_pos177_neg177_seed0_frac0.25_new_sys_prompt.parquet}"

python -m recipe.watermark_kd_ray.main \
    actor_rollout_ref.model.path=Qwen/Qwen3-14B \
    +actor_rollout_ref.model.override_config.rope_scaling.rope_type=yarn \
    +actor_rollout_ref.model.override_config.rope_scaling.factor=4.0 \
    +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=40960 \
    +actor_rollout_ref.model.override_config.max_position_embeddings=131072 \
    actor_rollout_ref.actor.optim.lr=7e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    +data.ref_prompt_column=prompt_ref \
    data.train_batch_size=8 \
    data.val_max_samples=-1 \
    data.val_batch_size=128 \
    watermark.ce_loss_weight=0.0 \
    watermark.green_loss_weight=0.0 \
    watermark.kl_biased_ref_actor_weight=0.0 \
    watermark.reverse_kl_biased_ref_actor_weight=0.0 \
    watermark.kl_ref_actor_weight=1.0 \
    watermark.reverse_kl_ref_actor_weight=0.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.reverse_kl_biased_actor_actor_weight=0.0 \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_green_seed=0 \
    watermark.eval_green_fraction=0.25 \
    watermark.strength=3.0 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=2 \
    trainer.test_freq=30 \
    trainer.save_freq=after_each_epoch \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name="v1_privileged_gt_strength_3.0_bsz_8__1.0kl_ref_actor_${DATE}"
