#!/usr/bin/env bash
# On-policy self-distillation training script.
# Mirrors watermark_kd_ray/scripts/run_train_prompt_v2.sh (KL-only, strength=5.0)
# but uses WatermarkOnPolicyTrainer — rollout happens every step, no precomputed responses.

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
export PYTHONWARNINGS="ignore::UserWarning:hydra._internal"

python -m recipe.watermark_onpolicy_ray.main \
    actor_rollout_ref.model.path=Qwen/Qwen3-14B \
    +actor_rollout_ref.model.override_config.rope_scaling.rope_type=yarn \
    +actor_rollout_ref.model.override_config.rope_scaling.factor=4.0 \
    +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=40960 \
    +actor_rollout_ref.model.override_config.max_position_embeddings=131072 \
    actor_rollout_ref.actor.optim.lr=7e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    data.train_files=/home/tianyichen/llm_watermark/verl/data/sft_modified_loss/vblagoje_lfqa/Qwen-Qwen3-14B_strength_3.0_filtered_promptv2_pos_5931_neg_0.parquet \
    data.val_files=/home/tianyichen/llm_watermark/verl/data/sft_modified_loss/vblagoje_lfqa/validation_pos177_neg177_seed0_frac0.25_new_sys_prompt.parquet \
    data.prompt_column=prompt \
    data.train_batch_size=8 \
    data.val_max_samples=-1 \
    data.val_batch_size=128 \
    actor_rollout_ref.rollout.n=1 \
    watermark.ce_loss_weight=0.0 \
    watermark.green_loss_weight=0.0 \
    watermark.kl_biased_ref_actor_weight=0.0 \
    watermark.reverse_kl_biased_ref_actor_weight=1.0 \
    watermark.kl_ref_actor_weight=0.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_wm_seed=0 \
    watermark.eval_wm_fraction=0.25 \
    watermark.strength=5.0 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=2 \
    trainer.test_freq=30 \
    trainer.save_freq=after_each_epoch \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name="onpolicy_strength_5.0_bsz_8__1.0reverse_kl_biased_ref_$(date +%Y%m%d%H%M)"
