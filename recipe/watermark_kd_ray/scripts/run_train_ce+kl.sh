# Explicitly set CUDA arch so megatron's JIT compile doesn't fail when
# Ray worker processes can't auto-detect GPU architecture at import time.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"

python -m recipe.watermark_kd_ray.main \
    actor_rollout_ref.model.path=Qwen/Qwen3-14B \
    +actor_rollout_ref.model.override_config.rope_scaling.rope_type=yarn \
    +actor_rollout_ref.model.override_config.rope_scaling.factor=4.0 \
    +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=40960 \
    +actor_rollout_ref.model.override_config.max_position_embeddings=131072 \
    actor_rollout_ref.actor.optim.lr=7e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    data.train_files=/home/tianyichen/llm_watermark/verl/data/sft_modified_loss/vblagoje_lfqa/Qwen-Qwen3-14B_strength_2.0_filtered_pos_9935_neg_0.parquet \
    data.val_files=/home/tianyichen/llm_watermark/verl/data/sft_modified_loss/vblagoje_lfqa/validation_pos64_neg64_align.parquet \
    data.train_batch_size=8 \
    data.val_max_samples=-1 \
    data.val_batch_size=64 \
    watermark.ce_loss_weight=1.0 \
    watermark.green_loss_weight=0.4 \
    watermark.kl_biased_ref_actor_weight=0.0 \
    watermark.kl_ref_actor_weight=0.2 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.gradient_accumulation_steps=1 \
    watermark.strength=2.0 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_training_steps=200 \
    trainer.test_freq=20 \
    trainer.save_freq=-1 \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name="bsz_8__1.0ce+0.4green+0.2kl_ref$(date +%Y%m%d%H%M)"
