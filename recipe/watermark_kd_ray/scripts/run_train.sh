DATE=$(date +%Y%m%d%H%M)

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
    data.train_files=/home/tianyichen/llm_watermark/verl/data/sft_modified_loss/vblagoje_lfqa/Qwen-Qwen3-14B_strength_4.0_filtered_pos_3592_neg_0.parquet \
    data.val_files=/home/tianyichen/llm_watermark/verl/data/sft_modified_loss/vblagoje_lfqa/validation_177.parquet \
    data.train_batch_size=8 \
    data.val_max_samples=64 \
    data.val_batch_size=32 \
    watermark.green_loss_weight=0.2 \
    watermark.kl_loss_weight=0.0 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=3 \
    trainer.test_freq=50 \
    trainer.save_freq=250 \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name=only_sft${DATE}
