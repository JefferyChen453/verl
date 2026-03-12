#!/bin/bash
# Watermark Distillation Training
#
# Trains a model to follow in-context watermark instructions via knowledge distillation:
#   loss = L_CE + λ1 * L_green + λ2 * L_KL
#
# Two models are loaded: actor (trainable) + ref (frozen, logit-biased teacher).
# Both are FSDP2-sharded across GPUs. Ref model can be a different scale.
#
# Per-sample watermark seed and fraction are read from the parquet data columns
# "seed" and "fraction". Green masks are cached per (seed, fraction) pair.

set -x

DATA_DIR=/home/tianyichen/llm_watermark/verl/data

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m recipe.watermark_distill.trainer \
    data.train_files=$DATA_DIR/sft_modified_loss/Qwen-Qwen3-32B_combined_LFQA_1893_OpenGen_5071.parquet \
    data.val_max_samples=32 \
    +data.prompt_key=prompt \
    +data.response_key=response \
    data.max_length=81920 \
    data.max_token_len_per_gpu=20480 \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=1 \
    data.truncation=error \
    +data.shuffle=true \
    +data.seed=42 \
    'data.custom_cls.path=pkg://recipe.watermark_distill.dataset' \
    data.custom_cls.name=WatermarkSFTDataset \
    model.path=Qwen/Qwen3-14B \
    model.enable_gradient_checkpointing=true \
    model.enable_activation_offload=false \
    model.use_remove_padding=true \
    model.use_liger=true \
    ref_model.path=Qwen/Qwen3-32B \
    ref_model.param_offload=true \
    engine.model_dtype=bf16 \
    engine.strategy=fsdp2 \
    engine.param_offload=false \
    engine.optimizer_offload=false \
    engine.use_torch_compile=false \
    engine.reshard_after_forward=true \
    engine.forward_prefetch=true \
    trainer.project_name=watermark-distill \
    trainer.experiment_name=lfqa-qwen3-14b-sft \
    trainer.total_epochs=3 \
    'trainer.logger=[console,wandb]' \
    trainer.save_freq=after_each_epoch \
    trainer.test_freq=50 \
    'checkpoint.save_contents=[model,optimizer,extra]' \
    optim.lr=1e-5 \
    engine.ulysses_sequence_parallel_size=4 \
    trainer.n_gpus_per_node=8 \
    watermark.strength=2.0 \
    watermark.only_english=false \
    watermark.green_loss_weight=0.2 \
    watermark.kl_loss_weight=0.4 \
    $@
