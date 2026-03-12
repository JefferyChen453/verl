#!/bin/bash
# From official perf guide (https://verl.readthedocs.io/en/latest/perf/perf_tuning.html):
# Long-context memory-saving options (tune as needed):
#   engine.param_offload=true         - model params to CPU
#   engine.optimizer_offload=true     - optimizer state to CPU
#   model.enable_activation_offload=true - activations to CPU (use with gradient checkpointing)
#   model.enable_gradient_checkpointing=true - recompute activations in backward (default)
#   engine.use_torch_compile=false     - disable torch compile to save GPU memory
#   data.micro_batch_size_per_gpu=1   - min micro-batch for long seqs (fewer seqs = less activation mem)
#   data.train_batch_size=8 or 16     - smaller global batch if OOM
#
#   model.use_remove_padding=true     - sequence packing for Qwen/Llama/Mistral/Gemma (throughput)
#   model.use_liger=true               - LigerKernel for SFT (pip install liger-kernel)
#   engine.forward_prefetch=true       - FSDP1 only: overlap all-gather with compute
#   engine.strategy=fsdp2              - optional: ~7% less GPU memory, PyTorch 2.1+

set -x

DATA_DIR=/home/tianyichen/llm_watermark/verl/data

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.sft_trainer \
    data.train_files=$DATA_DIR/sft_modified_loss/vblagoje_lfqa/Qwen-Qwen3-14B_strength_4.0_filtered_pos_3592_neg_0.parquet \
    data.val_files=$DATA_DIR/sft_modified_loss/vblagoje_lfqa/validation_177.parquet \
    data.val_max_samples=32 \
    +data.prompt_key=prompt \
    +data.response_key=response \
    data.max_length=81920 \
    data.max_token_len_per_gpu=40960 \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=4 \
    data.truncation=error \
    'data.custom_cls.path=pkg://verl.utils.dataset.sft_dataset' \
    data.custom_cls.name=SFTDataset \
    model.path=Qwen/Qwen3-14B \
    model.enable_gradient_checkpointing=true \
    model.enable_activation_offload=false \
    model.use_remove_padding=true \
    model.use_liger=true \
    engine.model_dtype=bf16 \
    engine.strategy=fsdp2 \
    engine.param_offload=false \
    engine.optimizer_offload=false \
    engine.use_torch_compile=false \
    engine.reshard_after_forward=true \
    engine.forward_prefetch=true \
    trainer.project_name=watermark-sft \
    trainer.experiment_name=vblagoje_lfqa_sft \
    trainer.total_epochs=3 \
    'trainer.logger=[console,wandb]' \
    trainer.save_freq=after_each_epoch \
    trainer.test_freq=500 \
    'checkpoint.save_contents=[model,optimizer,extra]' \
    optim.lr=1e-5 \
    engine.ulysses_sequence_parallel_size=2 \
    trainer.n_gpus_per_node=8 \
    $@

