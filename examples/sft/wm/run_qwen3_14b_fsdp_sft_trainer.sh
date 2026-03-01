#!/bin/bash
set -x


# LFQA parquet (prompt/response columns, flat structure)
DATA_DIR=/home/tianyichen/llm_watermark/verl/data

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_DIR/sft/Qwen-Qwen3-32B_LFQA_filtered_data_pos_1142_neg_680.parquet \
    data.val_files=$DATA_DIR/sft/Qwen-Qwen3-32B_LFQA_filtered_data_pos_1142_neg_680.parquet \
    data.val_max_samples=32 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=85000 \
    data.truncation=error \
    model.partial_pretrain=Qwen/Qwen3-14B \
    model.fsdp_config.model_dtype=bf16 \
    model.fsdp_config.offload_params=True \
    trainer.project_name=watermark-sft \
    trainer.experiment_name=lfqa-qwen3-14b \
    trainer.total_epochs=2 \
    trainer.logger='["console","wandb"]' \
    trainer.save_freq=after_each_epoch \
    trainer.test_freq=5000 \
    trainer.checkpoint.save_contents='["model"]' \
    optim.lr=1e-5 \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true \
    $@


# torchrun --standalone --nnodes=1 --nproc_per_node=8 \
#     -m verl.trainer.fsdp_sft_trainer \
#     data.train_files=$DATA_DIR/sft/Qwen-Qwen3-32B_OpenGen_filtered_data_pos_5640_neg_1443.parquet \
#     data.val_files=$DATA_DIR/sft/Qwen-Qwen3-32B_LFQA_filtered_data_pos_1142_neg_680.parquet \
#     data.val_max_samples=32 \
#     data.prompt_key=prompt \
#     data.response_key=response \
#     data.train_batch_size=32 \
#     data.micro_batch_size_per_gpu=4 \
#     data.max_length=85000 \
#     data.truncation=error \
#     model.partial_pretrain=Qwen/Qwen3-14B \
#     model.fsdp_config.model_dtype=bf16 \
#     model.fsdp_config.offload_params=True \
#     trainer.project_name=watermark-sft \
#     trainer.experiment_name=OpenGen-qwen3-14b \
#     trainer.total_epochs=2 \
#     trainer.logger='["console","wandb"]' \
#     trainer.save_freq=after_each_epoch \
#     trainer.test_freq=50 \
#     trainer.checkpoint.save_contents='["model"]' \
#     optim.lr=1e-5 \
#     ulysses_sequence_parallel_size=8 \
#     use_remove_padding=true \
#     $@


# torchrun --standalone --nnodes=1 --nproc_per_node=8 \
#     -m verl.trainer.fsdp_sft_trainer \
#     data.train_files=$DATA_DIR/sft/Qwen-Qwen3-32B_combined_filtered_data_pos_6782_neg_2123.parquet \
#     data.val_files=$DATA_DIR/sft/Qwen-Qwen3-32B_combined_filtered_data_pos_6782_neg_2123.parquet \
#     data.val_max_samples=32 \
#     data.prompt_key=prompt \
#     data.response_key=response \
#     data.train_batch_size=32 \
#     data.micro_batch_size_per_gpu=4 \
#     data.max_length=85000 \
#     data.truncation=error \
#     model.partial_pretrain=Qwen/Qwen3-14B \
#     model.fsdp_config.model_dtype=bf16 \
#     model.fsdp_config.offload_params=True \
#     trainer.project_name=watermark-sft \
#     trainer.experiment_name=OpenGen-qwen3-14b \
#     trainer.total_epochs=2 \
#     trainer.logger='["console","wandb"]' \
#     trainer.save_freq=after_each_epoch \
#     trainer.test_freq=50 \
#     trainer.checkpoint.save_contents='["model"]' \
#     optim.lr=1e-5 \
#     ulysses_sequence_parallel_size=8 \
#     use_remove_padding=true \
#     $@
