#!/bin/bash

# usage: python -m verl.model_merger merge [-h] --backend {fsdp,megatron} [--local_dir LOCAL_DIR] [--tie-word-embedding] [--is-value-model] [--use_cpu_initialization] [--target_dir TARGET_DIR]
#                      [--hf_upload_path HF_UPLOAD_PATH] [--private]

# options:
# -h, --help            show this help message and exit
# --backend {fsdp,megatron}
#                         The backend of the model
# --local_dir LOCAL_DIR
#                         Path to the saved model checkpoints
# --tie-word-embedding  Whether to tie word embedding weights (currently only Megatron supported)
# --is-value-model      Whether the model is a value model (currently only Megatron supported)
# --use_cpu_initialization
#                         Whether to use CPU initialization for the model. This is useful for large models that cannot fit into GPU memory during initialization.
# --target_dir TARGET_DIR
#                         Directory to save the merged huggingface model
# --hf_upload_path HF_UPLOAD_PATH
#                         Hugging Face repository ID to upload the model
# --private             Whether to upload the model to a private Hugging Face repository

CKPT_DIR=/home/tianyichen/llm_watermark/verl/checkpoints


python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ${CKPT_DIR}/watermark-sft/LFQA+OpenGen-qwen3-14b/global_step_556 \
    --target_dir ${CKPT_DIR}/watermark-sft/LFQA+OpenGen-qwen3-14b/global_step_556/hf_model

