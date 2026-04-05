"""
WatermarkPromptDataset — prompt-only dataset for on-policy self-distillation.

Reads the same parquet schema as WatermarkKDDataset but only uses the prompt side.
Each item carries:
  - input_ids / attention_mask / position_ids  (tensors, left-padded to max_prompt_length)
  - raw_prompt_ids                              (list[int], for raw_prompt_ids_agent loop)
  - raw_prompt                                 (str, for async postprocessor logging)
  - wm_seed / wm_fraction                      (scalar ints/floats, for green-mask construction)

WatermarkPromptCollator converts these into DataProto-compatible dicts:
  - torch.Tensor  keys → DataProto.batch  (via DataProto.from_single_dict)
  - np.ndarray    keys → DataProto.non_tensor_batch
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class WatermarkPromptDataset(Dataset):
    """
    Dataset yielding only the prompt side of each watermark training sample.

    __getitem__ returns a dict with:
        input_ids        (max_prompt_length,)  long   — left-padded prompt token ids
        attention_mask   (max_prompt_length,)  long   — 1 for real tokens, 0 for left-pad
        position_ids     (max_prompt_length,)  long   — 0-indexed within real tokens
        raw_prompt_ids   list[int]             — unpadded prompt ids for vLLM agent loop
        raw_prompt       str                   — decoded prompt string for logging
        wm_seed          int                   — watermark seed for green-mask construction
        wm_fraction      float                 — green-token fraction
    """

    def __init__(self, parquet_files, tokenizer, config, max_samples: int = -1):
        import pandas as pd
        from verl.utils.fs import copy_to_local

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.max_prompt_length = int(config.get("max_prompt_length", 131000))
        # Which column to use as the prompt (default: "prompt" = in-context watermark prompt)
        prompt_column = config.get("prompt_column", "prompt")

        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]

        dfs = []
        for pf in parquet_files:
            local = copy_to_local(pf, verbose=True)
            dfs.append(pd.read_parquet(local))
        df = pd.concat(dfs, ignore_index=True)

        total = len(df)
        if max_samples > 0 and max_samples < total:
            rng = np.random.default_rng(42)
            idx = rng.choice(total, size=max_samples, replace=False)
            df = df.iloc[idx.tolist()]
            print(f"WatermarkPromptDataset: selected {max_samples} / {total} samples")
        print(f"WatermarkPromptDataset: {len(df)} samples loaded from column '{prompt_column}'")

        self.prompts = df[prompt_column].tolist()
        self.wm_seeds = df["seed"].tolist()
        self.wm_fractions = df["fraction"].tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        prompt_str = self.prompts[item]
        tok = self.tokenizer

        # Tokenize prompt (already a chat-template-formatted string)
        raw_ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # Truncate if necessary (keep right side — last max_prompt_length tokens)
        if raw_ids.shape[0] > self.max_prompt_length:
            raw_ids = raw_ids[-self.max_prompt_length:]

        prompt_len = raw_ids.shape[0]

        # Left-pad to max_prompt_length
        pad_len = self.max_prompt_length - prompt_len
        if pad_len > 0:
            pad = torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            input_ids = torch.cat([pad, raw_ids])
        else:
            input_ids = raw_ids

        attention_mask = torch.zeros(self.max_prompt_length, dtype=torch.long)
        attention_mask[pad_len:] = 1

        # position_ids: 0-indexed within real tokens (padded positions get 0)
        position_ids = torch.zeros(self.max_prompt_length, dtype=torch.long)
        position_ids[pad_len:] = torch.arange(prompt_len, dtype=torch.long)

        return {
            "input_ids": input_ids,           # torch.Tensor → batch
            "attention_mask": attention_mask,  # torch.Tensor → batch
            "position_ids": position_ids,      # torch.Tensor → batch
            "raw_prompt_ids": raw_ids.tolist(),  # list[int] → non_tensor_batch (via collator)
            "raw_prompt": prompt_str,            # str → non_tensor_batch (via collator)
            "wm_seed": int(self.wm_seeds[item]),        # int → non_tensor_batch
            "wm_fraction": float(self.wm_fractions[item]),  # float → non_tensor_batch
        }


class WatermarkPromptCollator:
    """
    Collator for WatermarkPromptDataset.

    Produces a dict that DataProto.from_single_dict can parse:
      - torch.Tensor  values → DataProto.batch
      - np.ndarray    values → DataProto.non_tensor_batch

    Tensor keys are already padded by the dataset (left-pad to max_prompt_length),
    so we just stack them.  Non-tensor keys are boxed into dtype=object numpy arrays.
    """

    TENSOR_KEYS = ("input_ids", "attention_mask", "position_ids")
    NON_TENSOR_KEYS = ("raw_prompt_ids", "raw_prompt", "wm_seed", "wm_fraction")

    def __call__(self, batch: list[dict]) -> dict:
        result = {}

        # Stack tensor keys
        for key in self.TENSOR_KEYS:
            result[key] = torch.stack([item[key] for item in batch])

        # Box non-tensor keys as object numpy arrays
        for key in self.NON_TENSOR_KEYS:
            result[key] = np.array([item[key] for item in batch], dtype=object)

        return result
