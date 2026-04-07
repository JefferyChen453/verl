"""
WatermarkPromptDataset — prompt-only dataset for on-policy self-distillation.

Reads the same parquet schema as WatermarkKDDataset but only uses the prompt side.
Each item carries:
  - input_ids / attention_mask / position_ids  (tensors, left-padded to max_prompt_length)
  - raw_prompt_ids                              (list[int], actor rollout prompt for vLLM agent loop)
  - raw_prompt_ids_ref                          (list[int], clean prompt for ref forward)
  - raw_prompt                                 (str, for async postprocessor logging)
  - wm_seed / wm_fraction                      (scalar ints/floats, for green-mask construction)

Actor gets the in-context watermark prompt (with green token list).
Ref gets the clean prompt (no green token list) — same design as offline KD.

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
        input_ids           (max_prompt_length,)  long  — left-padded actor prompt token ids
        attention_mask      (max_prompt_length,)  long  — 1 for real tokens, 0 for left-pad
        position_ids        (max_prompt_length,)  long  — 0-indexed within real tokens
        raw_prompt_ids      list[int]             — actor prompt ids (in-context wm) for vLLM
        raw_prompt_ids_ref  list[int]             — ref prompt ids (clean, no green list)
        raw_prompt          str                   — decoded actor prompt string for logging
        wm_seed             int                   — watermark seed for green-mask construction
        wm_fraction         float                 — green-token fraction
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
        self.prompts_ref = df["prompt_no_incontext_wm"].tolist()
        self.wm_seeds = df["seed"].tolist()
        self.wm_fractions = df["fraction"].tolist()

    def __len__(self):
        return len(self.prompts)

    def _tokenize_prompt(self, prompt_str: str) -> torch.Tensor:
        """Tokenize a prompt string, truncating to max_prompt_length (keep right side)."""
        tok = self.tokenizer
        raw_ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if raw_ids.shape[0] > self.max_prompt_length:
            raw_ids = raw_ids[-self.max_prompt_length:]
        return raw_ids

    def __getitem__(self, item):
        prompt_str = self.prompts[item]
        prompt_ref_str = self.prompts_ref[item]

        # Tokenize both prompts (chat-template-formatted strings)
        raw_ids = self._tokenize_prompt(prompt_str)          # actor: in-context wm prompt
        raw_ids_ref = self._tokenize_prompt(prompt_ref_str)  # ref:   clean prompt

        prompt_len = raw_ids.shape[0]

        # Left-pad actor prompt to max_prompt_length (for batched vLLM rollout)
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
            "raw_prompt_ids": raw_ids.tolist(),      # list[int] → non_tensor_batch
            "raw_prompt_ids_ref": raw_ids_ref.tolist(),  # list[int] → non_tensor_batch
            "raw_prompt": prompt_str,                # str → non_tensor_batch
            "wm_seed": int(self.wm_seeds[item]),
            "wm_fraction": float(self.wm_fractions[item]),
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
    NON_TENSOR_KEYS = ("raw_prompt_ids", "raw_prompt_ids_ref", "raw_prompt", "wm_seed", "wm_fraction")

    def __call__(self, batch: list[dict]) -> dict:
        result = {}

        # Stack tensor keys
        for key in self.TENSOR_KEYS:
            result[key] = torch.stack([item[key] for item in batch])

        # Box non-tensor keys as object numpy arrays
        for key in self.NON_TENSOR_KEYS:
            result[key] = np.array([item[key] for item in batch], dtype=object)

        return result
