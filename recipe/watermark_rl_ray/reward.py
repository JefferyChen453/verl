"""Per-sample watermark z-score reward function for Stage 2 RL training.

Each sample carries its own (task, wm_seed, wm_fraction). We build a detector
per (task, seed, fraction) on demand, cache it, and compute z-score on the
rollout response tokens. The scalar z is placed at the last response position
(token-level reward convention in verl/DAPO).

Input (DataProto):
  data.batch["responses"]            (B, T)  long  — rollout response tokens
  data.non_tensor_batch["task"]      (B,)    object — per-sample task label
  data.non_tensor_batch["wm_seed"]   (B,)    object — per-sample seed
  data.non_tensor_batch["wm_fraction"] (B,)  object — per-sample fraction/gamma

Output:
  {
      "reward_tensor": torch.Tensor  (B, T) with z-score at last response position,
      "reward_extra_info": {
          "z_score":           list[float],  # per-sample scalar z (routed)
          "z_score_valid":     list[bool],   # False if response too short
          "z_score_{task}":    list[float],  # per-task detector z (NaN on task mismatch)
          "response_len":      list[int],
      }
  }
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import numpy as np
import torch


# Ensure project root on sys.path for gptwm imports when invoked as Ray remote
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _round_frac(f: float, decimals: int = 6) -> float:
    return float(round(float(f), decimals))


class PerSampleWatermarkZScoreRewardFn:
    """Reward function that builds a detector per-sample from (task, seed, fraction).

    Caches detectors by (task, seed, rounded_fraction) to amortize construction
    cost. Assumes the tokenizer/model_config are the same across samples.
    """

    MIN_LEN = 200

    def __init__(
        self,
        tokenizer,
        model_config,
        strength: float = 2.0,
        only_english: bool = True,
        stats_file: str = "data/initials_icw/leading_space_first_letter_stats.json",
        active_tasks: Optional[list] = None,
        acrostics_target: str = "asdf",
        acrostics_n_resample: int = 200,
    ):
        assert tokenizer is not None
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.model_config = model_config
        self.strength = float(strength)
        self.only_english = bool(only_english)
        self.stats_file = stats_file
        self.active_tasks = list(active_tasks) if active_tasks else ["green", "initials"]
        self.acrostics_target = acrostics_target
        self.acrostics_n_resample = int(acrostics_n_resample)

        # Detector cache: {(task, seed, frac_key): detector}
        self._cache: Dict[tuple, object] = {}

    def _get_detector(self, task: str, seed: int, fraction: float):
        frac_key = _round_frac(fraction)
        key = (task, int(seed), frac_key)
        if key in self._cache:
            return self._cache[key]

        if task == "green":
            from gptwm import GPTWatermarkDetector
            det = GPTWatermarkDetector(
                fraction=fraction,
                strength=self.strength,
                vocab_size=self.tokenizer.vocab_size,
                model_emb_length=self.model_config.vocab_size,
                watermark_key=int(seed),
                only_English=self.only_english,
                tokenizer=self.tokenizer,
            )
        elif task == "initials":
            from gptwm_initials import (
                InitialsDetector, partition_letters, compute_gamma_from_stats,
            )
            green, _ = partition_letters(int(seed))
            gamma = compute_gamma_from_stats(green, self.stats_file)
            det = InitialsDetector(
                gamma=gamma,
                seed=int(seed),
                strength=self.strength,
                vocab_size=self.tokenizer.vocab_size,
                model_emb_length=self.model_config.vocab_size,
                tokenizer=self.tokenizer,
            )
        elif task == "acrostics":
            from gptwm_acrostics import AcrosticsDetector
            det = AcrosticsDetector(
                target=self.acrostics_target,
                tokenizer=self.tokenizer,
                n_resample=self.acrostics_n_resample,
            )
        else:
            raise ValueError(f"unknown task {task!r}")

        self._cache[key] = det
        return det

    def __call__(self, data, return_dict: bool = True):
        responses = data.batch["responses"]       # (B, T)
        B, T = responses.shape
        reward_tensor = torch.zeros(B, T, dtype=torch.float32)

        tasks       = data.non_tensor_batch["task"]
        wm_seeds    = data.non_tensor_batch["wm_seed"]
        wm_fracs    = data.non_tensor_batch["wm_fraction"]

        z_scores: list = []
        z_valid:  list = []
        resp_lens: list = []
        per_task_z: Dict[str, list] = {t: [] for t in self.active_tasks}

        for i in range(B):
            task = str(tasks[i])
            seed = int(wm_seeds[i])
            frac = float(wm_fracs[i])

            ids = responses[i].tolist()
            token_list = [t for t in ids if t != self.pad_token_id]
            n = len(token_list)
            resp_lens.append(n)

            if n < self.MIN_LEN:
                z_scores.append(-1e6)
                z_valid.append(False)
                for t in per_task_z:
                    per_task_z[t].append(float("nan"))
                continue

            # Per-task z (fill only the one matching this sample's task)
            try:
                det = self._get_detector(task, seed, frac)
                z = float(det.unidetect(token_list))
            except Exception as e:
                print(f"[reward] detector error task={task} seed={seed} frac={frac}: {e}")
                z = 0.0

            z_scores.append(z)
            z_valid.append(True)
            reward_tensor[i, n - 1] = z

            for t in per_task_z:
                per_task_z[t].append(z if t == task else float("nan"))

        extra = {
            "z_score": z_scores,
            "z_score_valid": z_valid,
            "response_len": resp_lens,
        }
        for t, arr in per_task_z.items():
            extra[f"z_score_{t}"] = arr

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor
