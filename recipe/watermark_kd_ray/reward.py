"""
WatermarkZScoreRewardFn — val reward function for watermark KD Ray trainer.

Unified API: detectors are built from ``eval_tasks``. A single-element list
(e.g. ``[green]`` or ``[initials]``) acts as single-task val; multi-element
(``[green, initials]``) triggers per-task routing of the scalar reward based
on the val parquet's ``task`` column and emits per-detector z-scores so the
trainer can report per-task metrics.
"""

import sys
import os
from typing import Dict, List, Optional

import torch

# Ensure project root is on path for gptwm imports
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _build_green_detector(wm_seed, wm_fraction, strength, only_english, tokenizer, model_config):
    from gptwm import GPTWatermarkDetector
    return GPTWatermarkDetector(
        fraction=wm_fraction,
        strength=strength,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        watermark_key=wm_seed,
        only_English=only_english,
        tokenizer=tokenizer,
    )


def _build_initials_detector(wm_seed, strength, tokenizer, model_config, stats_file):
    from gptwm_initials import (
        InitialsDetector, partition_letters, compute_gamma_from_stats,
    )
    green, _ = partition_letters(wm_seed)
    gamma = compute_gamma_from_stats(green, stats_file)
    return InitialsDetector(
        gamma=gamma,
        seed=wm_seed,
        strength=strength,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        tokenizer=tokenizer,
    )


class WatermarkZScoreRewardFn:
    """Reward function that computes watermark z-scores as rewards.

    Builds one detector per entry in ``eval_tasks``. When multiple tasks are
    present, per-sample scalar z-score is routed by ``data.non_tensor_batch["task"]``
    and every detector's z is also included in ``reward_extra_info`` (under
    ``z_score_{task}``) so the trainer can report per-task metrics with a
    shared negative set.
    """

    MIN_LEN = 200

    def __init__(
        self,
        tokenizer,
        model_config,
        strength: float = 2.0,
        only_english: bool = True,
        stats_file: str = "data/initials_icw/leading_space_first_letter_stats.json",
        eval_tasks: Optional[List[str]] = None,
        eval_green_seed: int = 1,
        eval_green_fraction: float = 0.25,
        eval_initials_seed: int = 0,
    ):
        assert tokenizer is not None, "tokenizer must be provided"
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.model_config = model_config

        self.eval_tasks = list(eval_tasks) if eval_tasks else ["green"]
        unknown = [t for t in self.eval_tasks if t not in ("green", "initials")]
        if unknown:
            raise ValueError(f"eval_tasks must be subset of {{green, initials}}; got {unknown}")

        self.detectors: Dict[str, object] = {}
        if "green" in self.eval_tasks:
            self.detectors["green"] = _build_green_detector(
                wm_seed=eval_green_seed,
                wm_fraction=eval_green_fraction,
                strength=strength,
                only_english=only_english,
                tokenizer=tokenizer,
                model_config=model_config,
            )
        if "initials" in self.eval_tasks:
            self.detectors["initials"] = _build_initials_detector(
                wm_seed=eval_initials_seed,
                strength=strength,
                tokenizer=tokenizer,
                model_config=model_config,
                stats_file=stats_file,
            )
        self.default_detector_key = self.eval_tasks[0]
        self.mode = "mixed" if len(self.eval_tasks) > 1 else self.eval_tasks[0]

    # Back-compat: some callers may reference self.detector as a scalar
    @property
    def detector(self):
        return self.detectors[self.default_detector_key]

    def __call__(self, data, return_dict: bool = True):
        """
        Expected ``data.batch["responses"]`` shape (B, T). Task labels are read
        from ``data.non_tensor_batch["task"]`` when present; otherwise every
        sample falls back to the default detector.
        """
        response_ids = data.batch["responses"]
        batch_size, resp_len = response_ids.shape
        reward_tensor = torch.zeros(batch_size, resp_len, dtype=torch.float32)

        tasks = None
        if hasattr(data, "non_tensor_batch") and data.non_tensor_batch is not None:
            tasks = data.non_tensor_batch.get("task", None)

        per_detector_z: Dict[str, List[float]] = {k: [] for k in self.detectors}
        z_scores: List[float] = []
        z_score_valid: List[bool] = []

        for i in range(batch_size):
            token_ids = response_ids[i].tolist()
            token_list = [t for t in token_ids if t != self.pad_token_id]

            if len(token_list) < self.MIN_LEN:
                z_scores.append(-1_000_000.0)
                z_score_valid.append(False)
                for k in per_detector_z:
                    per_detector_z[k].append(-1_000_000.0)
                continue

            for k, det in self.detectors.items():
                per_detector_z[k].append(float(det.unidetect(token_list)))

            task_i = str(tasks[i]) if tasks is not None else None
            if task_i == "neg" or task_i is None or task_i not in self.detectors:
                # neg / unknown / single-task path: use default detector
                z_sample = per_detector_z[self.default_detector_key][-1]
            else:
                z_sample = per_detector_z[task_i][-1]

            z_scores.append(z_sample)
            z_score_valid.append(True)
            last_pos = len(token_list) - 1
            reward_tensor[i, last_pos] = z_sample

        extra = {
            "z_score": z_scores,
            "z_score_valid": z_score_valid,
        }
        for k, arr in per_detector_z.items():
            extra[f"z_score_{k}"] = arr

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor
