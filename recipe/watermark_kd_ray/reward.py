"""
WatermarkZScoreRewardFn — val reward function for watermark KD Ray trainer.

Computes per-sample z-scores using GPTWatermarkDetector.dynamic_threshold()
and assigns the z-score as a scalar reward at the last non-pad token position.
Used as val_reward_fn in PPORayTrainer._validate().
"""

import sys
import os

import torch

# Ensure project root is on path for gptwm imports
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class WatermarkZScoreRewardFn:
    """
    Reward function that computes watermark z-scores as rewards.

    Assigns the z-score to the last non-pad position in each response,
    matching the convention expected by PPORayTrainer._validate():
        reward_tensor.sum(-1)  →  per-sample z-score
    """

    def __init__(
        self,
        wm_seed: int = 1,
        wm_fraction: float = 0.2,
        strength: float = 2.0,
        only_english: bool = True,
        tokenizer=None,
        model_config=None,
    ):
        from gptwm import GPTWatermarkDetector

        assert tokenizer is not None, "tokenizer must be provided"
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.model_config = model_config
        self.detector = GPTWatermarkDetector(
            fraction=wm_fraction,
            strength=strength,
            vocab_size=tokenizer.vocab_size,
            model_emb_length=model_config.vocab_size,
            watermark_key=wm_seed,
            only_English=only_english,
            tokenizer=tokenizer,
        )

    def __call__(self, data, return_dict: bool = True):
        """
        Args:
            data: DataProto with data.batch["responses"] — (batch, max_resp_len) token ids
            return_dict: if True, return dict with reward_tensor and reward_extra_info

        Returns:
            dict with:
                "reward_tensor": (batch, max_resp_len) float — z-score at last non-pad position
                "reward_extra_info": {
                    "z_score": [...],
                    "z_score_valid": [...],  # whether the sample passed the min-length check
                }
        """
        response_ids = data.batch["responses"]  # (B, T) or (B, T) padded
        batch_size, resp_len = response_ids.shape
        reward_tensor = torch.zeros(batch_size, resp_len, dtype=torch.float32)

        z_scores = []
        z_score_valid = []

        for i in range(batch_size):
            token_ids = response_ids[i].tolist()
            # Strip padding
            token_list = [t for t in token_ids if t != self.pad_token_id]

            if len(token_list) < 200:
                z_scores.append(-1000000)
                z_score_valid.append(False)
                continue

            z = self.detector.unidetect(token_list)
            z_scores.append(z)
            z_score_valid.append(True)

            # Place scalar reward at last non-pad position (PPO convention)
            last_pos = len(token_list) - 1
            reward_tensor[i, last_pos] = z

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "z_score": z_scores,
                    "z_score_valid": z_score_valid,
                },
            }
        return reward_tensor
