"""WatermarkRLTrainer — Stage 2 on-policy RL with per-sample sequence reward.

Design decisions (per 2026-04-20 mentor meeting):
  - No ref model, no KD losses (use_reference_policy=False, no KL-in-reward)
  - GRPO advantage (group by prompt uid, normalize by group std)
  - Per-sample task/seed/fraction piped through to reward fn via non_tensor_batch
  - Native verl ActorRolloutRef (async) worker; native update_actor (PPO loss)

fit() loop per step:
  1. Prompt batch → _get_gen_batch (pops input_ids + non_tensor fields)
  2. Restore {task, wm_seed, wm_fraction} in batch so they survive rollout union
  3. Rollout via vLLM agent loop (n rollouts per prompt)
  4. batch_rep.union(gen_out) → full sequences
  5. compute_reward (our PerSampleWatermarkZScoreRewardFn)
  6. compute_response_mask → compute old_log_probs → compute_advantage(grpo)
  7. actor_rollout_wg.update_actor(batch_rep)

Inherits WatermarkKDRayTrainer for:
  - Worker infrastructure (resource pool + worker groups)
  - _validate() using WatermarkZScoreRewardFn
  - Checkpoint save/load
  - _create_dataloader() (accepts our prompt dataloader via kd_train_dataloader arg)
"""

import uuid
from collections import defaultdict
from time import perf_counter
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.utils import Role
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from recipe.watermark_kd_ray.trainer import WatermarkKDRayTrainer


# Non-tensor batch keys that must survive rollout (they're popped by _get_gen_batch
# into gen_batch but not restored by gen_batch.union with rollout output).
_PASSTHROUGH_KEYS = ("wm_seed", "wm_fraction", "task")


class WatermarkRLTrainer(WatermarkKDRayTrainer):
    """On-policy RL trainer with per-sample sequence reward (GRPO)."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict,
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        prompt_train_dataloader: Optional[StatefulDataLoader] = None,
        device_name: Optional[str] = None,
    ):
        # Pass our prompt loader as kd_train_dataloader so parent stores it
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            val_reward_fn=val_reward_fn,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            kd_train_dataloader=prompt_train_dataloader,
            device_name=device_name,
        )
        self.reward_fn = reward_fn
        # We intentionally disable critic + ref
        self.use_critic = False
        self.use_reference_policy = False

    # ------------------------------------------------------------------ #
    #  Worker init — use native async actor/rollout worker                #
    # ------------------------------------------------------------------ #

    def init_workers(self):
        from verl.single_controller.ray import RayClassWithInitArgs
        from verl.single_controller.ray.base import create_colocated_worker_cls

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role=str(Role.ActorRollout),
        )
        self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls

        wg_kwargs = {"device_name": self.device_name}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = (
                self.config.trainer.ray_wait_register_center_timeout
            )

        all_wg = {}
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        self.async_rollout_mode = self.config.actor_rollout_ref.rollout.mode == "async"
        if self.async_rollout_mode:
            from verl.experimental.agent_loop import AgentLoopManager
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=None,
            )

    # ------------------------------------------------------------------ #
    #  Fit loop                                                           #
    # ------------------------------------------------------------------ #

    def fit(self):
        tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        test_freq = self.config.trainer.get("test_freq", -1)
        save_freq = self.config.trainer.get("save_freq", -1)
        val_before_train = self.config.trainer.get("val_before_train", False)

        steps_per_epoch = len(self.train_dataloader)
        if save_freq == "after_each_epoch":
            save_freq = steps_per_epoch
        if test_freq == "after_each_epoch":
            test_freq = steps_per_epoch

        if val_before_train and self.val_dataloader is not None:
            val_metrics = self._validate()
            tracking.log(data=val_metrics, step=self.global_steps)

        n = self.config.actor_rollout_ref.rollout.n
        size_divisor = (
            self.config.actor_rollout_ref.rollout.agent.num_workers
            if self.async_rollout_mode
            else self.actor_rollout_wg.world_size
        )

        adv_estimator = self.config.algorithm.adv_estimator
        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

        for epoch in range(self.config.trainer.total_epochs):
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
            )
            for prompt_batch_data in pbar:
                self.global_steps += 1
                metrics: dict = {}

                # ---- Build initial batch ----
                batch = DataProto.from_single_dict(prompt_batch_data)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # ---- Get gen batch (pops input_ids + non-tensor fields) ----
                gen_batch = self._get_gen_batch(batch)
                # Restore wm passthrough fields so they survive union with gen_out
                for key in _PASSTHROUGH_KEYS:
                    if key in gen_batch.non_tensor_batch and key not in batch.non_tensor_batch:
                        batch.non_tensor_batch[key] = gen_batch.non_tensor_batch[key].copy()

                gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": True,
                    "validate": False,
                    "global_steps": self.global_steps,
                }

                # GRPO: n rollouts per prompt
                gen_batch_rep = gen_batch.repeat(n, interleave=True)
                gen_padded, pad_size = pad_dataproto_to_divisor(gen_batch_rep, size_divisor)

                rollout_start = perf_counter()
                if self.async_rollout_mode:
                    gen_out_padded = self.async_rollout_manager.generate_sequences(gen_padded)
                else:
                    gen_out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
                rollout_s = perf_counter() - rollout_start

                gen_out = unpad_dataproto(gen_out_padded, pad_size=pad_size)

                # Repeat prompt-side batch to align with n-per-prompt rollouts
                batch_rep = batch.repeat(n, interleave=True)
                full_batch = batch_rep.union(gen_out)

                # Sanity on first step
                if self.global_steps == 1:
                    for key in _PASSTHROUGH_KEYS:
                        assert key in full_batch.non_tensor_batch, (
                            f"{key} missing from full_batch.non_tensor_batch"
                        )
                    assert "responses" in full_batch.batch.keys(), "responses missing"

                # ---- Reward ----
                reward_start = perf_counter()
                reward_tensor, reward_extra = compute_reward(full_batch, self.reward_fn)
                full_batch.batch["token_level_scores"] = reward_tensor
                full_batch.batch["token_level_rewards"] = reward_tensor
                if reward_extra:
                    full_batch.non_tensor_batch.update(
                        {k: np.array(v) for k, v in reward_extra.items()}
                    )
                reward_s = perf_counter() - reward_start

                # ---- Response mask + old_log_probs (needed for PPO ratio) ----
                full_batch.batch["response_mask"] = compute_response_mask(full_batch)

                # Recompute old_log_probs through the actor (needed for PPO clip ratio).
                logp_start = perf_counter()
                old_logp = self.actor_rollout_wg.compute_log_prob(full_batch)
                if "entropys" in old_logp.batch.keys():
                    old_logp.batch.pop("entropys")
                full_batch = full_batch.union(old_logp)
                logp_s = perf_counter() - logp_start

                # ---- Advantages: valid-subset GRPO + invalid loss mask ----
                # Only samples with z_score_valid=True participate in the group
                # mean/std. Invalid samples (response too short) get advantage=0
                # AND response_mask=0 so they contribute nothing to PG / entropy
                # / KL loss aggregation. Reward value for invalid samples is
                # therefore inert (the MIN_LEN sentinel never enters the loss).
                full_batch.meta_info["global_token_num"] = torch.sum(
                    full_batch.batch["attention_mask"], dim=-1
                ).tolist()

                valid_np = full_batch.non_tensor_batch["z_score_valid"].astype(bool)
                uids_np  = full_batch.non_tensor_batch["uid"]
                scores   = full_batch.batch["token_level_rewards"].sum(dim=-1)
                device   = scores.device

                advantages_scalar = torch.zeros_like(scores)
                valid_t = torch.from_numpy(valid_np).to(device)

                group_all_invalid = 0
                group_partial_invalid = 0
                group_singleton_valid = 0
                unique_uids = np.unique(uids_np)
                for uid in unique_uids:
                    g_mask_np  = (uids_np == uid)
                    g_valid_np = g_mask_np & valid_np
                    n_valid = int(g_valid_np.sum())
                    n_total = int(g_mask_np.sum())

                    if n_valid == 0:
                        group_all_invalid += 1
                        continue
                    if n_valid < n_total:
                        group_partial_invalid += 1
                    if n_valid < 2:
                        group_singleton_valid += 1
                        continue  # no within-group signal; advantage stays 0

                    g_valid_t = torch.from_numpy(g_valid_np).to(device)
                    s = scores[g_valid_t]
                    mean_g = s.mean()
                    if norm_adv_by_std_in_grpo:
                        std_g = s.std(unbiased=False) + 1e-6
                        advantages_scalar[g_valid_t] = (s - mean_g) / std_g
                    else:
                        advantages_scalar[g_valid_t] = s - mean_g

                adv_tok = advantages_scalar.unsqueeze(-1) * full_batch.batch["response_mask"]
                full_batch.batch["advantages"] = adv_tok
                full_batch.batch["returns"]    = adv_tok

                invalid_t = ~valid_t
                full_batch.batch["response_mask"][invalid_t] = 0

                metrics["reward/n_valid"] = int(valid_np.sum())
                metrics["reward/n_total"] = int(valid_np.size)
                metrics["reward/group_count"] = int(len(unique_uids))
                metrics["reward/group_all_invalid"] = group_all_invalid
                metrics["reward/group_partial_invalid"] = group_partial_invalid
                metrics["reward/group_singleton_valid"] = group_singleton_valid

                # ---- Update actor ----
                update_start = perf_counter()
                actor_out = self.actor_rollout_wg.update_actor(full_batch)
                update_s = perf_counter() - update_start
                actor_metrics = reduce_metrics(actor_out.meta_info.get("metrics", {}))
                metrics.update(actor_metrics)

                # ---- Log ----
                metrics.update(self._compute_reward_metrics(reward_extra, full_batch))
                metrics["timing/rollout_s"] = rollout_s
                metrics["timing/reward_s"] = reward_s
                metrics["timing/log_prob_s"] = logp_s
                metrics["timing/update_s"] = update_s

                log_metrics = {
                    (k if (k.startswith("train/") or k.startswith("timing/") or k.startswith("val/") or k.startswith("actor/")) else f"train/{k}"): v
                    for k, v in metrics.items()
                }
                tracking.log(data=log_metrics, step=self.global_steps)
                pbar.set_postfix(
                    rwd_mean=f"{metrics.get('reward/z_mean', 0.0):.2f}",
                    g_z=f"{metrics.get('reward/z_green_mean', 0.0):.2f}",
                    i_z=f"{metrics.get('reward/z_initials_mean', 0.0):.2f}",
                )

                if test_freq > 0 and self.global_steps % test_freq == 0:
                    val_metrics = self._validate()
                    tracking.log(data=val_metrics, step=self.global_steps)

                if save_freq > 0 and self.global_steps % save_freq == 0:
                    self._save_checkpoint()

                if self.global_steps >= self.total_training_steps:
                    break

            if self.global_steps >= self.total_training_steps:
                break

        # Final validation only if test_freq > 0 was requested
        if self.val_dataloader is not None and test_freq > 0:
            val_metrics = self._validate()
            tracking.log(data=val_metrics, step=self.global_steps)

        self._save_checkpoint()
        print(f"Training complete. Total steps: {self.global_steps}")

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_reward_metrics(reward_extra: dict, batch: DataProto) -> dict:
        """Aggregate per-sample reward metrics into scalars."""
        m: dict = {}
        if not reward_extra:
            return m

        z = np.array(reward_extra.get("z_score", []), dtype=np.float32)
        valid = np.array(reward_extra.get("z_score_valid", []), dtype=bool)
        if valid.sum() > 0:
            m["reward/z_mean"] = float(z[valid].mean())
            m["reward/z_std"] = float(z[valid].std())
        m["reward/valid_ratio"] = float(valid.mean()) if valid.size else 0.0

        # Per-task aggregates (use nanmean to ignore non-matching samples)
        tasks = batch.non_tensor_batch.get("task")
        if tasks is not None:
            for t in ("green", "initials", "acrostics"):
                key = f"z_score_{t}"
                if key in reward_extra:
                    arr = np.array(reward_extra[key], dtype=np.float32)
                    # count = samples actually from this task with valid response
                    task_mask = np.array([str(x) == t for x in tasks])
                    good = task_mask & valid
                    if good.sum() > 0:
                        m[f"reward/z_{t}_mean"] = float(arr[good].mean())
                        m[f"reward/z_{t}_std"]  = float(arr[good].std())
                        m[f"reward/z_{t}_count"] = int(good.sum())

        # Response length stats
        rl = reward_extra.get("response_len", [])
        if rl:
            rl_np = np.array(rl, dtype=np.float32)
            m["response/len_mean"] = float(rl_np.mean())
            m["response/len_min"]  = float(rl_np.min())
            m["response/len_max"]  = float(rl_np.max())

        return m
