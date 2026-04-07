"""
WatermarkOnPolicyTrainer — on-policy self-distillation trainer.

Inherits WatermarkKDRayTrainer for:
  - Worker infrastructure, init_workers, checkpoint save/load
  - _validate() with z-score metrics
  - _create_dataloader() for val set

Overrides:
  - __init__: takes prompt_train_dataloader instead of kd_train_dataloader
  - fit(): rollout-then-update loop instead of offline KD

Training loop per step:
  1. Sample prompt batch (prompt + wm_seed + wm_fraction)
  2. vLLM rollout via generate_sequences() — actor under current weights, no green bias
  3. Union prompt + rollout → full (prompt + response) sequences
  4. update_actor_onpolicy() — ref forward + actor forward + KL(D̂_ref ‖ D_actor) + backward
"""

import uuid
from time import perf_counter
from typing import Optional

import numpy as np

from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.utils import Role
from verl.utils.tracking import Tracking

from recipe.watermark_kd_ray.trainer import WatermarkKDRayTrainer


class WatermarkOnPolicyTrainer(WatermarkKDRayTrainer):
    """
    On-policy self-distillation trainer.

    Each training step:
        1. vLLM rollout from actor (no green-bias logit intervention)
        2. Ref + actor forward on the same (prompt + rollout) sequences
        3. KL(D̂_ref ‖ D_actor) loss + backward

    Unlike WatermarkKDRayTrainer.fit(), there is no pre-generated response
    — each step generates fresh samples under the current actor weights.

    Args:
        prompt_train_dataloader: StatefulDataLoader yielding prompt-only batches
            from WatermarkPromptDataset.  Each batch contains:
                input_ids, attention_mask, position_ids  (tensors, left-padded)
                raw_prompt_ids, raw_prompt, wm_seed, wm_fraction  (object numpy arrays)
        (all other args same as WatermarkKDRayTrainer / RayPPOTrainer)
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict,
        resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        val_reward_fn=None,
        val_dataset=None,
        collate_fn=None,
        prompt_train_dataloader=None,
        device_name: Optional[str] = None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            val_reward_fn=val_reward_fn,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            # Pass as kd_train_dataloader so parent's _create_dataloader() accepts it
            kd_train_dataloader=prompt_train_dataloader,
            device_name=device_name,
        )

    # ------------------------------------------------------------------ #
    #  Worker init override                                               #
    # ------------------------------------------------------------------ #

    def init_workers(self):
        """
        Same as WatermarkKDRayTrainer.init_workers() but uses WatermarkOnPolicyWorker.
        """
        from recipe.watermark_onpolicy_ray.worker import WatermarkOnPolicyWorker
        from verl.single_controller.ray import RayClassWithInitArgs
        from verl.single_controller.ray.base import create_colocated_worker_cls

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
        worker_cfg = self._build_actor_rollout_worker_config()
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRolloutRef],
            config=worker_cfg,
            role=str(Role.ActorRolloutRef),
        )
        self.resource_pool_to_cls[resource_pool][str(Role.ActorRolloutRef)] = actor_rollout_cls

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

        self.actor_rollout_wg = all_wg[str(Role.ActorRolloutRef)]
        self.actor_rollout_wg.init_model()

        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager
            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=None,
            )

        self.use_critic = False
        self.use_reference_policy = False

    # ------------------------------------------------------------------ #
    #  Training loop                                                      #
    # ------------------------------------------------------------------ #

    def fit(self):
        """
        On-policy self-distillation training loop.

        Steps:
          1. For each prompt batch from train_dataloader:
               a. vLLM rollout via generate_sequences() — no green bias at inference
               b. Union prompt batch + rollout output → full (prompt + response) batch
               c. actor_rollout_wg.update_actor_onpolicy(full_batch) → metrics
          2. At test_freq: _validate() → z-score metrics
          3. At save_freq: _save_checkpoint()
        """
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

        for epoch in range(self.config.trainer.total_epochs):
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
            )
            for prompt_batch_data in pbar:
                self.global_steps += 1

                # Build DataProto from prompt batch
                batch = DataProto.from_single_dict(prompt_batch_data)
                # Add uid (used by async agent loop for tracing)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # --- Rollout ---
                gen_batch = self._get_gen_batch(batch)
                # _get_gen_batch pops non_tensor fields (wm_seed, wm_fraction,
                # raw_prompt_ids_ref, …) from batch into gen_batch.
                # generate_sequences() / agent loop won't preserve them in gen_out,
                # so copy them back to batch so they survive through batch_rep.union(gen_out).
                for key in ("wm_seed", "wm_fraction", "raw_prompt_ids_ref"):
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

                gen_batch_rep = gen_batch.repeat(n, interleave=True)
                gen_padded, pad_size = pad_dataproto_to_divisor(gen_batch_rep, size_divisor)

                rollout_start = perf_counter()
                if self.async_rollout_mode:
                    gen_out_padded = self.async_rollout_manager.generate_sequences(gen_padded)
                else:
                    gen_out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
                rollout_s = perf_counter() - rollout_start

                gen_out = unpad_dataproto(gen_out_padded, pad_size=pad_size)

                # Repeat prompt-side batch to match n rollouts per prompt
                # then union with rollout output (adds input_ids, responses, etc.)
                batch_rep = batch.repeat(n, interleave=True)
                full_batch = batch_rep.union(gen_out)

                # Quick assertion to catch field routing issues early (first step only)
                if self.global_steps == 1:
                    for key in ("wm_seed", "wm_fraction", "raw_prompt_ids_ref"):
                        assert key in full_batch.non_tensor_batch, (
                            f"{key} missing from full_batch.non_tensor_batch"
                        )
                    assert "responses" in full_batch.batch.keys(), (
                        "responses missing from full_batch.batch — "
                        "check that generate_sequences populates it"
                    )

                # --- On-policy KD update ---
                update_start = perf_counter()
                output = self.actor_rollout_wg.update_actor_onpolicy(full_batch)
                update_s = perf_counter() - update_start

                metrics = self._reduce_worker_metrics(output.meta_info.get("metrics", {}))
                metrics["train/timing/rollout_s"] = rollout_s
                metrics["train/timing/update_s"] = update_s

                log_metrics = {f"train/{k}" if not k.startswith("train/") else k: v
                               for k, v in metrics.items()}
                tracking.log(data=log_metrics, step=self.global_steps)
                pbar.set_postfix(
                    loss=self._format_metric(metrics, "total_loss", 4),
                    z_green=self._format_metric(metrics, "avg_green_prob", 3),
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

        if self.val_dataloader is not None:
            val_metrics = self._validate()
            tracking.log(data=val_metrics, step=self.global_steps)

        self._save_checkpoint()
        print(f"Training complete. Total steps: {self.global_steps}")
