
"""
WatermarkKDRayTrainer — Ray-based watermark knowledge distillation trainer.

Inherits RayPPOTrainer for:
  - Ray worker infrastructure (init_workers, resource pools)
  - vLLM-based validation generation
  - Checkpoint save/load

Overrides:
  - _create_dataloader(): uses WatermarkSFTDataset (padded) for train,
    RLHFDataset (prompts only) for val
  - _validate(): logs only watermark-specific validation metrics
  - fit(): simple KD loop — no rollout, no PPO; calls update_actor_kd per batch

PPO-specific code removed: no critic, no advantage, no compute_ref_log_prob RPC.
"""

import uuid
from typing import Optional

import numpy as np

from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager
from verl.trainer.ppo.utils import Role
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking


class WatermarkKDRayTrainer(RayPPOTrainer):
    """
    Distributed watermark KD trainer using Ray backend.

    Training loop:
        For each step:
            actor_rollout_wg.update_actor_kd(batch)
            → ref forward (no_grad) + actor forward + KD loss + backward

    Validation loop:
        actor_rollout_wg.generate_sequences(val_prompts)
        → WatermarkZScoreRewardFn → valid ratio + valid z-score summary

    Args:
        kd_train_dataloader: Pre-built StatefulDataLoader yielding batches from
            WatermarkSFTDataset. Each batch must contain padded tensors:
                input_ids, attention_mask, loss_mask, wm_seed, wm_fraction
        (all other args same as RayPPOTrainer)
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict,
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls=RayWorkerGroup,
        val_reward_fn=None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        kd_train_dataloader: Optional[StatefulDataLoader] = None,
        device_name: Optional[str] = None,
    ):
        # Store KD train dataloader BEFORE super().__init__ which calls _create_dataloader
        self._kd_train_dataloader = kd_train_dataloader

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=None,            # no train reward (KD loss replaces it)
            val_reward_fn=val_reward_fn,
            train_dataset=None,        # handled via kd_train_dataloader
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            device_name=device_name,
        )

    # ------------------------------------------------------------------ #
    #  Dataloader override                                                #
    # ------------------------------------------------------------------ #

    def _create_dataloader(
        self,
        train_dataset,
        val_dataset,
        collate_fn,
        train_sampler: Optional[Sampler],
    ):
        """
        Override: use the pre-built KD train dataloader; create val dataloader normally.
        """
        from verl.trainer.main_ppo import create_rl_dataset
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        # --- Val dataloader (RL-style, prompt-only) ---
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                processor=None,
                is_train=False,
                max_samples=self.config.data.get("val_max_samples", -1),
            )

        if collate_fn is None:
            collate_fn = default_collate_fn

        val_batch_size = self.config.data.get("val_batch_size", None)
        if val_batch_size is None:
            val_batch_size = len(val_dataset)

        num_workers = self.config.data.get("dataloader_num_workers", 0)
        self.val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", False),
            drop_last=False,
            collate_fn=collate_fn,
        )

        # --- Train dataloader (KD-style, from WatermarkSFTDataset) ---
        assert self._kd_train_dataloader is not None, (
            "kd_train_dataloader must be provided to WatermarkKDRayTrainer"
        )
        self.train_dataloader = self._kd_train_dataloader

        # --- Compute total training steps ---
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.get("total_training_steps") is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, "
            f"Size of val dataloader: {len(self.val_dataloader)}"
        )
        print(f"Total training steps: {self.total_training_steps}")

        # Propagate to actor optimizer schedule if present
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                        total_training_steps
                    )
        except Exception as e:
            print(f"Warning: could not set total_training_steps in config: {e}")

    # ------------------------------------------------------------------ #
    #  Worker init override                                               #
    # ------------------------------------------------------------------ #

    def _build_actor_rollout_worker_config(self):
        """Pass root watermark settings into the worker-local actor_rollout_ref config."""
        worker_cfg = OmegaConf.create(
            OmegaConf.to_container(self.config.actor_rollout_ref, resolve=True)
        )
        root_watermark = OmegaConf.select(self.config, "watermark")
        if root_watermark is not None:
            with open_dict(worker_cfg):
                worker_cfg.watermark = OmegaConf.to_container(root_watermark, resolve=True)
        return worker_cfg

    def init_workers(self):
        """
        Override to use WatermarkActorRolloutRefWorker and skip critic.

        The role is always ActorRolloutRef (actor + rollout + co-located ref).
        """
        from recipe.watermark_kd_ray.worker import WatermarkActorRolloutRefWorker
        from verl.single_controller.ray import RayClassWithInitArgs
        from verl.single_controller.ray.base import create_colocated_worker_cls

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # Register WatermarkActorRolloutRefWorker as ActorRolloutRef
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
        worker_cfg = self._build_actor_rollout_worker_config()
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRolloutRef],
            config=worker_cfg,
            role=str(Role.ActorRolloutRef),
        )
        self.resource_pool_to_cls[resource_pool][str(Role.ActorRolloutRef)] = actor_rollout_cls

        # No critic, no separate ref policy worker
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

        # Set async rollout mode based on config
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager
            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=None,
            )

        # Override attrs so _validate() / profiling don't try to access
        # ref_policy_wg or critic_wg (which we don't create).
        # use_reference_policy was set True in __init__ because ActorRolloutRef is in mapping,
        # but we co-locate ref inside the same worker — no separate ref_policy_wg.
        self.use_critic = False
        self.use_rm = False
        self.use_reference_policy = False

    # ------------------------------------------------------------------ #
    #  Validation helpers                                                 #
    # ------------------------------------------------------------------ #

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Override: log first 10 val samples to a wandb table.

        Columns: prefix (first 300 chars of input), completion, z_score.
        The full input_prompt is intentionally omitted as it is too long.
        """
        n = min(10, len(inputs))
        if n == 0:
            return

        try:
            import wandb

            if wandb.run is None:
                return

            table = wandb.Table(columns=["prefix", "completion", "z_score"])
            for i in range(n):
                prefix = inputs[i][-200:]
                completion = outputs[i]
                z_score = scores[i]
                table.add_data(prefix, completion, z_score)

            wandb.log({"val/generations": table}, step=self.global_steps)
        except Exception as e:
            print(f"Warning: failed to log wandb val table: {e}")

    @staticmethod
    def _reduce_worker_metrics(metrics: dict) -> dict:
        """Collapse per-worker metric lists into scalar values for logging."""
        if not metrics:
            return {}

        reduced = reduce_metrics(dict(metrics))
        normalized = {}
        for key, value in reduced.items():
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    pass
            normalized[key] = value
        return normalized

    @staticmethod
    def _format_metric(metrics: dict, key: str, precision: int) -> str:
        value = metrics.get(key, 0.0)
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        try:
            return f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            return str(value)

    def _validate(self):
        """Run validation generation and log only watermark-specific aggregate metrics."""
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        z_scores = []
        z_score_valid = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # Preserve a stable id for debug logging in the async raw_prompt_ids agent loop.
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_ids = test_batch.batch["input_ids"]
            sample_inputs.extend(self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            output_ids = test_output_gen_batch.batch["responses"]
            sample_outputs.extend(self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")

            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_extra_info = result.get("reward_extra_info", {})

            batch_z_scores = reward_extra_info.get("z_score")
            if batch_z_scores is None:
                batch_z_scores = result["reward_tensor"].sum(-1).cpu().tolist()

            batch_valid = reward_extra_info.get("z_score_valid")
            if batch_valid is None:
                batch_valid = [True] * len(batch_z_scores)

            if len(batch_z_scores) != len(batch_valid):
                raise ValueError(
                    f"Validation z-score length mismatch: {len(batch_z_scores)=}, {len(batch_valid)=}"
                )

            z_scores.extend(float(score) for score in batch_z_scores)
            z_score_valid.extend(bool(flag) for flag in batch_valid)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=z_scores)

        reward_extra_infos_dict = {
            "z_score": z_scores,
            "z_score_valid": z_score_valid,
        }
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=z_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        total_count = len(z_scores)
        if total_count == 0:
            return {"val/valid_ratio": 0.0}

        valid_mask = np.asarray(z_score_valid, dtype=bool)
        if len(valid_mask) != total_count:
            raise ValueError(f"Validation z-score bookkeeping mismatch: {len(valid_mask)=}, {total_count=}")

        valid_ratio = float(valid_mask.mean())
        metric_dict = {"val/valid_ratio": valid_ratio}

        if valid_mask.any():
            valid_z_scores = np.asarray(z_scores, dtype=np.float64)[valid_mask]
            metric_dict["val/valid_z_score_mean"] = float(valid_z_scores.mean())
            metric_dict["val/valid_z_score_min"] = float(valid_z_scores.min())
            metric_dict["val/valid_z_score_max"] = float(valid_z_scores.max())

        return metric_dict

    # ------------------------------------------------------------------ #
    #  Training loop                                                      #
    # ------------------------------------------------------------------ #

    def fit(self):
        """
        Watermark KD training loop (no rollout, no PPO).

        Steps:
          1. For each batch from kd_train_dataloader:
               actor_rollout_wg.update_actor_kd(batch) → metrics
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

        # Optional validation before training begins
        if val_before_train and self.val_dataloader is not None:
            val_metrics = self._validate()
            tracking.log(data=val_metrics, step=self.global_steps)

        for epoch in range(self.config.trainer.total_epochs):
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
            )
            for batch_data in pbar:
                self.global_steps += 1

                # Convert dict of tensors to DataProto
                batch = DataProto.from_single_dict(batch_data)

                # Dispatch KD update to all actor workers
                output = self.actor_rollout_wg.update_actor_kd(batch)
                metrics = self._reduce_worker_metrics(output.meta_info.get("metrics", {}))

                # Log training metrics
                log_metrics = {f"train/{k}": v for k, v in metrics.items()}
                tracking.log(data=log_metrics, step=self.global_steps)
                pbar.set_postfix(
                    loss=self._format_metric(metrics, "total_loss", 4),
                    z_green=self._format_metric(metrics, "avg_green_prob", 3),
                )

                # Validation
                if test_freq > 0 and self.global_steps % test_freq == 0:
                    val_metrics = self._validate()
                    tracking.log(data=val_metrics, step=self.global_steps)

                # Checkpoint
                if save_freq > 0 and self.global_steps % save_freq == 0:
                    self._save_checkpoint()

                if self.global_steps >= self.total_training_steps:
                    break

            if self.global_steps >= self.total_training_steps:
                break

        # Final validation and checkpoint
        if self.val_dataloader is not None:
            val_metrics = self._validate()
            tracking.log(data=val_metrics, step=self.global_steps)

        self._save_checkpoint()
        print(f"Training complete. Total steps: {self.global_steps}")
