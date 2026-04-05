"""
Entry point for on-policy watermark self-distillation.

Usage:
    python -m recipe.watermark_onpolicy_ray.main \\
        actor_rollout_ref.model.path=<model_path> \\
        data.train_files=<prompt_parquet> \\
        data.val_files=<val_jsonl_or_parquet> \\
        watermark.kl_biased_ref_actor_weight=1.0 \\
        watermark.strength=5.0 \\
        ...

Differences from watermark_kd_ray/main.py:
  - Uses WatermarkPromptDataset (prompt-only) + WatermarkPromptCollator
  - Uses WatermarkOnPolicyTrainer (rollout-then-update loop)
  - Uses WatermarkOnPolicyWorker (update_actor_onpolicy RPC)
"""

import os
import random
import socket

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role


@hydra.main(config_path="config", config_name="watermark_onpolicy_ray", version_base=None)
def main(config):
    run_watermark_onpolicy(config)


def run_watermark_onpolicy(config, task_runner_class=None):
    """Initialize Ray and run on-policy watermark self-distillation."""
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        worker_env_vars = {}
        cuda_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if cuda_arch:
            worker_env_vars["TORCH_CUDA_ARCH_LIST"] = cuda_arch
        task_runner_class = ray.remote(
            num_cpus=1,
            runtime_env={"env_vars": worker_env_vars} if worker_env_vars else {},
        )(TaskRunner)

    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))


class TaskRunner:
    """Ray remote class executing the on-policy watermark training workflow."""

    def run(self, config):
        from pprint import pprint
        from transformers import AutoConfig

        from verl.utils.fs import copy_to_local
        from verl.utils import hf_tokenizer

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        seed = config.data.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # ---- Tokenizer ----
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        model_config = AutoConfig.from_pretrained(local_path)

        # ---- Role → Worker mapping ----
        from recipe.watermark_onpolicy_ray.worker import WatermarkOnPolicyWorker

        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(WatermarkOnPolicyWorker),
        }
        mapping = {Role.ActorRolloutRef: "global_pool"}

        # ---- Resource pool ----
        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        # ---- Prompt-only train dataloader ----
        prompt_train_dataloader = _build_prompt_train_dataloader(config, tokenizer)

        # ---- Val dataset (RL-style, prompts only) ----
        from verl.trainer.main_ppo import create_rl_dataset
        from verl.utils.dataset.rl_dataset import collate_fn as rl_collate_fn

        val_dataset = None
        if config.data.get("val_files"):
            val_dataset = create_rl_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor=None,
                is_train=False,
                max_samples=config.data.get("val_max_samples", -1),
            )

        # ---- Val reward function (z-score) ----
        from recipe.watermark_kd_ray.reward import WatermarkZScoreRewardFn

        val_reward_fn = WatermarkZScoreRewardFn(
            wm_seed=config.watermark.get("eval_wm_seed", 1),
            wm_fraction=config.watermark.get("eval_wm_fraction", 0.2),
            strength=config.watermark.get("strength", 2.0),
            only_english=config.watermark.get("only_english", True),
            tokenizer=tokenizer,
            model_config=model_config,
        )

        # ---- Trainer ----
        from recipe.watermark_onpolicy_ray.trainer import WatermarkOnPolicyTrainer
        from verl.single_controller.ray import RayWorkerGroup

        trainer = WatermarkOnPolicyTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            val_reward_fn=val_reward_fn,
            val_dataset=val_dataset,
            collate_fn=rl_collate_fn,
            prompt_train_dataloader=prompt_train_dataloader,
        )
        trainer.init_workers()
        trainer.fit()


def _build_prompt_train_dataloader(config, tokenizer):
    """
    Build the prompt-only training dataloader for on-policy self-distillation.

    Uses WatermarkPromptDataset which reads only the prompt side of each sample.
    """
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torch.utils.data import RandomSampler

    from recipe.watermark_onpolicy_ray.dataset import WatermarkPromptDataset, WatermarkPromptCollator

    train_dataset = WatermarkPromptDataset(
        parquet_files=config.data.train_files,
        tokenizer=tokenizer,
        config=config.data,
        max_samples=config.data.get("train_max_samples", -1),
    )

    seed = config.data.get("seed", 42)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(train_dataset, generator=generator)

    batch_size = config.data.get("train_batch_size", 8)
    num_workers = config.data.get("dataloader_num_workers", 4)

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=WatermarkPromptCollator(),
        num_workers=num_workers,
        drop_last=True,
    )

    return train_dataloader


if __name__ == "__main__":
    main()
