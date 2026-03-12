"""
WatermarkDistillTrainer: trains a model to follow in-context watermark instructions
via knowledge distillation from a logit-biased reference model.

loss = L_CE + λ1 * L_green + λ2 * L_KL

The ref model can be a different scale from the actor (same tokenizer family).
Per-sample watermark seed and fraction come from the data ("seed" and "fraction"
columns in the parquet), so each batch item gets its own green-list mask.
Masks are cached by (seed, fraction) to avoid recomputation.
"""

import os
import sys

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging

import hydra
import torch
import torch.distributed
import torch.nn.functional as F
from omegaconf import OmegaConf
from tensordict.tensorclass import NonTensorData
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint import CheckpointHandler
from verl.utils.device import get_device_name, get_device_id
from verl.utils.distributed import destroy_global_process_group
from verl.utils.logger import log_with_rank
from verl.utils.tracking import Tracking
from verl.workers.engine.utils import prepare_micro_batches

from recipe.watermark_distill.loss import compute_watermark_distill_loss

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class WatermarkDistillTrainer:
    """
    Trainer for watermark knowledge distillation.

    Builds on top of SFTTrainer's dataset/dataloader/checkpoint infrastructure,
    but uses a custom forward-backward loop that:
      1. Forwards both actor and frozen ref model to get raw logits
      2. Computes combined loss (CE + green + KL) with per-sample green masks
    """

    def __init__(self, config):
        self.config = config
        self.rank = torch.distributed.get_rank()

        self._build_config()
        self._build_dataset()
        self._build_engine()
        self._build_dataloader()
        self._init_engine()
        self._build_ref_model()
        self._build_ckpt_handler()

        self._green_mask_cache: dict[tuple[int, float], torch.Tensor] = {}
        self._debug_first_step = True

        if self.engine.ulysses_device_mesh is not None:
            from verl.utils.ulysses import set_ulysses_sequence_parallel_group

            set_ulysses_sequence_parallel_group(
                self.engine.ulysses_device_mesh["sp"].get_group()
            )

        self.resume_global_step = self.ckpt_handler.load_checkpoint()
        self.device_name = self.config.trainer.device

        if self.rank == 0:
            print(self.config)

    # ---- Reused from SFTTrainer (identical) ----

    def _build_config(self):
        from verl.utils.config import omega_conf_to_dataclass

        self.model_config = omega_conf_to_dataclass(self.config.model)
        self.engine_config = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)

    def _build_engine(self):
        from functools import partial

        from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
        from verl.workers.utils.losses import sft_loss

        self.loss_fn = partial(sft_loss, config=None)

        config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

        self.training_client = TrainingWorker(config=config)
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        self.engine = self.training_client.engine

    def _build_dataset(self):
        from recipe.watermark_distill.dataset import WatermarkSFTDataset

        config = self.config
        tokenizer = self.model_config.tokenizer
        data_config = config.data

        # If custom_cls is set, use it; otherwise default to WatermarkSFTDataset
        if data_config.custom_cls.get("path", None):
            from verl.utils.import_utils import load_extern_object

            dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
        else:
            dataset_cls = WatermarkSFTDataset

        train_dataset = dataset_cls(
            parquet_files=config.data.train_files, tokenizer=tokenizer, config=config.data,
            max_samples=config.data.get("train_max_samples", -1),
        )
        if config.data.val_files:
            val_dataset = dataset_cls(
                parquet_files=config.data.val_files, tokenizer=tokenizer, config=config.data,
                max_samples=config.data.get("val_max_samples", -1),
            )
        else:
            val_dataset = None
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

    def _build_dataloader(self):
        from torch.utils.data import DistributedSampler
        from torchdata.stateful_dataloader import StatefulDataLoader

        from recipe.watermark_distill.dataset import WatermarkCollator

        config = self.config
        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True,
        )
        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.collate_fn = WatermarkCollator(config.data.pad_mode)

        device_name = get_device_name()
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True,
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None

    def _init_engine(self):
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps
        self.steps_per_epoch = len(self.train_dataloader)

        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch

        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch

        self.training_client.reset()

    def _build_ckpt_handler(self):
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)

        self.ckpt_handler = CheckpointHandler(
            engine=self.engine,
            train_dataloader=self.train_dataloader,
            default_local_dir=self.config.trainer.default_local_dir,
            max_ckpt_to_keep=max_ckpt_to_keep,
            default_hdfs_dir=default_hdfs_dir,
            resume_mode=resume_mode,
            resume_from_path=resume_from_path,
        )

    # ---- Ref model (independent config, different scale allowed) ----

    def _build_ref_model(self):
        """Load a frozen ref model (potentially different scale), wrapped in FSDP2."""
        from transformers import AutoModelForCausalLM
        from verl.utils.fsdp_utils import apply_fsdp2

        ref_config = self.config.ref_model
        ref_path = ref_config.path
        assert ref_path is not None, "ref_model.path must be set"

        if self.rank == 0:
            print(f"[WatermarkDistill] Loading ref model from {ref_path}...")

        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        if self.engine_config.strategy == "fsdp2":
            from torch.distributed._composable.fsdp import MixedPrecisionPolicy

            offload_policy = None
            if ref_config.get("param_offload", False):
                from torch.distributed._composable.fsdp import CPUOffloadPolicy
                offload_policy = CPUOffloadPolicy()

            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, cast_forward_inputs=True,
            )
            fsdp_kwargs = {"mp_policy": mp_policy, "reshard_after_forward": True}
            if offload_policy is not None:
                fsdp_kwargs["offload_policy"] = offload_policy
            apply_fsdp2(ref_model, fsdp_kwargs, self.engine_config)
        else:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload

            ref_model = FSDP(
                ref_model,
                cpu_offload=CPUOffload(offload_params=ref_config.get("param_offload", False)),
                use_orig_params=True,
            )

        self.ref_model = ref_model

        if self.rank == 0:
            print("[WatermarkDistill] Ref model loaded and frozen.")

    # ---- Per-sample green mask (cached) ----

    def _get_green_mask(self, seed: int, fraction: float) -> torch.Tensor:
        """Return a cached boolean green-list mask for (seed, fraction)."""
        key = (int(seed), round(float(fraction), 6))
        if key not in self._green_mask_cache:
            from gptwm import _make_green_list_mask

            tokenizer = self.model_config.tokenizer
            vocab_size = tokenizer.vocab_size
            model_emb_length = self.engine.module.config.vocab_size
            wm_config = self.config.watermark

            mask = _make_green_list_mask(
                watermark_key=key[0],
                fraction=key[1],
                vocab_size=vocab_size,
                model_emb_length=model_emb_length,
                only_English=wm_config.get("only_english", False),
                tokenizer=tokenizer,
            )
            self._green_mask_cache[key] = mask.bool()
        return self._green_mask_cache[key]

    def _build_sample_green_masks(self, wm_seeds, wm_fractions):
        """Build a stacked (num_samples, V) bool mask tensor from per-sample seeds/fractions."""
        masks = []
        for seed, frac in zip(wm_seeds.tolist(), wm_fractions.tolist()):
            masks.append(self._get_green_mask(seed, frac))
        return torch.stack(masks)  # (num_samples, V)

    # ---- Utility ----

    def _get_batch_seqlens(self, data):
        is_nested = data["input_ids"].is_nested
        if is_nested:
            batch_seqlens: torch.Tensor = data["input_ids"].offsets().diff()
        else:
            batch_seqlens: torch.Tensor = data["attention_mask"].sum(dim=-1)
        batch_seqlens = batch_seqlens.to(self.device_name)

        output_tensor = torch.empty(
            (batch_seqlens.shape[0] * self.engine.get_data_parallel_size(),),
            dtype=batch_seqlens.dtype,
            device=self.device_name,
        )
        torch.distributed.all_gather_into_tensor(
            output_tensor=output_tensor,
            input_tensor=batch_seqlens,
            group=self.engine.get_data_parallel_group(),
        )
        return output_tensor.tolist()

    # ---- Custom forward-backward ----

    def _build_sample_index(self, offsets, total_len, sp_size, pad_size):
        """
        Build a (chunk_len,) long tensor mapping each flattened token position
        to its sample index. Handles SP padding and slicing.
        """
        num_samples = len(offsets) - 1
        sample_index = torch.zeros(offsets[-1].item(), dtype=torch.long)
        for i in range(num_samples):
            sample_index[offsets[i]:offsets[i + 1]] = i

        if sp_size > 1:
            if pad_size > 0:
                sample_index = F.pad(sample_index, (0, pad_size), value=0)
            from verl.utils.ulysses import slice_input_tensor
            sample_index = slice_input_tensor(sample_index.unsqueeze(0), dim=1, padding=False).squeeze(0)

        return sample_index

    def _slice_loss_mask_for_sp(self, loss_mask_flat, pad_size):
        """Pad and slice loss_mask to align with SP-local logits."""
        from verl.utils.ulysses import slice_input_tensor

        loss_mask_2d = loss_mask_flat.unsqueeze(0)
        if pad_size > 0:
            loss_mask_2d = F.pad(loss_mask_2d, (0, pad_size), value=0.0)
        loss_mask_2d = slice_input_tensor(loss_mask_2d, dim=1, padding=False)
        return loss_mask_2d.squeeze(0)

    def _custom_forward_backward(self, data):
        """
        Custom forward-backward that gets raw logits from both actor and ref model,
        then computes the combined watermark distillation loss with per-sample green masks.
        """
        device_name = get_device_name()
        wm_config = self.config.watermark
        sp_size = self.engine.ulysses_sequence_parallel_size

        tu.assign_non_tensor(data, sp_size=sp_size)

        local_num_tokens = data["loss_mask"].values().sum().item() if data["loss_mask"].is_nested else data["loss_mask"].sum().item()
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.engine.get_data_parallel_group(),
        )
        batch_num_tokens_val = batch_num_tokens.item()
        if self._debug_first_step and self.rank == 0:
            lm = data["loss_mask"]
            print(f"[DEBUG batch] local_num_tokens={local_num_tokens}, "
                  f"is_nested={lm.is_nested}, "
                  f"shape={lm.shape if not lm.is_nested else 'nested'}, "
                  f"values_shape={lm.values().shape if lm.is_nested else 'N/A'}, "
                  f"values_sum={lm.values().sum().item() if lm.is_nested else lm.sum().item()}, "
                  f"values_dtype={lm.values().dtype if lm.is_nested else lm.dtype}")
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens_val)
        tu.assign_non_tensor(data, dp_size=self.engine.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.engine.get_data_parallel_group(), same_micro_num_in_dp=True,
        )

        all_metrics = []
        self.engine.optimizer_zero_grad()
        self.engine.module.train()

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())

            # Extract per-sample watermark info before flattening
            wm_seeds = micro_batch["wm_seed"]      # (num_samples,) regular tensor
            wm_fractions = micro_batch["wm_fraction"]  # (num_samples,) regular tensor
            offsets = micro_batch["input_ids"].offsets()  # (num_samples+1,) cumulative offsets

            green_masks = self._build_sample_green_masks(wm_seeds, wm_fractions)
            green_masks = green_masks.to(get_device_id())  # (num_samples, V)

            model_inputs, output_args = self.engine.prepare_model_inputs(micro_batch)

            # Build sample_index aligned with the (possibly SP-sliced) logits
            pad_size = output_args.get("pad_size", 0)
            chunk_len = output_args["input_ids_rmpad_rolled"].shape[0]
            sample_index = self._build_sample_index(offsets, chunk_len, sp_size, pad_size)
            sample_index = sample_index.to(get_device_id())

            # Roll loss_mask on full sequence BEFORE SP slicing (matches input_ids_rolled)
            raw_lm = micro_batch["loss_mask"]
            raw_values = raw_lm.values() if raw_lm.is_nested else raw_lm
            loss_mask_flat = raw_values.float()
            loss_mask_flat = torch.roll(loss_mask_flat, shifts=-1, dims=0)

            if sp_size > 1:
                loss_mask_flat = self._slice_loss_mask_for_sp(loss_mask_flat, pad_size)

            if self._debug_first_step and self.rank == 0:
                from verl.utils.ulysses import get_ulysses_sequence_parallel_rank, get_ulysses_sequence_parallel_world_size
                print(f"[DEBUG micro] raw sum={raw_values.sum().item()}, "
                      f"rolled sum={raw_values.float().roll(-1,0).sum().item()}, "
                      f"sp_rank={get_ulysses_sequence_parallel_rank()}, "
                      f"sp_ws={get_ulysses_sequence_parallel_world_size()}, "
                      f"after SP slice: sum={loss_mask_flat.sum().item()}, shape={loss_mask_flat.shape}, "
                      f"nonzero_positions={raw_values.nonzero().squeeze().tolist()[:5]}...")

            with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
                # Run ref first (no_grad) so its activations are freed before actor forward.
                # This avoids holding both models' activations at once (~2x activation memory).
                with torch.no_grad():
                    ref_raw_output = self.ref_model(**model_inputs, use_cache=False)
                    ref_logits = ref_raw_output.logits.squeeze(0).detach()  # (chunk_len, V)

                actor_raw_output = self.engine.module(**model_inputs, use_cache=False)
                actor_logits = actor_raw_output.logits.squeeze(0)  # (chunk_len, V)

                input_ids_rolled = output_args["input_ids_rmpad_rolled"]

                loss, metrics = compute_watermark_distill_loss(
                    actor_logits=actor_logits,
                    ref_logits=ref_logits,
                    input_ids_rolled=input_ids_rolled,
                    loss_mask_flat=loss_mask_flat,
                    sample_index=sample_index,
                    green_masks=green_masks,
                    strength=wm_config.strength,
                    green_loss_weight=wm_config.green_loss_weight,
                    kl_loss_weight=wm_config.kl_loss_weight,
                    batch_num_tokens=batch_num_tokens_val,
                    dp_size=self.engine.get_data_parallel_size(),
                )

                if sp_size > 1:
                    loss = loss * sp_size

                loss.backward()

            if self._debug_first_step and self.rank == 0:
                print(f"[DEBUG step] batch_num_tokens={batch_num_tokens_val}, "
                      f"loss_mask_flat sum={loss_mask_flat.sum().item()}, "
                      f"actor_logits shape={tuple(actor_logits.shape)}, "
                      f"actor_logits range=[{actor_logits.min().item():.4f}, {actor_logits.max().item():.4f}], "
                      f"loss={loss.item():.6f}, metrics={metrics}")

            all_metrics.append(metrics)

        self._debug_first_step = False
        grad_norm = self.engine.optimizer_step()

        agg = {}
        for key in all_metrics[0]:
            agg[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        agg["grad_norm"] = grad_norm

        return agg

    # ---- Main training loop ----

    def fit(self):
        is_logging = (
            self.engine.is_mp_src_rank_with_outputs()
            and self.engine.get_data_parallel_rank() == 0
        )

        if is_logging:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps}",
            logger=logger, rank=0, log_only_rank_0=True,
        )

        if global_step > 0:
            log_with_rank(
                f"Resuming from global step: {global_step}",
                logger=logger, rank=0, log_only_rank_0=True,
            )

        start_epoch = global_step // self.steps_per_epoch

        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

        total_tokens = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=not is_logging,
                )
            ):
                global_step += 1

                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                batch_seqlens_list = self._get_batch_seqlens(data=data)
                batch_seqlens = NonTensorData(batch_seqlens_list)
                tu.assign_non_tensor(data, update_lr_scheduler=True, global_token_num=batch_seqlens)

                metrics = self._custom_forward_backward(data)

                lr = self.engine.lr_scheduler_step()
                metrics["lr"] = lr

                if self.engine.is_mp_src_rank_with_outputs():
                    total_tokens += sum(batch_seqlens_list)
                    log_metrics = {
                        "train/total_loss": metrics["total_loss"],
                        "train/ce_loss": metrics["ce_loss"],
                        "train/green_loss": metrics.get("green_loss", 0.0),
                        "train/kl_loss": metrics.get("kl_loss", 0.0),
                        "train/avg_green_prob": metrics.get("avg_green_prob", 0.0),
                        "train/grad_norm": metrics["grad_norm"],
                        "train/lr": lr,
                        "train/global_tokens": sum(batch_seqlens_list),
                        "train/total_tokens(B)": total_tokens / 1e9,
                    }
                    if self.engine.get_data_parallel_rank() == 0:
                        tracking.log(data=log_metrics, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.test_freq == 0 if self.test_freq > 0 else False
                is_save_step = global_step % self.save_freq == 0 if self.save_freq > 0 else False

                # Validation
                if (is_last_step and self.val_dataloader is not None) or is_valid_step:
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
                        output = self.training_client.infer_batch(val_data)
                        if self.engine.is_mp_src_rank_with_outputs():
                            val_m = tu.get(output, "metrics")
                            val_losses.append(val_m["loss"])

                    if self.engine.is_mp_src_rank_with_outputs():
                        val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))
                        torch.distributed.all_reduce(
                            val_loss, op=torch.distributed.ReduceOp.AVG,
                            group=self.engine.get_data_parallel_group(),
                        )
                    if is_logging:
                        metric = {"val/loss": val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                # Checkpoint
                if is_last_step or is_save_step:
                    self.ckpt_handler.save_checkpoint(step=global_step)

                if is_last_step:
                    if is_logging:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_watermark_distill(config):
    from verl.utils.distributed import initialize_global_process_group

    initialize_global_process_group()
    trainer = WatermarkDistillTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="watermark_distill", version_base=None)
def main(config):
    run_watermark_distill(config)


if __name__ == "__main__":
    main()
