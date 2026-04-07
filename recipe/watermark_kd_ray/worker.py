"""
WatermarkActorRolloutRefWorker — Ray worker for watermark KD training.

Extends ActorRolloutRefWorker (fsdp_workers.py) with:
  - update_actor_kd(): custom forward-backward with watermark KD loss
    (ref forward no_grad + actor forward + CE + L_green + L_KL)

The ref model is co-located with the actor on the same GPUs using the
existing _is_ref=True mechanism (role="actor_rollout_ref").  CPU offload
is controlled by actor_rollout_ref.ref.fsdp_config.param_offload in the config.

generate_sequences() is inherited unchanged — vLLM handles eval rollouts.
"""

import os
import sys

import torch

# Ensure project root is on path for gptwm imports
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from recipe.watermark_kd_ray.loss import compute_watermark_kd_loss

import logging

logger = logging.getLogger(__name__)


class WatermarkActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """
    Extends ActorRolloutRefWorker with watermark KD training.

    New method:
        update_actor_kd(data) -> DataProto with metrics
            - Runs ref forward (no_grad) + actor forward
            - Computes CE + L_green + L_KL loss with per-sample green masks
            - Performs backward + optimizer step
            - Returns metrics in DataProto.meta_info["metrics"]

    All rollout/generation methods are inherited unchanged.
    """

    # ------------------------------------------------------------------ #
    #  Model init override                                                #
    # ------------------------------------------------------------------ #

    def _build_rollout(self, trust_remote_code=False):
        """Inject YaRN rope config into vLLM engine_kwargs before building rollout."""
        wm_cfg = self.config.get("watermark", {})
        if wm_cfg.get("yarn", False):
            from omegaconf import OmegaConf, open_dict

            rope_parameters = {
                "rope_type": "yarn",
                "factor": float(wm_cfg.get("yarn_factor", 4.0)),
                "original_max_position_embeddings": int(
                    wm_cfg.get("yarn_original_max_position_embeddings", 40960)
                ),
            }
            configured_max_len = int(self.config.rollout.get("max_model_len", 0) or 0)
            effective_max_len = int(self.config.rollout.get("prompt_length", 0) or 0) + int(
                self.config.rollout.get("response_length", 0) or 0
            )
            target_max_len = effective_max_len or configured_max_len or 131072

            with open_dict(self.config.rollout):
                existing = OmegaConf.to_container(
                    self.config.rollout.get("engine_kwargs", OmegaConf.create({})),
                    resolve=True,
                )
                vllm_kwargs = existing.setdefault("vllm", {})
                hf_overrides = dict(vllm_kwargs.get("hf_overrides", {}) or {})

                # vLLM applies nested dict overrides after it has already normalized
                # RoPE fields. For Qwen3, the runtime model path consumes
                # `rope_parameters`, not `rope_scaling`, so override the field
                # that the vLLM model executor actually reads.
                hf_overrides.update(
                    {
                        "rope_parameters": rope_parameters,
                        "max_position_embeddings": target_max_len,
                    }
                )
                vllm_kwargs["hf_overrides"] = hf_overrides
                self.config.rollout.engine_kwargs = OmegaConf.create(existing)

            logger.info(
                "YaRN enabled for rollout: configured_max_model_len=%s, "
                "effective_max_model_len=%s, hf_overrides.max_position_embeddings=%s, "
                "rope_parameters=%s",
                configured_max_len or None,
                effective_max_len or None,
                target_max_len,
                rope_parameters,
            )

        super()._build_rollout(trust_remote_code=trust_remote_code)

    def _need_ref_model(self) -> bool:
        """Check if any ref-dependent loss term or feature is enabled."""
        wm_cfg = self.config.get("watermark", {})
        return (
            float(wm_cfg.get("kl_biased_ref_actor_weight", 0.0)) > 0
            or float(wm_cfg.get("reverse_kl_biased_ref_actor_weight", 0.0)) > 0
            or float(wm_cfg.get("kl_ref_actor_weight", 0.0)) > 0
            or float(wm_cfg.get("reverse_kl_ref_actor_weight", 0.0)) > 0
            or int(wm_cfg.get("quality_green_topk", 0)) > 0
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Call parent init_model(), then freeze and set ref model to eval mode.

        If no ref-dependent loss terms are enabled, skip ref model loading
        entirely by setting _is_ref=False before parent init.
        """
        if not self._need_ref_model():
            logger.info("No ref-dependent loss terms enabled — skipping ref model loading")
            self._is_ref = False

        super().init_model()
        if self._is_ref and hasattr(self, "ref_module_fsdp"):
            self.ref_module_fsdp.eval()
            for p in self.ref_module_fsdp.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    #  Green mask helpers (cached by (seed, fraction))                    #
    # ------------------------------------------------------------------ #

    def _get_green_mask(self, seed: int, fraction: float) -> torch.Tensor:
        """Return a cached boolean green-list mask for (seed, fraction)."""
        if not hasattr(self, "_green_mask_cache"):
            self._green_mask_cache: dict = {}

        key = (int(seed), round(float(fraction), 6))
        if key not in self._green_mask_cache:
            from gptwm import _make_green_list_mask_numpy

            vocab_size = self.tokenizer.vocab_size
            # actor_model_config is set by init_model() from _build_model_optimizer()
            model_emb_length = self.actor_model_config.vocab_size
            only_english = self.config.get("watermark", {}).get("only_english", True)

            mask = torch.tensor(_make_green_list_mask_numpy(
                watermark_key=key[0],
                fraction=key[1],
                vocab_size=vocab_size,
                model_emb_length=model_emb_length,
                only_English=only_english,
                tokenizer=self.tokenizer,
            ), dtype=torch.float32)
            self._green_mask_cache[key] = mask.bool()
        return self._green_mask_cache[key]

    def _get_english_vocab_mask(self):
        """Return a (model_emb_length,) bool mask of English token positions, or None if only_english=False."""
        if not hasattr(self, "_english_vocab_mask_cached"):
            only_english = self.config.get("watermark", {}).get("only_english", True)
            if only_english:
                from gptwm import _get_english_token_ids

                vocab_size = self.tokenizer.vocab_size
                model_emb_length = self.actor_model_config.vocab_size
                english_ids = _get_english_token_ids(self.tokenizer, vocab_size)
                mask = torch.zeros(model_emb_length, dtype=torch.bool)
                mask[english_ids] = True
                self._english_vocab_mask_cached = mask
            else:
                self._english_vocab_mask_cached = None
        return self._english_vocab_mask_cached

    def _build_sample_green_masks(
        self,
        wm_seeds: torch.Tensor,
        wm_fractions: torch.Tensor,
    ) -> torch.Tensor:
        """Build a stacked (num_samples, V) bool mask tensor."""
        masks = []
        for seed, frac in zip(wm_seeds.tolist(), wm_fractions.tolist()):
            masks.append(self._get_green_mask(seed, frac))
        return torch.stack(masks)  # (N, V)

    # ------------------------------------------------------------------ #
    #  KD training step                                                   #
    # ------------------------------------------------------------------ #

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor_kd(self, data: DataProto) -> DataProto:
        """
        Custom forward-backward for watermark knowledge distillation.

        Expected data.batch keys (all regular, padded tensors):
            input_ids            (N, L_actor)  long  — actor: incontext wm prompt + response
            attention_mask       (N, L_actor)  long
            loss_mask            (N, L_actor)  float — 1 for response tokens
            input_ids_ref        (N, L_ref)    long  — ref: clean prompt + response
            attention_mask_ref   (N, L_ref)    long
            loss_mask_ref        (N, L_ref)    float — 1 for response tokens
            wm_seed              (N,)          long  — per-sample watermark seed
            wm_fraction          (N,)          float — per-sample green fraction
        Optional:
            position_ids         (N, L_actor)  long
            position_ids_ref     (N, L_ref)    long

        Returns DataProto with meta_info["metrics"] containing:
            total_loss, ce_loss, green_loss, kl_loss, avg_green_prob, grad_norm, lr
        """
        assert self._is_actor, "update_actor_kd requires actor role"

        wm_cfg = self.config.get("watermark", {})
        strength                     = float(wm_cfg.get("strength", 2.0))
        ce_loss_weight               = float(wm_cfg.get("ce_loss_weight", 1.0))
        green_loss_weight            = float(wm_cfg.get("green_loss_weight", 0.0))
        kl_biased_ref_actor_weight         = float(wm_cfg.get("kl_biased_ref_actor_weight", 0.0))
        reverse_kl_biased_ref_actor_weight = float(wm_cfg.get("reverse_kl_biased_ref_actor_weight", 0.0))
        kl_ref_actor_weight                = float(wm_cfg.get("kl_ref_actor_weight", 0.0))
        reverse_kl_ref_actor_weight        = float(wm_cfg.get("reverse_kl_ref_actor_weight", 0.0))
        kl_biased_actor_actor_weight = float(wm_cfg.get("kl_biased_actor_actor_weight", 0.0))
        reverse_kl_biased_actor_actor_weight = float(wm_cfg.get("reverse_kl_biased_actor_actor_weight", 0.0))
        max_grad_norm                = float(wm_cfg.get("max_grad_norm", 1.0))
        grad_accum_steps             = int(wm_cfg.get("gradient_accumulation_steps", 1))
        green_target_ratio           = float(wm_cfg.get("green_target_ratio", 0.0))
        quality_green_topk           = int(wm_cfg.get("quality_green_topk", 0))
        need_green_masks = green_loss_weight > 0 or kl_biased_ref_actor_weight > 0 or reverse_kl_biased_ref_actor_weight > 0 or kl_biased_actor_actor_weight > 0 or reverse_kl_biased_actor_actor_weight > 0
        need_ref_forward = kl_biased_ref_actor_weight > 0 or reverse_kl_biased_ref_actor_weight > 0 or kl_ref_actor_weight > 0 or reverse_kl_ref_actor_weight > 0 or quality_green_topk > 0
        if need_ref_forward:
            assert self._is_ref, (
                "Ref-dependent loss terms are enabled but ref model was not loaded. "
                "Check watermark config — _need_ref_model() must return True for ref terms."
            )

        # Offload management
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            from verl.utils.fsdp_utils import load_fsdp_optimizer
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        data = data.to("cpu")  # will be moved to GPU inside

        with self.ulysses_sharding_manager:
            device = get_device_id()
            batch = data.batch

            # --- Unpack actor inputs ---
            input_ids = batch["input_ids"].to(device)            # (N, L_actor)
            attention_mask = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device).float()

            if "position_ids" in batch.keys():
                position_ids = batch["position_ids"].to(device)
            else:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

            N, L_actor = input_ids.shape
            assert N % grad_accum_steps == 0, (
                f"Batch size per DP rank ({N}) must be divisible by "
                f"gradient_accumulation_steps ({grad_accum_steps})"
            )
            micro_batch_size = N // grad_accum_steps

            english_vocab_mask = self._get_english_vocab_mask()
            if english_vocab_mask is not None:
                english_vocab_mask = english_vocab_mask.to(device)

            if need_green_masks:
                wm_seeds = batch["wm_seed"].to(device)               # (N,)
                wm_fractions = batch["wm_fraction"].to(device)       # (N,)

                # --- Build per-sample green masks (N, V) ---
                green_masks = self._build_sample_green_masks(wm_seeds, wm_fractions).to(device)
            else:
                green_masks = None

            if need_ref_forward:

                # --- Unpack ref inputs (clean prompt, same response, may differ in length) ---
                input_ids_ref = batch["input_ids_ref"].to(device)    # (N, L_ref)
                attention_mask_ref = batch["attention_mask_ref"].to(device)
                loss_mask_ref = batch["loss_mask_ref"].to(device).float()

                if "position_ids_ref" in batch.keys():
                    position_ids_ref = batch["position_ids_ref"].to(device)
                else:
                    position_ids_ref = (attention_mask_ref.long().cumsum(-1) - 1).clamp(min=0)

                L_ref = input_ids_ref.shape[1]

            # --- Shift actor tensors for next-token prediction ---
            input_ids_rolled_flat = input_ids[:, 1:].contiguous().view(-1)       # (N*(L_actor-1),)
            loss_mask_flat = loss_mask[:, 1:].contiguous().view(-1)              # (N*(L_actor-1),)
            sample_index = (
                torch.arange(N, device=device).unsqueeze(1).expand(-1, L_actor - 1).contiguous().view(-1)
            )

            # --- Compute batch_num_tokens via actor loss_mask (all-reduce across DP) ---
            batch_num_tokens_local = loss_mask_flat.sum()
            torch.distributed.all_reduce(
                batch_num_tokens_local, op=torch.distributed.ReduceOp.SUM
            )
            batch_num_tokens_val = batch_num_tokens_local.item()

            # --- Response-position indices ---
            resp_idx_actor = loss_mask_flat.bool()       # (N*(L_actor-1),)
            if need_ref_forward:
                # --- Shift ref loss_mask for next-token prediction ---
                loss_mask_ref_flat = loss_mask_ref[:, 1:].contiguous().view(-1)      # (N*(L_ref-1),)
                resp_idx_ref = loss_mask_ref_flat.bool()     # (N*(L_ref-1),)

                # Sanity: both must select the same number of response tokens per batch
                assert resp_idx_actor.sum() == resp_idx_ref.sum(), (
                    f"Actor response tokens ({resp_idx_actor.sum()}) != "
                    f"ref response tokens ({resp_idx_ref.sum()}). "
                    "Check that both sequences contain identical response content."
                )

            # --- Gradient accumulation loop ---
            self.actor_optimizer.zero_grad()
            self.actor_module_fsdp.train()

            accumulated_metrics = []

            for step_i in range(grad_accum_steps):
                s = step_i * micro_batch_size
                e = s + micro_batch_size

                # Slice actor tensors for this micro-batch
                mb_input_ids      = input_ids[s:e]       # (mb_N, L_actor)
                mb_attention_mask = attention_mask[s:e]
                mb_loss_mask      = loss_mask[s:e]
                mb_position_ids   = position_ids[s:e]
                mb_N              = e - s

                mb_input_ids_rolled_flat = mb_input_ids[:, 1:].contiguous().view(-1)
                mb_loss_mask_flat        = mb_loss_mask[:, 1:].contiguous().view(-1)
                mb_sample_index = (
                    torch.arange(mb_N, device=device)
                    .unsqueeze(1).expand(-1, L_actor - 1).contiguous().view(-1)
                )
                mb_resp_idx_actor = mb_loss_mask_flat.bool()

                mb_green_masks = green_masks[s:e] if need_green_masks else None
                mb_fractions = wm_fractions[s:e] if need_green_masks else None

                with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                    if need_ref_forward:
                        mb_input_ids_ref      = input_ids_ref[s:e]
                        mb_attention_mask_ref = attention_mask_ref[s:e]
                        mb_position_ids_ref   = position_ids_ref[s:e]
                        mb_loss_mask_ref_flat = loss_mask_ref[s:e, 1:].contiguous().view(-1)
                        mb_resp_idx_ref       = mb_loss_mask_ref_flat.bool()

                        with torch.no_grad():
                            ref_output = self.ref_module_fsdp(
                                input_ids=mb_input_ids_ref,
                                attention_mask=mb_attention_mask_ref,
                                position_ids=mb_position_ids_ref,
                                use_cache=False,
                            )
                            mb_chunk_len_ref = mb_N * (L_ref - 1)
                            mb_ref_logits = (
                                ref_output.logits[:, :-1].contiguous().view(mb_chunk_len_ref, -1)[mb_resp_idx_ref]
                            )
                            del ref_output

                    # Actor forward — incontext wm prompt, trainable
                    actor_output = self.actor_module_fsdp(
                        input_ids=mb_input_ids,
                        attention_mask=mb_attention_mask,
                        position_ids=mb_position_ids,
                        use_cache=False,
                    )
                    mb_chunk_len_actor = mb_N * (L_actor - 1)
                    mb_actor_logits = (
                        actor_output.logits[:, :-1].contiguous().view(mb_chunk_len_actor, -1)[mb_resp_idx_actor]
                    )
                    del actor_output

                    mb_input_ids_resp    = mb_input_ids_rolled_flat[mb_resp_idx_actor]
                    mb_sample_index_resp = mb_sample_index[mb_resp_idx_actor]

                    if not need_ref_forward:
                        mb_ref_logits = mb_actor_logits.new_empty((0, mb_actor_logits.shape[-1]))
                    if not need_green_masks:
                        mb_green_masks = torch.empty((0, 0), dtype=torch.bool, device=mb_actor_logits.device)

                    # Normalize by total batch tokens (not micro-batch) so gradients
                    # accumulate correctly: sum(micro_loss_i) == full_batch_loss
                    loss, mb_metrics = compute_watermark_kd_loss(
                        actor_logits=mb_actor_logits,
                        ref_logits=mb_ref_logits,
                        input_ids_rolled=mb_input_ids_resp,
                        sample_index=mb_sample_index_resp,
                        green_masks=mb_green_masks,
                        strength=strength,
                        ce_loss_weight=ce_loss_weight,
                        green_loss_weight=green_loss_weight,
                        kl_biased_ref_actor_weight=kl_biased_ref_actor_weight,
                        reverse_kl_biased_ref_actor_weight=reverse_kl_biased_ref_actor_weight,
                        kl_ref_actor_weight=kl_ref_actor_weight,
                        reverse_kl_ref_actor_weight=reverse_kl_ref_actor_weight,
                        kl_biased_actor_actor_weight=kl_biased_actor_actor_weight,
                        reverse_kl_biased_actor_actor_weight=reverse_kl_biased_actor_actor_weight,
                        batch_num_tokens=batch_num_tokens_val,
                        dp_size=self.world_size,
                        english_vocab_mask=english_vocab_mask,
                        green_target_ratio=green_target_ratio,
                        sample_fractions=mb_fractions,
                        quality_green_topk=quality_green_topk,
                    )

                    loss.backward()

                accumulated_metrics.append(mb_metrics)

            # Aggregate metrics: loss terms are normalized by total tokens, safe to sum.
            # avg_green_prob is a per-micro-batch average; divide to get overall average.
            metrics = {k: sum(m[k] for m in accumulated_metrics) for k in accumulated_metrics[0]}
            if "avg_green_prob" in metrics:
                metrics["avg_green_prob"] /= grad_accum_steps

            # --- Optimizer step ---
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module_fsdp.parameters(), max_norm=max_grad_norm
            )
            self.actor_optimizer.step()
            if self.actor_lr_scheduler is not None:
                lr = self.actor_lr_scheduler.get_last_lr()[0]
                self.actor_lr_scheduler.step()
            else:
                lr = self.actor_optimizer.param_groups[0]["lr"]

        # Collect metrics
        metrics["grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        metrics["lr"] = lr.item() if torch.is_tensor(lr) else float(lr)

        # Offload back to CPU if configured
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor_kd", logger=logger)
        if self._is_offload_optimizer:
            from verl.utils.fsdp_utils import offload_fsdp_optimizer
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        output = DataProto(meta_info={"metrics": metrics})
        return output.to("cpu")
