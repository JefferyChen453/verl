"""
WatermarkOnPolicyWorker — Ray worker for on-policy watermark self-distillation.

Extends WatermarkActorRolloutRefWorker (from watermark_kd_ray) with a single
new RPC: update_actor_onpolicy().

On-policy design:
  - Batch arrives from trainer after vLLM rollout: full (prompt + response) sequences
  - loss_mask is derived on-the-fly from attention_mask + response length
  - Actor and ref forward on the SAME input_ids (no separate clean-prompt ref path)
  - wm_seed / wm_fraction come from non_tensor_batch (not batch tensors)

The KD loss function (compute_watermark_kd_loss) is unchanged.
"""

import torch

from verl import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

from recipe.watermark_kd_ray.worker import WatermarkActorRolloutRefWorker
from recipe.watermark_kd_ray.loss import compute_watermark_kd_loss


def _build_loss_mask_from_response(attention_mask: torch.Tensor, response_length: int) -> torch.Tensor:
    """
    Derive loss_mask from attention_mask and the response span.

    attention_mask : (N, L)  — 1 for real (non-pad) tokens
    response_length: int     — length of the response suffix in the full sequence

    Returns loss_mask: (N, L).  loss_mask[t] = 1 iff input_ids[t+1] is a real
    response token.  This matches the worker's [:, 1:] shift convention so that:
      loss_mask[:, 1:][t] = 1  →  logit at position t+1 predicts a response token

    Convention (same as WatermarkKDDataset / SFTDataset):
      - Position (total_len - response_length - 1): 1 → logit there predicts r0
      - Last real response position: 0 (nothing after it, or only padding)
    """
    # response indicator: 1 only at real response positions
    resp_slice = attention_mask[:, -response_length:].float()   # (N, response_length)
    full = torch.zeros_like(attention_mask, dtype=torch.float32)
    full[:, -response_length:] = resp_slice

    # shift left by 1: loss_mask[t] = full[t+1]
    loss_mask = torch.zeros_like(full)
    loss_mask[:, :-1] = full[:, 1:]
    return loss_mask


class WatermarkOnPolicyWorker(WatermarkActorRolloutRefWorker):
    """
    Extends WatermarkActorRolloutRefWorker with on-policy KD update.

    Inherits:
      - generate_sequences() — vLLM rollout (used by trainer.fit() every step)
      - init_model() — co-locates actor + frozen ref on same GPUs
      - _build_sample_green_masks(), _get_english_vocab_mask()

    New RPC:
      - update_actor_onpolicy(): forward-backward on (prompt + rollout) sequences
    """

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor_onpolicy(self, data: DataProto) -> DataProto:
        """
        On-policy KD update step.

        Expected data.batch keys (all padded tensors from rollout output):
            input_ids       (N, L)   long   — full sequence: left-pad + prompt + response + right-pad
            attention_mask  (N, L)   long   — 1 for real tokens
            position_ids    (N, L)   long
            responses       (N, R)   long   — response-only tokens (R = response_length)

        Expected data.non_tensor_batch keys:
            wm_seed         (N,)     object array of ints
            wm_fraction     (N,)     object array of floats

        Returns DataProto with meta_info["metrics"] containing:
            total_loss, kl_biased_ref_actor (and other active terms), avg_green_prob, grad_norm, lr
        """
        assert self._is_actor, "update_actor_onpolicy requires actor role"
        assert self._is_ref, "update_actor_onpolicy requires ref model (role must include 'ref')"

        wm_cfg = self.config.get("watermark", {})
        strength                          = float(wm_cfg.get("strength", 2.0))
        ce_loss_weight                    = float(wm_cfg.get("ce_loss_weight", 0.0))
        green_loss_weight                 = float(wm_cfg.get("green_loss_weight", 0.0))
        kl_biased_ref_actor_weight        = float(wm_cfg.get("kl_biased_ref_actor_weight", 1.0))
        reverse_kl_biased_ref_actor_weight = float(wm_cfg.get("reverse_kl_biased_ref_actor_weight", 0.0))
        kl_ref_actor_weight               = float(wm_cfg.get("kl_ref_actor_weight", 0.0))
        kl_biased_actor_actor_weight      = float(wm_cfg.get("kl_biased_actor_actor_weight", 0.0))
        max_grad_norm                     = float(wm_cfg.get("max_grad_norm", 1.0))
        grad_accum_steps                  = int(wm_cfg.get("gradient_accumulation_steps", 1))
        green_target_ratio                = float(wm_cfg.get("green_target_ratio", 0.0))

        need_green_masks = (
            green_loss_weight > 0
            or kl_biased_ref_actor_weight > 0
            or reverse_kl_biased_ref_actor_weight > 0
            or kl_biased_actor_actor_weight > 0
        )
        need_ref_forward = (
            kl_biased_ref_actor_weight > 0
            or reverse_kl_biased_ref_actor_weight > 0
            or kl_ref_actor_weight > 0
        )

        # Offload management
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            from verl.utils.fsdp_utils import load_fsdp_optimizer
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        data = data.to("cpu")

        with self.ulysses_sharding_manager:
            device = get_device_id()
            batch = data.batch

            # --- Unpack actor inputs (full prompt + response sequences from rollout) ---
            input_ids      = batch["input_ids"].to(device)       # (N, L)
            attention_mask = batch["attention_mask"].to(device)  # (N, L)

            if "position_ids" in batch.keys():
                position_ids = batch["position_ids"].to(device)
            else:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

            # Derive loss_mask from response span
            response_length = batch["responses"].shape[1]
            loss_mask = _build_loss_mask_from_response(attention_mask, response_length)

            N, L = input_ids.shape
            assert N % grad_accum_steps == 0, (
                f"Batch size per DP rank ({N}) must be divisible by "
                f"gradient_accumulation_steps ({grad_accum_steps})"
            )
            micro_batch_size = N // grad_accum_steps

            # --- Read wm_seed / wm_fraction from non_tensor_batch ---
            wm_seeds     = torch.tensor(
                [int(x) for x in data.non_tensor_batch["wm_seed"]],
                dtype=torch.long, device=device
            )
            wm_fractions = torch.tensor(
                [float(x) for x in data.non_tensor_batch["wm_fraction"]],
                dtype=torch.float32, device=device
            )

            english_vocab_mask = self._get_english_vocab_mask()
            if english_vocab_mask is not None:
                english_vocab_mask = english_vocab_mask.to(device)

            if need_green_masks:
                green_masks = self._build_sample_green_masks(wm_seeds, wm_fractions).to(device)
            else:
                green_masks = None

            # --- On-policy: ref and actor forward on the SAME sequences ---
            # input_ids_ref = input_ids (both see the in-context watermark prompt + rollout)
            input_ids_ref      = input_ids
            attention_mask_ref = attention_mask
            position_ids_ref   = position_ids
            loss_mask_ref      = loss_mask

            # --- Shifted tensors for next-token prediction ---
            input_ids_rolled_flat = input_ids[:, 1:].contiguous().view(-1)   # (N*(L-1),)
            loss_mask_flat        = loss_mask[:, 1:].contiguous().view(-1)   # (N*(L-1),)
            sample_index = (
                torch.arange(N, device=device).unsqueeze(1).expand(-1, L - 1).contiguous().view(-1)
            )

            # --- Compute batch_num_tokens (response tokens, all-reduce across DP) ---
            batch_num_tokens_local = loss_mask_flat.sum()
            torch.distributed.all_reduce(
                batch_num_tokens_local, op=torch.distributed.ReduceOp.SUM
            )
            batch_num_tokens_val = batch_num_tokens_local.item()

            resp_idx = loss_mask_flat.bool()

            # Sanity: actor and ref select same positions (they are the same tensor)
            if need_ref_forward:
                loss_mask_ref_flat = loss_mask_ref[:, 1:].contiguous().view(-1)
                resp_idx_ref = loss_mask_ref_flat.bool()
                assert resp_idx.sum() == resp_idx_ref.sum(), (
                    f"Actor/ref response token count mismatch: "
                    f"{resp_idx.sum()} vs {resp_idx_ref.sum()}"
                )

            # --- Gradient accumulation loop ---
            self.actor_optimizer.zero_grad()
            self.actor_module_fsdp.train()

            accumulated_metrics = []

            for step_i in range(grad_accum_steps):
                s = step_i * micro_batch_size
                e = s + micro_batch_size

                mb_input_ids       = input_ids[s:e]
                mb_attention_mask  = attention_mask[s:e]
                mb_position_ids    = position_ids[s:e]
                mb_loss_mask       = loss_mask[s:e]
                mb_N               = e - s

                mb_input_ids_rolled_flat = mb_input_ids[:, 1:].contiguous().view(-1)
                mb_loss_mask_flat        = mb_loss_mask[:, 1:].contiguous().view(-1)
                mb_sample_index = (
                    torch.arange(mb_N, device=device)
                    .unsqueeze(1).expand(-1, L - 1).contiguous().view(-1)
                )
                mb_resp_idx = mb_loss_mask_flat.bool()

                mb_green_masks = green_masks[s:e] if need_green_masks else None
                mb_fractions   = wm_fractions[s:e] if need_green_masks else None

                with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                    if need_ref_forward:
                        with torch.no_grad():
                            ref_output = self.ref_module_fsdp(
                                input_ids=mb_input_ids,
                                attention_mask=mb_attention_mask,
                                position_ids=mb_position_ids,
                                use_cache=False,
                            )
                            mb_chunk_len = mb_N * (L - 1)
                            mb_ref_logits = (
                                ref_output.logits[:, :-1].contiguous().view(mb_chunk_len, -1)[mb_resp_idx]
                            )
                            del ref_output

                    actor_output = self.actor_module_fsdp(
                        input_ids=mb_input_ids,
                        attention_mask=mb_attention_mask,
                        position_ids=mb_position_ids,
                        use_cache=False,
                    )
                    mb_chunk_len = mb_N * (L - 1)
                    mb_actor_logits = (
                        actor_output.logits[:, :-1].contiguous().view(mb_chunk_len, -1)[mb_resp_idx]
                    )
                    del actor_output

                    mb_input_ids_resp    = mb_input_ids_rolled_flat[mb_resp_idx]
                    mb_sample_index_resp = mb_sample_index[mb_resp_idx]

                    if not need_ref_forward:
                        mb_ref_logits = mb_actor_logits.new_empty((0, mb_actor_logits.shape[-1]))
                    if not need_green_masks:
                        mb_green_masks = torch.empty((0, 0), dtype=torch.bool, device=mb_actor_logits.device)

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
                        kl_biased_actor_actor_weight=kl_biased_actor_actor_weight,
                        batch_num_tokens=batch_num_tokens_val,
                        dp_size=self.world_size,
                        english_vocab_mask=english_vocab_mask,
                        green_target_ratio=green_target_ratio,
                        sample_fractions=mb_fractions,
                    )

                    loss.backward()

                accumulated_metrics.append(mb_metrics)

            # Aggregate metrics
            metrics = {k: sum(m[k] for m in accumulated_metrics) for k in accumulated_metrics[0]}
            if "avg_green_prob" in metrics:
                metrics["avg_green_prob"] /= grad_accum_steps

            # Optimizer step
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module_fsdp.parameters(), max_norm=max_grad_norm
            )
            self.actor_optimizer.step()
            if self.actor_lr_scheduler is not None:
                lr = self.actor_lr_scheduler.get_last_lr()[0]
                self.actor_lr_scheduler.step()
            else:
                lr = self.actor_optimizer.param_groups[0]["lr"]

        metrics["grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        metrics["lr"] = lr.item() if torch.is_tensor(lr) else float(lr)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            from verl.utils.fsdp_utils import offload_fsdp_optimizer
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        output = DataProto()
        output.meta_info["metrics"] = metrics
        return output
