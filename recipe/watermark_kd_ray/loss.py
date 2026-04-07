"""
Watermark KD loss: ce_loss_weight * L_CE + green_loss_weight * L_green
                 + kl_biased_ref_actor_weight          * KL(D̂_ref  ‖ D_actor)
                 + reverse_kl_biased_ref_actor_weight  * KL(D_actor ‖ D̂_ref)
                 + kl_ref_actor_weight                 * KL(D_ref   ‖ D_actor)
                 + reverse_kl_ref_actor_weight         * KL(D_actor ‖ D_ref)
                 + kl_biased_actor_actor_weight         * KL(stopgrad(D̂_actor) ‖ D_actor)
                 + reverse_kl_biased_actor_actor_weight * KL(D_actor ‖ stopgrad(D̂_actor))

Notation:
  D_ref        = softmax(ref_logits)                          — unbiased reference
  D̂_ref        = softmax(ref_logits + strength * green_mask)  — biased reference (teacher)
  D_actor      = softmax(actor_logits)                        — unbiased actor
  D̂_actor      = softmax(actor_logits + strength * green_mask) — biased actor ("ideal self")

- L_CE:    standard cross-entropy on response tokens
- L_green: encourages actor to place probability mass on green tokens
           = -log(sum(actor_probs[green_mask])) averaged over response tokens
           Each sample uses its own green mask (from per-sample seed/fraction).
- KL(D̂_ref ‖ D_actor):            align actor with watermarked teacher (mean-seeking)
- KL(D_actor ‖ D̂_ref):            reverse KL — actor avoids mass where biased ref is low (mode-seeking)
- KL(D_ref  ‖ D_actor):            forward KD stability anchor (mean-seeking, no watermark)
- KL(D_actor ‖ D_ref):             reverse KD stability anchor (mode-seeking, no watermark)
- KL(stopgrad(D̂_actor) ‖ D_actor): forward self-distillation (mean-seeking), teacher stopgraded
- KL(D_actor ‖ stopgrad(D̂_actor)): reverse self-distillation (mode-seeking), teacher stopgraded
"""

from typing import Optional

import torch
import torch.nn.functional as F


def compute_watermark_kd_loss(
    actor_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    input_ids_rolled: torch.Tensor,
    sample_index: torch.Tensor,
    green_masks: torch.Tensor,
    strength: float,
    ce_loss_weight: float,
    green_loss_weight: float,
    kl_biased_ref_actor_weight: float,
    reverse_kl_biased_ref_actor_weight: float,
    kl_ref_actor_weight: float,
    reverse_kl_ref_actor_weight: float,
    kl_biased_actor_actor_weight: float,
    reverse_kl_biased_actor_actor_weight: float,
    batch_num_tokens: float,
    dp_size: int,
    english_vocab_mask: Optional[torch.Tensor] = None,
    green_target_ratio: float = 0.0,
    sample_fractions: Optional[torch.Tensor] = None,
    quality_green_topk: int = 0,
):
    """
    Compute combined watermark KD loss on response-only flattened tensors.

    All tensors are pre-filtered to response positions only (prompt/pad excluded).

    Args:
        actor_logits:                (num_resp, vocab_size) — with grad, response tokens only
        ref_logits:                  (num_resp, vocab_size) — detached, response tokens only
        input_ids_rolled:            (num_resp,) — next-token labels at response positions
        sample_index:                (num_resp,) long — maps each token to its sample idx
        green_masks:                 (num_samples, vocab_size) bool — per-sample green-list masks
        strength:                    scalar bias added to logits on green positions
        ce_loss_weight:              weight for L_CE (λ_ce)
        green_loss_weight:           weight for L_green (λ_green)
        kl_biased_ref_actor_weight:          weight for KL(D̂_ref ‖ D_actor) (λ_kl1)
        reverse_kl_biased_ref_actor_weight:  weight for KL(D_actor ‖ D̂_ref) (λ_kl1r)
        kl_ref_actor_weight:                 weight for KL(D_ref  ‖ D_actor) (λ_kl2)
        reverse_kl_ref_actor_weight:         weight for KL(D_actor ‖ D_ref)  (λ_kl2r)
        kl_biased_actor_actor_weight: weight for KL(stopgrad(D̂_actor) ‖ D_actor) (λ_kl3)
        reverse_kl_biased_actor_actor_weight: weight for KL(D_actor ‖ stopgrad(D̂_actor)) (λ_kl3r)
        batch_num_tokens:            total response tokens across all dp ranks (for normalization)
        dp_size:                     data parallel world size
        english_vocab_mask:          (vocab_size,) bool — if provided, KL terms are computed over
                                     English tokens only (distributions renormalized over English
                                     sub-vocabulary). CE and L_green always use the full vocab.

    Returns:
        (loss, metrics_dict)
    """
    num_samples = green_masks.shape[0]
    loss = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
    metrics = {}

    # Shared: actor log-probs (used by CE, green, and all KL terms)
    log_probs_all = F.log_softmax(actor_logits, dim=-1)

    # ---- L_CE (no per-sample loop needed) ----
    if ce_loss_weight > 0:
        log_probs_target = log_probs_all.gather(dim=-1, index=input_ids_rolled.unsqueeze(-1)).squeeze(-1)
        l_ce = -log_probs_target.sum() / batch_num_tokens * dp_size
        loss = loss + ce_loss_weight * l_ce
        metrics["ce_loss"] = l_ce.detach().item()

    # ---- Per-sample losses: single loop ----
    need_green = green_loss_weight > 0
    need_kl_biased_ref = kl_biased_ref_actor_weight > 0
    need_reverse_kl_biased_ref = reverse_kl_biased_ref_actor_weight > 0
    need_kl_ref = kl_ref_actor_weight > 0
    need_reverse_kl_ref = reverse_kl_ref_actor_weight > 0
    need_kl_biased_actor = kl_biased_actor_actor_weight > 0
    need_reverse_kl_biased_actor = reverse_kl_biased_actor_actor_weight > 0
    need_loop = need_green or need_kl_biased_ref or need_reverse_kl_biased_ref or need_kl_ref or need_reverse_kl_ref or need_kl_biased_actor or need_reverse_kl_biased_actor

    if need_loop:
        if need_green:
            actor_probs = log_probs_all.exp()
            l_green_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
            use_green_hinge = green_target_ratio > 0.0 and sample_fractions is not None
            if use_green_hinge:
                # Sequence-level hinge: target_i = fraction_i * ratio
                # Loss per sample = max(0, log(target_i / mean_t(green_prob_t)))
                # Gradient is zero for ALL positions when sample avg green prob >= target.
                green_targets = (sample_fractions.float() * green_target_ratio).clamp(max=1.0 - 1e-8)
                green_target_log = torch.log(green_targets)  # (num_samples,)
        if need_kl_biased_ref:
            l_kl_biased_ref_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        if need_reverse_kl_biased_ref:
            l_kl_biased_ref_rev_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        if need_kl_ref:
            l_kl_ref_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        if need_reverse_kl_ref:
            l_kl_ref_rev_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        if need_kl_biased_actor:
            l_kl_ba_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        if need_reverse_kl_biased_actor:
            l_kl_ba_rev_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)

        need_green_bias = need_kl_biased_ref or need_reverse_kl_biased_ref or need_kl_biased_actor or need_reverse_kl_biased_actor

        for i in range(num_samples):
            token_mask = sample_index == i
            if token_mask.sum() == 0:
                continue

            # Shared indexing per sample
            sample_log_q = log_probs_all[token_mask]  # (n_tokens, V)

            if need_green_bias:
                green_bias = strength * green_masks[i].float().unsqueeze(0)  # (1, V)
                if english_vocab_mask is not None:
                    green_bias_eng = green_bias[:, english_vocab_mask]          # (1, E)

            # English-only actor logits slice (reused across KL terms if english_vocab_mask set)
            if english_vocab_mask is not None and (need_kl_biased_ref or need_reverse_kl_biased_ref or need_kl_ref or need_reverse_kl_ref or need_kl_biased_actor or need_reverse_kl_biased_actor):
                actor_logits_eng = actor_logits[token_mask][:, english_vocab_mask]  # (n, E)

            # L_green
            if need_green:
                sample_probs = actor_probs[token_mask]
                green_prob = sample_probs[:, green_masks[i]].sum(dim=-1).clamp(min=1e-8)
                if use_green_hinge:
                    # Sequence-level: hinge on sample average, zero grad when avg >= target
                    sample_avg_green = green_prob.mean()
                    sample_loss = torch.clamp(-torch.log(sample_avg_green) - (-green_target_log[i]), min=0.0)
                    l_green_total = l_green_total + sample_loss * green_prob.shape[0]
                else:
                    per_token_loss = -torch.log(green_prob)
                    l_green_total = l_green_total + per_token_loss.sum()

            # KL(D̂_ref ‖ D_actor)
            if need_kl_biased_ref:
                sample_ref = ref_logits[token_mask]
                if english_vocab_mask is not None:
                    log_p = F.log_softmax(sample_ref[:, english_vocab_mask] + green_bias_eng, dim=-1)
                    log_q = F.log_softmax(actor_logits_eng, dim=-1)
                else:
                    log_p = F.log_softmax(sample_ref + green_bias, dim=-1)
                    log_q = sample_log_q
                p = log_p.exp()
                kl_per_token = torch.sum(p * (log_p - log_q), dim=-1)
                l_kl_biased_ref_total = l_kl_biased_ref_total + kl_per_token.sum()

            # KL(D_actor ‖ D̂_ref)  — reverse biased ref
            if need_reverse_kl_biased_ref:
                sample_ref = ref_logits[token_mask] if not need_kl_biased_ref else sample_ref
                if english_vocab_mask is not None:
                    log_p = F.log_softmax(actor_logits_eng, dim=-1)
                    log_q = F.log_softmax(sample_ref[:, english_vocab_mask] + green_bias_eng, dim=-1)
                else:
                    log_p = sample_log_q
                    log_q = F.log_softmax(sample_ref + green_bias, dim=-1)
                p = log_p.exp()
                kl_per_token = torch.sum(p * (log_p - log_q), dim=-1)
                l_kl_biased_ref_rev_total = l_kl_biased_ref_rev_total + kl_per_token.sum()

            # KL(D_ref ‖ D_actor) — forward quality anchor
            if need_kl_ref:
                sample_ref = ref_logits[token_mask] if not need_kl_biased_ref else sample_ref
                if english_vocab_mask is not None:
                    log_p = F.log_softmax(sample_ref[:, english_vocab_mask], dim=-1)
                    log_q = F.log_softmax(actor_logits_eng, dim=-1)
                else:
                    log_p = F.log_softmax(sample_ref, dim=-1)
                    log_q = sample_log_q
                p = log_p.exp()
                kl_per_token = torch.sum(p * (log_p - log_q), dim=-1)
                l_kl_ref_total = l_kl_ref_total + kl_per_token.sum()

            # KL(D_actor ‖ D_ref) — reverse quality anchor
            if need_reverse_kl_ref:
                sample_ref = ref_logits[token_mask] if not need_kl_biased_ref and not need_kl_ref else sample_ref
                if english_vocab_mask is not None:
                    log_p = F.log_softmax(actor_logits_eng, dim=-1)
                    log_q = F.log_softmax(sample_ref[:, english_vocab_mask], dim=-1)
                else:
                    log_p = sample_log_q
                    log_q = F.log_softmax(sample_ref, dim=-1)
                p = log_p.exp()
                kl_per_token = torch.sum(p * (log_p - log_q), dim=-1)
                l_kl_ref_rev_total = l_kl_ref_rev_total + kl_per_token.sum()

            # --- Quality-filtered green bias for self-distill terms ---
            # When quality_green_topk > 0, replace uniform green_bias with
            # per-position bias on green ∩ ref_topk tokens only.
            # This teaches "increase green tokens that ref also approves of".
            if quality_green_topk > 0 and (need_kl_biased_actor or need_reverse_kl_biased_actor):
                _ref_qg = ref_logits[token_mask]  # (n_tokens, V)
                if english_vocab_mask is not None:
                    _ref_qg_sub = _ref_qg[:, english_vocab_mask]  # (n_tokens, E)
                    _, _topk_idx = _ref_qg_sub.topk(quality_green_topk, dim=-1)
                    _topk_mask = torch.zeros_like(_ref_qg_sub, dtype=torch.bool)
                    _topk_mask.scatter_(1, _topk_idx, True)
                    _green_eng = green_masks[i][english_vocab_mask].unsqueeze(0)  # (1, E)
                    sd_bias_eng = strength * (_green_eng & _topk_mask).float()  # (n_tokens, E)
                else:
                    _, _topk_idx = _ref_qg.topk(quality_green_topk, dim=-1)
                    _topk_mask = torch.zeros_like(_ref_qg, dtype=torch.bool)
                    _topk_mask.scatter_(1, _topk_idx, True)
                    _green_full = green_masks[i].unsqueeze(0)  # (1, V)
                    sd_bias = strength * (_green_full & _topk_mask).float()  # (n_tokens, V)
            else:
                # Fall back to uniform green bias (original behavior)
                if need_kl_biased_actor or need_reverse_kl_biased_actor:
                    if english_vocab_mask is not None:
                        sd_bias_eng = green_bias_eng  # (1, E) broadcast
                    else:
                        sd_bias = green_bias  # (1, V) broadcast

            # KL(stopgrad(D̂_actor) ‖ D_actor) — forward self-distillation
            if need_kl_biased_actor:
                if english_vocab_mask is not None:
                    log_p = F.log_softmax(actor_logits_eng.detach() + sd_bias_eng, dim=-1)
                    log_q = F.log_softmax(actor_logits_eng, dim=-1)
                else:
                    log_p = F.log_softmax(actor_logits[token_mask].detach() + sd_bias, dim=-1)
                    log_q = sample_log_q
                p = log_p.exp()
                kl_per_token = torch.sum(p * (log_p - log_q), dim=-1)
                l_kl_ba_total = l_kl_ba_total + kl_per_token.sum()

            # KL(D_actor ‖ stopgrad(D̂_actor)) — reverse self-distillation
            if need_reverse_kl_biased_actor:
                if english_vocab_mask is not None:
                    log_p = F.log_softmax(actor_logits_eng, dim=-1)
                    log_q = F.log_softmax(actor_logits_eng.detach() + sd_bias_eng, dim=-1)
                else:
                    log_p = sample_log_q
                    log_q = F.log_softmax(actor_logits[token_mask].detach() + sd_bias, dim=-1)
                p = log_p.exp()
                kl_per_token = torch.sum(p * (log_p - log_q), dim=-1)
                l_kl_ba_rev_total = l_kl_ba_rev_total + kl_per_token.sum()

        # Reduce and accumulate
        if need_green:
            l_green = l_green_total / batch_num_tokens * dp_size
            loss = loss + green_loss_weight * l_green
            metrics["green_loss"] = l_green.detach().item()

        if need_kl_biased_ref:
            l_kl_biased_ref_actor = l_kl_biased_ref_total / batch_num_tokens * dp_size
            loss = loss + kl_biased_ref_actor_weight * l_kl_biased_ref_actor
            metrics["kl_biased_ref_actor"] = l_kl_biased_ref_actor.detach().item()

        if need_reverse_kl_biased_ref:
            l_reverse_kl_biased_ref_actor = l_kl_biased_ref_rev_total / batch_num_tokens * dp_size
            loss = loss + reverse_kl_biased_ref_actor_weight * l_reverse_kl_biased_ref_actor
            metrics["reverse_kl_biased_ref_actor"] = l_reverse_kl_biased_ref_actor.detach().item()

        if need_kl_ref:
            l_kl_ref_actor = l_kl_ref_total / batch_num_tokens * dp_size
            loss = loss + kl_ref_actor_weight * l_kl_ref_actor
            metrics["kl_ref_actor"] = l_kl_ref_actor.detach().item()

        if need_reverse_kl_ref:
            l_reverse_kl_ref_actor = l_kl_ref_rev_total / batch_num_tokens * dp_size
            loss = loss + reverse_kl_ref_actor_weight * l_reverse_kl_ref_actor
            metrics["reverse_kl_ref_actor"] = l_reverse_kl_ref_actor.detach().item()

        if need_kl_biased_actor:
            l_kl_biased_actor_actor = l_kl_ba_total / batch_num_tokens * dp_size
            loss = loss + kl_biased_actor_actor_weight * l_kl_biased_actor_actor
            metrics["kl_biased_actor_actor"] = l_kl_biased_actor_actor.detach().item()

        if need_reverse_kl_biased_actor:
            l_kl_biased_actor_actor_reverse = l_kl_ba_rev_total / batch_num_tokens * dp_size
            loss = loss + reverse_kl_biased_actor_actor_weight * l_kl_biased_actor_actor_reverse
            metrics["kl_biased_actor_actor_reverse"] = l_kl_biased_actor_actor_reverse.detach().item()

    # ---- Always: green prob metrics (no grad, independent of loss weights) ----
    with torch.no_grad():
        _probs = log_probs_all.detach().exp()
        _raw_sum = torch.tensor(0.0, device=actor_logits.device)
        _ratio_sum = torch.tensor(0.0, device=actor_logits.device)
        for i in range(num_samples):
            _mask = sample_index == i
            if _mask.sum() == 0:
                continue
            _gp = _probs[_mask][:, green_masks[i]].sum(dim=-1)
            _raw_sum += _gp.sum()
            if sample_fractions is not None:
                _ratio_sum += (_gp / sample_fractions[i]).sum()
        total_tokens_this_rank = batch_num_tokens / dp_size
        metrics["avg_green_prob"] = (_raw_sum / total_tokens_this_rank).item()
        if sample_fractions is not None:
            metrics["avg_green_prob_ratio"] = (_ratio_sum / total_tokens_this_rank).item()

    metrics["total_loss"] = loss.detach().item()
    return loss, metrics
