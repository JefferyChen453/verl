"""
Watermark KD loss: ce_loss_weight * L_CE + green_loss_weight * L_green
                 + kl_biased_ref_actor_weight  * KL(D̂_ref  ‖ D_actor)
                 + kl_ref_actor_weight         * KL(D_ref   ‖ D_actor)
                 + kl_biased_actor_actor_weight * KL(D̂_actor ‖ D_actor)

Notation:
  D_ref        = softmax(ref_logits)                          — unbiased reference
  D̂_ref        = softmax(ref_logits + strength * green_mask)  — biased reference (teacher)
  D_actor      = softmax(actor_logits)                        — unbiased actor
  D̂_actor      = softmax(actor_logits + strength * green_mask) — biased actor ("ideal self")

- L_CE:    standard cross-entropy on response tokens
- L_green: encourages actor to place probability mass on green tokens
           = -log(sum(actor_probs[green_mask])) averaged over response tokens
           Each sample uses its own green mask (from per-sample seed/fraction).
- KL(D̂_ref ‖ D_actor):   align actor with watermarked teacher (original KL term)
- KL(D_ref  ‖ D_actor):   standard KD stability anchor (no watermark)
- KL(D̂_actor ‖ D_actor):  encourage actor to already favour green tokens naturally
"""

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
    kl_ref_actor_weight: float,
    kl_biased_actor_actor_weight: float,
    batch_num_tokens: float,
    dp_size: int,
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
        kl_biased_ref_actor_weight:  weight for KL(D̂_ref ‖ D_actor) (λ_kl1)
        kl_ref_actor_weight:         weight for KL(D_ref  ‖ D_actor) (λ_kl2)
        kl_biased_actor_actor_weight: weight for KL(D̂_actor ‖ D_actor) (λ_kl3)
        batch_num_tokens:            total response tokens across all dp ranks (for normalization)
        dp_size:                     data parallel world size

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
    need_kl_ref = kl_ref_actor_weight > 0
    need_kl_biased_actor = kl_biased_actor_actor_weight > 0
    need_loop = need_green or need_kl_biased_ref or need_kl_ref or need_kl_biased_actor

    if need_loop:
        if need_green:
            actor_probs = log_probs_all.exp()
            l_green_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
            green_prob_sum = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        if need_kl_biased_ref:
            l_kl_biased_ref_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        if need_kl_ref:
            l_kl_ref_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        if need_kl_biased_actor:
            l_kl_ba_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)

        need_green_bias = need_kl_biased_ref or need_kl_biased_actor

        for i in range(num_samples):
            token_mask = sample_index == i
            if token_mask.sum() == 0:
                continue

            # Shared indexing per sample
            sample_log_q = log_probs_all[token_mask]  # (n_tokens, V)

            if need_green_bias:
                green_bias = strength * green_masks[i].float().unsqueeze(0)  # (1, V)

            # L_green
            if need_green:
                sample_probs = actor_probs[token_mask]
                green_prob = sample_probs[:, green_masks[i]].sum(dim=-1).clamp(min=1e-8)
                l_green_total = l_green_total + (-torch.log(green_prob)).sum()
                green_prob_sum = green_prob_sum + green_prob.sum()

            # KL(D̂_ref ‖ D_actor)
            if need_kl_biased_ref:
                sample_ref = ref_logits[token_mask]
                log_p = F.log_softmax(sample_ref + green_bias, dim=-1)
                p = log_p.exp()
                kl_per_token = torch.sum(p * (log_p - sample_log_q), dim=-1)
                l_kl_biased_ref_total = l_kl_biased_ref_total + kl_per_token.sum()

            # KL(D_ref ‖ D_actor)
            if need_kl_ref:
                sample_ref = ref_logits[token_mask] if not need_kl_biased_ref else sample_ref
                log_p = F.log_softmax(sample_ref, dim=-1)
                p = log_p.exp()
                kl_per_token = torch.sum(p * (log_p - sample_log_q), dim=-1)
                l_kl_ref_total = l_kl_ref_total + kl_per_token.sum()

            # KL(D̂_actor ‖ D_actor)
            if need_kl_biased_actor:
                log_p = F.log_softmax(actor_logits[token_mask] + green_bias, dim=-1)
                p = log_p.exp()
                kl_per_token = torch.sum(p * (log_p - sample_log_q), dim=-1)
                l_kl_ba_total = l_kl_ba_total + kl_per_token.sum()

        # Reduce and accumulate
        if need_green:
            l_green = l_green_total / batch_num_tokens * dp_size
            loss = loss + green_loss_weight * l_green
            metrics["green_loss"] = l_green.detach().item()
            with torch.no_grad():
                total_tokens_this_rank = batch_num_tokens / dp_size
                avg_green_prob = green_prob_sum / total_tokens_this_rank
                metrics["avg_green_prob"] = avg_green_prob.item()

        if need_kl_biased_ref:
            l_kl_biased_ref_actor = l_kl_biased_ref_total / batch_num_tokens * dp_size
            loss = loss + kl_biased_ref_actor_weight * l_kl_biased_ref_actor
            metrics["kl_biased_ref_actor"] = l_kl_biased_ref_actor.detach().item()

        if need_kl_ref:
            l_kl_ref_actor = l_kl_ref_total / batch_num_tokens * dp_size
            loss = loss + kl_ref_actor_weight * l_kl_ref_actor
            metrics["kl_ref_actor"] = l_kl_ref_actor.detach().item()

        if need_kl_biased_actor:
            l_kl_biased_actor_actor = l_kl_ba_total / batch_num_tokens * dp_size
            loss = loss + kl_biased_actor_actor_weight * l_kl_biased_actor_actor
            metrics["kl_biased_actor_actor"] = l_kl_biased_actor_actor.detach().item()

    metrics["total_loss"] = loss.detach().item()
    return loss, metrics
