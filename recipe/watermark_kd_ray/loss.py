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
    num_response = batch_num_tokens
    loss = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)

    # ---- L_CE ----
    if ce_loss_weight > 0:
        log_probs_all = F.log_softmax(actor_logits, dim=-1)
        log_probs_target = log_probs_all.gather(dim=-1, index=input_ids_rolled.unsqueeze(-1)).squeeze(-1)
        l_ce = -log_probs_target.sum() / num_response * dp_size

        metrics = {"ce_loss": l_ce.detach().item()}
        loss = loss + ce_loss_weight * l_ce

        num_samples = green_masks.shape[0]

    # ---- L_green: green token probability loss ----
    if green_loss_weight > 0:
        actor_probs = log_probs_all.exp()
        l_green_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        green_prob_sum = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        response_count = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)

        for i in range(num_samples):
            token_mask = sample_index == i
            if token_mask.sum() == 0:
                continue

            sample_probs = actor_probs[token_mask]           # (n_tokens, V)
            green_prob = sample_probs[:, green_masks[i]].sum(dim=-1)  # (n_tokens,)
            green_prob = green_prob.clamp(min=1e-8)

            l_green_total = l_green_total + (-torch.log(green_prob)).sum()
            green_prob_sum = green_prob_sum + green_prob.sum()
            response_count = response_count + token_mask.sum().float()

        l_green = l_green_total / num_response * dp_size
        loss = loss + green_loss_weight * l_green
        metrics["green_loss"] = l_green.detach().item()

        with torch.no_grad():
            avg_green_prob = green_prob_sum / response_count.clamp(min=1)
            metrics["avg_green_prob"] = avg_green_prob.item()

    # ---- KL(D̂_ref ‖ D_actor): biased reference as teacher ----
    if kl_biased_ref_actor_weight > 0:
        l_kl_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)

        for i in range(num_samples):
            token_mask = sample_index == i
            if token_mask.sum() == 0:
                continue

            green_bias = strength * green_masks[i].float().unsqueeze(0)  # (1, V)
            sample_ref_biased = ref_logits[token_mask] + green_bias      # (n_tokens, V)
            sample_log_q = log_probs_all[token_mask]                     # (n_tokens, V) actor log probs

            log_p = F.log_softmax(sample_ref_biased, dim=-1)
            p = log_p.exp()

            kl_per_token = torch.sum(p * (log_p - sample_log_q), dim=-1)
            l_kl_total = l_kl_total + kl_per_token.sum()

        l_kl_biased_ref_actor = l_kl_total / num_response * dp_size
        loss = loss + kl_biased_ref_actor_weight * l_kl_biased_ref_actor
        metrics["kl_biased_ref_actor"] = l_kl_biased_ref_actor.detach().item()

    # ---- KL(D_ref ‖ D_actor): unbiased reference as stability anchor ----
    if kl_ref_actor_weight > 0:
        l_kl_ref_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)

        for i in range(num_samples):
            token_mask = sample_index == i
            if token_mask.sum() == 0:
                continue

            log_p = F.log_softmax(ref_logits[token_mask], dim=-1)  # D_ref (ref_logits detached)
            p = log_p.exp()
            log_q = log_probs_all[token_mask]                       # D_actor

            kl_per_token = torch.sum(p * (log_p - log_q), dim=-1)
            l_kl_ref_total = l_kl_ref_total + kl_per_token.sum()

        l_kl_ref_actor = l_kl_ref_total / num_response * dp_size
        loss = loss + kl_ref_actor_weight * l_kl_ref_actor
        metrics["kl_ref_actor"] = l_kl_ref_actor.detach().item()

    # ---- KL(D̂_actor ‖ D_actor): biased actor vs unbiased actor ----
    if kl_biased_actor_actor_weight > 0:
        l_kl_ba_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)

        for i in range(num_samples):
            token_mask = sample_index == i
            if token_mask.sum() == 0:
                continue

            green_bias = strength * green_masks[i].float().unsqueeze(0)   # (1, V)
            actor_biased = actor_logits[token_mask] + green_bias           # D̂_actor logits
            log_p = F.log_softmax(actor_biased, dim=-1)                    # D̂_actor
            p = log_p.exp()
            log_q = log_probs_all[token_mask]                              # D_actor

            kl_per_token = torch.sum(p * (log_p - log_q), dim=-1)
            l_kl_ba_total = l_kl_ba_total + kl_per_token.sum()

        l_kl_biased_actor_actor = l_kl_ba_total / num_response * dp_size
        loss = loss + kl_biased_actor_actor_weight * l_kl_biased_actor_actor
        metrics["kl_biased_actor_actor"] = l_kl_biased_actor_actor.detach().item()

    metrics["total_loss"] = loss.detach().item()
    return loss, metrics
