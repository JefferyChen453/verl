"""
Watermark KD loss: L_CE + λ1 * L_green + λ2 * L_KL

Thin re-export of the watermark_distill loss under a new name.
All math is identical; only the function name changes.

- L_CE:    standard cross-entropy on response tokens
- L_green: encourages actor to place probability mass on green tokens
           = -log(sum(actor_probs[green_mask])) averaged over response tokens
           Each sample uses its own green mask (from per-sample seed/fraction).
- L_KL:    KL(D_ref_biased || D_actor) where D_ref_biased = softmax(ref_logits + strength * green_mask)
           Each sample uses its own green mask for the bias.
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
    green_loss_weight: float,
    kl_loss_weight: float,
    batch_num_tokens: float,
    dp_size: int,
):
    """
    Compute combined watermark KD loss on response-only flattened tensors.

    All tensors are pre-filtered to response positions only (prompt/pad excluded).

    Args:
        actor_logits:     (num_resp, vocab_size) — with grad, response tokens only
        ref_logits:       (num_resp, vocab_size) — detached, response tokens only
        input_ids_rolled: (num_resp,) — next-token labels at response positions
        sample_index:     (num_resp,) long — maps each token to its sample idx
        green_masks:      (num_samples, vocab_size) bool — per-sample green-list masks
        strength:         scalar bias added to ref logits on green positions
        green_loss_weight: λ1
        kl_loss_weight:    λ2
        batch_num_tokens:  total response tokens across all dp ranks (for normalization)
        dp_size:           data parallel world size

    Returns:
        (loss, metrics_dict)
    """
    num_response = batch_num_tokens

    # ---- L_SFT ----
    log_probs_all = F.log_softmax(actor_logits, dim=-1)
    log_probs_target = log_probs_all.gather(dim=-1, index=input_ids_rolled.unsqueeze(-1)).squeeze(-1)
    l_ce = -log_probs_target.sum() / num_response * dp_size

    metrics = {"ce_loss": l_ce.detach().item()}
    loss = l_ce

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

    # ---- KL(D_ref_biased || D_actor) ----
    if kl_loss_weight > 0:
        l_kl_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)

        for i in range(num_samples):
            token_mask = sample_index == i
            if token_mask.sum() == 0:
                continue

            green_bias = strength * green_masks[i].float().unsqueeze(0)  # (1, V)
            sample_ref_biased = ref_logits[token_mask] + green_bias      # (n_tokens, V)
            sample_log_q = log_probs_all[token_mask]                     # (n_tokens, V) actor log probs

            log_p = F.log_softmax(sample_ref_biased, dim=-1)
            p = F.softmax(sample_ref_biased, dim=-1)

            kl_per_token = torch.sum(p * (log_p - sample_log_q), dim=-1)
            l_kl_total = l_kl_total + kl_per_token.sum()

        l_kl = l_kl_total / num_response * dp_size
        loss = loss + kl_loss_weight * l_kl
        metrics["kl_loss"] = l_kl.detach().item()

    metrics["total_loss"] = loss.detach().item()
    return loss, metrics
