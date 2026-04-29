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

Per-sample pos/neg routing
--------------------------
When ``sample_is_negative`` is supplied (per-sample bool), the loop uses a hard
dispatch:
  - Positive samples (is_negative[i] == False): run L_green, KL biased_ref
    (fwd/rev), and the biased self-distill terms. Skip clean-ref KL terms.
  - Negative samples (is_negative[i] == True): run only the clean-ref KL terms
    (``KL(D_ref ‖ D_actor)`` and reverse). Skip all biased-teacher terms.
L_CE (if enabled) applies to all samples. The batch normalizer is still the
total response-token count, so pos/neg terms are naturally weighted by their
token share.
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
    distill_topk_biased_ref: int = 0,
    sample_is_negative: Optional[torch.Tensor] = None,
    sample_strengths: Optional[torch.Tensor] = None,
    sample_task_ids: Optional[torch.Tensor] = None,
    acrostic_bias_letter_idx: Optional[torch.Tensor] = None,
    acrostic_letter_bucket_masks: Optional[torch.Tensor] = None,
    loss_normalization_mode: str = "global",
    batch_num_pos_nonacr_tokens: Optional[float] = None,
    batch_num_neg_tokens: Optional[float] = None,
    batch_num_acr_active_tokens: Optional[float] = None,
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
        loss_normalization_mode:     "global" (legacy default) or "per_task". In "global" all
                                     terms divide by ``batch_num_tokens``. In "per_task" each
                                     term divides by its applicable token count:
                                       - L_green / pos-non-acr biased KL / self-distill →
                                         ``batch_num_pos_nonacr_tokens``
                                       - acrostic biased KL → ``batch_num_acr_active_tokens``
                                       - clean-ref KL (neg) → ``batch_num_neg_tokens``
                                       - L_CE → still ``batch_num_tokens`` (every response token)
        batch_num_pos_nonacr_tokens, batch_num_neg_tokens, batch_num_acr_active_tokens:
                                     per-task token totals, all-reduced across DP ranks. Required
                                     when loss_normalization_mode="per_task". Pass any value in
                                     "global" mode (ignored).
        distill_topk_biased_ref:     if > 0, biased-ref KL branches compute per-position top-K of
                                     the biased teacher logits (ref + bias, restricted to
                                     ``english_vocab_mask`` if provided) and do proper KL on that
                                     subspace. Clean-ref and self-distill branches unaffected.
                                     Top-K is taken on the biased teacher so bias-promoted green
                                     tokens enter the KL support.

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
            # Split into pos-non-acr (green/initials) and acrostic accumulators so
            # per-task normalization mode can divide by their respective denominators.
            l_kl_biased_ref_pos_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
            l_kl_biased_ref_acr_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        if need_reverse_kl_biased_ref:
            l_kl_biased_ref_rev_pos_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
            l_kl_biased_ref_rev_acr_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        if need_kl_ref:
            l_kl_ref_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        if need_reverse_kl_ref:
            l_kl_ref_rev_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        if need_kl_biased_actor:
            l_kl_ba_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)
        if need_reverse_kl_biased_actor:
            l_kl_ba_rev_total = torch.zeros(1, device=actor_logits.device, dtype=actor_logits.dtype)

        need_green_bias = need_kl_biased_ref or need_reverse_kl_biased_ref or need_kl_biased_actor or need_reverse_kl_biased_actor

        # Task IDs: 0=green, 1=initials, 2=neg, 3=acrostic (matches dataset.TASK_NAMES)
        TASK_ID_ACROSTIC = 3

        for i in range(num_samples):
            token_mask = sample_index == i
            if token_mask.sum() == 0:
                continue

            # Per-sample pos/neg/acrostic routing
            is_neg_i = sample_is_negative is not None and bool(sample_is_negative[i])
            is_acrostic_i = (
                sample_task_ids is not None
                and int(sample_task_ids[i].item()) == TASK_ID_ACROSTIC
            )
            # Pos-only (NON-acrostic): biased_ref KL (fwd/rev), green loss, biased self-distill
            run_biased_ref = need_kl_biased_ref and not is_neg_i and not is_acrostic_i
            run_reverse_biased_ref = need_reverse_kl_biased_ref and not is_neg_i and not is_acrostic_i
            run_green = need_green and not is_neg_i and not is_acrostic_i
            run_kl_biased_actor = need_kl_biased_actor and not is_neg_i and not is_acrostic_i
            run_reverse_kl_biased_actor = need_reverse_kl_biased_actor and not is_neg_i and not is_acrostic_i
            # Neg-only: clean ref KL (fwd/rev)
            run_kl_ref = need_kl_ref and is_neg_i
            run_reverse_kl_ref = need_reverse_kl_ref and is_neg_i
            # Acrostic-only: per-position biased KL at sentence-start positions
            run_acrostic_biased_ref = (
                need_kl_biased_ref and is_acrostic_i
                and acrostic_bias_letter_idx is not None
                and acrostic_letter_bucket_masks is not None
            )
            run_acrostic_reverse_biased_ref = (
                need_reverse_kl_biased_ref and is_acrostic_i
                and acrostic_bias_letter_idx is not None
                and acrostic_letter_bucket_masks is not None
            )

            # Shared indexing per sample
            sample_log_q = log_probs_all[token_mask]  # (n_tokens, V)

            need_green_bias_i = run_biased_ref or run_reverse_biased_ref or run_kl_biased_actor or run_reverse_kl_biased_actor
            if need_green_bias_i:
                s_i = (
                    float(sample_strengths[i].item())
                    if sample_strengths is not None
                    else strength
                )
                green_bias = s_i * green_masks[i].float().unsqueeze(0)  # (1, V)
                if english_vocab_mask is not None:
                    green_bias_eng = green_bias[:, english_vocab_mask]          # (1, E)

            # English-only actor logits slice (reused across KL terms if english_vocab_mask set)
            if english_vocab_mask is not None and (run_biased_ref or run_reverse_biased_ref or run_kl_ref or run_reverse_kl_ref or run_kl_biased_actor or run_reverse_kl_biased_actor):
                actor_logits_eng = actor_logits[token_mask][:, english_vocab_mask]  # (n, E)

            # L_green
            if run_green:
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

            # Cache sample_ref once if any ref-dependent term runs for this sample.
            _need_sample_ref_i = run_biased_ref or run_reverse_biased_ref or run_kl_ref or run_reverse_kl_ref
            if _need_sample_ref_i:
                sample_ref = ref_logits[token_mask]

            # Precompute per-position top-K on the BIASED teacher when
            # distill_topk_biased_ref is enabled. Top-K is taken on (ref + bias)
            # restricted to english if english_vocab_mask is provided.  We gather
            # raw ref / actor logits at top-K positions and re-softmax on the
            # K-subspace — proper KL on a per-position-selected vocabulary.
            if distill_topk_biased_ref > 0 and (run_biased_ref or run_reverse_biased_ref):
                _n_tok = sample_ref.shape[0]
                _biased_full = sample_ref + green_bias  # (n, V) — green_bias broadcasts from (1, V)
                if english_vocab_mask is not None:
                    # Restrict top-K pool to english.  Non-english logits are set to
                    # -inf so they rank last; but if K exceeds |english|, topk would
                    # still pull non-english into the index set (their raw logits
                    # would leak back when gathered below), polluting the support.
                    # Cap K at the english-token count to avoid that.
                    _biased_full = _biased_full.masked_fill(
                        ~english_vocab_mask.unsqueeze(0), float("-inf")
                    )
                    _max_k = int(english_vocab_mask.sum().item())
                else:
                    _max_k = _biased_full.shape[-1]
                _k = min(distill_topk_biased_ref, _max_k)
                _brtopk_idx = _biased_full.topk(_k, dim=-1).indices  # (n, K)
                # Gather raw ref + bias at top-K. green_bias is (1, V); expand to
                # (n, V) as a view so gather's dim-0 matches the index shape.
                _ref_gather = sample_ref.gather(-1, _brtopk_idx)                      # (n, K)
                _bias_gather = green_bias.expand(_n_tok, -1).gather(-1, _brtopk_idx)  # (n, K)
                _actor_gather = actor_logits[token_mask].gather(-1, _brtopk_idx)      # (n, K)

            # KL(D̂_ref ‖ D_actor) — pos-non-acrostic (green / initials) bucket
            if run_biased_ref:
                if distill_topk_biased_ref > 0:
                    log_p = F.log_softmax(_ref_gather + _bias_gather, dim=-1)
                    log_q = F.log_softmax(_actor_gather, dim=-1)
                elif english_vocab_mask is not None:
                    log_p = F.log_softmax(sample_ref[:, english_vocab_mask] + green_bias_eng, dim=-1)
                    log_q = F.log_softmax(actor_logits_eng, dim=-1)
                else:
                    log_p = F.log_softmax(sample_ref + green_bias, dim=-1)
                    log_q = sample_log_q
                p = log_p.exp()
                kl_per_token = torch.sum(p * (log_p - log_q), dim=-1)
                l_kl_biased_ref_pos_total = l_kl_biased_ref_pos_total + kl_per_token.sum()

            # KL(D_actor ‖ D̂_ref)  — reverse biased ref, pos-non-acrostic bucket
            if run_reverse_biased_ref:
                if distill_topk_biased_ref > 0:
                    log_p = F.log_softmax(_actor_gather, dim=-1)
                    log_q = F.log_softmax(_ref_gather + _bias_gather, dim=-1)
                elif english_vocab_mask is not None:
                    log_p = F.log_softmax(actor_logits_eng, dim=-1)
                    log_q = F.log_softmax(sample_ref[:, english_vocab_mask] + green_bias_eng, dim=-1)
                else:
                    log_p = sample_log_q
                    log_q = F.log_softmax(sample_ref + green_bias, dim=-1)
                p = log_p.exp()
                kl_per_token = torch.sum(p * (log_p - log_q), dim=-1)
                l_kl_biased_ref_rev_pos_total = l_kl_biased_ref_rev_pos_total + kl_per_token.sum()

            # KL(D_ref ‖ D_actor) — forward quality anchor (neg samples only)
            if run_kl_ref:
                if english_vocab_mask is not None:
                    log_p = F.log_softmax(sample_ref[:, english_vocab_mask], dim=-1)
                    log_q = F.log_softmax(actor_logits_eng, dim=-1)
                else:
                    log_p = F.log_softmax(sample_ref, dim=-1)
                    log_q = sample_log_q
                p = log_p.exp()
                kl_per_token = torch.sum(p * (log_p - log_q), dim=-1)
                l_kl_ref_total = l_kl_ref_total + kl_per_token.sum()

            # KL(D_actor ‖ D_ref) — reverse quality anchor (neg samples only)
            if run_reverse_kl_ref:
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
            if quality_green_topk > 0 and (run_kl_biased_actor or run_reverse_kl_biased_actor):
                _ref_qg = ref_logits[token_mask]  # (n_tokens, V)
                if english_vocab_mask is not None:
                    _ref_qg_sub = _ref_qg[:, english_vocab_mask]  # (n_tokens, E)
                    _, _topk_idx = _ref_qg_sub.topk(quality_green_topk, dim=-1)
                    _topk_mask = torch.zeros_like(_ref_qg_sub, dtype=torch.bool)
                    _topk_mask.scatter_(1, _topk_idx, True)
                    _green_eng = green_masks[i][english_vocab_mask].unsqueeze(0)  # (1, E)
                    sd_bias_eng = s_i * (_green_eng & _topk_mask).float()  # (n_tokens, E)
                else:
                    _, _topk_idx = _ref_qg.topk(quality_green_topk, dim=-1)
                    _topk_mask = torch.zeros_like(_ref_qg, dtype=torch.bool)
                    _topk_mask.scatter_(1, _topk_idx, True)
                    _green_full = green_masks[i].unsqueeze(0)  # (1, V)
                    sd_bias = s_i * (_green_full & _topk_mask).float()  # (n_tokens, V)
            else:
                # Fall back to uniform green bias (original behavior)
                if run_kl_biased_actor or run_reverse_kl_biased_actor:
                    if english_vocab_mask is not None:
                        sd_bias_eng = green_bias_eng  # (1, E) broadcast
                    else:
                        sd_bias = green_bias  # (1, V) broadcast

            # KL(stopgrad(D̂_actor) ‖ D_actor) — forward self-distillation
            if run_kl_biased_actor:
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
            if run_reverse_kl_biased_actor:
                if english_vocab_mask is not None:
                    log_p = F.log_softmax(actor_logits_eng, dim=-1)
                    log_q = F.log_softmax(actor_logits_eng.detach() + sd_bias_eng, dim=-1)
                else:
                    log_p = sample_log_q
                    log_q = F.log_softmax(actor_logits[token_mask].detach() + sd_bias, dim=-1)
                p = log_p.exp()
                kl_per_token = torch.sum(p * (log_p - log_q), dim=-1)
                l_kl_ba_rev_total = l_kl_ba_rev_total + kl_per_token.sum()

            # ---- Acrostic per-position biased KL (sentence-start positions only) ----
            # Bias is per-token: at active positions (acrostic_bias_letter_idx >= 0),
            # bias = strength * letter_bucket_mask[letter_idx]. KL is computed only
            # at active positions. Top-K and english_vocab_mask are applied the
            # same way as the green/initials biased path for consistency.
            if run_acrostic_biased_ref or run_acrostic_reverse_biased_ref:
                acro_bias_idx_i = acrostic_bias_letter_idx[token_mask]   # (n_tokens,)
                active_mask_i = acro_bias_idx_i >= 0
                if int(active_mask_i.sum().item()) > 0:
                    s_i = (
                        float(sample_strengths[i].item())
                        if sample_strengths is not None
                        else strength
                    )
                    # Active position slices
                    sample_actor_active = actor_logits[token_mask][active_mask_i]      # (n_act, V)
                    sample_ref_active = ref_logits[token_mask][active_mask_i]          # (n_act, V)
                    active_letter_idx = acro_bias_idx_i[active_mask_i].long()          # (n_act,)
                    active_bias_full = (
                        s_i
                        * acrostic_letter_bucket_masks[active_letter_idx].to(
                            sample_actor_active.dtype
                        )
                    )                                                                  # (n_act, V)
                    n_act = sample_actor_active.shape[0]

                    if distill_topk_biased_ref > 0:
                        biased_full = sample_ref_active + active_bias_full
                        if english_vocab_mask is not None:
                            biased_full = biased_full.masked_fill(
                                ~english_vocab_mask.unsqueeze(0), float("-inf")
                            )
                            max_k = int(english_vocab_mask.sum().item())
                        else:
                            max_k = biased_full.shape[-1]
                        k = min(distill_topk_biased_ref, max_k)
                        topk_idx = biased_full.topk(k, dim=-1).indices                # (n_act, K)
                        ref_g = sample_ref_active.gather(-1, topk_idx)
                        bias_g = active_bias_full.gather(-1, topk_idx)
                        actor_g = sample_actor_active.gather(-1, topk_idx)
                        log_p_acro = F.log_softmax(ref_g + bias_g, dim=-1)
                        log_q_acro = F.log_softmax(actor_g, dim=-1)
                    elif english_vocab_mask is not None:
                        log_p_acro = F.log_softmax(
                            sample_ref_active[:, english_vocab_mask]
                            + active_bias_full[:, english_vocab_mask],
                            dim=-1,
                        )
                        log_q_acro = F.log_softmax(
                            sample_actor_active[:, english_vocab_mask], dim=-1
                        )
                    else:
                        log_p_acro = F.log_softmax(
                            sample_ref_active + active_bias_full, dim=-1
                        )
                        log_q_acro = F.log_softmax(sample_actor_active, dim=-1)

                    if run_acrostic_biased_ref:
                        p_acro = log_p_acro.exp()
                        kl_per = torch.sum(p_acro * (log_p_acro - log_q_acro), dim=-1)
                        l_kl_biased_ref_acr_total = l_kl_biased_ref_acr_total + kl_per.sum()
                    if run_acrostic_reverse_biased_ref:
                        q_acro = log_q_acro.exp()
                        kl_per = torch.sum(q_acro * (log_q_acro - log_p_acro), dim=-1)
                        l_kl_biased_ref_rev_acr_total = l_kl_biased_ref_rev_acr_total + kl_per.sum()

        # ----- Reduce and accumulate -----
        # Each loss TERM has a single denominator = count of tokens that
        # contribute to that term ("needed-loss tokens").
        #   global   : every term uses batch_num_tokens (legacy)
        #   per_task : each term uses its own contributing-token count
        #              - biased_ref (pos non-acr + acrostic active)
        #              - clean_ref  (neg response only)
        #              - self-distill / L_green (pos non-acr response only)
        #              - L_CE (every response token, regardless of task)
        #
        # Clamp to 1.0 to avoid div-by-zero when a task is absent (numerator
        # is also 0 in that case so the term contributes 0).
        per_task = (loss_normalization_mode == "per_task")
        n_pos = float(batch_num_pos_nonacr_tokens or 0.0)
        n_neg = float(batch_num_neg_tokens or 0.0)
        n_acr = float(batch_num_acr_active_tokens or 0.0)
        if per_task:
            denom_biased_ref     = max(n_pos + n_acr, 1.0)   # biased-ref needs pos + acrostic active
            denom_neg            = max(n_neg, 1.0)
            denom_pos_only       = max(n_pos, 1.0)            # green/self-distill need only pos non-acr
        else:
            denom_biased_ref = denom_neg = denom_pos_only = float(batch_num_tokens)

        # CE always uses full batch_num_tokens (one CE term per response token)
        # — already accumulated above. No change here.

        # ---- L_green (only fires for green/initials samples) ----
        if need_green:
            l_green = l_green_total / denom_pos_only * dp_size
            loss = loss + green_loss_weight * l_green
            metrics["green_loss"] = l_green.detach().item()

        # ---- KL(D̂_ref ‖ D_actor) — single combined bucket ----
        if need_kl_biased_ref:
            l_kl_biased_ref_actor = (
                (l_kl_biased_ref_pos_total + l_kl_biased_ref_acr_total)
                / denom_biased_ref * dp_size
            )
            loss = loss + kl_biased_ref_actor_weight * l_kl_biased_ref_actor
            metrics["kl_biased_ref_actor"] = l_kl_biased_ref_actor.detach().item()

        if need_reverse_kl_biased_ref:
            l_reverse_kl_biased_ref_actor = (
                (l_kl_biased_ref_rev_pos_total + l_kl_biased_ref_rev_acr_total)
                / denom_biased_ref * dp_size
            )
            loss = loss + reverse_kl_biased_ref_actor_weight * l_reverse_kl_biased_ref_actor
            metrics["reverse_kl_biased_ref_actor"] = l_reverse_kl_biased_ref_actor.detach().item()

        # ---- Clean-ref KL (neg-only anchor) ----
        if need_kl_ref:
            l_kl_ref_actor = l_kl_ref_total / denom_neg * dp_size
            loss = loss + kl_ref_actor_weight * l_kl_ref_actor
            metrics["kl_ref_actor"] = l_kl_ref_actor.detach().item()

        if need_reverse_kl_ref:
            l_reverse_kl_ref_actor = l_kl_ref_rev_total / denom_neg * dp_size
            loss = loss + reverse_kl_ref_actor_weight * l_reverse_kl_ref_actor
            metrics["reverse_kl_ref_actor"] = l_reverse_kl_ref_actor.detach().item()

        # ---- Self-distill (pos-non-acr only — already gated above) ----
        if need_kl_biased_actor:
            l_kl_biased_actor_actor = l_kl_ba_total / denom_pos_only * dp_size
            loss = loss + kl_biased_actor_actor_weight * l_kl_biased_actor_actor
            metrics["kl_biased_actor_actor"] = l_kl_biased_actor_actor.detach().item()

        if need_reverse_kl_biased_actor:
            l_kl_biased_actor_actor_reverse = l_kl_ba_rev_total / denom_pos_only * dp_size
            loss = loss + reverse_kl_biased_actor_actor_weight * l_kl_biased_actor_actor_reverse
            metrics["kl_biased_actor_actor_reverse"] = l_kl_biased_actor_actor_reverse.detach().item()

    # ---- Per-task prob metrics (no grad, independent of loss weights) ----
    # Two independent running totals: green-only and initials-only pos samples.
    # Neg samples are skipped (fraction=0 → empty mask → degenerate ratio).
    #
    # Emitted keys (absent when corresponding task has no samples this batch):
    #   avg_green_prob           mean actor green-mass on green pos
    #   avg_green_prob_ratio     mean(p_green / fraction) on green pos
    #   avg_initial_prob         mean actor green-mass on initials pos
    #   avg_initial_prob_ratio   mean(p_green / γ) on initials pos
    #
    # Uses the dataset's task_id mapping (0=green, 1=initials, 2=neg). If
    # sample_task_ids is None, everything falls back to the green bucket
    # (preserves behavior for single-task runs without task_id column).
    with torch.no_grad():
        _probs = log_probs_all.detach().exp()
        buckets = {
            "green":   {"raw": torch.tensor(0.0, device=actor_logits.device),
                        "ratio": torch.tensor(0.0, device=actor_logits.device),
                        "tok": torch.tensor(0.0, device=actor_logits.device)},
            "initial": {"raw": torch.tensor(0.0, device=actor_logits.device),
                        "ratio": torch.tensor(0.0, device=actor_logits.device),
                        "tok": torch.tensor(0.0, device=actor_logits.device)},
        }
        # task_id → bucket name (None = skip, e.g. neg)
        TID_TO_BUCKET = {0: "green", 1: "initial"}

        for i in range(num_samples):
            if sample_is_negative is not None and bool(sample_is_negative[i]):
                continue
            if sample_task_ids is not None:
                bucket_name = TID_TO_BUCKET.get(int(sample_task_ids[i].item()))
                if bucket_name is None:
                    continue
            else:
                bucket_name = "green"  # fallback for single-task runs
            _mask = sample_index == i
            n_mask = _mask.sum()
            if n_mask == 0:
                continue
            _gp = _probs[_mask][:, green_masks[i]].sum(dim=-1)
            b = buckets[bucket_name]
            b["raw"] += _gp.sum()
            b["tok"] += n_mask
            if sample_fractions is not None:
                b["ratio"] += (_gp / sample_fractions[i]).sum()

        for bucket_name, b in buckets.items():
            if b["tok"].item() == 0:
                continue
            denom = b["tok"].clamp(min=1.0)
            metrics[f"avg_{bucket_name}_prob"] = (b["raw"] / denom).item()
            if sample_fractions is not None:
                metrics[f"avg_{bucket_name}_prob_ratio"] = (b["ratio"] / denom).item()

    metrics["total_loss"] = loss.detach().item()
    return loss, metrics
