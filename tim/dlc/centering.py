"""The centering term C_t of DLC (docs/DLC_LOSS.md; the paper's Eq. for the DLC update): the math, torch only.

Per position, with the sampler's top-k head H: q = sampler, p = trainer now (theta_t), p0 = trainer at the sampler's
weights (theta_old). The PPO-clip surrogate on the lag ratio rho = p/p0 has the gradient A * m(a) * rho(a) * grad log p(a)
on the sampled token a (m = 1 where PPO does not clip). Under a ~ q instead of a ~ p0 its expectation is not zero:

    C = sum_k q_k m_k rho_k grad log p_k          (the drift of the numerical discrepancy, times A)

With the sampler's tail modelled as the theta_old tail rescaled, q_k = kappa * p0_k outside H,
kappa = (1 - sum_H q) / (1 - sum_H p0), and no clipping on the tail:

    C = sum_H [ q_k m_k rho_k - kappa p_k ] grad log p_k

so the loss gets  + A * sum_H sg[q_k m_k rho_k - kappa p_k] * log p_k,  as verl's score centering enters REINFORCE
(verl/trainer/ppo/score_centering.py score_centering_correction: weight w(p/q) -> m rho, alpha -> kappa).
In the code below kappa is called rho0.

The clip indicator depends on the sign of A (verl compute_policy_loss_vanilla, the gradient of its clipped loss):
    A > 0:  m = [rho <= 1 + clip_high]
    A < 0:  m = [1 - clip_low <= rho < clip_c]
so the correction is returned for both signs and the loss picks per token (`pick`).
"""
from __future__ import annotations

import torch

LOG_RATIO_CLAMP = 20.0   # as compute_policy_loss_vanilla clamps log_prob - old_log_prob


def clip_masks(ratio: torch.Tensor, clip_low: float, clip_high: float, clip_c: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Where vanilla PPO's clipped loss has a gradient, for A > 0 and for A < 0 (bool tensors like `ratio`)."""
    return ratio <= 1.0 + clip_high, (ratio >= 1.0 - clip_low) & (ratio < clip_c)


def split_sc_correction(train_head_log_probs: torch.Tensor, sampler_head_log_probs: torch.Tensor,
                        ref_head_log_probs: torch.Tensor, clip_low: float, clip_high: float, clip_c: float = 3.0,
                        eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """The centering term of DLC and its diagnostics.

    Args:
        train_head_log_probs: log p at the head ids, shape (N, k), with gradient to the trainer logits.
        sampler_head_log_probs: log q at the same ids, shape (N, k).
        ref_head_log_probs: log p0 at the same ids, shape (N, k).
        clip_low, clip_high, clip_c: the actor's PPO clip range and dual-clip bound.
        eps: floor of the tail masses in rho0 (verl's value).

    Returns, each of shape (N,), float32:
        pos, neg: sum_H sg[q m r - rho0 p] log p with the A > 0 / A < 0 clip indicator; gradient to the logits.
        q_mass, p0_mass, p_mass: head masses of the sampler, theta_0 and theta_t (detached).
        clip_pos, clip_neg: the sampler's head mass on clipped tokens, sum_H q (1 - m) (detached).
    """
    logp = train_head_log_probs.float()
    with torch.no_grad():
        logq, logp0 = sampler_head_log_probs.float(), ref_head_log_probs.float()
        p, q, p0 = logp.exp(), logq.exp(), logp0.exp()
        q_mass, p0_mass, p_mass = q.sum(-1), p0.sum(-1), p.sum(-1)
        rho0 = (1 - q_mass).clamp_min(eps) / (1 - p0_mass).clamp_min(eps)
        ratio = (logp - logp0).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP).exp()
        m_pos, m_neg = clip_masks(ratio, clip_low, clip_high, clip_c)
        tail = rho0.unsqueeze(-1) * p
        res_pos = q * ratio * m_pos - tail
        res_neg = q * ratio * m_neg - tail
        clip_pos, clip_neg = (q * ~m_pos).sum(-1), (q * ~m_neg).sum(-1)
    return {"pos": (res_pos * logp).sum(-1), "neg": (res_neg * logp).sum(-1),
            "q_mass": q_mass, "p0_mass": p0_mass, "p_mass": p_mass, "clip_pos": clip_pos, "clip_neg": clip_neg}


def pick(advantages: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Per token, the term of the token's advantage sign (A = 0: multiplied by A afterwards, either)."""
    return torch.where(advantages < 0, neg, pos)
