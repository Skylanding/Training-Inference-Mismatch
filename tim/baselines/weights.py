"""Per-batch math of the two published baselines, M2PO and VCPO (docs/BASELINES.md).

Pure torch on [B, T] tensors, no verl imports, so tests/test_baseline_weights.py runs on CPU. The hooks that feed these
functions inside verl are in tim/baselines/hooks.py; the source lines quoted below are the official code.

M2PO  Zheng, Zhao, Chen, "Prosperity before Collapse" (arXiv 2510.01161), github Infini-AI-Lab/M2PO @ af54a3e,
      verl/trainer/ppo/core_algos.py `_solve_tau_from_sorted_delta2`, `_get_trust_region_tokens_delta_sq`,
      `kpo_clip_harmful_tokens`, `compute_m2po_policy_loss` (miniclip_low/high = None as in the run scripts).
      Trust-region tokens: (A > 0 and r > 1) or (A < 0 and r < 1), r = p_theta / p_old, the sign of A taken from the first
      response position (outcome reward). If the mean of delta^2 = (log r)^2 over them exceeds the budget tau = 0.04, a
      single cap c on delta^2 is solved so that sum_i min(delta_i^2, c) = tau * n, and the PPO clip with bounds
      [e^{-sqrt c}, e^{sqrt c}] is applied. With one optimizer step per batch the ratio at the update is r(theta_t), so
      the clip is the 0/1 keep mask returned by `m2po_keep_mask` (dropped tokens stay in the token-mean denominator).

VCPO  Huang, Zhang, Hu, Yang, Han, "Stable Asynchrony" (arXiv 2602.17616 v2), github mit-han-lab/vcpo @ c7f030f,
      recipe/fully_async_policy/staleness_utils.py `compute_ess_info`, `compute_opob_baseline`, `get_weighted_mean`;
      verl/workers/actor/megatron_actor.py L881-896 (lr scaling), L756-790 `_compute_grad_norms`, L993-1060.
      Sequence-level ratio w_i = exp(sum_t log p_theta - log q), truncated at c = 8 in the loss; ESS ratio on the
      UNtruncated w; lr scale sqrt(min(1, rho / rho_on)), rho_on = 1.0 (Countdown scripts); optimal off-policy baseline
      b_g = sum_i W_i R_i / (sum_i W_i + 1e-8) per prompt group with W_i = ||g_i||^2 w_i^2, where the code's ||g_i|| is the
      norm of grad(wbar_i * sum_t log p_theta) (the trajectory's own score gradient times its truncated weight).
"""
from __future__ import annotations

import math

import torch

M2PO_BUDGET = 0.04
M2PO_NO_CLIP = 100000.0          # the official code's "no clipping needed" tau
VCPO_TIS_C = 8.0
VCPO_RHO_ON = 1.0
EPS = 1e-8


# ---------------------------------------------------------------------------------------------------------------- M2PO
def m2po_solve_tau(sorted_delta2: torch.Tensor, target_sum: float) -> tuple[float, float]:
    """Port of `_solve_tau_from_sorted_delta2` (ascending v_k = delta_k^2, target S = tau_budget * n).
    Returns (tau, M2 after capping). tau = M2PO_NO_CLIP when no cap is needed, 0.0 when everything is capped."""
    n = sorted_delta2.numel()
    if n == 0:
        return M2PO_NO_CLIP, 0.0
    v = sorted_delta2.double()
    total = float(v.sum())
    if target_sum >= total - 1e-12:
        return M2PO_NO_CLIP, total / n
    if target_sum <= 1e-12:
        return 0.0, 0.0
    csum = torch.cumsum(v, dim=0)
    k_idx = torch.arange(n, dtype=torch.float64)
    reach = (v - 1e-12) * (n - k_idx - 1) + csum >= target_sum - 1e-12
    k = int(torch.nonzero(reach)[0])                    # first k, as the official loop
    if k == 0:
        return 0.0, float(csum[-1]) / n                 # official return value kept as is
    m2_after = (float(v[k - 1]) * (n - k) + float(csum[k - 1])) / n
    return math.sqrt(float(v[k - 1]) - 1e-12), m2_after


def m2po_keep_mask(log_prob: torch.Tensor, old_log_prob: torch.Tensor, advantages: torch.Tensor,
                   response_mask: torch.Tensor, budget: float = M2PO_BUDGET) -> tuple[torch.Tensor, dict]:
    """0/1 weight per token equal to the official M2PO PPO clip at r = exp(log_prob - old_log_prob).

    log_prob: trainer at theta_t (the weights the single update starts from); old_log_prob: trainer at the stale sampler's
    weights (M2PO's denominator, the recomputed log-prob of the generating model). Shapes [B, T]."""
    mask = response_mask.bool()
    delta = (old_log_prob - log_prob).double()          # official sign: log p_old - log p_new
    ratio = torch.exp(-delta)
    seq_sign = advantages[:, :1].double()               # official: advantages[:, 0]
    pos, neg = seq_sign > 1e-12, seq_sign < -1e-12
    tr = mask & ((pos & (ratio > 1.0 + 1e-12)) | (neg & (ratio < 1.0 - 1e-12)))
    d2 = delta.pow(2)[tr]
    n = d2.numel()
    m2_now = float(d2.mean()) if n else 0.0
    tau, m2_after = M2PO_NO_CLIP, m2_now
    if n and m2_now > budget + 1e-12:
        tau, m2_after = m2po_solve_tau(torch.sort(d2).values, budget * n)
    hi, lo = math.exp(min(tau, 50.0)), math.exp(-min(tau, 50.0))
    clipped = mask & ((pos & (ratio > hi)) | (neg & (ratio < lo)))   # PPO max/min: zero gradient exactly here
    keep = (~clipped).to(log_prob.dtype)
    n_tok = max(int(mask.sum()), 1)
    stats = {"m2po/M2": m2_now, "m2po/M2_after": m2_after, "m2po/tau": tau, "m2po/tr_frac": n / n_tok,
             "m2po/masked_frac": float(clipped.sum()) / n_tok}
    return keep, stats


# ---------------------------------------------------------------------------------------------------------------- VCPO
def seq_log_ratio(log_prob: torch.Tensor, rollout_log_prob: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """sum_t (log p_theta - log q) per sequence, [B]."""
    return ((log_prob - rollout_log_prob).double() * response_mask.double()).sum(-1)


def vcpo_ess_ratio(seq_w: torch.Tensor) -> float:
    """compute_ess_info on the untruncated sequence weights: (sum w)^2 / (sum w^2 + 1e-8) / B."""
    w = seq_w.double()
    return float(w.sum() ** 2 / (w.pow(2).sum() + EPS) / max(w.numel(), 1))


def vcpo_lr_scale(ess_ratio: float, rho_on: float = VCPO_RHO_ON, rule: str = "sqrt") -> float:
    """megatron_actor.py L881-896: lr = lr0 * sqrt(min(1, rho / rho_on)) (rule 'linear': without the sqrt)."""
    s = min(1.0, float(ess_ratio) / max(float(rho_on), 1e-8))
    if rule == "sqrt":
        return s ** 0.5
    if rule == "linear":
        return s
    raise ValueError(f"rule={rule!r}: sqrt or linear")


def vcpo_opob_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, seq_w: torch.Tensor,
                         score_norm_sq: torch.Tensor, c: float = VCPO_TIS_C) -> tuple[torch.Tensor, dict]:
    """Per-sequence advantage (R_i - b_g) / sigma_g with the optimal off-policy baseline of each prompt group.

    rewards [B]; group_ids [B] (same id = same prompt); seq_w [B] untruncated sequence weights; score_norm_sq [B] =
    ||grad_theta sum_t log p_theta(y_t)||^2 of each sequence on its own (the pre-pass of tim.baselines.hooks).
    The official ||g_i|| is the norm of grad(wbar_i * sum_t log p), so W_i = wbar_i^2 * score_norm_sq_i * w_i^2.
    sigma_g: population std of the group's rewards; no division when sigma_g <= 1e-8."""
    r, w = rewards.double(), seq_w.double()
    wbar = w.clamp(max=c)
    weight = wbar.pow(2) * score_norm_sq.double() * w.pow(2)
    adv = torch.empty_like(r)
    baselines = []
    for g in torch.unique(group_ids):
        idx = group_ids == g
        b = (weight[idx] * r[idx]).sum() / (weight[idx].sum() + EPS)
        sd = r[idx].std(unbiased=False)
        adv[idx] = (r[idx] - b) / sd if float(sd) > 1e-8 else r[idx] - b
        baselines.append(float(b))
    return adv.to(rewards.dtype), {"vcpo/opob_baseline_mean": sum(baselines) / len(baselines)}
