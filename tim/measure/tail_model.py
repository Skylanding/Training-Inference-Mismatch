"""Head + modelled-tail algebra for sampler top-k heads (numpy, no torch).

The sampler exposes only its top-k head H (exact). Its tail is modelled as the trainer's tail rescaled to the sampler's
tail mass (Score Centering paper App. A.1 / Eq. 7), exactly as verl PR #8010 does it:

    # Copied (math) from verl/trainer/ppo/score_centering.py::score_centering_correction @ verl 8718ca30
    p_mass, q_mass = sum_H p, sum_H q                       # trainer / sampler head masses
    rho   = max(1 - q_mass, eps) / max(1 - p_mass, eps)     # eps = 1e-6
    q_tail(v) = rho * p(v)   for v not in H                 # the tail model

so the sampler-vs-trainer KL over the full vocabulary needs only the two heads:

    KL(q || p) = sum_H q (log q - log p)  +  sum_tail rho p log rho  =  sum_H q (log q - log p) + (1 - q_mass) log rho

(second term = tail mass of q times log rho). The GPU test tests/gpu/test_tail_model_vs_verl.py asserts that head
masses and rho agree with verl's `score_centering_correction`/its internals on random heads.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6   # verl score_centering_correction default


def head_mass(log_probs: np.ndarray) -> np.ndarray:
    """sum over the head of exp(log p); shape (..., K) -> (...)."""
    return np.exp(np.asarray(log_probs, dtype=np.float64)).sum(-1)


def rho(q_mass: np.ndarray, p_mass: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Tail rescaling factor (1 - q_mass) / (1 - p_mass) with the same eps floors as verl."""
    return np.maximum(1.0 - q_mass, eps) / np.maximum(1.0 - p_mass, eps)


def kl_with_tail(q_lp: np.ndarray, p_lp: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Per-row KL(q || p), q = sampler, p = trainer, both given as log-probs on the SAME head ids (n, K).

    The head ids are the sampler's top-k, p_lp the trainer's full-vocab-normalised log-probs at those ids.
    Returns float64 (n,). Non-negative up to the tail-model approximation (tiny negatives are not clipped on purpose).
    """
    q_lp = np.asarray(q_lp, dtype=np.float64)
    p_lp = np.asarray(p_lp, dtype=np.float64)
    q = np.exp(q_lp)
    q_mass, p_mass = q.sum(-1), np.exp(p_lp).sum(-1)
    head = (q * (q_lp - p_lp)).sum(-1)
    tail_q = np.maximum(1.0 - q_mass, 0.0)
    return head + tail_q * np.log(rho(q_mass, p_mass, eps))


def reverse_kl_with_tail(q_lp: np.ndarray, p_lp: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Per-row KL(p || q) under the same tail model (tail: p_tail * log(1/rho))."""
    q_lp = np.asarray(q_lp, dtype=np.float64)
    p_lp = np.asarray(p_lp, dtype=np.float64)
    p = np.exp(p_lp)
    q_mass, p_mass = np.exp(q_lp).sum(-1), p.sum(-1)
    head = (p * (p_lp - q_lp)).sum(-1)
    tail_p = np.maximum(1.0 - p_mass, 0.0)
    return head - tail_p * np.log(rho(q_mass, p_mass, eps))
