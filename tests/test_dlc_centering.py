"""tim.dlc.centering (docs/DLC_LOSS.md): the centering term C_t of DLC, torch on the CPU.
    CUDA_VISIBLE_DEVICES= python -m pytest tests/test_dlc_centering.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

torch = pytest.importorskip("torch")
from tim.dlc.centering import clip_masks, pick, split_sc_correction  # noqa: E402

N, V, K = 48, 60, 12
WIDE = dict(clip_low=1.0, clip_high=1e9, clip_c=1e12)     # no token clipped for either advantage sign


def _case(seed=0, lag=0.3, engine=0.3):
    """theta_0 logits, theta_t logits (lag on the head ids only, so r is one constant on the tail), a sampler q that is
    p0 * exp(noise) on the head and rho0 * p0 on the tail (the tail model of the correction, exactly)."""
    g = torch.Generator().manual_seed(seed)
    logits0 = torch.randn(N, V, generator=g, dtype=torch.float64) * 2
    logp0 = logits0.log_softmax(-1)
    ids = logp0.argsort(-1, descending=True)[:, :K]
    delta = torch.zeros(N, V, dtype=torch.float64).scatter(1, ids, torch.randn(N, K, generator=g, dtype=torch.float64) * lag)
    logits = (logits0 + delta).requires_grad_(True)
    p0_head = logp0.gather(1, ids).exp()
    q_head = p0_head * (torch.randn(N, K, generator=g, dtype=torch.float64) * engine).exp()
    q_head = q_head * (0.5 * (1 + p0_head.sum(-1, keepdim=True)) / q_head.sum(-1, keepdim=True)).clamp(max=1.0)  # mass < 1
    rho0 = (1 - q_head.sum(-1, keepdim=True)) / (1 - p0_head.sum(-1, keepdim=True))
    q = (rho0 * logp0.exp()).scatter(1, ids, q_head)
    assert torch.allclose(q.sum(-1), torch.ones(N, dtype=torch.float64))
    return logits, logp0, q, ids


def _heads(logits, logp0, q, ids):
    return logits.log_softmax(-1).gather(1, ids), q.gather(1, ids).log(), logp0.gather(1, ids)


def test_no_engine_error_gives_zero():
    logits, logp0, _, ids = _case()
    train, _, ref = _heads(logits, logp0, logp0.exp(), ids)
    out = split_sc_correction(train, ref, ref, **WIDE)               # q = p0 on the head (and on the tail: rho0 = 1)
    for key in ("pos", "neg"):
        grad, = torch.autograd.grad(out[key].sum(), logits, retain_graph=True)
        assert grad.abs().max() < 1e-4, key


def test_no_lag_is_verl_score_centering():
    verl_sc = pytest.importorskip("verl.trainer.ppo.score_centering")
    logits, logp0, q, ids = _case(lag=0.0)
    train, sampler, _ = _heads(logits, logp0, q, ids)
    out = split_sc_correction(train, sampler, train.detach(), clip_low=0.2, clip_high=0.2)   # r = 1: nothing clipped
    ref, q_mass, p_mass = verl_sc.score_centering_correction(train, sampler, torch.ones_like)
    want, = torch.autograd.grad(ref.sum(), logits, retain_graph=True)
    for key in ("pos", "neg"):
        assert torch.allclose(out[key], ref, atol=1e-4), key
        got, = torch.autograd.grad(out[key].sum(), logits, retain_graph=True)
        assert torch.allclose(got, want, atol=1e-4), key
    assert torch.allclose(out["q_mass"], q_mass, atol=1e-6) and torch.allclose(out["p_mass"], p_mass, atol=1e-6)
    assert out["clip_pos"].abs().max() == 0 and out["clip_neg"].abs().max() == 0


@pytest.mark.parametrize("sign", ["pos", "neg"])
def test_gradient_is_the_full_vocab_expectation(sign):
    """grad of the term == sum over the WHOLE vocabulary of q_k m_k r_k grad log p_k, when the sampler's tail follows
    the rho0 model and no tail token is clipped (the tail's r is one constant near 1, inside the clip range)."""
    logits, logp0, q, ids = _case(lag=0.5, engine=0.4)
    clip = dict(clip_low=0.2, clip_high=0.2, clip_c=3.0)
    logp = logits.log_softmax(-1)
    ratio = (logp - logp0).detach().exp()
    m_pos, m_neg = clip_masks(ratio, **clip)
    m = m_pos if sign == "pos" else m_neg
    head = torch.zeros(N, V, dtype=torch.bool).scatter(1, ids, True)
    rows = (m | head).all(-1)                                         # rows whose tail is unclipped
    assert rows.sum() >= 8 and (~m & head)[rows].any()                # and some head tokens are clipped: a real test
    exact, = torch.autograd.grad(((q * m * ratio) * logp)[rows].sum(), logits, retain_graph=True)
    out = split_sc_correction(*_heads(logits, logp0, q, ids), **clip)
    got, = torch.autograd.grad(out[sign][rows].sum(), logits)
    assert torch.allclose(got.double(), exact, atol=1e-4), (got.double() - exact).abs().max()
    clipped = (q * ~m * head).sum(-1)
    assert torch.allclose(out["clip_" + sign].double(), clipped, atol=1e-6)


def test_pick_follows_the_advantage_sign():
    adv = torch.tensor([1.0, -1.0, 0.0])
    assert pick(adv, torch.tensor([1.0, 1.0, 1.0]), torch.tensor([2.0, 2.0, 2.0])).tolist() == [1.0, 2.0, 1.0]
