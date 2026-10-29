"""CPU tests of tim/baselines/weights.py against the official M2PO / VCPO definitions (docs/BASELINES.md)."""
import math

import pytest

torch = pytest.importorskip("torch")

from tim.baselines.weights import (M2PO_NO_CLIP, m2po_keep_mask, m2po_solve_tau, vcpo_ess_ratio, vcpo_lr_scale,
                                         vcpo_opob_advantages)


def official_solve_tau(sorted_delta2, target_sum):
    """The official loop, verbatim logic (Infini-AI-Lab/M2PO core_algos.py)."""
    total = float(sorted_delta2.sum().item())
    if target_sum >= total - 1e-12:
        return 100000
    if target_sum <= 1e-12:
        return 0.0
    csum = torch.cumsum(sorted_delta2, dim=0)
    n = sorted_delta2.numel()
    for k in range(n):
        left_sum = float(csum[k].item())
        rest = n - k - 1
        m2 = sorted_delta2[k].item() - 1e-12
        if m2 * rest + left_sum >= target_sum - 1e-12:
            if k == 0:
                return 0.0
            return float(sorted_delta2[k - 1].item() - 1e-12) ** 0.5
    return 100000


def test_solve_tau_matches_official_loop():
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        v = torch.sort(torch.rand(int(torch.randint(1, 60, (1,), generator=g)), generator=g, dtype=torch.float64)
                       ** 4 * 3).values
        target = float(torch.rand(1, generator=g)) * float(v.sum()) * 1.2
        ours, _ = m2po_solve_tau(v, target)
        ref = official_solve_tau(v, target)
        assert math.isclose(ours, ref, rel_tol=1e-9, abs_tol=1e-12) or (ours == M2PO_NO_CLIP and ref == 100000)


def test_capped_m2_meets_budget():
    v = torch.sort(torch.tensor([0.001, 0.002, 0.01, 0.5, 2.0, 9.0], dtype=torch.float64)).values
    tau, m2_after = m2po_solve_tau(v, 0.04 * v.numel())
    assert tau < M2PO_NO_CLIP
    assert float(torch.minimum(v, torch.tensor(tau ** 2)).mean()) <= 0.04 + 1e-9
    assert m2_after <= 0.04 + 1e-9


def test_keep_mask_drops_only_harmful_large_ratios():
    # harmful (trust-region) d^2: 1e-4, 4e-4, 0.09, 2.25 (n = 4, S = 0.16). The official loop stops at k = 2 and caps at
    # v_1 = 4e-4 (it returns v_{k-1}, minus 1e-12), so the harmful tokens with d^2 >= 4e-4 lose their gradient.
    lp_old = torch.zeros(2, 4)
    lp = torch.tensor([[0.01, 0.3, -1.5, 0.0],       # A > 0: r = e^0.01, e^0.3 harmful; r = e^-1.5 not harmful
                       [-0.02, -1.5, 1.5, 0.0]])     # A < 0: r = e^-0.02, e^-1.5 harmful; r = e^1.5 not harmful
    adv = torch.tensor([[1.0] * 4, [-1.0] * 4])
    keep, stats = m2po_keep_mask(lp, lp_old, adv, torch.ones(2, 4), budget=0.04)
    assert math.isclose(stats["m2po/tau"], 0.02, rel_tol=1e-6)
    assert keep.tolist() == [[1, 0, 1, 1], [0, 0, 1, 1]]


def test_keep_mask_no_clip_under_budget():
    lp_old = torch.zeros(1, 3)
    lp = torch.tensor([[0.01, -0.01, 0.1]])
    keep, stats = m2po_keep_mask(lp, lp_old, torch.ones(1, 3), torch.ones(1, 3))
    assert stats["m2po/tau"] == M2PO_NO_CLIP and keep.sum() == 3


def test_keep_mask_ignores_padding():
    lp_old = torch.zeros(1, 3)
    lp = torch.tensor([[0.0, 0.0, 5.0]])
    keep, stats = m2po_keep_mask(lp, lp_old, torch.ones(1, 3), torch.tensor([[1.0, 1.0, 0.0]]))
    assert stats["m2po/M2"] == 0.0 and keep.sum() == 3


def test_ess_ratio_and_lr_scale():
    assert math.isclose(vcpo_ess_ratio(torch.ones(512)), 1.0, rel_tol=1e-9)
    w = torch.zeros(512, dtype=torch.float64)
    w[0] = 1.0
    assert math.isclose(vcpo_ess_ratio(w), 1 / 512, rel_tol=1e-6)
    assert math.isclose(vcpo_lr_scale(0.25), 0.5)
    assert vcpo_lr_scale(1.7) == 1.0                      # clamp at 1 (code)
    assert math.isclose(vcpo_lr_scale(0.25, rule="linear"), 0.25)


def test_opob_reduces_to_weighted_group_mean():
    r = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
    gid = torch.tensor([0, 0, 0, 0, 1, 1])
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 0.5])
    s = torch.tensor([1.0, 3.0, 1.0, 3.0, 1.0, 1.0])
    adv, _ = vcpo_opob_advantages(r, gid, w, s)
    b0 = (1 * 1 - 3 - 0 + 1 - 3) / 8                   # weights 1, 3, 1, 3 on rewards 1, -1, 1, -1
    assert torch.allclose(adv[:4], (r[:4] - b0) / r[:4].std(unbiased=False), atol=1e-6)
    assert torch.allclose(adv[4:], torch.zeros(2), atol=1e-6)   # constant rewards: R - b = 0, no std division


def test_opob_uses_truncated_weight_in_norm():
    r = torch.tensor([1.0, -1.0])
    gid = torch.tensor([0, 0])
    s = torch.ones(2)
    adv_big, _ = vcpo_opob_advantages(r, gid, torch.tensor([20.0, 1.0]), s)
    # W = min(w, 8)^2 * w^2 = 64 * 400 vs 1: baseline close to R_0 = 1
    assert adv_big[0].abs() < 1e-3
