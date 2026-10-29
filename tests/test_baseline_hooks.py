"""CPU tests of tim/baselines/hooks.py: the TransferQueue (nested jagged) side of the M2PO / VCPO hooks gives the same
numbers as tim/baselines/weights.py on the padded tensors."""
import pytest

torch = pytest.importorskip("torch")
TensorDict = pytest.importorskip("tensordict").TensorDict

from tim.baselines.hooks import STASH, m2po_weights, vcpo_scale
from tim.baselines.weights import m2po_keep_mask, vcpo_ess_ratio, vcpo_lr_scale


def nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def make(seed=0, lens=(5, 3, 7, 4)):
    g = torch.Generator().manual_seed(seed)
    f = lambda scale: [torch.randn(n, generator=g) * scale for n in lens]
    lt, l0, q = f(0.3), f(0.3), f(0.3)
    adv = [torch.full((n,), float(s)) for n, s in zip(lens, (1.0, -1.0, 0.5, -2.0))]
    mask = [torch.ones(n) for n in lens]
    td = TensorDict({STASH: nested(lt), "old_log_probs": nested(l0), "advantages": nested(adv),
                     "response_mask": nested(mask), "rollout_log_probs": nested(q)}, batch_size=len(lens))
    return td, lt, l0, adv, mask, q


def pad(rows):
    out = torch.zeros(len(rows), max(len(r) for r in rows))
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def test_m2po_weights_match_math():
    td, lt, l0, adv, mask, _ = make()
    keep, stats = m2po_weights(td)
    keep2, stats2 = m2po_keep_mask(pad(lt), pad(l0), pad(adv), pad(mask), budget=0.04)
    assert torch.equal(keep, keep2.float())
    assert stats == stats2
    assert stats["m2po/masked_frac"] > 0          # random log-probs: the budget binds


def test_vcpo_scale_matches_math_and_is_overflow_safe():
    td, _, l0, _, mask, q = make(1)
    s, stats = vcpo_scale(td)
    w = torch.exp(((pad(l0) - pad(q)) * pad(mask)).sum(-1).double())
    assert abs(stats["vcpo/ess_ratio"] - vcpo_ess_ratio(w)) < 1e-6   # float32 inputs
    assert abs(s - vcpo_lr_scale(vcpo_ess_ratio(w))) < 1e-6
    big = TensorDict({k: (nested([r + 400.0 for r in l0]) if k == "old_log_probs" else v) for k, v in td.items()},
                     batch_size=td.batch_size)
    s_big, st_big = vcpo_scale(big)                # a common factor exp(400 * len) per row is not common: finite only
    assert s_big == s_big and 0 < s_big <= 1


def test_opob_log_space_matches_math():
    from tim.baselines.hooks import opob_advantages
    from tim.baselines.weights import vcpo_opob_advantages
    g = torch.Generator().manual_seed(3)
    rewards = torch.randint(0, 2, (16,), generator=g).double()
    groups = torch.arange(16) // 8
    log_w = torch.randn(16, generator=g, dtype=torch.float64) * 1.5
    norms = torch.rand(16, generator=g, dtype=torch.float64) * 10 + 0.1
    a1, _ = opob_advantages(rewards, groups, log_w, norms)
    a2, _ = vcpo_opob_advantages(rewards, groups, torch.exp(log_w), norms)
    assert torch.allclose(a1, a2.double(), atol=1e-6)
    a3, _ = opob_advantages(rewards, groups, log_w + 500.0, norms)   # exp(500) overflows the direct formula
    assert torch.isfinite(a3).all()
