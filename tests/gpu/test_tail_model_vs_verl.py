"""numpy tail algebra == verl score_centering_correction (head masses, rho, centering residual). torch (CPU ok)."""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.gpu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

torch = pytest.importorskip("torch")
from verl.trainer.ppo.score_centering import score_centering_correction  # noqa: E402

from tim.measure import tail_model as T  # noqa: E402


def test_masses_and_correction_match_verl():
    rng = np.random.default_rng(0)
    n, K, V = 64, 16, 500
    logits_t = rng.normal(0, 2, (n, V))
    logits_s = logits_t + rng.normal(0, 0.1, (n, V))
    lt = logits_t - np.log(np.exp(logits_t).sum(-1, keepdims=True))
    ls = logits_s - np.log(np.exp(logits_s).sum(-1, keepdims=True))
    ids = np.argsort(-ls, 1)[:, :K]
    p_lp, q_lp = np.take_along_axis(lt, ids, 1), np.take_along_axis(ls, ids, 1)
    corr, q_mass, p_mass = score_centering_correction(torch.tensor(p_lp, dtype=torch.float32), torch.tensor(q_lp, dtype=torch.float32), torch.ones_like)
    assert np.allclose(q_mass.numpy(), T.head_mass(q_lp), atol=1e-6)
    assert np.allclose(p_mass.numpy(), T.head_mass(p_lp), atol=1e-6)
    rho = T.rho(T.head_mass(q_lp), T.head_mass(p_lp))
    p, q = np.exp(p_lp), np.exp(q_lp)
    ref = ((q - rho[:, None] * p) * p_lp).sum(-1)      # alpha = rho * weight_fn(1/rho) = rho for ones
    assert np.allclose(corr.detach().numpy(), ref, atol=1e-4)
