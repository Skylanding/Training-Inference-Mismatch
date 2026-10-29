"""Replay formulas, the flat prompt-logprob extractor and the measure.py helpers (pure numpy)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tim.measure.tail_model import kl_with_tail  # noqa: E402
from tim.measure import formulas as F  # noqa: E402
from tim.measure.heads import flat_prompt_heads, rows_by_sequence  # noqa: E402


def test_centred_advantages_per_group_no_std():
    r = [1, -1, -1, -1, 1, 1, 1, 1]
    g = [0, 0, 0, 0, 1, 1, 1, 1]
    a = F.centred_advantages(r, g)
    assert np.allclose(a[:4], [1.5, -0.5, -0.5, -0.5])
    assert np.allclose(a[4:], 0.0)


def test_tis_weight_and_ratio_stats():
    lr = np.log([0.5, 1.0, 2.0, 3.0])
    assert np.allclose(F.tis_weight(lr, 2.0), [0.5, 1.0, 2.0, 2.0])
    s = F.ratio_stats(lr, 2.0)
    assert s["frac_ratio_gt_cap"] == pytest.approx(0.25)          # only p/q = 3 exceeds the cap (2.0 is not > 2)
    assert s["ratio_w_mean"] == pytest.approx(np.mean([-0.5, 0.0, 1.0, 1.0]))
    assert s["ratio_w_abs_mean"] == pytest.approx(np.mean([0.5, 0.0, 1.0, 1.0]))


def test_eps_stats_zero_when_equal_and_known_offset():
    rng = np.random.default_rng(0)
    q = -np.sort(-np.log(rng.dirichlet(np.ones(10), size=5)), axis=1)[:, :8]   # by rank: column 0 = argmax
    s = F.eps_stats(q, q, q[:, 0], q[:, 0])
    assert s["eps_abs_mean"] == 0 and s["delta_abs_mean"] == 0 and s["flip_rate"] == 0
    assert s["kl_q_p_tail_mean"] == pytest.approx(0.0, abs=1e-12)
    s = F.eps_stats(q, q + 0.1, q[:, 0], q[:, 0] - 0.2)
    assert s["eps_abs_mean"] == pytest.approx(0.1) and s["delta_mean"] == pytest.approx(-0.2)
    assert s["kl_q_p_tail_mean"] == pytest.approx(kl_with_tail(q, q + 0.1).mean())


def test_repeat_diff():
    ids = np.array([[1, 2, 3], [4, 5, 6]])
    lp = np.log(np.array([[0.5, 0.3, 0.1], [0.6, 0.2, 0.1]]))
    nxt = np.array([-0.1, -0.2])
    same = F.repeat_diff(ids, lp, nxt, ids, lp, nxt)
    assert same["head_abs_max"] == 0 and same["next_abs_max"] == 0 and same["rows_equal_exactly"] == 1.0
    ids2 = np.array([[1, 3, 2], [4, 5, 7]])               # row 0 reordered (same set), row 1 one id swapped
    lp2 = np.array([[lp[0, 0], lp[0, 2], lp[0, 1] + 0.01], [lp[1, 0], lp[1, 1], -3.0]])
    d = F.repeat_diff(ids, lp, nxt, ids2, lp2, nxt + 0.05)
    assert d["rows_ids_differ"] == 1.0
    assert d["head_abs_max"] == pytest.approx(0.01)
    assert d["head_coverage"] == pytest.approx(5 / 6)
    assert d["next_abs_mean"] == pytest.approx(0.05)


def test_pearson_cosine_nan_handling():
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    assert F.pearson(x, 2 * x + 1) == pytest.approx(1.0)
    assert F.cosine(x, -x) == pytest.approx(-1.0)
    assert np.isnan(F.pearson([1, 1, 1], [1, 2, 3]))


def test_lag_pairs():
    p = F.lag_pairs(range(0, 41), [1, 5, 10, 20])
    assert (0, 1, 1) in p and (20, 40, 20) in p and (21, 41, 20) not in p
    assert len(p) == 40 + 36 + 31 + 21
    assert F.lag_pairs([0, 2, 3], [1]) == [(2, 3, 1)]


def test_eps_lag_identical_and_shifted_ids():
    rng = np.random.default_rng(2)
    ids = np.stack([rng.permutation(30)[:5] for _ in range(40)])
    e = rng.normal(size=ids.shape)
    d = rng.normal(size=40)
    r = F.eps_lag(ids, e, d, ids, e, d)
    assert r["delta_pearson"] == pytest.approx(1.0) and r["eps_pearson"] == pytest.approx(1.0)
    perm = ids[:, ::-1]
    r2 = F.eps_lag(ids, e, d, perm, e[:, ::-1], -d)          # same values reordered with the ids; delta flipped
    assert r2["eps_pearson"] == pytest.approx(1.0) and r2["delta_pearson"] == pytest.approx(-1.0)


def test_stale_split_adds_up():
    rng = np.random.default_rng(3)
    p_t, p_0, q = rng.normal(size=(3, 100))
    s = F.stale_split(p_t, p_0, q)
    assert s["total_ms"] == pytest.approx(s["weight_ms"] + s["engine_ms"] + 2 * s["cross_ms"])


def test_dict_cosine():
    a = {"x": np.array([1.0, 0.0]), "y": np.array([[0.0, 1.0]])}
    b = {"x": np.array([1.0, 0.0]), "y": np.array([[0.0, -1.0]])}
    assert F.dict_cosine(a, a) == pytest.approx(1.0)
    assert F.dict_cosine(a, b) == pytest.approx(0.0)


def test_flat_prompt_heads():
    # sequence of 5 tokens, prompt of 3: rows for entries 3 and 4. k = 2.
    seq = [10, 11, 12, 13, 14]
    entries = [None,
               [(11, -0.1, 1), (20, -2.0, 2)],
               [(12, -0.2, 1), (21, -2.1, 2)],
               [(30, -0.3, 1), (31, -1.5, 2), (13, -4.0, 9)],      # actual token outside the head
               [(14, -0.05, 1), (32, -3.0, 2)]]
    starts, ends, toks, lps, rks = [], [], [], [], []
    for e in entries:
        starts.append(len(toks))
        for t, lp, r in e or []:
            toks.append(t); lps.append(lp); rks.append(r)
        ends.append(len(toks))
    ids, lp, nxt = flat_prompt_heads(starts, ends, toks, lps, rks, seq, 3, 2)
    assert ids.tolist() == [[30, 31], [14, 32]]
    assert np.allclose(lp, [[-0.3, -1.5], [-0.05, -3.0]])
    assert np.allclose(nxt, [-4.0, -0.05])
    with pytest.raises(RuntimeError):
        flat_prompt_heads(starts, ends, toks, lps, rks, seq, 3, 3)        # head slot 3 is empty


def test_rows_by_sequence():
    assert rows_by_sequence([3, 2], [5, 6]).tolist() == [0, 2, 6]


def test_measure_config_fixed():
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "measure", "qwen3_0.6b.json")))
    rows = sorted(int(x) for x in np.random.default_rng(0).choice(19936, 64, replace=False))
    assert cfg["p2"]["rows"] == rows
    assert cfg["topk"] == 128 and cfg["tis_cap"] == 2.0 and cfg["lags"] == [1, 5, 10, 20]


def _measure_cli():
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "measure.py")
    spec = importlib.util.spec_from_file_location("measure_cli", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_parse_steps_and_sources():
    m = _measure_cli()
    assert m.parse_steps("0-3,10") == [0, 1, 2, 3, 10] and m.parse_steps([10, 5, 5]) == [5, 10]
    assert m.SOURCE_NAMES["quantized_stale"] == "fp8_stale" and m.SAMPLER_KIND["fp8_stale"] == "fp8"


def test_micro_batches_fixed_groups():
    from tim.measure.heads import micro_batches

    class M:
        _tim_mode = "padded"
    assert micro_batches(M(), [[1]] * 5, 2) == [[0, 1], [2, 3], [4]]


