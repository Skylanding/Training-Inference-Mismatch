"""Loop measurement, pure parts: per-token weights, subset choice, scalars from inner products, 6a-2 reductions. No torch."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tim.measure.loop import pure  # noqa: E402

OFF = np.array([0, 3, 3, 7])          # sequences of 3, 0 and 4 response tokens


def test_exclusive_cumsum_and_positions():
    x = np.arange(1.0, 8.0)
    assert pure.exclusive_cumsum(x, OFF).tolist() == [0, 1, 3, 0, 4, 9, 15]
    seq, pos = pure.positions(OFF)
    assert seq.tolist() == [0, 0, 0, 2, 2, 2, 2] and pos.tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert pure.per_row([1.0, 5.0, -2.0], OFF).tolist() == [1, 1, 1, -2, -2, -2, -2]


def test_bias_weights():
    lag = np.array([0.1, 0.2, 0.3, 1.0, 1.0, 1.0, 1.0])
    A = pure.per_row([2.0, 0.0, -1.0], OFF)
    w = pure.bias_weights(lag, A, OFF, "sc")
    assert np.allclose(w["token"], -A * lag)
    assert np.allclose(w["prefix"][:3], -2.0 * np.array([0, 0.1, 0.3]))
    assert np.allclose(w["prefix"][3:], 1.0 * np.array([0, 1, 2, 3]))
    assert set(pure.bias_weights(lag, A, OFF, "tis")) == {"prefix"}
    with pytest.raises(ValueError):
        pure.bias_weights(lag, A, OFF, "pg")


def test_bias_weights_exact_matches_linear_when_small():
    """Exact weights: with small |S| and |l| the exact weights equal the linearised ones (to second order)."""
    rng = np.random.default_rng(3)
    lag = rng.normal(size=7) * 1e-4
    A = pure.per_row([2.0, 0.0, -1.0], OFF)
    lin, ex = pure.bias_weights(lag, A, OFF, "sc"), pure.bias_weights(lag, A, OFF, "sc", exact=True)
    for k in ("token", "prefix"):
        assert np.allclose(ex[k], lin[k], rtol=1e-3, atol=1e-10)
    tl, te = pure.bias_weights(lag, A, OFF, "tis"), pure.bias_weights(lag, A, OFF, "tis", exact=True)
    assert np.allclose(te["prefix"], tl["prefix"], rtol=1e-3, atol=1e-10)
    assert set(te) == {"prefix", "token"} and np.all(te["token"] == 0)     # e^l < 2: nothing truncated
    wa, wb = pure.subset_weights(lag, A, {1: 0.5, 4: -0.5}, 7)
    xa, xb = pure.subset_weights(lag, A, {1: 0.5, 4: -0.5}, 7, exact=True)
    assert np.allclose(xa, wa, rtol=1e-3, atol=1e-12) and np.allclose(xb, wb, rtol=1e-3, atol=1e-12)


def test_bias_weights_exact_values():
    lag = np.array([0.1, -2.0, 1.0, 0.0, 1.0, -50.0, 2.0])
    A = pure.per_row([2.0, 0.0, -1.0], OFF)
    ex = pure.bias_weights(lag, A, OFF, "sc", exact=True)
    S = pure.exclusive_cumsum(lag, OFF)
    assert np.allclose(ex["prefix"], A * (1 - np.exp(S)))
    assert np.allclose(ex["token"], A * (1 - np.exp(lag)))
    assert ex["prefix"][5] == pytest.approx(-1.0 * (1 - np.exp(1.0)))
    assert np.all(np.abs(ex["prefix"]) <= np.abs(A) * np.maximum(1.0, np.exp(S)))   # |1 - e^S| < 1 for S < 0
    t = pure.bias_weights(lag, A, OFF, "tis", exact=True, cap=2.0)
    assert np.allclose(t["token"], -A * np.maximum(np.exp(lag) - 2.0, 0.0))
    assert t["token"][6] == pytest.approx(1.0 * (np.exp(2.0) - 2.0)) and t["token"][0] == 0.0
    big = pure.bias_weights(np.array([100.0, 100.0]), np.ones(2), np.array([0, 2]), "sc", exact=True)
    assert np.all(np.isfinite(big["prefix"])) and np.all(np.isfinite(big["token"]))       # clipped at EXP_CLIP
    d = pure.exact_diagnostics(lag, OFF)
    assert d["abs_S_max"] == pytest.approx(np.abs(S).max()) and d["n_clipped"] == 0
    assert d["frac_ratio_gt_cap"] == pytest.approx(np.mean(lag > np.log(2.0)))


def test_choose_rows_fixed_seed():
    a, b = pure.choose_rows(5000, 512, 60005), pure.choose_rows(5000, 512, 60005)
    assert (a == b).all() and len(set(a.tolist())) == 512 and a.max() < 5000 and (np.diff(a) > 0).all()
    assert not (a == pure.choose_rows(5000, 512, 60010)).all()
    assert pure.choose_rows(7, 512, 1).tolist() == list(range(7))


def test_abar_and_subset_weights():
    ab = pure.abar_from_rewards([[1, 0, 0, 1], [0, 0]], [0.25, 0.5])
    assert np.allclose(ab, [0.25, -0.5])
    assert pure.abar_ratio([0.5, -0.5], [1.0, -1.0]) == pytest.approx(0.5)
    lag, A = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, -1.0, 2.0, 0.5])
    wa, wb = pure.subset_weights(lag, A, {1: 0.5, 3: -0.5}, 4)
    assert wa.tolist() == [0, 2.0, 0, -2.0] and wb.tolist() == [0, 3.0, 0, -4.0]


def _vecs(rng, n=40):
    return {k: rng.normal(size=n) for k in ("c", "o", "o2", "tok", "pre")}


def test_predicted_scalars_match_direct():
    rng = np.random.default_rng(0)
    v = _vecs(rng)
    v["c"] = v["o"] + 0.5 * v["tok"] + 0.3 * rng.normal(size=40)
    d = rng.normal(size=40)
    gram = {(a, b): float(v[a] @ v[b]) for a in v for b in v}
    dd = {k: float(x @ d) for k, x in v.items()}
    out = pure.predicted_scalars(gram, dd, float(d @ d), {"token": {"tok": 1.0}, "prefix": {"pre": 1.0},
                                                          "total": {"tok": 1.0, "pre": 1.0}})
    b, n, tot = v["c"] - v["o"], v["o2"] - v["o"], v["tok"] + v["pre"]
    cos = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
    assert out["b_norm"] == pytest.approx(np.linalg.norm(b)) and out["noise_norm"] == pytest.approx(np.linalg.norm(n))
    assert out["total"]["cos_b"] == pytest.approx(cos(tot, b)) and out["total"]["cos_noise"] == pytest.approx(cos(tot, n))
    assert out["total"]["cos_delta"] == pytest.approx(cos(tot, d))
    assert out["total"]["ratio"] == pytest.approx(np.linalg.norm(tot) / np.linalg.norm(b))
    assert out["token"]["cos_b"] == pytest.approx(cos(v["tok"], b)) and out["prefix"]["cos_delta"] == pytest.approx(cos(v["pre"], d))
    assert pure.cos_vectors(gram, {"tok": 1.0}, {"pre": 1.0}) == pytest.approx(cos(v["tok"], v["pre"]))
    g1 = {k: g for k, g in gram.items() if (k[1], k[0]) not in gram or k[0] <= k[1]}      # one order only
    assert pure.predicted_scalars(g1, dd, float(d @ d), {"t": {"tok": 1.0}})["t"]["cos_b"] == pytest.approx(cos(v["tok"], b))


def test_lag_split():
    lag = np.array([-1.0, -3.0, 2.0, 0.0, 4.0])
    A = np.array([-1.0, -1.0, 1.0, 0.0, 2.0])
    r = pure.lag_split(lag, A, delta_sq=2.0)
    assert r["neg"]["n"] == 2 and r["neg"]["mean"] == -2.0 and r["neg"]["share_sum"] == pytest.approx(-4.0 / 2.0)
    assert r["pos"]["mean"] == 3.0 and r["pos"]["share_sum"] == pytest.approx(3.0)
    assert r["zero"]["n"] == 1 and r["neg"]["abs_mean_over_delta_sq"] == pytest.approx(1.0)
    assert r["neg"]["share_sum"] + r["pos"]["share_sum"] + r["zero"]["share_sum"] == pytest.approx(1.0)
    assert math.isnan(pure.lag_split(np.array([1.0, -1.0]), np.array([1.0, -1.0]), 1.0)["pos"]["share_sum"])


def test_top_tokens():
    rng = np.random.default_rng(1)
    n = 1000
    lag = rng.normal(size=n) * 0.01
    lag[7], lag[500] = 5.0, -3.0
    A = rng.choice([-1.0, 0.0, 1.0], size=n)
    p = np.full(n, 0.5)
    p[7] = 1e-5
    pos = rng.integers(0, 50, size=n).astype(float)
    ln = pos + rng.integers(1, 50, size=n)
    r = pure.top_tokens(lag, A, p, pos, ln)
    t1, t01 = r["0.01"], r["0.001"]
    assert t1["k"] == 10 and t01["k"] == 1
    assert t01["share_sum"] == pytest.approx(5.0 / lag.sum()) and t01["frac_p_below"] == 1.0
    assert t01["pos_mean"] == pos[7] and t01["rel_pos_mean"] == pytest.approx(pos[7] / ln[7])
    assert t1["frac_adv_neg"] + t1["frac_adv_pos"] + t1["frac_adv_zero"] == pytest.approx(1.0)
    assert len(t1["pos_quartiles"]) == 3 and len(t1["rel_pos_quartiles"]) == 3
    assert t1["abs_l_min"] >= np.sort(np.abs(lag))[-10] - 1e-12


def test_summary_exact(tmp_path):
    import json
    from tim.measure.loop.summary import summarize_exact
    pred = {"cos_b": 0.3, "ratio": 1.5, "cos_noise": 0.01, "cos_delta": 0.2, "norm": 1.0}
    for run, method in (("s2_stale64_sc_seed0", "sc"), ("s2_stale64_tis_seed0", "tis")):
        for t in (5, 10):
            lin = {"pred": {"prefix": pred, "total": pred, "b_norm": 1.0, "noise_norm": 0.5, "delta_cum_norm": 2.0}}
            ex = {"pred": {"token": pred, "prefix": pred, "total": pred, "b_norm": 1.0, "noise_norm": 0.5, "delta_cum_norm": 2.0},
                  "cos_token_prefix": 0.1}
            if method == "sc":
                lin["pred"]["token"] = pred
                lin["subset"] = ex["subset"] = {"sub_A": pred, "sub_Abar": pred, "cos_sub_A_vs_Abar": 0.9}
            rec = {"run": run, "step": t, "source": "stale64", "method": method, "seed": 0, "lin": lin, "exact": ex,
                   "diag": {"exact": {"abs_S_max": 3.0, "S_max": 1.0, "frac_abs_S_gt_1": 0.2, "frac_ratio_gt_cap": 0.01,
                                      "mean_one_minus_exp_S": 0.1, "n_clipped": 0}, "lag_mean": -0.1, "delta_mean": -0.1,
                            "abar_ratio": 0.3, "n_tokens": 100}}
            (tmp_path / run).mkdir(exist_ok=True)
            (tmp_path / run / f"step_{t}.json").write_text(json.dumps(rec))
    summarize_exact(tmp_path, [5, 10])
    txt = (tmp_path / "TABLES.md").read_text()
    assert "stale64 sc" in txt and "stale64 tis" in txt and "prefix cos exact" in txt and "token cos lin" in txt
    assert "| 10 |" in txt and (tmp_path / "steps.csv").read_text().count("\n") == 5


def test_loop_summary(tmp_path):
    import json
    root = Path(__file__).resolve().parent.parent
    c6 = json.loads((root / "configs" / "measure" / "loop.json").read_text())
    assert c6["steps"] == list(range(5, 45, 5))
    from tim.measure.split.pure import parse_run
    assert all(parse_run(x)["lagged"] for x in c6["runs_06b"] + c6["runs_17b"])
    split = {"n": 4, "mean": 0.1, "sum": 0.4, **{g: {"n": 1, "mean": 0.1, "share_sum": 0.5, "abs_mean_over_delta_sq": 0.2} for g in ("neg", "pos", "zero")}}
    top = {f: {"share_sum": 0.5, "frac_p_below": 0.1, "frac_adv_neg": 0.3, "frac_adv_pos": 0.3, "frac_adv_zero": 0.4, "pos_mean": 3.0,
               "rel_pos_mean": 0.2} for f in ("0.01", "0.001")}
    pred = {"cos_b": 0.1, "ratio": 1.0, "cos_noise": 0.0, "cos_delta": 0.2, "norm": 1.0}
    for run in ("s2_stale64_sc_seed0", "s5_stale64_tis_seed0"):
        for t in (5, 10):
            rec = {"run": run, "step": t, "source": "stale64", "method": run.split("_")[2], "seed": 0,
                   "6a2": {"lag_mean": 0.1, "delta_cum_sq": 2.0, "by_A": split, "subset": {"abar_ratio": 0.4, "by_A_minus_Abar": split, "by_A": split}, "top": top}}
            if run.startswith("s2"):
                rec["6a1"] = {"pred": {"token": pred, "prefix": pred, "total": pred, "b_norm": 1.0, "noise_norm": 0.5, "delta_cum_norm": 2.0},
                              "lag": {"mean": 0.1, "abs_mean": 0.2}, "subset": {"abar_ratio": 0.4, "cos_sub_A_vs_Abar": 0.9, "sub_A": pred, "sub_Abar": pred}}
            (tmp_path / run).mkdir(exist_ok=True)
            (tmp_path / run / f"step_{t}.json").write_text(json.dumps(rec))
    from tim.measure.loop.summary import summarize
    summarize(tmp_path, c6)
    txt = (tmp_path / "TABLES.md").read_text()
    assert "6a-1 s2 stale64 sc" in txt and "6a-2 s5 stale64 tis" in txt and "6a-1 s5" not in txt
