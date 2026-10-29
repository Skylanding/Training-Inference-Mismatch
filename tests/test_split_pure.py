"""Split and residual bias, pure parts: run parsing, prompt sets, 4a scalars from inner products, 4b / 4c statistics and
the summary. No torch."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tim.measure.split import pure  # noqa: E402


def test_parse_run_and_samplers():
    p = pure.parse_run("s2_fp8_stale64_tis_sc_seed2")
    assert (p["source"], p["method"], p["seed"], p["kind"], p["sampler_at"]) == ("fp8_stale64", "tis_sc", 2, "fp8", "0")
    assert pure.parse_run("s2_stale64_sc_seed0")["lagged"] and not pure.parse_run("s2_fp8_tis_seed1")["lagged"]
    assert pure.parse_run("s2_eager_pg_seed0")["kind"] == "bf16_eager"
    with pytest.raises(ValueError):
        pure.parse_run("s3_fp8_pg_seed0")
    assert len(pure.STAGE4_RUNS) == 21 and len(set(pure.STAGE4_RUNS)) == 21
    assert pure.objectives_for("s2_stale64_tis_seed0")[0] == "tis"
    assert pure.objectives_for("s2_fp8_sc_seed0") == ["sc"] and pure.objectives_for("s2_eager_pg_seed1") == ["pg"]


def test_prompt_rows():
    r = pure.p2_rows(19936)
    assert len(r) == 64 and r == sorted(r) and r == pure.p2_rows(19936)
    c5, c10 = pure.rows_4c(5, 19936, 64, r), pure.rows_4c(10, 19936, 64, r)
    assert len(c5) == 64 and not set(c5) & set(r) and c5 != c10 and c5 == pure.rows_4c(5, 19936, 64, r)


def _gram(vecs):
    names = list(vecs)
    return {(a, b): float(vecs[a] @ vecs[b]) for i, a in enumerate(names) for b in names[i:]}


def test_bias_scalars_match_direct_vectors():
    rng = np.random.default_rng(0)
    c, o, o2, dr, dc = (rng.normal(size=50) for _ in range(5))
    gram = _gram({"c": c, "o": o, "o2": o2})
    wd = {(k, d): float(v @ dv) for k, v in (("c", c), ("o", o), ("o2", o2)) for d, dv in (("recent", dr), ("cum", dc))}
    out = pure.bias_scalars(gram, wd, {"recent": float(dr @ dr), "cum": float(dc @ dc)})
    b, n = c - o, o2 - o
    cos = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
    assert out["b_norm"] == pytest.approx(np.linalg.norm(b))
    assert out["noise_norm"] == pytest.approx(np.linalg.norm(n))
    assert out["b_cos_recent"] == pytest.approx(cos(b, dr)) and out["b_cos_cum"] == pytest.approx(cos(b, dc))
    assert out["noise_cos_cum"] == pytest.approx(cos(n, dc))
    assert out["cos_on_on2"] == pytest.approx(cos(o, o2))


def test_tis_token_stats_and_split():
    lr = np.log(np.array([0.1, 0.4, 1.0, 1.0, 3.0]))
    s = pure.tis_token_stats(lr)
    assert s["frac_low"] == pytest.approx(0.4) and s["frac_truncated"] == pytest.approx(0.2)
    assert s["w_trunc_mean"] == pytest.approx((0.1 + 0.4 + 1 + 1 + 2) / 5)
    assert sum(b["frac"] for b in s["bins"]) == pytest.approx(1.0)
    lp_t, lp_old, q = np.array([-1.0, -2.0, -0.5]), np.array([-1.2, -2.0, -0.4]), np.array([-1.1, -2.5, -0.5])
    m = pure.split_means(lp_t, lp_old, q, offsets=[0, 2, 3])
    assert m["lag_mean"] + m["engine_mean"] == pytest.approx(m["delta_mean"])
    assert m["delta_seq_mean"] == pytest.approx(np.mean([np.mean([0.1, 0.5]), 0.0]))


def test_first_separated_and_logs():
    sep = pure.first_separated([5, 10, 15], {"a": [0.0, 1.0, 1.0], "b": [1.0, 0.0, 1.0]}, [0.1, 0.1, 0.1])
    assert sep == {"a": 10, "b": 15}
    rows = [{"data": {"training/global_step": 2, "actor/rollout_corr/log_ppl_diff": 0.3}},
            {"data": {"training/global_step": 1, "rollout_corr/log_ppl_diff": -0.1, "rollout_corr/log_ppl_abs_diff": 0.2}},
            {"data": {"training/global_step": 3}}]
    out = pure.log_split_from_metrics(rows)
    assert [r["step"] for r in out] == [1, 2] and out[0]["delta_seq_mean"] == pytest.approx(0.1)
    assert out[1]["delta_seq_mean"] == pytest.approx(-0.3)


def _rec(run, t, b=1.0):
    p = pure.parse_run(run)
    rec = {"run": run, "step": t, "source": p["source"], "method": p["method"], "seed": p["seed"],
           "4a": {"b_norm": b * t, "noise_norm": 1.0, "b_cos_recent": 0.1, "b_cos_cum": 0.2, "noise_cos_recent": 0.0,
                  "noise_cos_cum": 0.0, "g_corr_norm": 2.0, "g_on_norm": 2.0, "g_on2_norm": 2.0, "cos_corr_on": 0.9,
                  "cos_on_on2": 0.8, "delta_recent_norm": 1.0, "delta_cum_norm": 2.0},
           "4c": {"delta_mean": 0.01 * t, "lag_mean": 0.01 * t, "engine_mean": 0.0, "rows": [1, "..."]},
           "check": {"repeat_rel_diff": 1e-4}, "seconds": 10.0, "timing_s": {}}
    if p["method"] == "tis":
        rec["4b"] = {"tokens": pure.tis_token_stats(np.log([0.3, 1.0, 2.5])),
                     "proj": {f"{g}:{d}": 0.1 for g in ("tis", "sc", "tis_low", "sc_low") for d in ("recent", "cum")}}
    return rec


def test_summary(tmp_path):
    from tim.measure.split.summary import summarize
    cfg4 = {"steps": [5, 10], "results_dir": str(tmp_path)}
    for run in ("s2_stale64_tis_seed0", "s2_stale64_tis_seed1", "s2_eager_pg_seed0"):
        for t in (5, 10):
            (tmp_path / run).mkdir(exist_ok=True)
            rec = _rec(run, t, 0.0 if "eager" in run else 1.0)
            if "eager" in run:
                rec["4c"].update(delta_mean=0.001, lag_mean=0.0, engine_mean=0.001)
            (tmp_path / run / f"step_{t}.json").write_text(json.dumps(rec))
    out = summarize(tmp_path, cfg4)
    text = (tmp_path / "TABLES.md").read_text()
    assert "6 step results" in out[0] and "### stale64 tis" in text and "4b TIS runs" in text
    assert "lag_mean: stale64 tis -> 5" in text
