"""Replay summary, pure parts: S2/S3 formulas and the replay tables (no torch)."""
import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from tim.measure.formulas import dist_stats, linear_slope, seq_sums, truncation_stats  # noqa: E402
from tim.measure.replay_summary import summarize  # noqa: E402

CFG2 = json.load(open(os.path.join(ROOT, "configs", "measure", "qwen3_0.6b.json")))


def test_seq_and_truncation_stats():
    off = [0, 2, 5]
    d = np.array([1.0, 0.0, -1.0, np.log(3.0), 0.5])
    assert np.allclose(seq_sums(d, off), [1.0, -1.0 + np.log(3.0) + 0.5])
    t = truncation_stats(d, off, 2.0)
    
    over = d > np.log(2.0)
    assert t["frac_ratio_gt_cap"] == pytest.approx(over.mean())
    assert t["excess_ratio_mean"] == pytest.approx(np.maximum(np.exp(d) - 2, 0).mean())
    assert t["truncated_delta_seq_mean"] == pytest.approx(seq_sums(np.where(over, d, 0), off).mean())
    s = dist_stats([1, 2, 3, 4, 5])
    assert s["mean"] == 3 and s["median"] == 3 and s["n"] == 5
    assert linear_slope([0, 1, 2], [1, 3, 5]) == pytest.approx(2.0) and math.isnan(linear_slope([1], [1]))


def _step(run, src, step, s1_pairs, seq_mean, tail, trunc):
    return {"run": run, "source": src, "step": step,
            "sampler_repeat": {"repeat_same_batch": {"next_abs_mean": 0.0, "next_abs_max": 0.0},
                               "batch1_vs_batch64": {"next_abs_mean": 0.0, "next_abs_max": 0.0}},
            "trainer_repeat": {"next_abs_max": 0.0}, "eps": {"eps_abs_mean": 0.5, "kl_q_p_tail_mean": 0.01},
            "ratio": {"frac_ratio_gt_cap": trunc},
            "S2_probe": {"frac_ratio_gt_cap": trunc, "excess_ratio_mean": 0.1, "truncated_delta_seq_mean": 0.2,
                         "tail_mass_mean": tail},
            "S3": {"seq_sum_delta": {"mean": seq_mean, "sd": 1, "q05": -1, "median": 0, "q95": 1}},
            "S4": {"seq_sum_lag": {"mean": seq_mean - 1}, "seq_sum_engine": {"mean": 1.0}},
            "lags": {"eps": [{"lag": l, "eps_pearson": p, "delta_pearson": p} for l, p in s1_pairs], "missing": []}}


def test_summarize_replay(tmp_path):
    for seed in (0, 1):
        run = f"s2ref_bf16_pg_seed{seed}"
        for src, s1, slope, tail, trunc in (("fp8", 0.5, -0.03, 0.0002, 0.015), ("stale", 0.6, -0.7, 0.0003, 0.01),
                                            ("bf16", 0.04, 0.0, 0.0002, 0.0002)):
            d = tmp_path / "res" / run / src
            d.mkdir(parents=True)
            for t in (0, 5, 10, 20):
                pairs = [(20, s1)] if t == 20 else []
                (d / f"step_{t}.json").write_text(json.dumps(_step(run, src, t, pairs, 1.0 + slope * t, tail, trunc)))
    out = summarize(tmp_path / "res", CFG2)
    assert "6 (run, source)" in out[0]
    tab = (tmp_path / "res" / "TABLES.md").read_text()
    assert "S3 sequence level" in tab and "-0.7 [-0.7, -0.7]" in tab
    from tim.measure.replay_summary import load, per_seed, source_inputs
    inp = source_inputs(per_seed(*load(tmp_path / "res")))
    assert inp["stale"]["S3_slope"] == pytest.approx(-0.7) and inp["fp8"]["eps_pearson_lag20"] == pytest.approx(0.5)
