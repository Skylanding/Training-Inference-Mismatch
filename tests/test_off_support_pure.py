"""Off-support measurement, pure parts: run names, held-out split, own-sample and probe statistics, Spearman, the job list and the
summary on synthetic records. No torch."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tim.measure.off_support import pure  # noqa: E402

OFF = np.array([0, 3, 3, 7])          # sequences of 3, 0 and 4 response tokens


def test_parse_run_and_reference():
    p = pure.parse_run("s2_fp8_stale64_tis_sc_seed2")
    assert (p["prefix"], p["source"], p["method"], p["seed"], p["model"], p["ref"]) == ("s2", "fp8_stale64", "tis_sc", 2, "06b", False)
    assert pure.parse_run("s6c_budget_tis_seed1")["model"] == "17b"
    assert pure.parse_run("s5ref_bf16_pg_seed0")["ref"] and pure.parse_run("s2ref_bf16_pg_seed1")["model"] == "06b"
    assert pure.parse_run("theta0_17b")["theta0"] and pure.parse_run("theta0_06b")["model"] == "06b"
    assert pure.reference_of("s6c_stale64_split_seed2") == "s5ref_bf16_pg_seed2"
    assert pure.reference_of("s2_stale64_split_seed0") == "s2ref_bf16_pg_seed0"
    with pytest.raises(ValueError):
        pure.parse_run("s4_stale64_sc_seed0")


def test_heldout_split():
    # theta_0 0.2, ref 0.6: run 0.5 -> not learned 0.1; run 0.1 -> not learned 0.4 (all of ref's gain) + worse 0.1
    a = pure.heldout_split(0.5, 0.6, 0.2)
    assert a["loss"] == pytest.approx(0.1) and a["not_learned"] == pytest.approx(0.1) and a["worse"] == 0 and a["rest"] == pytest.approx(0)
    b = pure.heldout_split(0.1, 0.6, 0.2)
    assert b["not_learned"] == pytest.approx(0.4) and b["worse"] == pytest.approx(0.1) and b["rest"] == pytest.approx(0)
    assert b["run_minus_theta0"] == pytest.approx(-0.1)
    c = pure.heldout_split(0.7, 0.6, 0.2)                  # better than the reference
    assert c["loss"] == pytest.approx(-0.1) and c["not_learned"] == 0 and c["worse"] == 0 and c["rest"] == pytest.approx(-0.1)


def test_reductions():
    x = np.arange(1.0, 8.0)
    assert pure.seq_sums(x, OFF).tolist() == [6.0, 0.0, 22.0]
    m = pure.seq_max(x, OFF)
    assert m[0] == 3 and math.isnan(m[1]) and m[2] == 7
    assert pure.positions(OFF).tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert pure.first_index(np.array([0, 0, 1, 0, 1, 1, 0], bool), OFF).tolist() == [2, -1, 1]


def test_own_stats():
    off = np.array([0, 3, 7])
    lp_0 = np.log(np.array([0.5, 1e-4, 0.5, 0.5, 0.5, 1e-5, 0.5]))
    lp_t = lp_0 + np.array([0.1, 2.0, 0.1, 0.0, 0.0, 3.0, -0.2])
    rank = np.array([0, 25, 1, 0, 3, 40, 0])
    r = pure.own_stats(lp_t, lp_0, rank, off, [False, True], reward=[1.0, 0.0])
    assert r["d_own"] == pytest.approx((2.2 + 2.8) / 2)
    assert r["share_offsupport"] == pytest.approx(5.0 / 5.0) and r["frac_tokens_p0_low"] == pytest.approx(2 / 7)
    assert r["d_own_offsupport"] == pytest.approx(5.0 / 2)
    assert r["first_out_mean"] == pytest.approx((1 + 2) / 2) and r["first_out_frac_never"] == 0
    assert r["frac_truncated"] == 0.5 and r["frac_short"] == 1.0 and r["len_mean"] == 3.5 and r["reward_mean"] == 0.5


def test_probe_stats():
    off = np.array([0, 2, 70])
    n = 70
    kl = np.zeros(n); kl[1] = 0.5; kl[10] = 2.0
    e0, et = np.full(n, 0.01), np.full(n, 0.01)
    et[2:66] += 0.1                                          # rows 2..65 = positions 0..63 of the second sequence
    lp0 = np.full(n, -1.0); lpt = lp0 - 0.2
    r = pure.probe_stats(kl, et, e0, lpt, lp0, off)
    assert r["d_q"] == pytest.approx((0.5 + 2.0) / 2) and r["kl_mean"] == pytest.approx(2.5 / n)
    assert r["max_kl_mean"] == pytest.approx((0.5 + 2.0) / 2) and r["max_kl_max"] == 2.0
    first = np.r_[np.zeros(2), np.full(64, 0.1), np.zeros(4)]   # positions < 64: rows 0, 1 and 2..65
    assert r["d_eos_first"] == pytest.approx(first.sum() / 66)
    assert r["mu"] == pytest.approx(-0.2) and r["mu_abs"] == pytest.approx(0.2)


def test_spearman_and_percentile():
    rho, n = pure.spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7])
    assert n == 5 and rho == pytest.approx(0.8207826816681233)        # scipy.stats.spearmanr, ties averaged
    assert pure.spearman([1, 2, 3, np.nan], [3, 2, 1, 0]) == (pytest.approx(-1.0), 3)
    assert math.isnan(pure.spearman([1, 2], [1, 2])[0])
    assert pure.ranks([10, 20, 20, 5]).tolist() == [2.0, 3.5, 3.5, 1.0]
    assert pure.percentile_of(3.0, [1, 2, 3, 4]) == pytest.approx(0.625)


def _rec(run, t, acc, d_own, d_q, model="17b"):
    p = pure.parse_run(run)
    own = {"d_own": d_own, "share_offsupport": 0.5, "d_own_offsupport": d_own / 2, "len_mean": 100.0, "frac_truncated": 0.0,
           "frac_short": 0.1, "frac_tokens_p0_low": 0.01, "frac_tokens_out_top": 0.02, "first_out_mean": 5.0, "first_out_frac_never": 0.3}
    probe = {"d_q": d_q, "kl_mean": d_q / 100, "max_kl_mean": d_q / 10, "max_kl_max": d_q, "d_eos_first": 0.01, "mu": -d_q / 50,
             "mu_abs": d_q / 50}
    r = {"run": run, "step": t, "model": p["model"], "source": p["source"], "method": p["method"], "seed": p["seed"], "ref": p["ref"],
         "theta0": p["theta0"], "heldout": {"mean_accuracy": acc}, "own": own, "probe": probe}
    if model == "17b":
        r["probe_s1"] = dict(probe)
    return r


def test_config():
    c7 = json.loads((ROOT / "configs" / "measure" / "off_support.json").read_text())
    assert len(c7["runs_06b"]) == 33 and len(c7["runs_17b"]) == 15 and set(c7["collapsed"]) <= set(c7["runs_17b"])
    for r in c7["runs_06b"] + c7["runs_17b"] + c7["runs_7d"] + c7["runs_8b"] + c7["runs_9b_async"]:
        pure.parse_run(r)
    assert pure.reference_of("s9b_async4_split_async_seed0") == "s5ref_bf16_pg_seed0"


def test_summary(tmp_path):
    from tim.measure.off_support.summary import summarize
    c7 = json.loads((ROOT / "configs" / "measure" / "off_support.json").read_text())
    c7["existing"] = {"deviation": {"17b": "dev.csv"}, "noise_floor_run": {"17b": "s5_bf16_pg_seed{seed}"}, "mu_4c": {"17b": "mu.csv"}}
    (tmp_path / "dev.csv").write_text("run,step,kl_mean,acc_run\ns5_bf16_pg_seed0,10,0.1,0.5\ns5_bf16_pg_seed1,10,0.3,0.5\n"
                                      "s6c_budget_tis_seed0,10,0.4,0.01\ns5_stale64_sc_seed0,10,2.0,0.4\n")
    (tmp_path / "mu.csv").write_text("run,step,4c_lag_mean\ns5_stale64_sc_seed0,10,-0.5\n")
    res = tmp_path / "res"
    recs = [_rec("theta0_17b", 0, 0.3, 0.0, 0.0), _rec("s5ref_bf16_pg_seed0", 10, 0.6, 1.0, 0.5),
            _rec("s6c_budget_tis_seed0", 10, 0.01, 50.0, 2.0), _rec("s5_stale64_sc_seed0", 10, 0.4, 5.0, 8.0),
            _rec("s6c_stale64_split_seed0", 10, 0.1, 30.0, 1.0)]
    for r in recs:
        (res / r["run"]).mkdir(parents=True, exist_ok=True)
        (res / r["run"] / f"step_{r['step']}.json").write_text(json.dumps(r))
    summarize(res, c7, tmp_path)
    txt = (res / "TABLES.md").read_text()
    assert "Spearman" in txt and "s6c_budget_tis_seed0 | 10" in txt and "17b bf16 pg" in txt
    import csv
    rows = {r["run"]: r for r in csv.DictReader(open(res / "steps.csv"))}
    b = rows["s6c_budget_tis_seed0"]
    assert float(b["loss"]) == pytest.approx(0.59) and float(b["worse"]) == pytest.approx(0.29) and float(b["not_learned"]) == pytest.approx(0.3)
    assert float(b["dev_ratio"]) == pytest.approx(0.4 / 0.2) and float(rows["s5_stale64_sc_seed0"]["mu_4c_abs"]) == pytest.approx(0.5)
    assert rows["theta0_17b"]["kind"] == "theta0" and rows["s5ref_bf16_pg_seed0"]["kind"] == "ref"
