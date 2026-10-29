"""Report of DLC with resynchronization, the torch-free parts: the reference-reload check from the training logs, the
correlation rows that leave out the DLC runs and the report tables (tim/measure/ess.py)."""
import json
import sys
from pathlib import Path

import numpy as np  # noqa: F401
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tim.measure.off_support import pure as p7  # noqa: E402
from tim.measure import ess as pure  # noqa: E402

RUNS = [f"s7d_budget_split_seed{k}" for k in range(3)]


def _lines(lags, synced=None, chi2=None, seen=None):
    out = []
    for i, lag in enumerate(lags):
        c = (chi2[i] if chi2 else 0.002 + 0.001 * i)
        out.append({"step": i + 1, "data": {pure.LAG: lag, pure.CHI2: c, pure.SEEN: (seen[i] if seen else c),
                                            pure.SYNCED: (synced[i] if synced else 1.0)}})
    return out


def test_check_reload_passes_with_the_reload():
    ok, msgs = pure.check_reload(_lines([0.0106, 0.0109, 0.0102, 0.0111, 0.0108]), every_step_synced=True, min_checked=2)
    assert ok, msgs
    assert sum("ok" in m for m in msgs) == 4                    # steps 2-5 checked (step 5's sync has no step 6)


def _rec(run, t, acc, share=0.1, d_own=1.0):
    p = p7.parse_run(run)
    return {"run": run, "step": t, "model": p["model"], "source": p["source"], "method": p["method"], "seed": p["seed"],
            "ref": p["ref"], "theta0": p["theta0"], "heldout": {"mean_accuracy": acc},
            "own": {"d_own": d_own, "share_offsupport": share}, "probe": {"d_q": d_own / 2, "mu": 0.0, "mu_abs": 0.0}}


def test_check_reload_fails_without_it():
    rows = [_lines([0.0106, 0.021, 0.03, 0.045, 0.05])]
    ok, msgs = pure.check_reload(rows[0], every_step_synced=True, min_checked=2)    # DLC without the reload: lag grows
    assert not ok and any("TOO LARGE" in m for m in msgs)
    log = ROOT / "results" / "train_logs_resync_1.7b" / "s6c_stale64_split_seed0.jsonl"
    if log.exists():                                  # the shipped log of DLC without resynchronization (no reload)
        recs = [json.loads(x) for x in log.read_text().splitlines()]
        lags = [r["data"]["tim/split_lag_abs_mean"] for r in recs[:5]]
        ok, msgs = pure.check_reload(_lines(lags), every_step_synced=True, min_checked=2)
        assert not ok and any("TOO LARGE" in m for m in msgs)
    ok, _ = pure.check_reload(_lines([0.01, 0.011, 0.012], seen=[0.5, 0.5, 0.5]))       # the decision read another chi2
    assert not ok
    ok, msgs = pure.check_reload(_lines([0.01, 0.011, 0.012], synced=[0.0, 0.0, 0.0]))  # nothing to check
    assert not ok and "only 0" in msgs[-1]
    ok, _ = pure.check_reload(_lines([0.01, 0.011, 0.012], synced=[1.0, 0.0, 0.0]), every_step_synced=True)
    assert not ok
    resumed = _lines([0.01, 0.5]) + _lines([0.01, 0.011])                               # a re-logged step: last wins
    assert pure.check_reload(resumed)[0]


def _c7():
    c7 = json.loads((ROOT / "configs" / "measure" / "off_support.json").read_text())
    c7["existing"] = {"deviation": {}, "noise_floor_run": {}, "mu_4c": {}}
    return c7


def test_pooled_rows_leave_out_dlc(tmp_path):
    from tim.measure.off_support.summary import _pooled, flat
    c7 = _c7()
    recs = [_rec("theta0_17b", 0, 0.33)] + [_rec(f"s5ref_bf16_pg_seed{s}", 10, 0.5) for s in range(3)]
    recs += [_rec(f"s6c_budget_tis_seed{s}", 10, 0.4) for s in range(3)] + [_rec(run, 10, 0.45) for run in RUNS]
    F = flat(recs, c7, tmp_path)
    assert sorted(d["run"] for d in _pooled(F, c7)) == [f"s6c_budget_tis_seed{s}" for s in range(3)]
    assert len([d for d in F if d["run"].startswith("s7d_")]) == 3


def test_report_tables(tmp_path):
    from tim.measure.off_support.summary import flat
    c7 = _c7()
    recs = [_rec("theta0_17b", 0, 0.33)]
    for t in pure.S7A_STEPS:
        for s in range(3):
            recs += [_rec(f"s5ref_bf16_pg_seed{s}", t, 0.6), _rec(f"s7d_budget_split_seed{s}", t, 0.58, share=0.05),
                     _rec(f"s6c_stale64_split_seed{s}", t, 0.3, share=0.4)]
    A = flat(recs, c7, tmp_path)
    logs, dev = [], []
    for s in range(3):
        for t in range(1, 41):
            logs.append({"run": f"s7d_budget_split_seed{s}", "step": t, "split_chi2_token": 1.5 if t == 12 else 0.1,
                         "synced": 1.0 if t == 12 else 0.0, "split_lag_abs_mean": 0.011, "chi2_token": 0.002})
            logs.append({"run": f"s6c_budget_tis_seed{s}", "step": t, "chi2_token": 0.3, "synced": 0.0})
        for t in pure.DEV_STEPS:
            dev.append({"run": f"s7d_budget_split_seed{s}", "step": t, "ratio": 1.0 + s})
    md, rows = pure.report(logs, dev, A)
    text = "\n".join(md)
    assert "| 40 | 2 [1, 2, 3] | - | - |" in text
    assert "| s7d_budget_split_seed1 | 12 | 1 |" in text and "| s6c_budget_tis_seed0 | - | 0 |" in text
    assert "| s6c_stale64_split_seed0 | no log | - |" in text
    assert "| 12 | 1.5* | 1.5* | 1.5* |" in text
    assert text.count("| yes |") == 3
    r = {(d["run"], d["step"]): d for d in rows}
    assert r[("s7d_budget_split_seed2", 40)]["dev_ratio"] == 3.0 and r[("s7d_budget_split_seed2", 40)]["acc_ref"] == 0.6
    assert r[("s7d_budget_split_seed0", 12)]["chi2"] == 1.5 and r[("s6c_budget_tis_seed0", 5)]["chi2"] == 0.3
