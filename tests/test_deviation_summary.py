"""Stage 2 pure parts: KL statistics, run names, reference mapping, runlist, summary (no torch)."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from tim.measure.deviation import kl_stats  # noqa: E402
from tim.measure.deviation_summary import parse_run, summarize  # noqa: E402


def test_kl_stats():
    kl = np.array([0.0, 0.2, 0.4, 1.0])
    s = kl_stats(kl, [0, 2, 4], [0.1, 0.1, 0.3, 0.3])
    assert s["kl_mean"] == pytest.approx(0.4) and s["kl_seq_mean"] == pytest.approx((0.1 + 0.7) / 2)
    assert s["kl_max"] == 1.0 and s["next_abs_dlogp_mean"] == pytest.approx(0.2) and s["n_prefixes"] == 4


def test_parse_run():
    assert parse_run("s2_fp8_stale64_tis_sc_seed2") == {"prefix": "s2", "source": "fp8_stale64", "method": "tis_sc", "seed": 2}
    assert parse_run("s9b_async4_split_async_ess_seed0")["method"] == "split_async_ess"
    with pytest.raises(ValueError):
        parse_run("s2_int8_pg_seed0")


def test_summarize_deviation(tmp_path):
    cfg2 = json.load(open(os.path.join(ROOT, "configs", "measure", "qwen3_0.6b.json")))["deviation"]

    def put(run, step, kl, acc):
        d = tmp_path / run
        d.mkdir(exist_ok=True)
        json.dump({"run": run, "step": step, "kl": {"kl_mean": kl, "kl_seq_mean": kl, "next_abs_dlogp_mean": 0.0},
                   "heldout_run": {"mean_accuracy": acc}, "heldout_ref": {"mean_accuracy": 0.5},
                   "heldout_acc_diff": acc - 0.5}, open(d / f"step_{step}.json", "w"))
    for s in (0, 1):
        put(f"s2_bf16_pg_seed{s}", 40, 0.01, 0.5)
        put(f"s2_fp8_pg_seed{s}", 40, 0.05, 0.3)
    (tmp_path / "_ref").mkdir()                       # the reference cache is not a run
    summarize(tmp_path, cfg2)
    md = (tmp_path / "TABLES.md").read_text()
    assert "| bf16 | pg | 0.01 [0.01, 0.01] | +0 [+0, +0] | 1 [1, 1] | +0 [+0, +0] |" in md
    assert "| fp8 | pg | 0.05 [0.05, 0.05] | +0.04 [+0.04, +0.04] | 5 [5, 5] | -0.2 [-0.2, -0.2] |" in md
