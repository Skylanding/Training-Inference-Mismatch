#!/usr/bin/env python3
"""Loop gain on Qwen3-0.6B: the stale runs at learning rate 5e-3 (s6b_*, split records in results/loop_gain_0.6b) against
the runs at 1e-2 (s2_*, results/split_0.6b) at the same eta*k: 5e-3 steps 10/20/30/40 vs 1e-2 steps 5/10/15/20. Per pair:
|Delta_SC|, |Delta_TIS| (4a delta_cum_norm), the excess E = |Delta_SC| - |Delta_TIS|, |mu|/|Delta|^2 (mu = 4c lag_mean),
|b| (4a b_norm). Mean over seeds 0-2. Writes results/loop_gain_0.6b/compare.csv, prints markdown. Pure python.

    python3 scripts/report_loop_gain.py

The 5e-3 split records come from `scripts/measure.py split --config configs/measure/qwen3_0.6b.json --run s6b_stale64_sc_seed0
--results results/loop_gain_0.6b` (and the same for tis and seeds 1, 2).
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results"
PAIRS = [(10, 5), (20, 10), (30, 15), (40, 20)]


def rec(d, run, t):
    f = ROOT / d / run / f"step_{t}.json"
    return json.loads(f.read_text()) if f.exists() else None


rows = []
for lr, d, pre, idx in (("5e-3", "loop_gain_0.6b", "s6b", 0), ("1e-2", "split_0.6b", "s2", 1)):
    for pair in PAIRS:
        t = pair[idx]
        for m in ("sc", "tis"):
            for s in range(3):
                r = rec(d, f"{pre}_stale64_{m}_seed{s}", t)
                if r is None:
                    continue
                dn = r["4a"]["delta_cum_norm"]
                rows.append({"lr": lr, "eta_k": pair[0] * 5e-3, "step": t, "method": m, "seed": s, "delta_norm": dn,
                             "mu": r["4c"]["lag_mean"], "mu_over_d2": abs(r["4c"]["lag_mean"]) / dn ** 2, "b_norm": r["4a"]["b_norm"]})
if not rows:
    raise SystemExit("no split records of the s6b / s2 stale runs under results/ (run scripts/measure.py split first); nothing written")
(ROOT / "loop_gain_0.6b").mkdir(exist_ok=True)
with open(ROOT / "loop_gain_0.6b" / "compare.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print("| eta*k | lr | step | \\|Delta_SC\\| | \\|Delta_TIS\\| | E = SC - TIS | \\|mu\\|/\\|Delta\\|^2 SC | TIS | \\|b\\| SC | TIS |\n|---|---|---|---|---|---|---|---|---|---|")
for pair in PAIRS:
    for lr, idx in (("1e-2", 1), ("5e-3", 0)):
        g = lambda m, k: mean([r[k] for r in rows if r["lr"] == lr and r["step"] == pair[idx] and r["method"] == m])
        print(f"| {pair[0] * 5e-3:.2f} | {lr} | {pair[idx]} | {g('sc', 'delta_norm'):.4f} | {g('tis', 'delta_norm'):.4f} | "
              f"{g('sc', 'delta_norm') - g('tis', 'delta_norm'):+.4f} | {g('sc', 'mu_over_d2'):.0f} | {g('tis', 'mu_over_d2'):.0f} | "
              f"{g('sc', 'b_norm'):.3f} | {g('tis', 'b_norm'):.3f} |")
