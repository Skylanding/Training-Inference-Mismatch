#!/usr/bin/env python3
"""Training-log table of the Qwen3-1.7B runs with resynchronization and of DLC without it: per run and step, whether the
sampler was synced (tim/synced), the chi2 the decision used (tim/chi2_seen; resynchronized runs), chi2_token,
mu = -log_ppl_diff (log p_trainer - log q of the sampled tokens, nats / token; for DLC runs the trainer side is pi_old,
so the lag is tim/split_lag_mean) and the IS effective sample size, from the verl step metrics in
results/train_logs_resync_1.7b/<run>.jsonl (the file logger of scripts/train.sh, metrics.<attempt>.jsonl). For DLC
(s7d_*) also tim/split_chi2_token (E_q[r^2] - 1 of the lag ratio, the chi2 its resynchronization reads; ESS = 1 / (1 + chi2))
and tim/split_lag_abs_mean (back near the step-1 value at the step after each sync, where the reference took the synced
weights). A step logged twice (a resumed attempt) keeps its last line. Writes results/train_logs_resync_1.7b/steps.csv and
prints a markdown summary. Pure python.

    python3 scripts/report_train_logs.py          (scripts/report_dlc_resync.py imports `rows`)
"""
import csv
import json
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "results" / "train_logs_resync_1.7b"
K = {"chi2_token": "actor/rollout_corr/chi2_token", "log_ppl_diff": "actor/rollout_corr/log_ppl_diff",
     "ess": "actor/rollout_corr/rollout_is_eff_sample_size", "synced": "tim/synced", "chi2_seen": "tim/chi2_seen",
     "acc": "critic/score/mean"}
PATTERNS = ("s6c_*.jsonl", "s7d_*.jsonl")


def rows() -> list:
    out = []
    for f in sorted(p for pat in PATTERNS for p in D.glob(pat)):
        by_step = {}
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line); d = r["data"]
            # split runs (decoupled mode) log rollout_corr/* without the actor/ prefix, and the lag itself as tim/split_lag_mean
            get = lambda v: d.get(v, d.get(v[len("actor/"):]) if v.startswith("actor/") else None)
            row = {"run": f.stem, "step": r["step"], **{k: get(v) for k, v in K.items()}}
            row["mu"] = -row["log_ppl_diff"] if row["log_ppl_diff"] is not None else None
            row["split_lag_mean"] = d.get("tim/split_lag_mean")
            row["split_lag_abs_mean"] = d.get("tim/split_lag_abs_mean")
            row["split_chi2_token"] = d.get("tim/split_chi2_token")
            by_step[r["step"]] = row
        out += [by_step[s] for s in sorted(by_step)]
    return out


def main() -> None:
    rs = rows()
    if not rs:
        raise SystemExit(f"no training logs under {D}; nothing written")
    with open(D / "steps.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rs[0].keys())); w.writeheader(); w.writerows(rs)
    print("| run | sync steps (end of step s; weights sample s+1) | chi2_token max | mu at 10 / 20 / 40 | split lag at 10 / 20 / 40 | split chi2 max | ESS min |\n|---|---|---|---|---|---|---|")
    for run in sorted({r["run"] for r in rs}):
        rr = sorted((r for r in rs if r["run"] == run), key=lambda r: r["step"])
        syncs = [r["step"] for r in rr if r["synced"]]
        mu = {r["step"]: r["mu"] for r in rr}
        lag = {r["step"]: r["split_lag_mean"] for r in rr}
        sc = [r["split_chi2_token"] for r in rr if r["split_chi2_token"] is not None]
        chi2_max = max((r["chi2_token"] for r in rr if r["chi2_token"] is not None), default=float("nan"))
        ess_min = min((r["ess"] for r in rr if r["ess"] is not None), default=float("nan"))
        print(f"| {run} | {', '.join(map(str, syncs)) or '-'} | {chi2_max:.3g} | "
              + " / ".join("-" if mu.get(s) is None else f"{mu[s]:+.3f}" for s in (10, 20, 40))
              + " | " + " / ".join("-" if lag.get(s) is None else f"{lag[s]:+.3f}" for s in (10, 20, 40))
              + (f" | {max(sc):.3g}" if sc else " | -") + f" | {ess_min:.3f} |")


if __name__ == "__main__":
    main()
