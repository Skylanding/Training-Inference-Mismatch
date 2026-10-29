#!/usr/bin/env python3
"""Deviation table of the Qwen3-1.7B stale-sampler runs: KL(pi_run || pi_ref) on the probe prefixes (deviation records,
results/deviation_1.7b/<run>/step_<t>.json) divided by the same-seed noise floor (the on-policy re-run s5_bf16_pg_seed<s>
against s5ref), steps 5-40. Groups (internal names, README.md, "Run names"): budget x TIS = importance sampling with
resynchronization, stale64 x split = DLC without resynchronization, 5c stale64 x TIS / SC = importance sampling and score
centering, 7d budget x split = DLC. Writes results/train_logs_resync_1.7b/deviation.csv and prints markdown. Pure python.

    python3 scripts/report_resync_deviation.py   (scripts/report_dlc_resync.py imports `rows`)
"""
import csv
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results" / "deviation_1.7b"
OUT = Path(__file__).resolve().parent.parent / "results" / "train_logs_resync_1.7b" / "deviation.csv"
GROUPS = [("budget x TIS", "s6c_budget_tis"), ("stale64 x split", "s6c_stale64_split"),
          ("5c stale64 x TIS", "s5_stale64_tis"), ("5c stale64 x SC", "s5_stale64_sc"),
          ("7d budget x split", "s7d_budget_split")]
STEPS = [5, 10, 15, 20, 25, 30, 35, 40]


def _step(run, t):
    f = R / run / f"step_{t}.json"
    return json.loads(f.read_text()) if f.exists() else None


def kl(run, t):
    r = _step(run, t)
    return r["kl"]["kl_mean"] if r else None


def acc(run, t):
    r = _step(run, t)
    return r.get("heldout_acc_diff") if r else None


def rows(groups=GROUPS) -> list:
    out = []
    for name, pre in groups:
        if not any((R / f"{pre}_seed{s}").is_dir() for s in range(3)):
            continue                                      # a group with no deviation records yet
        for s in range(3):
            run = f"{pre}_seed{s}"
            for t in STEPS:
                k, fl = kl(run, t), kl(f"s5_bf16_pg_seed{s}", t)
                out.append({"group": name, "run": run, "seed": s, "step": t, "kl": k, "floor": fl,
                            "ratio": k / fl if k is not None and fl else None, "heldout_acc_diff": acc(run, t)})
    return out


def main() -> None:
    rs = rows()
    if not rs:
        raise SystemExit(f"no deviation records under {R} (run scripts/measure.py deviation first); nothing written")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rs[0].keys())); w.writeheader(); w.writerows(rs)
    f = lambda v: "-" if v is None else (f"{v:.3g}")
    print("KL / same-seed noise floor (mean over seeds, seeds in brackets)\n")
    print("| group | " + " | ".join(f"step {t}" for t in STEPS) + " | held-out acc run - ref at 40 |\n|---|" + "---|" * (len(STEPS) + 1))
    for name in dict.fromkeys(r["group"] for r in rs):
        cells = []
        for t in STEPS:
            v = [r["ratio"] for r in rs if r["group"] == name and r["step"] == t]
            ok = [x for x in v if x is not None]
            cells.append(f"{sum(ok) / len(ok):.3g} [{', '.join(f(x) for x in v)}]" if ok else "-")
        a = [r["heldout_acc_diff"] for r in rs if r["group"] == name and r["step"] == 40 and r["heldout_acc_diff"] is not None]
        print(f"| {name} | " + " | ".join(cells) + (f" | {sum(a) / len(a):+.3f} |" if a else " | - |"))


if __name__ == "__main__":
    main()
