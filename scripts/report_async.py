#!/usr/bin/env python3
"""Continuous-asynchrony tables on Qwen3-1.7B (the paper's async table), CPU, reads results/ only:

  python3 scripts/report_async.py            # TABLES.md to stdout
  python3 scripts/report_async.py --write    # results/async_1.7b/TABLES.md and results/async_1.7b/steps.csv

The five runs s9b_async4_<method>_seed0 (weights pushed after every step, maximum version gap 4) next to the stale-sampler
1.7B rows of the same corrections (scripts/report_scale_8b.py MODELS["Qwen3-1.7B"]). Both against the same reference and
noise floor:
  deviation = KL(run || s5ref_bf16_pg_seed0) on the probe prefixes (results/deviation_1.7b/<run>)
              / the same KL of the noise-floor run s5_bf16_pg_seed0 at the same step;
  accuracy  = the held-out accuracy of the off-support records, 64 x 8 (results/off_support/<run>), else the deviation
              record's (marked *).
Training-log columns from results/async_1.7b/logs/<run>.jsonl (verl step metrics): per-token version gap
(tim/async_tok_gap_mean, mean over steps; share of tokens with gap >= 1), the admission variant's deferral
(tim/async_ess_defer_frac mean, tim/async_ess_deferred_total at the last step), the step-40 training reward and the mean
wall-clock time per step (timing_s/step).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
_spec = importlib.util.spec_from_file_location("report_scale_8b", ROOT / "scripts" / "report_scale_8b.py")
s8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s8)

M17 = s8.MODELS["Qwen3-1.7B"]
DEV, FLOOR, REF, THETA0 = M17["dev"], M17["floor"], M17["ref"], M17["theta0"]   # deviation_1.7b, s5_bf16_pg_seed0, s5ref, theta0_17b
ASYNC = {"Policy gradient": "s9b_async4_pg_seed0", "Importance sampling": "s9b_async4_tis_seed0",
         "M2PO": "s9b_async4_m2po_seed0", "DLC": "s9b_async4_split_async_seed0",
         "DLC with admission": "s9b_async4_split_async_ess_seed0"}
STALE = {m: M17["rows"].get(m) for m in ASYNC}          # DLC with admission has no stale-sampler counterpart
DEV_STEPS = [5, 10, 15, 20, 25, 30, 35, 40]
S7A_STEPS = [10, 15, 20, 30, 40]
LOGS = R / "async_1.7b" / "logs"


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def kl(run: str, step: int):
    r = load(R / DEV / run / f"step_{step}.json")
    return r["kl"]["kl_mean"] if r else None


def ratio(run: str, step: int):
    k, f = kl(run, step), kl(FLOOR, step)
    return k / f if k is not None and f else None


def train_log(run: str) -> dict:
    """{step: data} of the run's verl step metrics (a step logged twice keeps its last record)."""
    f = LOGS / f"{run}.jsonl"
    out = {}
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[int(r["step"])] = r["data"]
    return out


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def chi2_token(d: dict):
    for k in ("tim/split_chi2_token", "rollout_corr/chi2_token", "actor/rollout_corr/chi2_token"):
        if _num(d.get(k)) is not None:
            return _num(d[k])
    return None


def log_summary(run: str) -> dict:
    lg = train_log(run)
    if not lg:
        return {}
    steps = sorted(lg)
    last = lg[steps[-1]]
    return {"n_steps": len(steps), "gap_mean": _mean([_num(lg[s].get("tim/async_tok_gap_mean")) for s in steps]),
            "gap_ge1": _mean([None if _num(lg[s].get("tim/async_tok_gap_frac_g0")) is None
                              else 1 - _num(lg[s]["tim/async_tok_gap_frac_g0"]) for s in steps]),
            "gap_max": max((_num(lg[s].get("tim/async_tok_gap_max")) or 0) for s in steps),
            "defer_frac": _mean([_num(lg[s].get("tim/async_ess_defer_frac")) for s in steps]),
            "deferred_total": _num(last.get("tim/async_ess_deferred_total")),
            "reward_40": _num(lg.get(40, {}).get("critic/score/mean")),
            "step_time": _mean([_num(lg[s].get("timing_s/step")) for s in steps])}


def steps_rows() -> list:
    rows = []
    for method, run in ASYNC.items():
        lg = train_log(run)
        for t in sorted(set(DEV_STEPS) | set(S7A_STEPS)):
            d = load(R / DEV / run / f"step_{t}.json")
            a = load(R / "off_support" / run / f"step_{t}.json")
            g = lg.get(t, {})
            rows.append({"run": run, "method": method, "step": t, "kl": kl(run, t), "kl_floor": kl(FLOOR, t),
                         "kl_over_floor": ratio(run, t),
                         "acc_dev": d["heldout_run"]["mean_accuracy"] if d else None,
                         "acc_ref_dev": d["heldout_ref"]["mean_accuracy"] if d else None,
                         "acc_7a": a["heldout"]["mean_accuracy"] if a else None,
                         "d_own": a["own"]["d_own"] if a else None,
                         "share_offsupport": a["own"]["share_offsupport"] if a else None,
                         "train_reward": _num(g.get("critic/score/mean")), "train_chi2_token": chi2_token(g),
                         "gap_mean": _num(g.get("tim/async_tok_gap_mean")), "gap_max": _num(g.get("tim/async_tok_gap_max")),
                         "ess_defer_frac": _num(g.get("tim/async_ess_defer_frac"))})
    return rows


def tables() -> str:
    fmt, out = s8.fmt, io.StringIO()
    p = lambda *a: print(*a, file=out)   # noqa: E731
    p("# Continuous asynchrony, Qwen3-1.7B, seed 0\n")
    p("Generated by `scripts/report_async.py`. Deviation = KL(run || s5ref_bf16_pg_seed0) on the probe prefixes divided "
      "by the same KL of the noise-floor run s5_bf16_pg_seed0 (the reference and floor of the stale-sampler 1.7B rows); "
      "accuracy = held-out accuracy (64 x 8, %) of the off-support records, * = from the deviation record. Stale columns = "
      "the stale-sampler 1.7B rows (`scripts/report_scale_8b.py`), async columns = maximum version gap 4.\n")
    p("## Step 40\n")
    p("| Correction | Stale deviation (x floor) | Stale accuracy | Async deviation (x floor) | Async accuracy | Async run |")
    p("|---|---|---|---|---|---|")
    a0, _ = s8.acc(DEV, THETA0, 0)
    p(f"| Initial model theta_0 | - | {fmt(a0, pct=True)} | - | {fmt(a0, pct=True)} | - |")
    ar, mk = s8.acc(DEV, REF)
    p(f"| On-policy training (s5ref) | 1 (reference) | {fmt(ar, pct=True)}{mk} | 1 (reference) | {fmt(ar, pct=True)}{mk} | - |")
    for method, run in ASYNC.items():
        st = STALE[method]
        if st:
            a, m = s8.acc(DEV, st)
            c8 = f"{fmt(ratio(st, 40))} | {fmt(a, pct=True)}{m if a is not None else ''}"
        else:
            c8 = "- | -"
        a, m = s8.acc(DEV, run)
        p(f"| {method} | {c8} | {fmt(ratio(run, 40))} | {fmt(a, pct=True)}{m if a is not None else ''} | `{run}` |")
    p(f"\nNoise-floor KL at step 40: {fmt(kl(FLOOR, 40), 4)}\n")
    p("## Deviation over steps (x floor, async runs)\n")
    p("| Correction | " + " | ".join(str(t) for t in DEV_STEPS) + " |")
    p("|---|" + "---|" * len(DEV_STEPS))
    for method, run in ASYNC.items():
        p(f"| {method} | " + " | ".join(fmt(ratio(run, t)) for t in DEV_STEPS) + " |")
    p("\n## Held-out accuracy over steps (%, async runs)\n")
    p("| Correction | " + " | ".join(str(t) for t in S7A_STEPS) + " |")
    p("|---|" + "---|" * len(S7A_STEPS))
    for method, run in ASYNC.items():
        cells = []
        for t in S7A_STEPS:
            r = load(R / "off_support" / run / f"step_{t}.json")
            cells.append(fmt(r["heldout"]["mean_accuracy"] if r else None, pct=True))
        p(f"| {method} | " + " | ".join(cells) + " |")
    p("\n## Training logs (`results/async_1.7b/logs`)\n")
    p("| Correction | Steps | Mean token version gap | Share of tokens with gap >= 1 | Max gap | Deferred share (mean) | "
      "Deferred total | Training reward at 40 | Step time (s) |")
    p("|---|---|---|---|---|---|---|---|---|")
    for method, run in ASYNC.items():
        s = log_summary(run)
        p(f"| {method} | {s.get('n_steps', '-')} | {fmt(s.get('gap_mean'))} | {fmt(s.get('gap_ge1'))} | "
          f"{fmt(s.get('gap_max'), 0)} | {fmt(s.get('defer_frac'), 3)} | {fmt(s.get('deferred_total'), 0)} | "
          f"{fmt(s.get('reward_40'), 3)} | {fmt(s.get('step_time'), 1)} |")
    return out.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write results/async_1.7b/TABLES.md and results/async_1.7b/steps.csv")
    a = ap.parse_args(argv)
    md = tables()
    if not a.write:
        print(md, end="")
        return
    if not any((R / DEV / run).is_dir() for run in ASYNC.values()):
        raise SystemExit(f"no deviation records of the async runs under {R / DEV}; nothing written")
    d = R / "async_1.7b"
    d.mkdir(parents=True, exist_ok=True)
    (d / "TABLES.md").write_text(md)
    rows = steps_rows()
    with open(d / "steps.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{d / 'TABLES.md'}, {d / 'steps.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
