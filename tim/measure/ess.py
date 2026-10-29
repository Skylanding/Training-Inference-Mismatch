"""Report of DLC with ESS-triggered resynchronization on Qwen3-1.7B, from training logs (pure python; tests/test_ess_report.py).

Internal names (README.md, "Run names"): s7d_budget_split = DLC (split with resynchronization when the ESS of the lag ratio
falls below one half, i.e. tim/split_chi2_token = E_q[rho^2] - 1 > 1), s6c_stale64_split = DLC without resynchronization,
s6c_budget_tis = importance sampling with resynchronization.

1. `check_reload`: the GPU-side test of the reference reload, read from a run's verl step metrics (metrics.*.jsonl lines
   {"step": s, "data": {...}}). Step s logs the lag log p_theta_{s-1} - log p_pi_old on step s's samples
   (tim/split_lag_abs_mean) and, at its end, whether the sampler was synced (tim/synced). With the reload, pi_old after a
   sync at the end of step s is theta_s, the trainer that computes step s + 1's log-probs, so step s + 1's lag is recompute
   noise only, the size of step 1's (DLC without resynchronization, no reload: step 2 already 2-3 x step 1, step 4 4-5 x).
   Also: the resynchronization decision used tim.dlc.hooks' chi2 (tim/chi2_seen == tim/split_chi2_token) at every step.
2. `report`: DLC (s7d) side by side with DLC without resynchronization and importance sampling with resynchronization:
   deviation / noise floor per step (scripts/report_resync_deviation.py rows), sync steps and the chi2 per step
   (scripts/report_train_logs.py rows), held-out accuracy and off-support share per step (tim.measure.off_support.summary.flat),
   and the checks stated before the runs.
"""
from __future__ import annotations

import math

LAG = "tim/split_lag_abs_mean"
CHI2 = "tim/split_chi2_token"
SEEN = "tim/chi2_seen"
SYNCED = "tim/synced"

GROUPS = [("budget x split (7d)", "s7d_budget_split"), ("stale64 x split (6c)", "s6c_stale64_split"),
          ("budget x TIS (6c)", "s6c_budget_tis")]
REF, THETA0 = "s5ref_bf16_pg", "theta0_17b"
DEV_STEPS = [5, 10, 15, 20, 25, 30, 35, 40]
S7A_STEPS = [10, 15, 20, 30, 40]
# checks stated before the DLC runs (case: a resynchronization happens before step 30)
OFFSUPPORT_MAX, ACC_GAP_MAX, THETA0_ACC = 0.14, 0.08, 0.33


def by_step(lines: list) -> dict:
    """{step: data} from metrics.jsonl records; a step logged twice (a resumed attempt) keeps its last record."""
    out = {}
    for r in lines:
        out[int(r["step"])] = r["data"]
    return out


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def check_reload(lines: list, factor: float = 1.5, slack: float = 0.005, every_step_synced: bool = False,
                 min_checked: int = 1) -> tuple:
    """-> (ok, report lines). After every sync at the end of step s, step s + 1's tim/split_lag_abs_mean must be at most
    max(factor x step 1's, step 1's + slack); every step must log a finite tim/split_chi2_token, and tim/chi2_seen (the
    value the resynchronization decision used) must equal it. every_step_synced: TIM_CHI2_BUDGET=-1 syncs at every step.
    At least min_checked post-sync steps must exist (otherwise the test checked nothing)."""
    S = by_step(lines)
    msgs, ok = [], True
    if 1 not in S or _num(S[1].get(LAG)) is None:
        return False, ["no step 1 with tim/split_lag_abs_mean (not a split run, or the log is empty)"]
    base = _num(S[1][LAG])
    limit = max(factor * base, base + slack)
    msgs.append(f"step 1 lag |log p_theta - log p_old| mean {base:.4g}; limit after a sync {limit:.4g}")
    checked = 0
    for s in sorted(S):
        d = S[s]
        c, seen = _num(d.get(CHI2)), _num(d.get(SEEN))
        if c is None:
            ok = False; msgs.append(f"step {s}: no finite {CHI2}")
        elif seen is None or abs(seen - c) > 1e-6 * max(1.0, abs(c)):
            ok = False; msgs.append(f"step {s}: {SEEN}={d.get(SEEN)} is not {CHI2}={c:.6g} (the resynchronization did not read the lag chi2)")
        synced = _num(d.get(SYNCED))
        if every_step_synced and synced != 1.0:
            ok = False; msgs.append(f"step {s}: {SYNCED}={d.get(SYNCED)}, expected 1 (budget -1 syncs every step)")
        if synced == 1.0 and s + 1 in S:
            lag = _num(S[s + 1].get(LAG))
            checked += 1
            good = lag is not None and lag <= limit
            ok &= good
            msgs.append(f"sync at end of step {s}: step {s + 1} lag {lag if lag is None else f'{lag:.4g}'} "
                        f"({'ok' if good else 'TOO LARGE: the reference did not take the synced weights'})")
    if checked < min_checked:
        ok = False; msgs.append(f"only {checked} post-sync steps to check (need {min_checked})")
    return ok, msgs


# ------------------------------------------------------------------------------------------------------- tables
def _cell(vals, fmt="{:.3g}"):
    v = [x for x in vals if x is not None]
    if not v:
        return "-"
    return fmt.format(sum(v) / len(v)) + " [" + ", ".join("-" if x is None else fmt.format(x) for x in vals) + "]"


def _lookup(rows: list, key_run: str, key_step: str) -> dict:
    return {(r[key_run], int(r[key_step])): r for r in rows}


def report(log_rows: list, dev_rows: list, s7a_rows: list, seeds=(0, 1, 2)) -> tuple:
    """-> (markdown lines, per (group, run, step) rows for steps.csv). Inputs: scripts/report_train_logs.rows(),
    scripts/report_resync_deviation.rows(), tim.measure.off_support.summary.flat(...) (missing parts print as '-')."""
    L = _lookup(log_rows, "run", "step")
    D = _lookup(dev_rows, "run", "step")
    A = _lookup(s7a_rows, "run", "step")

    def chi2(run, t):
        r = L.get((run, t))
        if r is None:
            return None
        return _num(r.get("split_chi2_token")) if "_split" in run else _num(r.get("chi2_token"))

    def g(table, run, t, k):
        r = table.get((run, t))
        return None if r is None else _num(r.get(k))

    out, csv_rows = [], []
    out += ["# DLC with resynchronization, Qwen3-1.7B", "",
            "Generated by `python3 scripts/report_dlc_resync.py`. Qwen3-1.7B, Countdown, 40 steps, seeds 0-2; "
            "cells: mean over seeds, per-seed values in brackets. budget x split = DLC (centering term, PPO clip on the lag "
            "ratio p_theta / p_theta_old) with a resynchronization whenever the step's `tim/split_chi2_token` (E_q[r^2] - 1 "
            "of the lag ratio) exceeds 1, i.e. its ESS falls below one half, the reference (pi_old) reloaded at each sync. "
            "Deviation: KL(pi_run || pi_s5ref) on the probe prefixes over the same-seed noise floor "
            "(`scripts/report_resync_deviation.py`). chi2: `tim/split_chi2_token` for DLC, verl's `rollout_corr/chi2_token` "
            "for importance sampling with resynchronization; not logged by DLC without resynchronization (its runs predate "
            "the metric). Held-out (64 prompts x 8, seed 0) and the off-support share of D_own (p_theta0(y) < 1e-3): "
            "the off-support measurement (`results/off_support/<run>/step_<t>.json`).", ""]

    def side_by_side(title, steps, value, fmt="{:.3g}", extra=None):
        hdr = [n for n, _ in GROUPS] + ([h for h, _ in extra] if extra else [])
        rows = [f"### {title}", "", "| t | " + " | ".join(hdr) + " |", "|---" * (len(hdr) + 1) + "|"]
        for t in steps:
            cells = [_cell([value(f"{pre}_seed{s}", t) for s in seeds], fmt) for _, pre in GROUPS]
            if extra:
                cells += [fn(t) for _, fn in extra]
            rows.append(f"| {t} | " + " | ".join(cells) + " |")
        return rows + [""]

    ref_acc = lambda t: _cell([g(A, f"{REF}_seed{s}", t, "acc") for s in seeds])
    th0 = g(A, THETA0, 0, "acc")
    out += ["## Per step, side by side", ""]
    out += side_by_side("Deviation / noise floor", DEV_STEPS, lambda r, t: g(D, r, t, "ratio"))
    out += side_by_side("Held-out accuracy", S7A_STEPS, lambda r, t: g(A, r, t, "acc"),
                        extra=[("s5ref", ref_acc), ("theta_0", lambda t: "-" if th0 is None else f"{th0:.3g}")])
    out += side_by_side("Off-support share of D_own", S7A_STEPS, lambda r, t: g(A, r, t, "share_offsupport"))
    out += side_by_side("D_own", S7A_STEPS, lambda r, t: g(A, r, t, "d_own"))
    out += side_by_side("chi2 at the step (split: tim/split_chi2_token; TIS: rollout_corr/chi2_token)", DEV_STEPS, chi2)

    out += ["## Sync steps (end of step s; the weights sample step s + 1)", "", "| run | sync steps | n |", "|---|---|---|"]
    for _, pre in GROUPS:
        for s in seeds:
            run = f"{pre}_seed{s}"
            steps = sorted(t for (r, t), x in L.items() if r == run and _num(x.get("synced")) == 1.0)
            have = any(r == run for r, _ in L)
            out.append(f"| {run} | {', '.join(map(str, steps)) or ('-' if have else 'no log')} | {len(steps) if have else '-'} |")
    out.append("")

    last = max([t for (r, t) in L if r.startswith(GROUPS[0][1])], default=0)
    out += ["## DLC per step: tim/split_chi2_token and tim/split_lag_abs_mean (* = synced at the end of the step)", "",
            "| t | " + " | ".join(f"chi2 seed {s}" for s in seeds) + " | " + " | ".join(f"lag seed {s}" for s in seeds) + " |",
            "|---" * (2 * len(seeds) + 1) + "|"]
    for t in range(1, last + 1):
        cells = []
        for s in seeds:
            r = L.get((f"{GROUPS[0][1]}_seed{s}", t))
            c = None if r is None else _num(r.get("split_chi2_token"))
            star = "*" if r is not None and _num(r.get("synced")) == 1.0 else ""
            cells.append("-" if c is None else f"{c:.3g}{star}")
        for s in seeds:
            r = L.get((f"{GROUPS[0][1]}_seed{s}", t))
            v = None if r is None else _num(r.get("split_lag_abs_mean"))
            cells.append("-" if v is None else f"{v:.3g}")
        out.append(f"| {t} | " + " | ".join(cells) + " |")
    out.append("")

    out += ["## Checks stated before the runs, per DLC seed", "",
            f"Case 1 (a sync before step 30): off-support share < {OFFSUPPORT_MAX} at 30 and 40, |acc - s5ref| <= {ACC_GAP_MAX} "
            f"at 40, acc >= theta_0 ({THETA0_ACC} stated; measured {'-' if th0 is None else f'{th0:.3g}'}) at every measured step. "
            "Case 2: no sync within 40 steps, DLC = DLC without resynchronization.", "",
            "| seed | syncs before 30 | first sync | off-support 30 / 40 | acc 40 - s5ref | min acc - theta_0 | case 1 holds |",
            "|---|---|---|---|---|---|---|"]
    for s in seeds:
        run = f"{GROUPS[0][1]}_seed{s}"
        syncs = sorted(t for (r, t), x in L.items() if r == run and _num(x.get("synced")) == 1.0)
        off = [g(A, run, t, "share_offsupport") for t in (30, 40)]
        a40, r40 = g(A, run, 40, "acc"), g(A, f"{REF}_seed{s}", 40, "acc")
        gap = None if a40 is None or r40 is None else a40 - r40
        accs = [g(A, run, t, "acc") for t in S7A_STEPS]
        floor = th0 if th0 is not None else THETA0_ACC
        dmin = min((a - floor for a in accs if a is not None), default=None)
        complete = all(x is not None for x in off) and gap is not None and all(a is not None for a in accs)
        holds = (complete and any(t < 30 for t in syncs) and all(x < OFFSUPPORT_MAX for x in off) and abs(gap) <= ACC_GAP_MAX
                 and dmin >= 0)
        f = lambda v, fmt="{:.3g}": "-" if v is None else fmt.format(v)
        out.append(f"| {s} | {sum(t < 30 for t in syncs)} | {syncs[0] if syncs else '-'} | {f(off[0])} / {f(off[1])} | "
                   f"{f(gap, '{:+.3f}')} | {f(dmin, '{:+.3f}')} | {('yes' if holds else 'no') if complete else 'incomplete'} |")
    out.append("")

    for name, pre in GROUPS:
        for s in seeds:
            run = f"{pre}_seed{s}"
            for t in sorted({t for (r, t) in list(L) + list(D) + list(A) if r == run}):
                lr = L.get((run, t)) or {}
                csv_rows.append({"group": name, "run": run, "seed": s, "step": t, "dev_ratio": g(D, run, t, "ratio"),
                                 "chi2": chi2(run, t), "synced": _num(lr.get("synced")),
                                 "split_lag_abs_mean": _num(lr.get("split_lag_abs_mean")), "acc": g(A, run, t, "acc"),
                                 "acc_ref": g(A, f"{REF}_seed{s}", t, "acc"),
                                 "share_offsupport": g(A, run, t, "share_offsupport"), "d_own": g(A, run, t, "d_own"),
                                 "d_q": g(A, run, t, "d_q")})
    return out, csv_rows
