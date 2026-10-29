"""summarize off-support: results/off_support/<run>/step_<t>.json -> results/off_support/steps.csv and TABLES.md (pure
python + numpy, as tim.measure.loop.summary). Joins the deviation (KL to the same-seed reference on the probe prefixes,
results/deviation_<model>/steps.csv, in multiples of the same-step noise floor = mean KL of the bf16 PG re-runs) and the 4c
mean lag (results/split_<model>/steps.csv, where those runs were measured).

Held-out loss = acc(ref) - acc(run), same seed and step; split by pure.heldout_split against theta_0's accuracy. Spearman
over all (run, step) rows that are neither a reference nor theta_0, pooled and per model.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tim.measure.split.summary import _fmt
from tim.measure.off_support import pure

PREDICTORS = [("D_own", "d_own"), ("D_q", "d_q"), ("max_t KL", "max_kl_mean"), ("dP(EOS) first 64", "d_eos_first"),
              ("deviation / floor", "dev_ratio"), ("\\|mu\\| (probe)", "mu_abs"), ("\\|mu\\| (4c)", "mu_4c_abs"),
              ("off-support share", "share_offsupport"), ("D_own off-support", "d_own_offsupport")]
TARGETS = [("loss (ref - run)", "loss"), ("worse", "worse"), ("not learned", "not_learned")]


def _rows(p: Path) -> list:
    return list(csv.DictReader(open(p))) if p.exists() else []


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def existing(c7: dict, repo: Path) -> tuple:
    """({(run, step): (kl_mean, floor, acc_run)}, {(run, step): 4c lag mean}) from the deviation and split CSVs."""
    dev, mu = {}, {}
    for model, rel in c7["existing"]["deviation"].items():
        rows = _rows(repo / rel)
        floor_run = c7["existing"]["noise_floor_run"][model]
        floors = {}
        for r in rows:
            if any(r["run"] == floor_run.format(seed=k) for k in range(3)):
                floors.setdefault(int(r["step"]), []).append(_f(r["kl_mean"]))
        for r in rows:
            t = int(r["step"])
            fl = float(np.mean(floors[t])) if t in floors else float("nan")
            dev[(r["run"], t)] = (_f(r["kl_mean"]), fl, _f(r["acc_run"]))
    for rel in c7["existing"]["mu_4c"].values():
        for r in _rows(repo / rel):
            mu[(r["run"], int(r["step"]))] = _f(r.get("4c_lag_mean"))
    return dev, mu


def flat(recs: list, c7: dict, repo: Path) -> list:
    by = {(r["run"], r["step"]): r for r in recs}
    acc0 = {m: by.get((name, 0), {}).get("heldout", {}).get("mean_accuracy") for m, name in pure.THETA0.items()}
    dev, mu = existing(c7, repo)
    out = []
    for r in sorted(recs, key=lambda x: (x["model"], x["run"], x["step"])):
        run, t = r["run"], r["step"]
        o, q = r.get("own", {}), r.get("probe", {})
        acc = r["heldout"]["mean_accuracy"]
        d = {"run": run, "model": r["model"], "source": r["source"], "method": r["method"], "seed": r["seed"], "step": t,
             "kind": "theta0" if r["theta0"] else ("ref" if r["ref"] else "run"), "acc": acc, "acc_theta0": acc0.get(r["model"]),
             "acc_trunc": r["heldout"].get("frac_truncated"), "acc_len": r["heldout"].get("mean_completion_len")}
        ref = None if r["theta0"] else by.get((pure.reference_of(run), t))
        d["acc_ref"] = ref["heldout"]["mean_accuracy"] if ref else None
        if d["acc_ref"] is not None and d["acc_theta0"] is not None:
            d.update(pure.heldout_split(acc, d["acc_ref"], d["acc_theta0"]))
        for k in ("d_own", "d_own_per_token", "share_offsupport", "d_own_offsupport", "frac_tokens_p0_low", "len_mean",
                  "frac_truncated", "frac_short", "frac_tokens_out_top", "first_out_frac_never", "first_out_mean", "reward_mean"):
            d[k if k.startswith(("d_own", "share", "first")) else f"own_{k}"] = o.get(k)
        for k in ("d_q", "kl_mean", "max_kl_mean", "max_kl_max", "max_kl_pos_mean", "d_eos_first", "d_eos_first_abs", "eos_0_first",
                  "mu", "mu_abs"):
            d[k] = q.get(k)
        if "probe_s1" in r:
            for k in ("d_q", "max_kl_mean", "d_eos_first", "kl_mean", "mu"):
                d[f"s1_{k}"] = r["probe_s1"].get(k)
        kl, fl, acc_dev = dev.get((run, t), (None, None, None))
        d.update(dev_kl=kl, dev_floor=fl, dev_ratio=kl / fl if kl is not None and fl and np.isfinite(fl) else None,
                 acc_stage2=acc_dev)
        m4 = mu.get((run, t))
        d["mu_4c"], d["mu_4c_abs"] = m4, (abs(m4) if m4 is not None and np.isfinite(m4) else None)
        out.append(d)
    return out


def _pooled(F: list, c7: dict) -> list:
    """The run x step rows of the correlations and percentiles: references, theta_0 and the prefixes in
    spearman_exclude_prefixes (runs added after the correlation table was fixed) left out."""
    skip = tuple(f"{x}_" for x in c7.get("spearman_exclude_prefixes", []))
    return [d for d in F if d["kind"] == "run" and not d["run"].startswith(skip)]


def spearman_table(F: list, c7: dict = None) -> list:
    runs = _pooled(F, c7 or {})
    scopes = [("all", runs), ("0.6B", [d for d in runs if d["model"] == "06b"]), ("1.7B", [d for d in runs if d["model"] == "17b"])]
    L = ["| target | predictor | " + " | ".join(s for s, _ in scopes) + " |", "|---" * (2 + len(scopes)) + "|"]
    for tn, tk in TARGETS:
        for pn, pk in PREDICTORS:
            cells = []
            for _, rows in scopes:
                rho, n = pure.spearman([_f(d.get(tk)) for d in rows], [_f(d.get(pk)) for d in rows])
                cells.append(f"{rho:+.2f} (n={n})" if np.isfinite(rho) else f"n/a (n={n})")
            L.append(f"| {tn} | {pn} | " + " | ".join(cells) + " |")
    return L + [""]


def summarize(root: Path, c7: dict, repo: Path) -> list:
    recs = [json.loads(f.read_text()) for f in sorted(root.glob("*/step_*.json"))]
    if not recs:
        return [f"no off-support results under {root}"]
    F = flat(recs, c7, repo)
    cols = list(dict.fromkeys(k for d in F for k in d))
    with open(root / "steps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(F)
    seeds = [0, 1, 2]
    by = {(d["model"], d["source"], d["method"], d["seed"], d["step"]): d for d in F}

    def table(g, steps, cols, fmt="{:.3g}"):
        L = ["| t | " + " | ".join(h for h, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for t in steps:
            cs = [[by.get((*g, s, t), {}).get(k) for s in seeds] for _, k in cols]
            if all(x is None for c in cs for x in c):
                continue
            L.append(f"| {t} | " + " | ".join(_fmt([_f(x) if x is not None else None for x in c], fmt) for c in cs) + " |")
        return L + [""]

    L = ["# Off-support tables", "",
         "Generated by `scripts/measure.py summarize off-support` from `results/off_support/<run>/step_<t>.json`; one row per "
         "(run, step) in `results/off_support/steps.csv`. Cells: mean over seeds, per-seed values in brackets. Trainer numerics (HF bf16, "
         "fa3_liger kernel set) for every log-prob; samples from the bf16 vLLM sampler at theta with the training sampling parameters. "
         "Held-out: 64 prompts x 8, seed 0 (as the deviation); loss = acc(ref) - acc(run), same seed and step; not learned = "
         "max(0, ref - max(run, theta_0)), worse = max(0, theta_0 - run). D_own: 256 own samples (64 held-out prompts x 4), "
         "mean of sum_t [log p_theta - log p_theta0]; off-support = tokens with p_theta0(y) < 1e-3. D_q: theta_0's samples on the "
         "probe prefixes (1.7B and 8B: the same rule from their own theta_0), mean of sum_t KL(p_theta || p_theta0); max_t KL per "
         "response; dP(EOS) = mean of p_theta(stop) - p_theta0(stop) over positions < 64. mu = mean (log p_theta - log p_theta0) "
         "on the same probe tokens (= the stale runs' mean lag on their sampler's samples). deviation / floor: the existing KL "
         "to the same-seed reference over the same-step mean KL of the bf16 PG re-runs.", ""]
    th = [d for d in F if d["kind"] == "theta0"]
    if th:
        L += ["## theta_0", "", "| model | held-out acc | own len | truncated | < 64 tokens | p0 < 1e-3 tokens | outside top-20 tokens |",
              "|---|---|---|---|---|---|---|"]
        for d in th:
            L.append(f"| {d['model']} | " + " | ".join(f"{_f(d.get(k)):.3g}" for k in ("acc", "own_len_mean", "own_frac_truncated",
                     "own_frac_short", "own_frac_tokens_p0_low", "own_frac_tokens_out_top")) + " |")
        L.append("")
    L += ["## Spearman correlations over run x step (references and theta_0 excluded)", ""] + spearman_table(F, c7)

    groups = sorted({(d["model"], d["source"], d["method"]) for d in F if d["kind"] != "theta0"})
    for g in groups:
        steps = c7["steps"][g[0]]
        name = " ".join(g)
        L += [f"## {name}: held-out and own samples", ""]
        L += table(g, steps, [("acc", "acc"), ("loss", "loss"), ("not learned", "not_learned"), ("worse", "worse"),
                              ("D_own", "d_own"), ("off-support share", "share_offsupport"), ("own len", "own_len_mean"),
                              ("trunc", "own_frac_truncated"), ("<64", "own_frac_short"), ("first out top-20", "first_out_mean"),
                              ("never out", "first_out_frac_never")])
        L += [f"### {name}: theta_0's samples (probe prefixes) and the existing measures", ""]
        cols = [("D_q", "d_q"), ("max_t KL", "max_kl_mean"), ("max KL (max)", "max_kl_max"), ("dP(EOS) <64", "d_eos_first"),
                ("KL mean", "kl_mean"), ("mu", "mu"), ("mu 4c", "mu_4c"), ("dev / floor", "dev_ratio"), ("acc (deviation job)", "acc_stage2")]
        if g[0] == "17b":
            cols[5:5] = [("D_q (0.6B probe)", "s1_d_q"), ("dP(EOS) (0.6B probe)", "s1_d_eos_first")]
        L += table(g, steps, cols, "{:.4g}")

    runs17 = [d for d in _pooled(F, c7) if d["model"] == "17b"]
    L += ["## The three collapsed runs (1.7B): value and percentile among all 1.7B run x step rows", ""]
    keys = [("loss", "loss"), ("worse", "worse"), ("D_own", "d_own"), ("off-support", "share_offsupport"), ("D_q", "d_q"),
            ("max_t KL", "max_kl_mean"), ("dP(EOS)", "d_eos_first"), ("dev/floor", "dev_ratio"), ("\\|mu\\|", "mu_abs")]
    L += ["| run | t | " + " | ".join(h for h, _ in keys) + " |", "|---" * (len(keys) + 2) + "|"]
    for run in c7["collapsed"]:
        for d in sorted((x for x in runs17 if x["run"] == run), key=lambda x: x["step"]):
            cells = []
            for _, k in keys:
                v = _f(d.get(k))
                pct = pure.percentile_of(v, [_f(x.get(k)) for x in runs17])
                cells.append(f"{v:.3g} ({pct:.0%})" if np.isfinite(v) else "n/a")
            L.append(f"| {run} | {d['step']} | " + " | ".join(cells) + " |")
    L += ["", "## References: D_q per step (s5ref is 7b's target)", ""]
    for g in [x for x in groups if x[1] == "bf16" and x[2] == "pg"]:
        L += [f"### {' '.join(g)}", ""]
        L += table(g, c7["steps"][g[0]], [("D_q", "d_q"), ("max_t KL", "max_kl_mean"), ("dP(EOS) <64", "d_eos_first"),
                                          ("D_own", "d_own"), ("acc", "acc")], "{:.4g}")
    (root / "TABLES.md").write_text("\n".join(L) + "\n")
    return [f"{len(recs)} step results -> {root / 'TABLES.md'}, {root / 'steps.csv'}"]
