"""Replay tables from results/replay_0.6b (pure python + numpy).

Input: results/replay_0.6b/<run>/<source>/step_<t>.json written by `scripts/measure.py replay` on the on-policy reference
checkpoints s2ref_bf16_pg_seed*: each sampler (on-policy bf16, quantized, stale, quantized and stale, eager) is replayed on
the same checkpoints and probe sequences. S1 = persistence of the sampled-token error across steps, S2 = ratio
truncation, S3 = per-sequence sum of delta (the sequence offset of the paper), S4 = the exact split of that sum into the
policy lag and the numerical discrepancy.

steps.csv     one row per (run, source, step): the S1-S4 scalars of that step
lags.csv      one row per (run, source, quantity, lag): mean over anchor pairs of the cross-step Pearson, n, coverage
per_seed.csv  one row per (run, source): S1 lag 20, S3 slope, tail mass, truncation and the means over steps
TABLES.md     S1-S4 per source, mean over seeds [per seed]
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from tim.measure.formulas import linear_slope

SOURCES = ("bf16", "fp8", "stale", "fp8_stale", "eager")
SEED = re.compile(r"_seed(\d+)$")
LAG_Q = ("eps_pearson", "delta_pearson")

STEP_FIELDS = (
    # S1 (a): repeat consistency of the sampler (same batch twice; batch 1 vs 64) and of the trainer
    ("rep_same_next_abs_mean", ("sampler_repeat", "repeat_same_batch", "next_abs_mean")),
    ("rep_same_next_abs_max", ("sampler_repeat", "repeat_same_batch", "next_abs_max")),
    ("rep_b1_next_abs_mean", ("sampler_repeat", "batch1_vs_batch64", "next_abs_mean")),
    ("rep_b1_next_abs_max", ("sampler_repeat", "batch1_vs_batch64", "next_abs_max")),
    ("trainer_rep_next_abs_max", ("trainer_repeat", "next_abs_max")),
    ("eps_abs_mean", ("eps", "eps_abs_mean")), ("kl_q_p_tail_mean", ("eps", "kl_q_p_tail_mean")),
    # S2: TIS side on the P2 batch and on the probe set; SC side = tail
    ("trunc_frac_p2", ("ratio", "frac_ratio_gt_cap")),
    ("trunc_frac_probe", ("S2_probe", "frac_ratio_gt_cap")),
    ("excess_ratio_probe", ("S2_probe", "excess_ratio_mean")),
    ("truncated_delta_seq_probe", ("S2_probe", "truncated_delta_seq_mean")),
    ("tail_mass", ("S2_probe", "tail_mass_mean")),
    # S3: per-sequence sum of delta over the probe set
    ("seq_delta_mean", ("S3", "seq_sum_delta", "mean")), ("seq_delta_sd", ("S3", "seq_sum_delta", "sd")),
    ("seq_delta_q05", ("S3", "seq_sum_delta", "q05")), ("seq_delta_median", ("S3", "seq_sum_delta", "median")),
    ("seq_delta_q95", ("S3", "seq_sum_delta", "q95")),
    # S4: sum delta = sum(log p_theta - log p_theta_old) + sum(log p_theta_old - log q)
    ("seq_lag_mean", ("S4", "seq_sum_lag", "mean")), ("seq_engine_mean", ("S4", "seq_sum_engine", "mean")),
)


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return float("nan")
        d = d[k]
    return float(d) if isinstance(d, (int, float)) else float("nan")


def _mean(v):
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def _fmt(vals, f="{:.3g}"):
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return "n/a"
    return f.format(np.mean(vals)) + " [" + ", ".join(f.format(v) for v in vals) + "]"


def seed_of(run: str) -> int:
    m = SEED.search(run)
    return int(m.group(1)) if m else -1


def load(root: Path):
    """Step rows and lag accumulators from the step JSONs under root."""
    rows = []
    lag_acc = defaultdict(list)          # (run, source, quantity, lag) -> values over anchor pairs
    lag_miss = defaultdict(int)
    for f in sorted(Path(root).glob("*/*/step_*.json")):
        if f.parent.parent.name.startswith("_"):
            continue
        r = json.loads(f.read_text())
        rows.append({"run": r["run"], "seed": seed_of(r["run"]), "source": r["source"], "step": r["step"],
                     **{n: _get(r, p) for n, p in STEP_FIELDS}})
        lg = r.get("lags", {})
        for e in lg.get("eps", []):
            for q in LAG_Q:
                lag_acc[(r["run"], r["source"], q, e["lag"])].append(float(e[q]))
        for m in lg.get("missing", []):
            if m["kind"] == "eps":
                for q in LAG_Q:
                    lag_miss[(r["run"], r["source"], q, m["lag"])] += 1
    rows.sort(key=lambda x: (x["source"], x["seed"], x["step"]))
    lag_rows = []
    for key in sorted(set(lag_acc) | set(lag_miss), key=lambda k: (k[1], k[0], k[2], k[3])):
        v = [x for x in lag_acc.get(key, []) if np.isfinite(x)]
        n_miss = lag_miss.get(key, 0)
        lag_rows.append({"run": key[0], "seed": seed_of(key[0]), "source": key[1], "quantity": key[2], "lag": key[3],
                         "mean": float(np.mean(v)) if v else float("nan"), "n": len(v), "n_missing": n_miss,
                         "coverage": len(v) / (len(v) + n_miss) if (v or n_miss) else float("nan")})
    return rows, lag_rows


def per_seed(rows, lag_rows) -> list:
    """One row per (run, source). S1 = eps Pearson at lag 20 (mean over anchors); S3 slope =
    least-squares slope of the per-step mean sum delta on the step; tail and truncation = means over steps."""
    by = defaultdict(list)
    for x in rows:
        by[(x["run"], x["source"])].append(x)
    lag = {(x["run"], x["source"], x["quantity"], x["lag"]): x for x in lag_rows}
    out = []
    for (run, src), xs in sorted(by.items(), key=lambda kv: (kv[0][1], seed_of(kv[0][0]))):
        xs.sort(key=lambda x: x["step"])
        rec = {"run": run, "seed": seed_of(run), "source": src, "n_steps": len(xs),
               "step_min": xs[0]["step"], "step_max": xs[-1]["step"]}
        for q in LAG_Q:
            for l in (1, 5, 10, 20):
                e = lag.get((run, src, q, l))
                rec[f"{q}_lag{l}"] = e["mean"] if e else float("nan")
                rec[f"{q}_lag{l}_n"] = e["n"] if e else 0
        rec["S3_slope"] = linear_slope([x["step"] for x in xs], [x["seq_delta_mean"] for x in xs])
        rec["S4_lag_slope"] = linear_slope([x["step"] for x in xs], [x["seq_lag_mean"] for x in xs])
        rec["S4_engine_slope"] = linear_slope([x["step"] for x in xs], [x["seq_engine_mean"] for x in xs])
        for n, _ in STEP_FIELDS:
            rec[n] = _mean([x[n] for x in xs])
        out.append(rec)
    return out


def source_inputs(seeds: list) -> dict:
    """Per source: mean over seeds (and the per-seed values) of the main per-seed quantities."""
    by = defaultdict(list)
    for x in seeds:
        by[x["source"]].append(x)
    out = {}
    for src, xs in by.items():
        xs.sort(key=lambda x: x["seed"])
        out[src] = {"seeds": [x["seed"] for x in xs],
                    **{k: _mean([x[k] for x in xs]) for k in ("eps_pearson_lag20", "S3_slope", "tail_mass", "trunc_frac_p2")},
                    "per_seed": {k: [x[k] for x in xs] for k in ("eps_pearson_lag20", "S3_slope", "tail_mass", "trunc_frac_p2")}}
    return out


def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(root: Path, cfg2: dict, title: str = None, intro: str = None, tail=None) -> list:
    """`title`, `intro` and `tail` (a function (seeds) -> extra markdown lines) override the default header and append
    extra lines."""
    root = Path(root)
    rows, lag_rows = load(root)
    seeds = per_seed(rows, lag_rows)
    _write_csv(root / "steps.csv", rows)
    _write_csv(root / "lags.csv", lag_rows)
    _write_csv(root / "per_seed.csv", seeds)
    srcs = [s for s in SOURCES if any(x["source"] == s for x in seeds)] + \
        sorted({x["source"] for x in seeds} - set(SOURCES))
    col = {s: [x for x in sorted(seeds, key=lambda x: x["seed"]) if x["source"] == s] for s in srcs}
    roles = cfg2.get("replay", {}).get("sources", {})

    def table(title, items, f="{:.3g}"):
        md = [f"## {title}", "", "| quantity | " + " | ".join(srcs) + " |", "|---|" + "---|" * len(srcs)]
        for label, key in items:
            md.append(f"| {label} | " + " | ".join(_fmt([x[key] for x in col[s]], f) for s in srcs) + " |")
        return md + [""]

    md = [title or "# Replay tables, Qwen3-0.6B", "",
          (intro or "Generated by `scripts/measure.py summarize replay` from `results/replay_0.6b/<run>/<source>/step_<t>.json` "
           "(replay on the `s2ref_bf16_pg_seed*` checkpoints).") + " Per source: mean over seeds, per-seed "
          "values in brackets (seed order " + ", ".join(str(x["seed"]) for x in col[srcs[0]]) + " for the first "
          "column). Per seed, a step quantity is the mean over the replayed steps.", "",
          "| source | role | seeds | steps |", "|---|---|---|---|"]
    for s in srcs:
        md.append(f"| {s} | {roles.get(s, '')} | {', '.join(str(x['seed']) for x in col[s])} | "
                  + ", ".join(f"{x['step_min']}-{x['step_max']} ({x['n_steps']})" for x in col[s]) + " |")
    md.append("")
    md += table("S1 persistence", [
        ("repeat, same batch: mean / max abs diff (sampled token)", "rep_same_next_abs_mean"),
        ("", "rep_same_next_abs_max"),
        ("repeat, batch 1 vs 64: mean / max", "rep_b1_next_abs_mean"), ("", "rep_b1_next_abs_max"),
        ("trainer repeat: max", "trainer_rep_next_abs_max"),
        *[(f"eps Pearson, lag {l}", f"eps_pearson_lag{l}") for l in (1, 5, 10, 20)],
        *[(f"delta Pearson, lag {l}", f"delta_pearson_lag{l}") for l in (1, 5, 10, 20)],
        *[(f"anchor pairs, lag {l}", f"eps_pearson_lag{l}_n") for l in (1, 5, 10, 20)],
        ("abs eps mean (top-128)", "eps_abs_mean"), ("KL(q||p) with tail", "kl_q_p_tail_mean")])
    md += table("S2 token-level remainder", [
        ("TIS: frac p/q > 2, P2 batch", "trunc_frac_p2"),
        ("TIS: frac p/q > 2, probe set", "trunc_frac_probe"),
        ("TIS: mean max(p/q - 2, 0), probe", "excess_ratio_probe"),
        ("TIS: truncated delta per sequence, probe", "truncated_delta_seq_probe"),
        ("SC: sampler tail mass outside top-128", "tail_mass")])
    md += table("S3 sequence level (sum of delta per probe sequence, nats)", [
        ("mean", "seq_delta_mean"), ("sd", "seq_delta_sd"), ("q05", "seq_delta_q05"),
        ("median", "seq_delta_median"), ("q95", "seq_delta_q95"),
        ("slope of the mean vs step (nats/step)", "S3_slope")], "{:+.3g}")
    md += table("S4 source split of the sum (nats per sequence)", [
        ("lag part: sum(log p_theta - log p_theta_old)", "seq_lag_mean"),
        ("engine part: sum(log p_theta_old - log q)", "seq_engine_mean"),
        ("slope of the lag part", "S4_lag_slope"), ("slope of the engine part", "S4_engine_slope")], "{:+.3g}")
    if tail is not None:
        md += tail(seeds)
    (root / "TABLES.md").write_text("\n".join(md) + "\n")
    return [f"{len(rows)} step results, {len(lag_rows)} lag rows, {len(seeds)} (run, source) -> "
            f"{root}/steps.csv, lags.csv, per_seed.csv, TABLES.md"]
