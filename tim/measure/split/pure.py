"""Split and residual bias, the numpy part: which sampler a run used, the prompt sets, the scalars of 4a / 4b / 4c from
gradient inner products and per-token arrays, and the summary helpers (tests/test_split_pure.py); the GPU side is
tim.measure.split.job.

4a reports, per (run, t), from inner products only (no gradient is stored or differenced on disk):
  b  = g_corr - g_on                          the remaining gradient bias of the run's correction
  n  = g_on' - g_on                           the on-policy sampling noise floor (two independent seeds)
  |b|, cos(b, dtheta_recent), cos(b, dtheta_cum), and the same three for n,
with dtheta_recent = theta_t - theta_{t-5}, dtheta_cum = theta_t - theta_0. Every quantity is expanded from the Gram
matrix of {g_corr, g_on, g_on'} and their inner products with the two parameter deltas (`bias_scalars`).
"""
from __future__ import annotations

import math
import re

import numpy as np

# s2_*: the Qwen3-0.6B runs; s5_*: the Qwen3-1.7B runs; s6b_*: the 0.6B runs at learning rate 5e-3
RUN_RE = re.compile(r"^(?P<prefix>s2|s5|s6b)_(?P<source>.+?)_(?P<method>tis_sc|pg4|pg|tis|sc|split|ppo)_seed(?P<seed>\d+)$")

# the sampler each source trained with (scripts/train.sh, tim/dlc/resync.py): engine kind and
# whether it holds theta_t or theta_0 (the stale64 sources never re-sync within 40 steps: the sampler is theta_0)
SOURCE_SAMPLER = {
    "bf16": ("bf16", "t"),
    "fp8": ("fp8", "t"),
    "stale64": ("bf16", "0"),
    "fp8_stale64": ("fp8", "0"),
    "eager": ("bf16_eager", "t"),
}

# the 21 Qwen3-0.6B runs of the split measurement
STAGE4_RUNS = ([f"s2_{s}_{m}_seed{k}" for s in ("fp8", "stale64", "fp8_stale64") for m in ("tis", "sc") for k in (0, 1, 2)]
               + [f"s2_eager_pg_seed{k}" for k in (0, 1, 2)])

# objectives (ascent direction, token-mean over the batch's N response tokens, as the training loss):
#   pg       A * log p(y)
#   tis      A * min(p/q, cap) * log p(y)                     the TIS runs' loss (verl bypass, rollout_is=token, cap 2)
#   sc       A * (log p(y) - c),  c = sum_H sg[q - alpha p] log p   the SC runs' loss (verl score_centering, weight 1)
#   tis_low / sc_low   the same restricted to tokens with p/q < LOW (4b)
LOW = 0.5


def parse_run(run: str) -> dict:
    m = RUN_RE.match(run)
    if not m:
        raise ValueError(f"not a run name of the split measurement: {run!r}")
    d = m.groupdict()
    if d["source"] not in SOURCE_SAMPLER:
        raise ValueError(f"unknown source in {run!r}")
    d["seed"] = int(d["seed"])
    d["kind"], d["sampler_at"] = SOURCE_SAMPLER[d["source"]]
    d["lagged"] = d["sampler_at"] == "0"
    return d


def objectives_for(run: str, with_4b: bool = True) -> list:
    """The gradients a run needs on its own sampler's batch: its corrected gradient first (g_corr), then 4b's.
    with_4b=False (Qwen3-1.7B: 4a and 4c only): the corrected gradient alone, which keeps a 1.7B TIS run at three fp32
    gradients (g_corr, g_on, g_on') instead of six."""
    p = parse_run(run)
    if p["method"] == "tis":
        if not with_4b:
            return ["tis"]
        return ["tis", "sc", "tis_low", "sc_low"]   # 4b: the SC counterpart and the down-weighted tokens, same data
    if p["method"] in ("sc", "pg"):
        return [p["method"]]
    raise ValueError(f"the split measurement has no {p['method']} run: {run}")


def p2_rows(n_train: int, n: int = 64, seed: int = 0) -> list:
    """The P2 rows (configs/measure/qwen3_0.6b.json p2.rows_rule): default_rng(0).choice(n_train, 64, replace=False), sorted."""
    return sorted(int(x) for x in np.random.default_rng(seed).choice(n_train, n, replace=False))


def rows_4c(step: int, n_train: int, n: int, exclude) -> list:
    """4c's train prompts: a different set at each step, disjoint from the P2 rows,
    default_rng(10000 + step)."""
    ex = set(int(x) for x in exclude)
    pool = np.array([r for r in range(n_train) if r not in ex])
    return sorted(int(x) for x in np.random.default_rng(10000 + int(step)).choice(pool, n, replace=False))


def sampling_seeds(n_prompts: int, base: int) -> list:
    """Per-prompt vLLM seeds base + j (one request of n samples per prompt; vLLM derives the samples' seeds from it)."""
    return [base + j for j in range(n_prompts)]


SEED_RUN, SEED_ON, SEED_ON2, SEED_4C = 4000, 5000, 6000, 7000


def _cos(dot, na2, nb2) -> float:
    return dot / math.sqrt(na2 * nb2) if na2 > 0 and nb2 > 0 else float("nan")


def bias_scalars(gram: dict, with_delta: dict, delta_sq: dict) -> dict:
    """4a scalars from inner products.
    gram: {("c","c"), ("c","o"), ("c","o2"), ("o","o"), ("o","o2"), ("o2","o2")} for g_corr (c), g_on (o), g_on' (o2);
    with_delta: {(x, d): <g_x, delta_d>} for x in c, o, o2 and d in "recent", "cum"; delta_sq: {d: |delta_d|^2}."""
    G = lambda a, b: gram[(a, b)] if (a, b) in gram else gram[(b, a)]
    out = {}
    for name, (x, y) in (("b", ("c", "o")), ("noise", ("o2", "o"))):
        n2 = G(x, x) + G(y, y) - 2 * G(x, y)
        out[f"{name}_norm"] = math.sqrt(max(n2, 0.0))
        for d in ("recent", "cum"):
            out[f"{name}_cos_{d}"] = _cos(with_delta[(x, d)] - with_delta[(y, d)], n2, delta_sq[d])
    out["g_corr_norm"], out["g_on_norm"], out["g_on2_norm"] = (math.sqrt(G(k, k)) for k in ("c", "o", "o2"))
    out["cos_corr_on"] = _cos(G("c", "o"), G("c", "c"), G("o", "o"))
    out["cos_on_on2"] = _cos(G("o", "o2"), G("o", "o"), G("o2", "o2"))
    out["delta_recent_norm"], out["delta_cum_norm"] = math.sqrt(delta_sq["recent"]), math.sqrt(delta_sq["cum"])
    return out


def projection(dot_with_delta: float, delta_sq: float) -> float:
    """<g, delta> / |delta|: a gradient's component along an update direction."""
    return dot_with_delta / math.sqrt(delta_sq) if delta_sq > 0 else float("nan")


def tis_token_stats(log_ratio, cap: float = 2.0, low: float = LOW, edges=(-np.inf, math.log(0.25), math.log(0.5),
                                                                           math.log(0.8), math.log(1.25), math.log(2.0),
                                                                           np.inf)) -> dict:
    """4b: the sampled tokens' p/q before and after truncation against delta = log p - log q (= log of p/q): share of
    tokens and mean delta per p/q bin, the share truncated (p/q > cap) and down-weighted (p/q < low), and the delta mass
    each carries."""
    lr = np.asarray(log_ratio, dtype=np.float64)
    w_raw, w = np.exp(lr), np.minimum(np.exp(lr), cap)
    out = {"n_tokens": int(lr.size), "delta_mean": float(lr.mean()), "w_raw_mean": float(w_raw.mean()),
           "w_trunc_mean": float(w.mean()), "frac_truncated": float((lr > math.log(cap)).mean()),
           "frac_low": float((lr < math.log(low)).mean()),
           "delta_sum_truncated": float(lr[lr > math.log(cap)].sum()), "delta_sum_low": float(lr[lr < math.log(low)].sum()),
           "delta_sum_all": float(lr.sum()), "bins": []}
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lr >= a) & (lr < b)
        out["bins"].append({"p_over_q": [float(math.exp(a)) if np.isfinite(a) else 0.0,
                                         float(math.exp(b)) if np.isfinite(b) else None],
                            "frac": float(m.mean()), "delta_mean": float(lr[m].mean()) if m.any() else None})
    return out


def split_means(lp_t, lp_old, q_next, offsets=None) -> dict:
    """4c: per-token means of delta = log p_theta - log q and its split into lag = log p_theta - log p_theta_old and
    engine = log p_theta_old - log q (lag + engine = delta exactly), plus the per-sequence-mean variant verl's
    log_ppl_diff uses (offsets: row offsets per sequence)."""
    lp_t, lp_old, q = (np.asarray(x, dtype=np.float64) for x in (lp_t, lp_old, q_next))
    d, lag, eng = lp_t - q, lp_t - lp_old, lp_old - q
    out = {"delta_mean": float(d.mean()), "lag_mean": float(lag.mean()), "engine_mean": float(eng.mean()),
           "delta_abs_mean": float(np.abs(d).mean()), "lag_abs_mean": float(np.abs(lag).mean()),
           "engine_abs_mean": float(np.abs(eng).mean()), "n_tokens": int(d.size)}
    if offsets is not None:
        off = np.asarray(offsets)
        seq = lambda x: np.array([x[off[i]:off[i + 1]].mean() for i in range(len(off) - 1) if off[i + 1] > off[i]])
        out.update(delta_seq_mean=float(seq(d).mean()), lag_seq_mean=float(seq(lag).mean()),
                   engine_seq_mean=float(seq(eng).mean()))
    return out


def first_separated(steps, values: dict, floor: dict, factor: float = 2.0) -> dict:
    """4c: per class, the first step from which |value - floor| > factor x |floor| holds at that step and every later one
    (None if never). values / floor: {class: [value per step]} / [floor value per step]."""
    steps = list(steps)
    out = {}
    for cls, v in values.items():
        ok = [abs(a - f) > factor * abs(f) if (a is not None and f is not None and np.isfinite(a) and np.isfinite(f))
              else False for a, f in zip(v, floor)]
        first = None
        for i in range(len(steps)):
            if all(ok[i:]):
                first = steps[i]
                break
        out[cls] = first
    return out


def linear_slope(xs, ys) -> float:
    x, y = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    x, y = x[m], y[m]
    return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())


def log_split_from_metrics(rows: list) -> list:
    """4c from the training logs (metrics.*.jsonl rows, one per step): per step the batch's mean per-sequence delta
    = -log_ppl_diff (verl rollout_corr_helper: log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training, sequence
    means), under either prefix (bypass runs: actor/rollout_corr/, decoupled: rollout_corr/). In the bypass runs (all runs
    of this measurement) "training" is theta_t, so this is the total delta; the logs hold no theta_old term for them, so the lag /
    engine split comes from the GPU job. Returns [{"step", "delta_seq_mean", "delta_abs_seq_mean"}]."""
    out = []
    for r in rows:
        d = r.get("data", r)
        step = d.get("training/global_step", d.get("step"))
        for p in ("actor/rollout_corr/", "rollout_corr/"):
            if p + "log_ppl_diff" in d:
                out.append({"step": int(step), "delta_seq_mean": -float(d[p + "log_ppl_diff"]),
                            "delta_abs_seq_mean": float(d.get(p + "log_ppl_abs_diff", float("nan")))})
                break
    return sorted(out, key=lambda x: x["step"])
