"""Off-support measurement, the numpy part (tests/test_off_support_pure.py); the GPU side is tim.measure.off_support.job.

Per checkpoint theta (and theta_0), trainer numerics for every log-prob:
  1. held-out accuracy of theta and theta_0; run - ref split into "not learned" and "worse" (`heldout_split`);
  2. D_own = mean over theta's own samples of sum_t [log p_theta(y_t) - log p_theta0(y_t)] (`own_stats`), with the response
     length distribution, the truncated share and the share shorter than 64 tokens;
  3. D_q = mean over theta_0's samples (the probe prefixes) of sum_t KL(p_theta || p_theta0)(h_t), the per-response max_t KL_t,
     the mean p_theta(EOS) - p_theta0(EOS) over the first 64 positions (`probe_stats`);
  4. on theta's own samples, the share of D_own from tokens with p_theta0(y) < 1e-3 and the position of the first token
     outside theta_0's top 20 (`own_stats`).
The summary correlates the held-out loss with D_own, D_q, the existing deviation and |mu| (`spearman`).
"""
from __future__ import annotations

import math
import re

import numpy as np

# Run prefixes (README.md, "Run names"): 0.6B s2_* with reference s2ref; 1.7B s5_*, s6c_* (resynchronized importance sampling,
# DLC without resynchronization), s7d_* (DLC) and s5ref; s8a_* / s8a6_* (M2PO, VCPO and further baselines on 1.7B / 0.6B);
# 8B s8b_* with reference s8ref; s9b_* (continuous asynchrony, 1.7B). theta0_<model>: the base model.
RUN_RE = re.compile(r"^(?P<prefix>s2ref|s5ref|s8ref|s9bref|s2|s5|s6c|s7d|s8a6|s8a|s8b|s9b)_(?P<source>.+?)_(?P<method>tis_sc|tis_raw|split_async_ess|split_async|split_supp|pg|tis|sc|split|ppo|m2po|vcpo)_seed(?P<seed>\d+)$")
# s8a = Qwen3-1.7B, s8a6 = Qwen3-0.6B; two 0.6B runs keep the s8a prefix (RUN_MODEL).
# s8b = Qwen3-8B (method runs and the noise-floor rerun s8b_bf16_pg_seed0), reference s8ref.
MODEL_OF = {"s2": "06b", "s2ref": "06b", "s5": "17b", "s5ref": "17b", "s6c": "17b", "s7d": "17b", "s8a": "17b", "s8a6": "06b",
            "s8b": "8b", "s8ref": "8b", "s9b": "17b", "s9bref": "17b"}
RUN_MODEL = {"s8a_budget_tis_seed0": "06b", "s8a_budget_split_seed0": "06b"}
REFERENCE = {"06b": "s2ref_bf16_pg_seed{seed}", "17b": "s5ref_bf16_pg_seed{seed}", "8b": "s8ref_bf16_pg_seed{seed}"}
THETA0 = {"06b": "theta0_06b", "17b": "theta0_17b", "8b": "theta0_8b"}

SEED_OWN = 9000                  # vLLM seed of the own-sample request of held-out prompt j = SEED_OWN + j
P0_FLOOR, TOP_K0, FIRST_N = 1e-3, 20, 64
SHORT = 64


def parse_run(run: str) -> dict:
    """{prefix, source, method, seed, model, ref, theta0}; theta0_06b / theta0_17b / theta0_8b are the base models (step 0)."""
    for m, name in THETA0.items():
        if run == name:
            return {"prefix": "theta0", "source": "base", "method": "none", "seed": -1, "model": m, "ref": False, "theta0": True}
    g = RUN_RE.match(run)
    if not g:
        raise ValueError(f"not a run name of the off-support measurement: {run!r}")
    d = g.groupdict()
    d["seed"] = int(d["seed"])
    d["model"] = RUN_MODEL.get(run, MODEL_OF[d["prefix"]])
    d["ref"] = d["prefix"].endswith("ref")
    d["theta0"] = False
    return d


def model_of_base(base_model: str) -> str:
    """The model key of a measurement process's base model (an HF id or a local directory of Qwen3-0.6B / -1.7B / -8B)."""
    if "1.7" in base_model:
        return "17b"
    if "Qwen3-8B" in base_model:
        return "8b"
    return "06b"


# The continuous-asynchrony runs (s9b_async4_<method>_seed<s>) use the 1.7B reference s5ref and its noise floor: sources
# in MODEL_REFERENCE_SOURCES skip PREFIX_REFERENCE and take the model's reference.
PREFIX_REFERENCE = {"s9b": "s9bref_bf16_pg_seed{seed}"}
MODEL_REFERENCE_SOURCES = ("async4",)


def reference_of(run: str) -> str:
    p = parse_run(run)
    if p["source"] in MODEL_REFERENCE_SOURCES:
        return REFERENCE[p["model"]].format(seed=p["seed"])
    return PREFIX_REFERENCE.get(p["prefix"], REFERENCE[p["model"]]).format(seed=p["seed"])


# ------------------------------------------------------------------------------------------------ 1. held-out split
def heldout_split(acc_run: float, acc_ref: float, acc_0: float) -> dict:
    """loss = ref - run (> 0: the run is worse than the reference), split relative to theta_0:
         not_learned = the part of the reference's gain over theta_0 the run did not reach = max(0, ref - max(run, theta_0))
         worse       = the part of the run below theta_0 = max(0, theta_0 - run)
         rest        = loss - not_learned - worse (0 when ref >= theta_0 and run <= ref; < 0 when the run beats the
                       reference; > 0 only when the reference itself is below theta_0).
    Also run - theta_0."""
    loss = acc_ref - acc_run
    nl = max(0.0, acc_ref - max(acc_run, acc_0))
    w = max(0.0, acc_0 - acc_run)
    return {"loss": loss, "not_learned": nl, "worse": w, "rest": loss - nl - w, "run_minus_theta0": acc_run - acc_0,
            "ref_minus_theta0": acc_ref - acc_0}


# ------------------------------------------------------------------------------------------------ per-sequence reductions
def seq_sums(x, off) -> np.ndarray:
    """sum of x over each sequence's response rows (off: row offsets, n+1)."""
    x, off = np.asarray(x, dtype=np.float64), np.asarray(off)
    c = np.concatenate([[0.0], np.cumsum(x)])
    return c[off[1:]] - c[off[:-1]]


def seq_max(x, off) -> np.ndarray:
    x, off = np.asarray(x, dtype=np.float64), np.asarray(off)
    return np.array([x[a:b].max() if b > a else float("nan") for a, b in zip(off[:-1], off[1:])])


def positions(off) -> np.ndarray:
    off = np.asarray(off)
    lens = np.diff(off)
    return np.arange(int(off[-1])) - np.repeat(off[:-1], lens)


def _q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> list:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.quantile(x, qs)] if x.size else [float("nan")] * len(qs)


def _mean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean()) if x.size else float("nan")


def first_index(mask, off) -> np.ndarray:
    """Per sequence: the response position (0-indexed) of its first row with mask True, -1 if none."""
    mask, off = np.asarray(mask, dtype=bool), np.asarray(off)
    out = np.full(len(off) - 1, -1, dtype=np.int64)
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        k = np.flatnonzero(mask[a:b])
        if k.size:
            out[i] = int(k[0])
    return out


# ------------------------------------------------------------------------------------------------ 2 + 4. own samples
def own_stats(lp_t, lp_0, rank_0, off, truncated, reward=None, p0_floor: float = P0_FLOOR, top_k0: int = TOP_K0,
              short: int = SHORT) -> dict:
    """theta's own samples: lp_t, lp_0 = log p_theta(y), log p_theta0(y) per response row (trainer numerics), rank_0 = rank of
    y under theta_0 (0 = argmax), truncated = per sequence (finish_reason "length").
    D_own = mean_seq sum_t (lp_t - lp_0); share_offsupport = the part of sum (lp_t - lp_0) from rows with p_theta0(y) < p0_floor
    (pooled over all samples, so it is the share of D_own); first_out = the first position whose token is outside theta_0's
    top `top_k0` (rank >= top_k0), with the share of samples that never leave it."""
    lp_t, lp_0, rank_0 = (np.asarray(x, dtype=np.float64) for x in (lp_t, lp_0, rank_0))
    off = np.asarray(off)
    d = lp_t - lp_0
    per = seq_sums(d, off)
    lens = np.diff(off)
    low = lp_0 < math.log(p0_floor)
    tot = float(d.sum())
    fo = first_index(rank_0 >= top_k0, off)
    hit = fo >= 0
    out = {"n": int(len(lens)), "n_tokens": int(off[-1]),
           "d_own": _mean(per), "d_own_quantiles": _q(per), "d_own_max": float(per.max()) if per.size else float("nan"),
           "d_own_per_token": float(d.mean()) if d.size else float("nan"),
           "len_mean": _mean(lens), "len_quantiles": _q(lens), "frac_truncated": _mean(np.asarray(truncated, dtype=bool)),
           "frac_short": _mean(lens < short),
           "frac_tokens_p0_low": _mean(low), "share_offsupport": float(d[low].sum()) / tot if tot != 0 else float("nan"),
           "d_own_offsupport": float(d[low].sum()) / max(len(lens), 1),
           "frac_seq_with_p0_low": _mean(seq_sums(low, off) > 0),
           "frac_tokens_out_top": _mean(rank_0 >= top_k0),
           "first_out_frac_never": _mean(~hit), "first_out_mean": _mean(fo[hit]), "first_out_quantiles": _q(fo[hit]),
           "first_out_rel_mean": _mean(fo[hit] / np.maximum(lens[hit], 1))}
    if reward is not None:
        out["reward_mean"] = _mean(reward)
    return out


# ------------------------------------------------------------------------------------------------ 3. probe prefixes
def probe_stats(kl, eos_t, eos_0, lp_t, lp_0, off, first_n: int = FIRST_N) -> dict:
    """theta_0's samples (probe prefixes): kl = KL(p_theta || p_theta0) per row (full vocabulary), eos_t / eos_0 = the
    probability of the stop tokens at that row, lp_t / lp_0 = log p(y) of the sampled token.
    D_q = mean_seq sum_t KL_t; max_t KL_t per sequence; dEOS = mean over rows at positions < first_n of eos_t - eos_0;
    kl_mean = mean over all rows (the existing deviation's averaging, against theta_0); mu = mean_t (lp_t - lp_0), the
    stale runs' mean lag on their sampler's samples (-KL(p_theta0 || p_theta) per token)."""
    kl, eos_t, eos_0, lp_t, lp_0 = (np.asarray(x, dtype=np.float64) for x in (kl, eos_t, eos_0, lp_t, lp_0))
    off = np.asarray(off)
    per, mx = seq_sums(kl, off), seq_max(kl, off)
    pos = positions(off)
    first = pos < first_n
    de = eos_t - eos_0
    return {"n": int(len(off) - 1), "n_tokens": int(off[-1]),
            "d_q": _mean(per), "d_q_quantiles": _q(per), "kl_mean": _mean(kl),
            "max_kl_mean": _mean(mx), "max_kl_quantiles": _q(mx), "max_kl_max": float(np.nanmax(mx)) if mx.size else float("nan"),
            "max_kl_pos_mean": _mean([int(np.argmax(kl[a:b])) for a, b in zip(off[:-1], off[1:]) if b > a]),
            "d_eos_first": _mean(de[first]), "d_eos_first_abs": _mean(np.abs(de[first])), "eos_0_first": _mean(eos_0[first]),
            "d_eos_all": _mean(de), "mu": _mean(lp_t - lp_0), "mu_abs": abs(_mean(lp_t - lp_0))}


# ------------------------------------------------------------------------------------------------ summary statistics
def ranks(x) -> np.ndarray:
    """Average ranks (1-based, ties share their mean rank)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y) -> tuple:
    """(rho, n) over the pairs where both are finite; rho NaN with fewer than 3 pairs or a constant side."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return float("nan"), n
    a, b = ranks(x[m]), ranks(y[m])
    a, b = a - a.mean(), b - b.mean()
    d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return (float((a * b).sum()) / d if d > 0 else float("nan")), n


def percentile_of(value: float, population) -> float:
    """Share of the finite population strictly below value, plus half the ties (mid-rank percentile, 0..1)."""
    p = np.asarray(population, dtype=np.float64)
    p = p[np.isfinite(p)]
    if not p.size or value is None or not np.isfinite(value):
        return float("nan")
    return float(((p < value).sum() + 0.5 * (p == value).sum()) / p.size)
