"""Loop measurements, the numpy part (tests/test_loop_pure.py); the GPU side is tim.measure.loop.job.

Per sampled token: l_t = log p_theta_t(y_t) - log p_theta_0(y_t), A_t the sequence's group-centred advantage.
6a-1 predicts the remaining bias b = g_corr - g_on from the weights (stop-grad constants)
    SC  : w_t = A_t l_t + A_t sum_{t'<t} l_t'        TIS : w_t = A_t sum_{t'<t} l_t'
as b_pred = -grad of (sum_t w_t log p(y_t) / N); the token and the prefix term are separate vectors. The exact version replaces
the linearised weights by the exact ones (`bias_weights(..., exact=True)`): prefix A (1 - e^{S_t}), SC token A (1 - e^{delta_t}),
TIS token -A max(e^{delta_t} - 2, 0). Their cosines with b,
the norm ratios and the cosines with the noise n = g_on' - g_on and with Delta_cum come from inner products only
(`predicted_scalars`). 6a-2 splits the lag over tokens by the sign of A (or A - Abar) and looks at the largest |l| tokens.
"""
from __future__ import annotations

import math

import numpy as np

SEED_ABAR = 8000                 # vLLM seed per continuation request = SEED_ABAR + request index
N_ABAR, N_CONT = 512, 8          # sampled positions per batch, continuations per position
SUBSET_SEED_4A, SUBSET_SEED_4C = 60000, 61000      # default_rng(base + t) picks the positions


# ------------------------------------------------------------------------------------------------ per-token arrays
def per_row(values, off) -> np.ndarray:
    """One value per sequence -> one per response row (off: row offsets per sequence, n+1)."""
    return np.repeat(np.asarray(values, dtype=np.float64), np.diff(np.asarray(off)))


def exclusive_cumsum(x, off) -> np.ndarray:
    """sum_{t' < t} x_t' within each sequence's response rows (0 at the first token)."""
    x, off = np.asarray(x, dtype=np.float64), np.asarray(off)
    out = np.zeros_like(x)
    for i in range(len(off) - 1):
        a, b = off[i], off[i + 1]
        if b > a:
            out[a + 1:b] = np.cumsum(x[a:b - 1])
    return out


EXP_CLIP = 30.0                  # exact weights: exponents (S_t, delta_t) clipped here; exact_diagnostics counts the clipped rows


def _one_minus_exp(x) -> np.ndarray:
    """1 - e^x without cancellation at small x, the exponent clipped at EXP_CLIP."""
    return -np.expm1(np.minimum(x, EXP_CLIP))


def bias_weights(lag, adv_rows, off, method: str, exact: bool = False, cap: float = 2.0) -> dict:
    """The weights of b_pred = -grad, i.e. already negated (so the ascent gradient of sum w log p / N is b_pred itself).
    exact=False (6a-1): {"token": -A l, "prefix": -A sum_{t'<t} l}; TIS has the prefix term only.
    exact=True, `lag` = delta_t (= l_t on stale64, log p_theta_t - log q on fp8_stale64), S_t = sum_{t'<t} delta:
        prefix  A (1 - e^{S_t})                          (SC and TIS; -A S_t to first order)
        token   SC: A (1 - e^{delta_t})                  (-A delta_t to first order)
                TIS: -A max(e^{delta_t} - cap, 0)        (the remainder of the ratio truncated at cap)."""
    lag, A = np.asarray(lag, dtype=np.float64), np.asarray(adv_rows, dtype=np.float64)
    if method not in ("sc", "tis"):
        raise ValueError(method)
    S = exclusive_cumsum(lag, off)
    if not exact:
        out = {"prefix": -A * S}
        if method == "sc":
            out["token"] = -A * lag
        return out
    out = {"prefix": A * _one_minus_exp(S)}
    if method == "sc":
        out["token"] = A * _one_minus_exp(lag)
    else:
        out["token"] = -A * np.maximum(np.exp(np.minimum(lag, EXP_CLIP)) - cap, 0.0)
    return out


def exact_diagnostics(lag, off, cap: float = 2.0) -> dict:
    """How far the batch is from the linearised regime: |S_t| and |delta_t| quantiles, the share of rows with |S_t| > 1,
    with e^{delta} > cap (TIS truncation), and the rows whose exponent was clipped at EXP_CLIP."""
    lag = np.asarray(lag, dtype=np.float64)
    S = exclusive_cumsum(lag, off)
    q = lambda x: [float(v) for v in np.quantile(np.abs(x), [0.5, 0.9, 0.99])] if x.size else [float("nan")] * 3
    return {"abs_S_q50_q90_q99": q(S), "abs_S_max": float(np.abs(S).max()) if S.size else float("nan"),
            "S_max": float(S.max()) if S.size else float("nan"), "frac_abs_S_gt_1": float((np.abs(S) > 1).mean()) if S.size else float("nan"),
            "abs_delta_q50_q90_q99": q(lag), "frac_ratio_gt_cap": float((lag > np.log(cap)).mean()) if lag.size else float("nan"),
            "mean_one_minus_exp_S": float(_one_minus_exp(S).mean()) if S.size else float("nan"),
            "n_clipped": int((S > EXP_CLIP).sum() + (lag > EXP_CLIP).sum())}


def positions(off) -> tuple:
    """(seq (n,), pos (n,)) of every response row; pos is 0-indexed within the response."""
    off = np.asarray(off)
    lens = np.diff(off)
    return np.repeat(np.arange(len(lens)), lens), np.arange(int(off[-1])) - np.repeat(off[:-1], lens)


def choose_rows(n_rows: int, k: int, seed: int) -> np.ndarray:
    """k distinct sampled-token rows, uniform, sorted (fixed numpy seed; all rows when fewer than k)."""
    if n_rows <= k:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(int(seed)).choice(n_rows, size=k, replace=False))


def abar_from_rewards(cont_rewards, group_baseline) -> np.ndarray:
    """Abar = mean reward of the continuations of each position - that prompt group's baseline (the batch's group mean)."""
    return np.asarray([float(np.mean(r)) for r in cont_rewards]) - np.asarray(group_baseline, dtype=np.float64)


def abar_ratio(abar, adv) -> float:
    """mean|Abar| / mean|A| on the subset."""
    d = float(np.mean(np.abs(adv)))
    return float(np.mean(np.abs(abar))) / d if d > 0 else float("nan")


def subset_weights(lag, adv_rows, abar_by_row: dict, n_rows: int, exact: bool = False) -> tuple:
    """SC token-term weights restricted to the subset, with A and with A - Abar (negated as bias_weights); abar_by_row
    {row: Abar}. Zero outside the subset. exact=True: (1 - e^{delta}) in place of -l (`lag` = delta, as bias_weights)."""
    rows = np.fromiter(abar_by_row.keys(), dtype=np.int64)
    ab = np.fromiter(abar_by_row.values(), dtype=np.float64)
    lag, A = np.asarray(lag, dtype=np.float64), np.asarray(adv_rows, dtype=np.float64)
    f = _one_minus_exp(lag[rows]) if exact else -lag[rows]
    wa, wb = np.zeros(n_rows), np.zeros(n_rows)
    wa[rows] = A[rows] * f
    wb[rows] = (A[rows] - ab) * f
    return wa, wb


# ------------------------------------------------------------------------------------- 6a-1 scalars from inner products
def _cos(dot, na2, nb2) -> float:
    return dot / math.sqrt(na2 * nb2) if na2 > 0 and nb2 > 0 else float("nan")


def _G(gram, a, b):
    return gram[(a, b)] if (a, b) in gram else gram[(b, a)]


def combo_dot(gram: dict, u: dict, v: dict) -> float:
    """<sum_a u_a a, sum_b v_b b> from the Gram entries {(a, b): <a, b>} (either order)."""
    return sum(ca * cb * _G(gram, a, b) for a, ca in u.items() for b, cb in v.items())


B_VEC, N_VEC = {"c": 1.0, "o": -1.0}, {"o2": 1.0, "o": -1.0}      # b = g_corr - g_on, n = g_on' - g_on


def predicted_scalars(gram: dict, delta_dot: dict, delta_sq: float, preds: dict) -> dict:
    """preds {name: {base vector: coefficient}} (e.g. total = {"tok": 1, "pre": 1}); gram over the base vectors
    c (g_corr), o (g_on), o2 (g_on'), and the predicted ones; delta_dot {vector: <vector, Delta_cum>}.
    Per prediction: norm, ratio = |x| / |b|, cos with b, cos with the noise n, cos with Delta_cum."""
    b2, n2 = combo_dot(gram, B_VEC, B_VEC), combo_dot(gram, N_VEC, N_VEC)
    out = {"b_norm": math.sqrt(max(b2, 0.0)), "noise_norm": math.sqrt(max(n2, 0.0)), "delta_cum_norm": math.sqrt(delta_sq)}
    for name, u in preds.items():
        x2 = combo_dot(gram, u, u)
        dd = sum(c * delta_dot[a] for a, c in u.items())
        out[name] = {"norm": math.sqrt(max(x2, 0.0)),
                     "ratio": math.sqrt(x2 / b2) if b2 > 0 and x2 >= 0 else float("nan"),
                     "cos_b": _cos(combo_dot(gram, u, B_VEC), x2, b2),
                     "cos_noise": _cos(combo_dot(gram, u, N_VEC), x2, n2),
                     "cos_delta": _cos(dd, x2, delta_sq)}
    return out


def cos_vectors(gram: dict, u: dict, v: dict) -> float:
    return _cos(combo_dot(gram, u, v), combo_dot(gram, u, u), combo_dot(gram, v, v))


# ------------------------------------------------------------------------------------------------ 6a-2 lag by token group
def sign_groups(score) -> dict:
    s = np.asarray(score, dtype=np.float64)
    return {"neg": s < 0, "pos": s > 0, "zero": s == 0}


def lag_split(lag, score, delta_sq: float) -> dict:
    """Per group of sign(score) (A, or A - Abar on the subset): token count, mean l, share of sum(l), and |mean l| / |Delta|^2."""
    lag = np.asarray(lag, dtype=np.float64)
    tot = float(lag.sum())
    out = {"n": int(lag.size), "mean": float(lag.mean()) if lag.size else float("nan"), "sum": tot}
    for g, m in sign_groups(score).items():
        k = int(m.sum())
        mean = float(lag[m].mean()) if k else float("nan")
        out[g] = {"n": k, "mean": mean, "share_sum": float(lag[m].sum()) / tot if tot != 0 else float("nan"),
                  "abs_mean_over_delta_sq": abs(mean) / delta_sq if k and delta_sq > 0 else float("nan")}
    return out


def _quart(x) -> list:
    return [float(v) for v in np.quantile(np.asarray(x, dtype=np.float64), [0.25, 0.5, 0.75])]


def top_tokens(lag, adv, p_t, pos, resp_len, fracs=(0.01, 0.001), p_floor: float = 1e-3) -> dict:
    """The top fraction of tokens by |l|: share of sum(l), fraction with p_theta_t(y) < p_floor, fraction with A < 0, A > 0,
    A = 0, and the position distribution (mean, quartiles of pos and of pos / response length, pos 0-indexed)."""
    lag, A, p_t = (np.asarray(x, dtype=np.float64) for x in (lag, adv, p_t))
    pos, ln = np.asarray(pos, dtype=np.float64), np.asarray(resp_len, dtype=np.float64)
    tot = float(lag.sum())
    order = np.argsort(-np.abs(lag), kind="stable")
    out = {}
    for f in fracs:
        k = max(1, int(math.ceil(f * lag.size)))
        i = order[:k]
        rel = pos[i] / ln[i]
        out[str(f)] = {"k": k, "share_sum": float(lag[i].sum()) / tot if tot != 0 else float("nan"),
                       "abs_l_min": float(np.abs(lag[i]).min()), "frac_p_below": float((p_t[i] < p_floor).mean()),
                       "frac_adv_neg": float((A[i] < 0).mean()), "frac_adv_pos": float((A[i] > 0).mean()),
                       "frac_adv_zero": float((A[i] == 0).mean()), "pos_mean": float(pos[i].mean()), "pos_quartiles": _quart(pos[i]),
                       "rel_pos_mean": float(rel.mean()), "rel_pos_quartiles": _quart(rel)}
    return out
