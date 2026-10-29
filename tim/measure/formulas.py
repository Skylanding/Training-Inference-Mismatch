"""Pure-numpy formulas of the sampler replay. No torch; unit tests in tests/test_measure_formulas.py.

Array conventions for one set of full sequences, flattened over all response positions (row j = one prefix h):
  q_ids (n,K) int32   sampler top-K ids at h, by sampler rank (column 0 = sampler argmax)
  q_lp  (n,K)         sampler log-probs at q_ids;          p_lp (n,K) trainer log-probs at q_ids (full-vocab normalised)
  q_next, p_next (n,) log-probs of the token that follows h in the fixed sequence (the "sampled" token y_t)
eps = p_lp - q_lp (log p_train - log p_sampler on the head), delta = p_next - q_next (the sampled-token error).

Keys of the replay records: S1 = persistence of eps across checkpoints, S2 = ratio truncation, S3 = per-sequence sum of
delta (the sequence offset), S4 = the exact split of delta into policy lag and numerical discrepancy.
"""
from __future__ import annotations

import numpy as np

from tim.measure.tail_model import head_mass, kl_with_tail


def centred_advantages(reward, group) -> np.ndarray:
    """A_i = R_i - mean of R over i's group (no std), as training (algorithm.adv_estimator=grpo, norm_adv_by_std=False)."""
    reward = np.asarray(reward, dtype=np.float64)
    group = np.asarray(group)
    out = np.empty_like(reward)
    for g in np.unique(group):
        m = group == g
        out[m] = reward[m] - reward[m].mean()
    return out


def tis_weight(log_ratio, cap: float = 2.0) -> np.ndarray:
    """Token TIS weight min(p/q, cap) from log(p/q), as verl rollout_is=token with threshold `cap` (upper clamp only)."""
    return np.minimum(np.exp(np.asarray(log_ratio, dtype=np.float64)), cap)


def ratio_stats(log_ratio, cap: float = 2.0) -> dict:
    """What TIS changes on the sampled tokens: mean and mean |.| of (min(p/q, cap) - 1), share of tokens with p/q > cap."""
    lr = np.asarray(log_ratio, dtype=np.float64)
    w = tis_weight(lr, cap) - 1.0
    return {"ratio_w_mean": float(w.mean()), "ratio_w_abs_mean": float(np.abs(w).mean()),
            "frac_ratio_gt_cap": float((lr > np.log(cap)).mean()), "log_ratio_abs_max": float(np.abs(lr).max()),
            "n_tokens": int(lr.size)}


def eps_stats(q_lp, p_lp, q_next, p_next) -> dict:
    """Scalar summary of eps on the sampler head, the tail, and delta on the sampled token, over all prefixes."""
    q_lp = np.asarray(q_lp, dtype=np.float64)
    p_lp = np.asarray(p_lp, dtype=np.float64)
    e = p_lp - q_lp
    q = np.exp(q_lp)
    d = np.asarray(p_next, dtype=np.float64) - np.asarray(q_next, dtype=np.float64)
    qm, pm = head_mass(q_lp), head_mass(p_lp)
    return {
        "n_prefixes": int(e.shape[0]),
        "eps_abs_mean": float(np.abs(e).mean()),
        "eps_abs_qweighted": float(((q * np.abs(e)).sum(-1) / np.maximum(q.sum(-1), 1e-12)).mean()),
        "eps_rms": float(np.sqrt((e ** 2).mean())),
        "delta_mean": float(d.mean()), "delta_abs_mean": float(np.abs(d).mean()), "delta_rms": float(np.sqrt((d ** 2).mean())),
        "kl_q_p_tail_mean": float(kl_with_tail(q_lp, p_lp).mean()),
        "q_tail_mass_mean": float((1.0 - qm).mean()), "p_tail_mass_mean": float((1.0 - pm).mean()),
        "flip_rate": float((p_lp.argmax(-1) != 0).mean()),
    }


def align_common(ids_a, ids_b, vals_b, chunk: int = 2048) -> np.ndarray:
    """vals_b (n,Kb) given at ids_b, re-expressed at ids_a (n,Ka): NaN where an id of a is not in b's row.

    Vectorised, chunked over rows so the (rows, Ka, Kb)
    equality tensor stays small. Ids within a row are distinct.
    """
    ids_a, ids_b = np.asarray(ids_a), np.asarray(ids_b)
    vals_b = np.asarray(vals_b, dtype=np.float64)
    out = np.full(ids_a.shape, np.nan)
    for s in range(0, ids_a.shape[0], chunk):
        eq = ids_a[s:s + chunk, :, None] == ids_b[s:s + chunk, None, :]          # (r, Ka, Kb)
        hit = eq.any(-1)
        j = eq.argmax(-1)
        v = np.take_along_axis(vals_b[s:s + chunk], j, axis=1)
        out[s:s + chunk] = np.where(hit, v, np.nan)
    return out


def repeat_diff(ids_a, lp_a, next_a, ids_b, lp_b, next_b) -> dict:
    """Two measurements of the same prefixes: share of rows whose top-K id lists differ, |difference| on common ids, and on the
    sampled token. Zero everywhere means the engine returns the same numbers each time."""
    b_at_a = align_common(ids_a, ids_b, lp_b)
    diff = np.abs(np.asarray(lp_a, dtype=np.float64) - b_at_a)
    nd = np.abs(np.asarray(next_a, dtype=np.float64) - np.asarray(next_b, dtype=np.float64))
    return {"rows_ids_differ": float((np.asarray(ids_a) != np.asarray(ids_b)).any(-1).mean()),
            "head_coverage": float(np.isfinite(diff).mean()),
            "head_abs_mean": float(np.nanmean(diff)), "head_abs_max": float(np.nanmax(diff)),
            "next_abs_mean": float(nd.mean()), "next_abs_max": float(nd.max()),
            "rows_equal_exactly": float((np.nan_to_num(diff, nan=np.inf).max(-1) == 0).mean())}


def pearson(x, y) -> float:
    """Pearson correlation over entries finite in both; NaN if fewer than 3 or a constant input."""
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x, y = x[m] - x[m].mean(), y[m] - y[m].mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def cosine(x, y) -> float:
    """Uncentred cosine over entries finite in both."""
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def lag_pairs(steps, lags) -> list:
    """(t, t + l, l) for every lag and every t with both steps present."""
    have = set(int(s) for s in steps)
    return [(t, t + l, l) for l in lags for t in sorted(have) if t + l in have]


def eps_lag(ids_t, eps_t, delta_t, ids_u, eps_u, delta_u) -> dict:
    """Is eps kept across training steps ? Same prefixes at two checkpoints t < u.

    delta (sampled token) is aligned exactly (fixed sequences); eps on the head is compared on t's head ids found in u's head.
    Pearson (centred) and cosine (uncentred: a constant shared offset counts as kept) are both reported.
    """
    eu = align_common(ids_t, ids_u, eps_u)
    return {"delta_pearson": pearson(delta_t, delta_u), "delta_cosine": cosine(delta_t, delta_u),
            "eps_pearson": pearson(eps_t, eu), "eps_cosine": cosine(eps_t, eu),
            "eps_coverage": float(np.isfinite(eu).mean())}


def stale_split(p_t, p_0, q) -> dict:
    """Stale eps = p_t - q split into the weight difference (p_t - p_0, trainer at theta_t vs theta_0) and the engine
    difference (p_0 - q, bf16 trainer vs vLLM, both at theta_0); all three on the same ids. Mean squares add up to the
    total's mean square plus twice the cross term, which is reported too."""
    p_t, p_0, q = (np.asarray(a, dtype=np.float64) for a in (p_t, p_0, q))
    w, e = p_t - p_0, p_0 - q
    return {"weight_abs_mean": float(np.abs(w).mean()), "engine_abs_mean": float(np.abs(e).mean()),
            "weight_ms": float((w * w).mean()), "engine_ms": float((e * e).mean()), "cross_ms": float((w * e).mean()),
            "total_ms": float(((w + e) ** 2).mean()), "weight_engine_pearson": pearson(w, e)}


def dict_dot(a: dict, b: dict) -> float:
    """sum over shared names of <a[n], b[n]> (float64); numpy reference of the torch version in heads.grad_dot."""
    return float(sum(np.dot(np.asarray(a[n], dtype=np.float64).ravel(), np.asarray(b[n], dtype=np.float64).ravel())
                     for n in a if n in b))


def dict_cosine(a: dict, b: dict) -> float:
    na, nb = np.sqrt(dict_dot(a, a)), np.sqrt(dict_dot(b, b))
    return dict_dot(a, b) / (na * nb) if na > 0 and nb > 0 else float("nan")


# ---------------------------------------------------------------------------------------------------- replay records S2-S4
def seq_sums(values, offsets) -> np.ndarray:
    """Per-sequence sums of per-row values (rows flattened in sequence order, offsets (n+1,))."""
    v = np.asarray(values, dtype=np.float64)
    off = np.asarray(offsets)
    return np.array([v[off[i]:off[i + 1]].sum() for i in range(len(off) - 1)])


def dist_stats(x) -> dict:
    """mean, sd and quantiles of a per-sequence quantity."""
    x = np.asarray(x, dtype=np.float64)
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"mean": float(x.mean()), "sd": float(x.std()), "q05": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
            "q75": float(q[3]), "q95": float(q[4]), "n": int(x.size)}


def truncation_stats(delta, offsets, cap: float = 2.0) -> dict:
    """S2, TIS side, on sampled tokens with delta = log p - log q: share with p/q > cap, the ratio mass the cap removes
    (mean of max(p/q - cap, 0)), and per sequence the delta carried by the truncated tokens (sum over them; mean over
    sequences)."""
    d = np.asarray(delta, dtype=np.float64)
    over = d > np.log(cap)
    return {"frac_ratio_gt_cap": float(over.mean()), "excess_ratio_mean": float(np.maximum(np.exp(d) - cap, 0.0).mean()),
            "truncated_delta_seq_mean": float(seq_sums(np.where(over, d, 0.0), offsets).mean()), "n_tokens": int(d.size)}


def linear_slope(xs, ys) -> float:
    """Least-squares slope of ys on xs (NaN with fewer than 2 points)."""
    x, y = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.ptp(x[m]) == 0:
        return float("nan")
    return float(np.polyfit(x[m], y[m], 1)[0])


# -------------------------------------------------------------------------------------------------- S3 by answer length
def length_relation(seq_sum, resp_len, edges) -> dict:
    """S3 against the answer length: per-sequence sum of delta vs the sequence's response length (tokens).
    slope / intercept of the least-squares line (nats per token, nats), Pearson, the mean per-token delta (sum / length,
    mean over sequences), and per length bin [edges[b], edges[b+1]) (the last bin closed above) n, mean length and the
    mean and sd of the sums."""
    y, x = np.asarray(seq_sum, dtype=np.float64), np.asarray(resp_len, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    out = {"n": int(m.sum()), "slope_nats_per_token": linear_slope(x, y), "pearson": pearson(x[m], y[m]),
           "per_token_mean": float((y[m] / np.maximum(x[m], 1.0)).mean()) if m.any() else float("nan"),
           "len_mean": float(x[m].mean()) if m.any() else float("nan"), "bins": []}
    out["intercept_nats"] = (float(y[m].mean() - out["slope_nats_per_token"] * x[m].mean())
                             if np.isfinite(out["slope_nats_per_token"]) else float("nan"))
    e = list(edges)
    for b in range(len(e) - 1):
        sel = m & (x >= e[b]) & ((x < e[b + 1]) | ((b == len(e) - 2) & (x <= e[b + 1])))
        out["bins"].append({"lo": e[b], "hi": e[b + 1], "n": int(sel.sum()),
                            "len_mean": float(x[sel].mean()) if sel.any() else float("nan"),
                            "sum_mean": float(y[sel].mean()) if sel.any() else float("nan"),
                            "sum_sd": float(y[sel].std()) if sel.any() else float("nan")})
    return out
