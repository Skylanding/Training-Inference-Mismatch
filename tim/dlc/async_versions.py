"""Pure parts of DLC under continuous asynchrony (docs/ASYNC.md): per-token generating versions k(t), per-version
selection / merge of log-probs and sampler-head log-probs, the per-response ESS admission, the version-gap statistics.
torch on the CPU (tests/test_async_versions.py), no verl; the verl glue is tim/dlc/async_hooks.py.

Conventions (verl main @8718ca30, v1 separate-async trainer):
  version   the `global_steps` the sampler's weights were pushed with (CheckpointEngineManager.update_weights(s) at the
            end of training step s; update_weights(0) before step 1). Version v = the trainer weights after v updates.
  cur       at training step s the trainer still has version s - 1 (its update of step s comes after the old-log-prob).
  gap(t)    cur - k(t): optimizer updates between the weights that generated token t and the trainer (0 = on-policy).
  layouts   response-level jagged tensors (old_log_probs, rollout_log_probs, rollout_token_versions) have one row per
            response token; full-sequence tensors (log_probs from the worker, rollout_topk_*, ref_topk_log_probs) have one
            row per prompt + response position, and response token r of a sequence ending at offset e is predicted by
            row e - R - 1 + r (verl padding.response_from_nested, score_centering.pad_rollout_topk).
"""
from __future__ import annotations

MISSING = -1            # token version unknown (the server-side hook did not report one)
LOG_RATIO_CLAMP = 20.0  # as verl's compute_policy_loss_vanilla / tim.dlc.centering
GAP_BUCKETS = 8         # histogram buckets g0..g7 and g8plus


# ---------------------------------------------------------------------------------------------------------------------
# rollout side: segments of the partial-rollout resume loop -> one version per generated token (pure python)
# ---------------------------------------------------------------------------------------------------------------------
def segment_token_versions(segments, n_tokens: int) -> list:
    """segments: [(version or None, number of tokens the segment returned), ...] in generation order (one per call of
    the resume loop of FullyAsyncLLMServerClient.generate). Returns one int per generated token, MISSING where a segment
    with tokens carried no version, cut or padded with MISSING to n_tokens (the client's final token count)."""
    out = []
    for version, n in segments:
        if n <= 0:
            continue
        out.extend([MISSING if version is None else int(version)] * int(n))
    if len(out) >= n_tokens:
        return out[:n_tokens]
    return out + [MISSING] * (n_tokens - len(out))


def versions_to_evict(kept, cur: int, keep: int) -> list:
    """CPU snapshots to drop after saving `cur`: keep the newest `keep` versions <= cur (cur included)."""
    if keep < 1:
        raise ValueError(f"keep={keep}: at least the current version")
    return sorted(v for v in kept if v <= cur - keep or v > cur)


# ---------------------------------------------------------------------------------------------------------------------
# trainer side (torch)
# ---------------------------------------------------------------------------------------------------------------------
def response_index(resp_lens):
    """Response number of every response token (response-level values order)."""
    import torch
    lens = torch.as_tensor(resp_lens, dtype=torch.long)
    return torch.repeat_interleave(torch.arange(lens.numel()), lens)


def full_row_versions(seq_offsets, resp_lens, token_versions):
    """Version of the token each full-sequence row predicts: row e - R - 1 + r of a sequence ending at offset e gets the
    version of its response token r; prompt rows and the last row get MISSING.

    Args:
        seq_offsets: (B + 1,) offsets of the full-sequence jagged tensor (prompt + response per row).
        resp_lens: (B,) response lengths.
        token_versions: (sum R,) response-level versions.
    Returns: (seq_offsets[-1],) long tensor.
    """
    import torch
    seq_offsets = torch.as_tensor(seq_offsets, dtype=torch.long)
    lens = torch.as_tensor(resp_lens, dtype=torch.long)
    vers = torch.as_tensor(token_versions, dtype=torch.long)
    if vers.numel() != int(lens.sum()):
        raise ValueError(f"{vers.numel()} token versions for {int(lens.sum())} response tokens")
    out = torch.full((int(seq_offsets[-1]),), MISSING, dtype=torch.long)
    ends = seq_offsets[1:]
    if bool(((ends - seq_offsets[:-1]) < lens + 1).any()):
        raise ValueError("a sequence is shorter than its response + 1 (no prompt row)")
    starts = ends - lens - 1
    rows = torch.repeat_interleave(starts, lens) + (torch.arange(vers.numel()) - torch.repeat_interleave(
        torch.cumsum(lens, 0) - lens, lens))
    out[rows] = vers
    return out


def merge_by_version(base_values, versions, per_version: dict):
    """Row r of the result is per_version[versions[r]][r] when that version is given, else base_values[r] (the current
    weights' value). base_values and every per_version value share the first dimension with versions."""
    import torch
    out = base_values.clone()
    versions = torch.as_tensor(versions, dtype=torch.long)
    if versions.shape[0] != out.shape[0]:
        raise ValueError(f"{versions.shape[0]} versions for {out.shape[0]} rows")
    for k, values in per_version.items():
        if values.shape != out.shape:
            raise ValueError(f"version {k}: shape {tuple(values.shape)} vs {tuple(out.shape)}")
        sel = versions == int(k)
        out[sel] = values[sel].to(out.dtype)
    return out


def per_response_mean(values, mask, resp_lens, empty: float = float("nan")):
    """Mean of `values` over the masked tokens of every response; `empty` where a response has no masked token."""
    import torch
    idx = response_index(resp_lens)
    n = len(resp_lens)
    m = torch.as_tensor(mask).to(torch.float64)
    v = values.to(torch.float64) * m
    s = torch.zeros(n, dtype=torch.float64).index_add_(0, idx, v)
    c = torch.zeros(n, dtype=torch.float64).index_add_(0, idx, m)
    out = s / c.clamp_min(1.0)
    out[c == 0] = empty
    return out


def ess_per_response(log_ratio, mask, resp_lens):
    """Per-response ESS of the lag ratio rho_t = p_theta / p_theta_k(t): 1 / mean_t(rho_t^2) over the response's valid
    tokens (token mean within the response, 1/E_q[rho^2]). 1.0 for a response with no valid
    token (never deferred)."""
    import torch
    lr = torch.as_tensor(log_ratio).to(torch.float64).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
    m2 = per_response_mean(torch.exp(2.0 * lr), mask, resp_lens, empty=1.0)
    return 1.0 / m2


def defer_responses(ess, threshold: float, valid):
    """Responses to defer: valid ones whose ESS is below the threshold."""
    import torch
    return (torch.as_tensor(ess) < float(threshold)) & torch.as_tensor(valid, dtype=torch.bool)


def drop_token_mask(mask, resp_lens, drop_resp):
    """Token mask with every token of the dropped responses zeroed (same dtype as mask)."""
    import torch
    idx = response_index(resp_lens)
    keep = ~torch.as_tensor(drop_resp, dtype=torch.bool)[idx]
    return torch.as_tensor(mask) * keep.to(torch.as_tensor(mask).dtype)


def gap_stats(versions, mask, resp_lens, cur: int, prefix: str = "tim/async_") -> dict:
    """Version-gap histogram of one batch, over the valid response tokens.

    Keys: tok_gap_mean / tok_gap_max, tok_gap_frac_g0..g7 and g8plus (token fractions), n_versions (distinct token
    versions in the batch), resp_versions_mean (distinct versions per response), resp_multi_version_frac (responses
    generated across >= 2 versions: partial rollout), resp_gap_mean (per-response mean gap, averaged), tokens."""
    import torch
    vers = torch.as_tensor(versions, dtype=torch.long)
    m = torch.as_tensor(mask).to(torch.bool)
    lens = torch.as_tensor(resp_lens, dtype=torch.long)
    idx = response_index(lens)
    v = vers[m]
    out = {f"{prefix}tokens": float(v.numel())}
    if v.numel() == 0:
        return out
    gaps = (cur - v).to(torch.long)
    out[f"{prefix}tok_gap_mean"] = float(gaps.double().mean())
    out[f"{prefix}tok_gap_max"] = float(gaps.max())
    for g in range(GAP_BUCKETS):
        out[f"{prefix}tok_gap_frac_g{g}"] = float((gaps == g).double().mean())
    out[f"{prefix}tok_gap_frac_g{GAP_BUCKETS}plus"] = float((gaps >= GAP_BUCKETS).double().mean())
    out[f"{prefix}n_versions"] = float(torch.unique(v).numel())
    per = []
    for r in torch.unique(idx[m]).tolist():
        per.append(torch.unique(vers[(idx == r) & m]).numel())
    per = torch.tensor(per, dtype=torch.float64)
    out[f"{prefix}resp_versions_mean"] = float(per.mean())
    out[f"{prefix}resp_multi_version_frac"] = float((per >= 2).double().mean())
    rg = per_response_mean((cur - vers).double(), m, lens)
    rg = rg[~torch.isnan(rg)]
    out[f"{prefix}resp_gap_mean"] = float(rg.mean())
    return out


def abs_diff_by_gap(a, b, versions, mask, cur: int, name: str, prefix: str = "tim/async_") -> dict:
    """mean |a - b| over the valid tokens, split into fresh (gap 0) and stale (gap >= 1) tokens; keys absent when a group
    is empty. Used for |log p_theta_k(t) - log q| (bf16 noise expected for every gap) and |log p_theta_t - log q|."""
    import torch
    m = torch.as_tensor(mask).to(torch.bool)
    d = (a.to(torch.float64) - b.to(torch.float64)).abs()
    gaps = cur - torch.as_tensor(versions, dtype=torch.long)
    out = {}
    for tag, sel in (("all", m), ("fresh", m & (gaps == 0)), ("stale", m & (gaps >= 1))):
        if bool(sel.any()):
            out[f"{prefix}{name}_abs_mean_{tag}"] = float(d[sel].mean())
    return out


def tag_token_consistency(tag_min, tag_max, versions, mask, resp_lens) -> dict:
    """Agreement of verl's trajectory tags (min/max_global_steps: the server's global_steps when each segment RETURNED)
    with the per-token versions (the server's global_steps when each segment was ADMITTED): fraction of responses whose
    token-version range differs from the tag range. Informational: an aborted segment can return after
    set_global_steps moved, so the tags may run one version ahead."""
    import torch
    vers = torch.as_tensor(versions, dtype=torch.long)
    m = torch.as_tensor(mask).to(torch.bool)
    idx = response_index(resp_lens)
    bad, n = 0, 0
    for r, (lo, hi) in enumerate(zip(tag_min, tag_max)):
        sel = (idx == r) & m
        if not bool(sel.any()) or lo is None or hi is None:
            continue
        n += 1
        v = vers[sel]
        bad += int(int(v.min()) != int(lo) or int(v.max()) != int(hi))
    return {"tim/async_tag_token_mismatch_frac": bad / n} if n else {}
