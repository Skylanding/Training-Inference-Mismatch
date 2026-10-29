"""Sampler and trainer heads on full fixed sequences (teacher forcing). GPU only except `flat_prompt_heads` and
`rows_by_sequence`, which are pure numpy.

Row layout: one row per response token y = seq[i] (i = prompt_len .. len-1), holding the distribution that predicts it from
seq[:i] (the probe prefix h). Rows are flattened in sequence order (tim.measure.artifacts.response_index).

Sampler side: vLLM `prompt_logprobs=K` on the full sequence, i.e. what the sampler's engine (bf16, block-fp8 weights with
an fp8 KV cache, or the theta_0 weights) assigns at every position; `flat_logprobs=True` keeps the K+1 entries per position
as primitive lists (vLLM FlatLogprobs). The rank-1..K entries are the head, by rank as in verl's `extract_prompt_logprobs`
(verl/workers/rollout/vllm_rollout/utils.py @ 8718ca30); the log-prob of the actual next token is also kept when it is
outside the head, because the sampled-token error delta and the importance ratio need it. This is a prefill of the whole
sequence; during training the response KV cache is written by decode steps.

Trainer side: HF bf16 with the run's kernel set (tim.measure.engines): `sdpa` runs right-padded micro-batches, `fa3_liger`
runs FA3 on packed sequences in verl's dynamic micro-batches plus Liger (`forward_logits`), as the trainings did. Head
gather by verl's `topk_log_probs_from_logits` (verl/trainer/ppo/score_centering.py), the function training uses for score
centering.

Gradients: `grad_dot`, `grad_norm`, `grad_cosine` on {parameter name: fp32 tensor} dicts.
"""
from __future__ import annotations

import os

import numpy as np


def flat_prompt_heads(starts, ends, token_ids, logprobs, ranks, seq, prompt_len: int, k: int) -> tuple:
    """vLLM prompt_logprobs of one sequence (FlatLogprobs lists) -> (ids (R,k) int32, lp (R,k) float32, next (R,) float32).

    Entry i of prompt_logprobs is the distribution of token i given tokens < i (entry 0 is empty). Rows are entries
    prompt_len .. len-1. Raises if a head slot is empty or the actual token is missing.
    """
    L = len(seq)
    st, en = np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)
    if len(st) != L:
        raise ValueError(f"{len(st)} positions for a sequence of {L} tokens")
    pos = np.repeat(np.arange(L), en - st)
    tok = np.asarray(token_ids, dtype=np.int64)
    lp = np.asarray(logprobs, dtype=np.float64)
    rk = np.asarray([0 if r is None else r for r in ranks], dtype=np.int64)
    R = L - prompt_len
    ids_out = np.full((R, k), -1, dtype=np.int64)
    lp_out = np.full((R, k), np.nan)
    nxt = np.full(R, np.nan)
    m = pos >= prompt_len
    head = m & (rk >= 1) & (rk <= k)
    ids_out[pos[head] - prompt_len, rk[head] - 1] = tok[head]
    lp_out[pos[head] - prompt_len, rk[head] - 1] = lp[head]
    seq_arr = np.asarray(seq, dtype=np.int64)
    act = m & (tok == seq_arr[pos])
    nxt[pos[act] - prompt_len] = lp[act]
    if (ids_out < 0).any() or np.isnan(nxt).any():
        raise RuntimeError("prompt_logprobs: empty head slot or missing actual-token entry")
    return ids_out.astype(np.int32), lp_out.astype(np.float32), nxt.astype(np.float32)


def _flat_lists(plp):
    """FlatLogprobs or list[dict[token, Logprob] | None] -> (starts, ends, token_ids, logprobs, ranks)."""
    if hasattr(plp, "start_indices"):
        return plp.start_indices, plp.end_indices, plp.token_ids, plp.logprobs, plp.ranks
    starts, ends, toks, lps, rks = [], [], [], [], []
    for d in plp:
        starts.append(len(toks))
        for t, x in (d or {}).items():
            toks.append(int(t)); lps.append(x.logprob); rks.append(x.rank)
        ends.append(len(toks))
    return starts, ends, toks, lps, rks


def rows_by_sequence(prompt_len, seq_len) -> np.ndarray:
    """Offsets (n+1,) of each sequence's rows in the flattened row arrays."""
    r = np.asarray(seq_len, dtype=np.int64) - np.asarray(prompt_len, dtype=np.int64)
    return np.concatenate([[0], np.cumsum(r)])


# ------------------------------------------------------------------------------------------------------------------ GPU
def sampler_sequence_heads(llm, sequences, prompt_len, topk: int = 128, chunk: int = 64) -> dict:
    """vLLM heads at every response position; `chunk` sequences per generate() call."""
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1, prompt_logprobs=topk,
                            flat_logprobs=True, detokenize=False)
    ids, lps, nxt = [], [], []
    for a in range(0, len(sequences), chunk):
        batch = [{"prompt_token_ids": list(map(int, s))} for s in sequences[a:a + chunk]]
        for j, res in enumerate(llm.generate(batch, params, use_tqdm=False)):
            i = a + j
            x = flat_prompt_heads(*_flat_lists(res.prompt_logprobs), sequences[i], int(prompt_len[i]), topk)
            ids.append(x[0]); lps.append(x[1]); nxt.append(x[2])
    return {"ids": np.concatenate(ids), "lp": np.concatenate(lps), "next": np.concatenate(nxt)}


def _batch(sequences, idx, pad_id: int = 0):
    import torch
    lens = [len(sequences[i]) for i in idx]
    ids = torch.full((len(idx), max(lens)), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for r, i in enumerate(idx):
        ids[r, :lens[r]] = torch.as_tensor(sequences[i], dtype=torch.long)
        mask[r, :lens[r]] = 1
    return ids, mask, lens


def forward_logits(model, sequences, idx, device: str = "cuda") -> list:
    """Logits (L_i, V) of each sequence in idx. The model's `_tim_mode` says how:
      padded  right-padded micro-batch with an attention mask (kernel set sdpa, no remove-padding);
      packed  all sequences of the micro-batch in one row, position_ids restarting per sequence: transformers' varlen FA
              path, verl's remove-padding layout (kernel set fa3_liger, as in training);
      single  one sequence per forward, batch 1 (TIM_MEASURE_TRAINER_PACK=0). Packed and single differ only at the bf16 level
              for length-dependent tiles, with no error across sequence boundaries."""
    import torch
    mode = getattr(model, "_tim_mode", "padded")
    if mode == "single":
        out = []
        for i in idx:
            ids = torch.as_tensor(np.asarray(sequences[i], dtype=np.int64))[None].to(device)
            out.append(model(input_ids=ids).logits[0])
        return out
    if mode == "packed":
        lens = [len(sequences[i]) for i in idx]
        ids = torch.as_tensor(np.concatenate([np.asarray(sequences[i], dtype=np.int64) for i in idx]))[None]
        pos = torch.as_tensor(np.concatenate([np.arange(n) for n in lens]))[None]
        logits = model(input_ids=ids.to(device), position_ids=pos.to(device)).logits[0]
        off = np.concatenate([[0], np.cumsum(lens)])
        return [logits[off[r]:off[r + 1]] for r in range(len(idx))]
    ids, mask, lens = _batch(sequences, idx)
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
    return [logits[r, :lens[r]] for r in range(len(idx))]


PACK_MAX_TOKENS = int(os.environ.get("TIM_MEASURE_PACK_MAX_TOKENS", "16384"))   # = ppo_max_token_len_per_gpu of the trainings


def micro_batches(model, sequences, micro_batch: int) -> list:
    """Index lists of the trainer's micro-batches. Packed mode: verl's own dynamic batching by token budget
    (verl.utils.seqlen_balancing.rearrange_micro_batches with max_token_len = PACK_MAX_TOKENS, as the trainings
    use for update_actor and log-prob passes); otherwise consecutive groups of `micro_batch` sequences."""
    n = len(sequences)
    if getattr(model, "_tim_mode", "padded") != "packed":
        return [list(range(a, min(a + micro_batch, n))) for a in range(0, n, micro_batch)]
    import torch
    from tensordict import TensorDict
    from verl.utils.seqlen_balancing import rearrange_micro_batches
    L = max(len(x) for x in sequences)
    mask = torch.zeros(n, L, dtype=torch.long)
    for i, x in enumerate(sequences):
        mask[i, :len(x)] = 1
    td = TensorDict({"input_ids": mask.clone(), "attention_mask": mask}, batch_size=[n])
    _, idx = rearrange_micro_batches(td, max_token_len=max(PACK_MAX_TOKENS, L))
    return [[int(i) for i in g] for g in idx]


def trainer_sequence_heads(model, sequences, prompt_len, q_ids, micro_batch: int = 16, device: str = "cuda") -> dict:
    """Trainer log-probs at the sampler head ids and at the actual next token, every response row: lp (n,K), next (n,)."""
    import torch
    from verl.trainer.ppo.score_centering import topk_log_probs_from_logits
    off = rows_by_sequence(prompt_len, [len(s) for s in sequences])
    q_ids = np.asarray(q_ids)
    out_lp = np.zeros(q_ids.shape, dtype=np.float32)
    out_next = np.zeros(q_ids.shape[0], dtype=np.float32)
    with torch.no_grad():
        for idx in micro_batches(model, sequences, micro_batch):
            per_seq = forward_logits(model, sequences, idx, device)
            for r, i in enumerate(idx):
                P, L = int(prompt_len[i]), len(sequences[i])
                rows = per_seq[r][P - 1:L - 1]
                gather = np.concatenate([q_ids[off[i]:off[i + 1]], np.asarray(sequences[i][P:L])[:, None]], axis=1)
                lp = topk_log_probs_from_logits(rows, torch.as_tensor(gather, device=device)).cpu().numpy()
                out_lp[off[i]:off[i + 1]] = lp[:, :-1]
                out_next[off[i]:off[i + 1]] = lp[:, -1]
            del per_seq
    return {"lp": out_lp, "next": out_next}


def grad_dot(a: dict, b: dict, chunk: int = 1 << 22) -> float:
    """sum over shared names of <a[n], b[n]> in fp64, 4M elements at a time (full-tensor fp64 copies of the 155M-element
    embedding gradient were ~1.2 GB each)."""
    tot = 0.0
    for n in a:
        if n not in b:
            continue
        x, y = a[n].reshape(-1), b[n].reshape(-1)
        for i in range(0, x.numel(), chunk):
            tot += float((x[i:i + chunk].double() * y[i:i + chunk].double()).sum())
    return tot


def grad_norm(a: dict) -> float:
    return grad_dot(a, a) ** 0.5


def grad_cosine(a: dict, b: dict) -> float:
    na, nb = grad_norm(a), grad_norm(b)
    return grad_dot(a, b) / (na * nb) if na > 0 and nb > 0 else float("nan")
