"""Loop measurements on a GPU: one job per stale run (`scripts/measure.py loop --run <run>`), t = 5, 10, ..., 40
(configs/measure/loop.json). Per step (results/loop_<model>/<run>/step_<t>.json, written last):

6a-1 (0.6B runs; loop.json with_6a1): the 4a batch of the split measurement is regenerated (the run's sampler, rows_4a x n_samples,
    seed 4000) and g_corr, g_on, g_on' exactly as tim.measure.split.job.step_record (b = g_corr - g_on, n = g_on' - g_on). With
    l_t = log p_theta_t(y_t) - log p_theta_0(y_t) (trainer numerics, HF bf16 fa3_liger, both) the predicted bias b_pred = -grad
    of sum_t w_t log p(y_t) / N is formed for the SC token term (w = A l), the prefix term (w = A sum_{t'<t} l) and, from the
    Gram matrix, their sum; scalars via pure.predicted_scalars (cos with b, |b_pred| / |b|, cos with the noise n, cos with
    Delta_cum = theta_t - theta_0). Abar_t: 512 sampled tokens (default_rng(60000 + t)), 8 bf16 continuations from theta_t
    each, scored; the SC token term restricted to the subset with A and with A - Abar.
6a-2 (all runs): the 4c batch (rows_4c(t) x n_samples_4c, seed 7000, the run's sampler): per token l, A, p_t, p_0, position;
    the lag split by sign of A and (on 512 positions, default_rng(61000 + t), Abar from continuations) of A - Abar, and the
    largest-|l| tokens (pure.lag_split, pure.top_tokens). No gradients, so 1.7B runs do this alone.

exact (`scripts/measure.py residual-bias-exact --run <s2 run>`, step_record_exact): 6a-1 again with the linearised and
    the exact weights on one batch, results/residual_bias_exact_0.6b.

GPU memory: the engines at gmu 0.15, two trainers (run + old slot), up to 7 fp32
gradients (0.6B: 17 GB).
"""
from __future__ import annotations

import time

import numpy as np

from tim.measure.split import pure as p4
from tim.measure.split.job import _setup, delta_dots, generate, gradients, sampled_logprobs  # noqa: F401  (_setup re-exported)
from tim.measure.loop import pure


def weighted_gradients(model, batch: dict, weights: dict, micro_batch: int = 8, device: str = "cuda") -> dict:
    """{name: {param: fp32 grad}} of sum_t weights[name][t] log p(y_t) / N at the model's weights, the weights constants
    over the batch's N response rows (tim.measure.split.job.gradients' loop with per-token weights instead of A)."""
    import torch
    from verl.trainer.ppo.score_centering import topk_log_probs_from_logits

    from tim.measure.heads import forward_logits, micro_batches, rows_by_sequence
    seqs, plen = batch["sequences"], batch["prompt_len"]
    off = rows_by_sequence(plen, [len(s) for s in seqs])
    N = int(off[-1])
    names = list(weights)
    params = [(n, p) for n, p in model.named_parameters()]
    for _, p in params:
        p.requires_grad_(True)
    acc = {g: {n: torch.zeros(p.shape, dtype=torch.float32, device=device) for n, p in params} for g in names}
    for idx in micro_batches(model, seqs, micro_batch):
        per_seq = forward_logits(model, seqs, idx, device)
        J = {g: 0.0 for g in names}
        for r, i in enumerate(idx):
            P, L = int(plen[i]), len(seqs[i])
            y = torch.as_tensor(np.asarray(seqs[i][P:L]), device=device)[:, None]
            lp_y = topk_log_probs_from_logits(per_seq[r][P - 1:L - 1], y).squeeze(-1)
            for g in names:
                w = torch.as_tensor(np.asarray(weights[g][off[i]:off[i + 1]]), dtype=torch.float32, device=device)
                J[g] = J[g] + (w * lp_y).sum() / N
        for k, g in enumerate(names):
            if not torch.is_tensor(J[g]) or not J[g].requires_grad:
                continue
            grads = torch.autograd.grad(J[g], [p for _, p in params], retain_graph=k < len(names) - 1, allow_unused=True)
            for (n, _), gr in zip(params, grads):
                if gr is not None:
                    acc[g][n] += gr.float()
        del per_seq, J
    for _, p in params:
        p.requires_grad_(False)
    return acc


def cum_delta_sq(path_t: str, path_0: str, names, chunk: int = 1 << 22, device: str = "cuda") -> float:
    """|theta_t - theta_0|^2 over the named parameters, streamed from the two checkpoints (delta_dots' loop without gradients)."""
    import os

    from safetensors import safe_open

    def index(path):
        files = sorted(x for x in os.listdir(path) if x.endswith(".safetensors"))
        hs = [safe_open(os.path.join(path, f), framework="pt", device=device) for f in files]
        return {k: h for h in hs for k in h.keys()}
    it, i0 = index(path_t), index(path_0)
    sq = 0.0
    for n in names:
        xt, x0 = it[n].get_tensor(n).reshape(-1), i0[n].get_tensor(n).reshape(-1)
        for a in range(0, xt.numel(), chunk):
            d = xt[a:a + chunk].double() - x0[a:a + chunk].double()
            sq += float((d * d).sum())
    return sq


def continuation_rewards(llm, S: dict, batch: dict, off, rows, seed_base: int, n_cont: int) -> list:
    """For each sampled-token row: n_cont continuations of its prefix (prompt + response tokens before that position) to the
    end with `llm` (temperature 1, the stop ids and the max-length rule of generate(): the response is capped at
    MAX_RESPONSE tokens in total and the sequence at model_len), scored on the full response text. -> [[reward] * n_cont]."""
    from vllm import SamplingParams
    seq_of = np.searchsorted(np.asarray(off), np.asarray(rows), side="right") - 1
    reqs, params, meta = [], [], []
    for j, (r, i) in enumerate(zip(rows, seq_of)):
        seq, P = batch["sequences"][i], int(batch["prompt_len"][i])
        k = int(r - off[i])
        prefix = list(seq[:P + k])
        reqs.append({"prompt_token_ids": prefix})
        params.append(SamplingParams(n=n_cont, temperature=1.0, top_p=1.0, top_k=-1, stop_token_ids=S["stop"],
                                     seed=seed_base + j, max_tokens=max(1, min(S["max_response"] - k, S["model_len"] - len(prefix)))))
        meta.append((list(seq[P:P + k]), S["table"][batch["row"][i]]["reward_model"]["ground_truth"]))
    outs = llm.generate(reqs, params, use_tqdm=False)
    return [[S["reward"](S["tok"].decode(head + list(c.token_ids), skip_special_tokens=True), gt) for c in o.outputs]
            for o, (head, gt) in zip(outs, meta)]


def step_record(eng, run: str, t: int, ck_t: str, cfg: dict, c6: dict, S: dict) -> dict:
    import torch

    from tim.measure.heads import grad_dot, rows_by_sequence, sampler_sequence_heads
    p = p4.parse_run(run)
    with_61 = p["prefix"] in c6["with_6a1"]
    base = cfg["base_model"]
    rows = c6["rows_4a"]
    n, cap, n_abar, n_cont = int(c6["n_samples"]), float(c6["tis_cap"]), int(c6["n_abar"]), int(c6["n_cont"])
    model = eng.trainer_model(ck_t, "fa3_liger")
    old = eng.trainer_model(base, "fa3_liger", slot="old")
    timing, rec = {}, {"run": run, "step": t, "source": p["source"], "method": p["method"], "seed": p["seed"],
                       "sampler": f"{p['kind']}@theta_{p['sampler_at']}"}
    run_at = ck_t if p["sampler_at"] == "t" else base

    # phase A: the run's sampler (theta_0 for the stale runs): the 4a batch and its heads, the 4c batch
    t0 = time.time()
    torch.cuda.empty_cache()                            # vLLM checks free memory when an engine is first built
    llm = eng.sampler(p["kind"], run_at)
    if with_61:
        b_run = generate(llm, S, rows, n, p4.SEED_RUN)
        heads = sampler_sequence_heads(llm, b_run["sequences"], b_run["prompt_len"], cfg["topk"])
    r4c = p4.rows_4c(t, len(S["table"]), int(c6["n_prompts_4c"]), rows)
    b4 = generate(llm, S, r4c, int(c6["n_samples_4c"]), p4.SEED_4C)
    timing["sample_run"] = time.time() - t0; t0 = time.time()

    # phase B: the bf16 engine at theta_t (the trainer's own policy): the on-policy batches and the Abar continuations
    torch.cuda.empty_cache()
    llm_on = eng.sampler("bf16", ck_t)
    if with_61:
        b_on, b_on2 = generate(llm_on, S, rows, n, p4.SEED_ON), generate(llm_on, S, rows, n, p4.SEED_ON2)
    timing["on_policy_batches"] = time.time() - t0; t0 = time.time()

    def lag_of(batch):
        off = rows_by_sequence(batch["prompt_len"], [len(s) for s in batch["sequences"]])
        lp_t = sampled_logprobs(model, batch["sequences"], batch["prompt_len"])
        lp_0 = sampled_logprobs(old, batch["sequences"], batch["prompt_len"])
        return off, lp_t, lp_t - lp_0

    def abar_of(batch, off, sel, seed_base):
        cont = continuation_rewards(llm_on, S, batch, off, sel, seed_base, n_cont)
        seq_of = np.searchsorted(off, sel, side="right") - 1
        base_r = np.asarray(batch["reward"])[seq_of] - np.asarray(batch["advantage"])[seq_of]     # the group mean reward
        return pure.abar_from_rewards(cont, base_r)

    if with_61:
        off, lp_t, lag = lag_of(b_run)
        N = int(off[-1])
        A = pure.per_row(b_run["advantage"], off)
        W = pure.bias_weights(lag, A, off, p["method"])
        sel = pure.choose_rows(N, n_abar, pure.SUBSET_SEED_4A + t)
        abar = abar_of(b_run, off, sel, pure.SEED_ABAR)
        timing["abar_4a"] = time.time() - t0; t0 = time.time()
        W_sub = {}
        if p["method"] == "sc":
            W_sub["sA"], W_sub["sB"] = pure.subset_weights(lag, A, dict(zip(sel.tolist(), abar.tolist())), N)
        g_c = gradients(model, b_run, heads, [p["method"]], cap=cap)[0][p["method"]]
        g_on = gradients(model, b_on, None, ["pg"])[0]["pg"]
        g_on2 = gradients(model, b_on2, None, ["pg"])[0]["pg"]
        timing["grad_batches"] = time.time() - t0; t0 = time.time()
        V = {"c": g_c, "o": g_on, "o2": g_on2}
        names = {"pre": "prefix", "tok": "token"}
        wg = weighted_gradients(model, b_run, {k: W[v] for k, v in names.items() if v in W} | W_sub)
        V.update(wg)
        timing["grad_pred"] = time.time() - t0; t0 = time.time()
        keys = list(V)
        gram = {(a, b): grad_dot(V[a], V[b]) for i, a in enumerate(keys) for b in keys[i:]}
        dd, sq = delta_dots(V, ck_t, ck_t, base)        # path_prev = ck_t: the "recent" delta is zero and ignored
        delta_dot = {k: dd[(k, "cum")] for k in keys}
        preds = {"prefix": {"pre": 1.0}, "total": {"pre": 1.0}}
        if "tok" in V:
            preds = {"token": {"tok": 1.0}, **preds, "total": {"tok": 1.0, "pre": 1.0}}
        r61 = {"pred": pure.predicted_scalars(gram, delta_dot, sq["cum"], preds),
               "lag": {"mean": float(lag.mean()), "abs_mean": float(np.abs(lag).mean()), "abs_max": float(np.abs(lag).max()),
                       "n_tokens": N},
               "batch": {"run_reward_mean": float(np.mean(b_run["reward"])), "on_reward_mean": float(np.mean(b_on["reward"]))}}
        if "tok" in V:
            sub = pure.predicted_scalars(gram, delta_dot, sq["cum"], {"sub_A": {"sA": 1.0}, "sub_Abar": {"sB": 1.0}})
            r61["subset"] = {"n": int(sel.size), "abar_ratio": pure.abar_ratio(abar, A[sel]), "abar_mean": float(abar.mean()),
                             "cos_sub_A_vs_Abar": pure.cos_vectors(gram, {"sA": 1.0}, {"sB": 1.0}),
                             "sub_A": sub["sub_A"], "sub_Abar": sub["sub_Abar"]}
        else:
            r61["subset"] = {"n": int(sel.size), "abar_ratio": pure.abar_ratio(abar, A[sel]), "abar_mean": float(abar.mean())}
        rec["6a1"] = r61
        delta_sq = sq["cum"]
        del V, wg, g_c, g_on, g_on2
        torch.cuda.empty_cache()
        timing["dots"] = time.time() - t0; t0 = time.time()
    else:
        delta_sq = cum_delta_sq(ck_t, base, [n_ for n_, _ in model.named_parameters()])

    # 6a-2 on the 4c batch
    off4, lp_t4, lag4 = lag_of(b4)
    lp_04 = lp_t4 - lag4
    A4 = pure.per_row(b4["advantage"], off4)
    seq4, pos4 = pure.positions(off4)
    ln4 = np.diff(off4)[seq4]
    sel4 = pure.choose_rows(int(off4[-1]), n_abar, pure.SUBSET_SEED_4C + t)
    abar4 = abar_of(b4, off4, sel4, pure.SEED_ABAR + 100000)
    rec["6a2"] = {"n_tokens": int(off4[-1]), "delta_cum_sq": delta_sq, "lag_mean": float(lag4.mean()),
                  "p_t_mean": float(np.exp(lp_t4).mean()), "p_0_mean": float(np.exp(lp_04).mean()),
                  "by_A": pure.lag_split(lag4, A4, delta_sq),
                  "subset": {"n": int(sel4.size), "abar_ratio": pure.abar_ratio(abar4, A4[sel4]),
                             "by_A_minus_Abar": pure.lag_split(lag4[sel4], A4[sel4] - abar4, delta_sq),
                             "by_A": pure.lag_split(lag4[sel4], A4[sel4], delta_sq)},
                  "top": pure.top_tokens(lag4, A4, np.exp(lp_t4), pos4, ln4), "rows": r4c[:4] + ["..."]}
    timing["6a2"] = time.time() - t0
    rec["timing_s"] = {k: round(v, 1) for k, v in timing.items()}
    return rec


# ------------------------------------------------------------------------------------- exact-weight residual bias
def step_record_exact(eng, run: str, t: int, ck_t: str, cfg: dict, c6: dict, S: dict) -> dict:
    """6a-1 on the same regenerated 4a batch (the run's sampler, seed 4000) with the linearised weights
    (exactly step_record's) AND the exact weights (pure.bias_weights exact=True), side by side against the same b and noise n.
    delta_t: l_t on stale64 (where e_t < 1e-3), log p_theta_t(y) - log q(y) on fp8_stale64 (q = the sampler's own
    log-prob of the sampled token, from the prefill heads of the regenerated batch). The SC token term on the 512-position
    subset with A and with A - Abar, both versions. Two gradient passes (linearised, exact), each with g_corr, g_on, g_on'
    resident, so the peak is 7 fp32 gradients as in 6a. No 6a-2. Results results/residual_bias_exact_0.6b/<run>/step_<t>.json."""
    import torch

    from tim.measure.heads import grad_dot, rows_by_sequence, sampler_sequence_heads
    p = p4.parse_run(run)
    base = cfg["base_model"]
    rows = c6["rows_4a"]
    n, cap, n_abar, n_cont = int(c6["n_samples"]), float(c6["tis_cap"]), int(c6["n_abar"]), int(c6["n_cont"])
    model = eng.trainer_model(ck_t, "fa3_liger")
    old = eng.trainer_model(base, "fa3_liger", slot="old")
    timing, rec = {}, {"run": run, "step": t, "source": p["source"], "method": p["method"], "seed": p["seed"],
                       "sampler": f"{p['kind']}@theta_{p['sampler_at']}"}
    run_at = ck_t if p["sampler_at"] == "t" else base

    t0 = time.time()
    torch.cuda.empty_cache()                            # vLLM checks free memory when an engine is first built
    llm = eng.sampler(p["kind"], run_at)
    b_run = generate(llm, S, rows, n, p4.SEED_RUN)
    heads = sampler_sequence_heads(llm, b_run["sequences"], b_run["prompt_len"], cfg["topk"])
    torch.cuda.empty_cache()
    llm_on = eng.sampler("bf16", ck_t)
    b_on, b_on2 = generate(llm_on, S, rows, n, p4.SEED_ON), generate(llm_on, S, rows, n, p4.SEED_ON2)
    timing["sample"] = time.time() - t0; t0 = time.time()

    off = rows_by_sequence(b_run["prompt_len"], [len(s) for s in b_run["sequences"]])
    lp_t = sampled_logprobs(model, b_run["sequences"], b_run["prompt_len"])
    lag = lp_t - sampled_logprobs(old, b_run["sequences"], b_run["prompt_len"])
    q = np.asarray(heads["next"], dtype=np.float64)
    delta = lp_t - q if p["source"].startswith("fp8") else lag
    N = int(off[-1])
    A = pure.per_row(b_run["advantage"], off)
    sel = pure.choose_rows(N, n_abar, pure.SUBSET_SEED_4A + t)
    cont = continuation_rewards(llm_on, S, b_run, off, sel, pure.SEED_ABAR, n_cont)
    seq_of = np.searchsorted(off, sel, side="right") - 1
    abar = pure.abar_from_rewards(cont, np.asarray(b_run["reward"])[seq_of] - np.asarray(b_run["advantage"])[seq_of])
    timing["logprobs_abar"] = time.time() - t0; t0 = time.time()

    g_c = gradients(model, b_run, heads, [p["method"]], cap=cap)[0][p["method"]]
    g_on = gradients(model, b_on, None, ["pg"])[0]["pg"]
    g_on2 = gradients(model, b_on2, None, ["pg"])[0]["pg"]
    timing["grad_batches"] = time.time() - t0; t0 = time.time()
    names = [n_ for n_, _ in model.named_parameters()]
    delta_sq = cum_delta_sq(ck_t, base, names)

    def version(exact: bool) -> dict:
        x = delta if exact else lag
        W = pure.bias_weights(x, A, off, p["method"], exact=exact, cap=cap)
        Wg = {"pre": W["prefix"]} | ({"tok": W["token"]} if "token" in W else {})
        if p["method"] == "sc":
            Wg["sA"], Wg["sB"] = pure.subset_weights(x, A, dict(zip(sel.tolist(), abar.tolist())), N, exact=exact)
        V = {"c": g_c, "o": g_on, "o2": g_on2, **weighted_gradients(model, b_run, Wg)}
        keys = list(V)
        gram = {(a, b): grad_dot(V[a], V[b]) for i, a in enumerate(keys) for b in keys[i:]}
        dd, _ = delta_dots(V, ck_t, ck_t, base)
        delta_dot = {k: dd[(k, "cum")] for k in keys}
        preds = {"prefix": {"pre": 1.0}, "total": {"pre": 1.0}}
        if "tok" in V:
            preds = {"token": {"tok": 1.0}, "prefix": {"pre": 1.0}, "total": {"tok": 1.0, "pre": 1.0}}
        r = {"pred": pure.predicted_scalars(gram, delta_dot, delta_sq, preds),
             "weights_abs_mean": {k: float(np.abs(v).mean()) for k, v in W.items()}}
        if "tok" in V:
            r["cos_token_prefix"] = pure.cos_vectors(gram, {"tok": 1.0}, {"pre": 1.0})
        if p["method"] == "sc":
            sub = pure.predicted_scalars(gram, delta_dot, delta_sq, {"sub_A": {"sA": 1.0}, "sub_Abar": {"sB": 1.0}})
            r["subset"] = {"sub_A": sub["sub_A"], "sub_Abar": sub["sub_Abar"],
                           "cos_sub_A_vs_Abar": pure.cos_vectors(gram, {"sA": 1.0}, {"sB": 1.0})}
        del V
        torch.cuda.empty_cache()
        return r

    rec["lin"] = version(False)
    timing["lin"] = time.time() - t0; t0 = time.time()
    rec["exact"] = version(True)
    timing["exact"] = time.time() - t0
    rec["delta_source"] = "log p_theta_t - log q (sampler)" if p["source"].startswith("fp8") else "l = log p_theta_t - log p_theta_0"
    rec["diag"] = {"exact": pure.exact_diagnostics(delta, off, cap), "lag_mean": float(lag.mean()), "delta_mean": float(delta.mean()),
                   "engine_mean": float((delta - lag).mean()) if p["source"].startswith("fp8") else 0.0, "n_tokens": N,
                   "abar_ratio": pure.abar_ratio(abar, A[sel]), "n_subset": int(sel.size),
                   "run_reward_mean": float(np.mean(b_run["reward"])), "on_reward_mean": float(np.mean(b_on["reward"])),
                   "delta_cum_sq": delta_sq}
    del g_c, g_on, g_on2
    torch.cuda.empty_cache()
    rec["timing_s"] = {k: round(v, 1) for k, v in timing.items()}
    return rec
