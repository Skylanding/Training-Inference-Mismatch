"""Split and residual bias on a GPU: one job per run (`scripts/measure.py split --run <run>`), steps t = 5, 10, ..., 40
(the `split` block of configs/measure/<model>.json). Per step (results/split_<model>/<run>/step_<t>.json, written last):

4a  the run's own sampler (fp8 / bf16 at theta_t, bf16 / fp8 at theta_0, bf16 eager at theta_t; tim.measure.split.pure
    SOURCE_SAMPLER) generates the fixed 64 train prompts (the P2 rows) x 8 (seeds 4000 + j); its top-128 heads by
    prefill on the same sequences; the run's corrected gradient g_corr on that batch (TIS: A min(p/q, 2) log p; SC:
    A (log p - c) with verl's score_centering_correction; PG for eager). Then the bf16 engine refit to theta_t (= the
    trainer's own policy) generates the same prompts twice (seeds 5000 + j, 6000 + j) -> g_on, g_on'. All gradients at the
    trainer theta_t (HF bf16, KERNELS=fa3_liger as the run), fp32 accumulators, token-mean over the batch's response tokens.
    Scalars from inner products with dtheta_recent = theta_t - theta_{t-5} and dtheta_cum = theta_t - theta_0, streamed
    from the checkpoints' safetensors (no delta or gradient is ever written to disk).
4b  (TIS runs) on the same batch: the SC gradient, and both restricted to tokens with p/q < 0.5; projections of all four on
    the two deltas; the sampled tokens' p/q against delta (pure.tis_token_stats).
4c  the run's sampler on 64 train prompts that change with t (pure.rows_4c) x 4 (seeds 7000 + j): per-token delta =
    log p_theta - log q split into lag = log p_theta - log p_theta_old and engine = log p_theta_old - log q, theta_old the
    sampler's weights (theta_0 for the stale sources, theta_t otherwise, where lag = 0 exactly).

GPU memory (Qwen3-0.6B): two vLLM engines at gmu 0.15 each, the trainer (1.2 GB), up to 6 fp32 gradients (TIS runs:
4 + g_on + g_on' = 14.4 GB) + activations.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from tim.measure.split import pure


def _setup(cfg: dict) -> dict:
    """Prompts (token ids, chat-rendered as in training), ground truths, stop ids, reward."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from tim.measure.config import init_eos_ids
    from tim.measure.prefixes import MAX_RESPONSE, _render_prompts
    from tim.tasks.registry import reward_fn
    table = pq.read_table(cfg["train_parquet"]).to_pylist()
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    template = open(cfg["template"]).read()
    return {"table": table, "tok": tok, "template": template, "render": lambda rows: _render_prompts(
        tok, [table[r]["prompt"] for r in rows], template), "stop": init_eos_ids(cfg["base_model"]),
        "reward": reward_fn("countdown"), "max_response": MAX_RESPONSE, "model_len": 512}


def generate(llm, S: dict, rows: list, n: int, seed_base: int) -> dict:
    """n samples per prompt row (one request per row, seed seed_base + j), scored, group-centred advantages
    (tim.measure.artifacts.from_rollouts: A = R - group mean, as adv_estimator=grpo without std)."""
    from vllm import SamplingParams

    from tim.measure import artifacts
    prompts = S["render"](rows)
    params = [SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=-1, stop_token_ids=S["stop"], seed=seed_base + j,
                             max_tokens=min(S["max_response"], S["model_len"] - len(p))) for j, p in enumerate(prompts)]
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
    rolls = []
    for r, p, o in zip(rows, prompts, outs):
        gt = S["table"][r]["reward_model"]["ground_truth"]
        rolls.append((r, p, [list(c.token_ids) for c in o.outputs], [S["reward"](c.text, gt) for c in o.outputs]))
    return artifacts.from_rollouts(rolls)


def gradients(model, batch: dict, heads, objectives: list, cap: float = 2.0, micro_batch: int = 8,
              device: str = "cuda") -> tuple:
    """{objective: {param: fp32 grad}} at the model's weights (ascent direction, token-mean over the batch's N response
    tokens), and the sampled tokens' log(p/q) and advantage. heads None: PG only (no sampler log-probs needed). The
    objectives are listed in the module doc of tim.measure.split.pure; the score-centering term is verl's
    score_centering_correction."""
    import torch
    from verl.trainer.ppo.score_centering import score_centering_correction, topk_log_probs_from_logits

    from tim.measure.heads import forward_logits, micro_batches, rows_by_sequence
    seqs, plen = batch["sequences"], batch["prompt_len"]
    A = np.asarray(batch["advantage"], dtype=np.float64)
    off = rows_by_sequence(plen, [len(s) for s in seqs])
    N = int(off[-1])
    params = [(n, p) for n, p in model.named_parameters()]
    for _, p in params:
        p.requires_grad_(True)
    acc = {g: {n: torch.zeros(p.shape, dtype=torch.float32, device=device) for n, p in params} for g in objectives}
    lr_all, a_all = [], []
    low = float(np.log(pure.LOW))
    for idx in micro_batches(model, seqs, micro_batch):
        per_seq = forward_logits(model, seqs, idx, device)
        J = {g: 0.0 for g in objectives}
        for r, i in enumerate(idx):
            P, L = int(plen[i]), len(seqs[i])
            rows = per_seq[r][P - 1:L - 1]
            y = torch.as_tensor(np.asarray(seqs[i][P:L]), device=device)[:, None]
            lp_y = topk_log_probs_from_logits(rows, y).squeeze(-1)
            a = float(A[i])
            if heads is None:
                J["pg"] = J["pg"] + a * lp_y.sum() / N
                continue
            qn = torch.as_tensor(np.asarray(heads["next"][off[i]:off[i + 1]]), dtype=torch.float32, device=device)
            lr = lp_y.detach() - qn
            lr_all.append(lr.cpu().numpy()); a_all.append(np.full(L - P, a))
            w = torch.clamp(lr.exp(), max=cap)
            m_low = (lr < low).float()
            corr = None
            if any(g.startswith("sc") for g in objectives):
                h_ids = torch.as_tensor(np.asarray(heads["ids"][off[i]:off[i + 1]]), device=device)
                h_lp = torch.as_tensor(np.asarray(heads["lp"][off[i]:off[i + 1]]), dtype=torch.float32, device=device)
                corr, _, _ = score_centering_correction(topk_log_probs_from_logits(rows, h_ids), h_lp, torch.ones_like)
            for g in objectives:
                if g == "pg":
                    term = lp_y
                elif g == "tis":
                    term = w * lp_y
                elif g == "tis_low":
                    term = m_low * w * lp_y
                elif g == "sc":
                    term = lp_y - corr
                elif g == "sc_low":
                    term = m_low * (lp_y - corr)
                else:
                    raise ValueError(g)
                J[g] = J[g] + a * term.sum() / N
        for k, g in enumerate(objectives):
            if not torch.is_tensor(J[g]) or not J[g].requires_grad:
                continue
            grads = torch.autograd.grad(J[g], [p for _, p in params], retain_graph=k < len(objectives) - 1,
                                        allow_unused=True)
            for (n, _), gr in zip(params, grads):
                if gr is not None:
                    acc[g][n] += gr.float()
        del per_seq, J
    for _, p in params:
        p.requires_grad_(False)
    tok = {"log_ratio": np.concatenate(lr_all) if lr_all else np.zeros(0),
           "adv": np.concatenate(a_all) if a_all else np.zeros(0), "n_tokens": N}
    return acc, tok


def sampled_logprobs(model, seqs, plen, micro_batch: int = 16, device: str = "cuda") -> np.ndarray:
    """log p(y) of every response token at the model's weights (no gradient), in response-row order."""
    import torch
    from verl.trainer.ppo.score_centering import topk_log_probs_from_logits

    from tim.measure.heads import forward_logits, micro_batches, rows_by_sequence
    off = rows_by_sequence(plen, [len(s) for s in seqs])
    out = np.zeros(int(off[-1]))
    with torch.no_grad():
        for idx in micro_batches(model, seqs, micro_batch):
            per_seq = forward_logits(model, seqs, idx, device)
            for r, i in enumerate(idx):
                P, L = int(plen[i]), len(seqs[i])
                y = torch.as_tensor(np.asarray(seqs[i][P:L]), device=device)[:, None]
                out[off[i]:off[i + 1]] = topk_log_probs_from_logits(per_seq[r][P - 1:L - 1], y).squeeze(-1).double().cpu().numpy()
            del per_seq
    return out


def delta_dots(grads: dict, path_t: str, path_prev: str, path_0: str, chunk: int = 1 << 22,
               device: str = "cuda") -> tuple:
    """{(grad, "recent"|"cum"): <g, delta>} and {"recent"|"cum": |delta|^2}, delta_recent = theta_t - theta_prev,
    delta_cum = theta_t - theta_0, streamed one parameter tensor at a time from the three checkpoints' safetensors in
    fp64 chunks (parameters the gradients have; a tied lm_head is the embedding)."""
    from safetensors import safe_open

    def index(path):
        files = sorted(x for x in os.listdir(path) if x.endswith(".safetensors"))
        hs = [safe_open(os.path.join(path, f), framework="pt", device=device) for f in files]
        return {k: h for h in hs for k in h.keys()}
    it, ip, i0 = index(path_t), index(path_prev), index(path_0)
    names = next(iter(grads.values())).keys()
    missing = [n for n in names if n not in it or n not in ip or n not in i0]
    if missing:
        raise RuntimeError(f"parameters missing from a checkpoint: {missing[:5]}")
    dots = {(g, d): 0.0 for g in grads for d in ("recent", "cum")}
    sq = {"recent": 0.0, "cum": 0.0}
    for n in names:
        xt, xp, x0 = (ix[n].get_tensor(n).reshape(-1) for ix in (it, ip, i0))
        for a in range(0, xt.numel(), chunk):
            t = xt[a:a + chunk].double()
            dr, dc = t - xp[a:a + chunk].double(), t - x0[a:a + chunk].double()
            sq["recent"] += float((dr * dr).sum()); sq["cum"] += float((dc * dc).sum())
            for g, gd in grads.items():
                v = gd[n].reshape(-1)[a:a + chunk].double()
                dots[(g, "recent")] += float((v * dr).sum()); dots[(g, "cum")] += float((v * dc).sum())
            del t, dr, dc
        del xt, xp, x0
    return dots, sq


def step_record(eng, run: str, t: int, ck_t: str, ck_prev: str, cfg: dict, cfg4: dict, S: dict, check: bool = False) -> dict:
    import torch

    from tim.measure.heads import grad_dot, sampler_sequence_heads, rows_by_sequence
    p = pure.parse_run(run)
    with_4b = bool(cfg4.get("with_4b", True))         # Qwen3-1.7B: False (4a and 4c only)
    objs = pure.objectives_for(run, with_4b)
    base = cfg["base_model"]
    rows = cfg4["rows_4a"]
    n, cap = int(cfg4["n_samples"]), float(cfg4["tis_cap"])
    model = eng.trainer_model(ck_t, "fa3_liger")
    timing = {}

    t0 = time.time()
    torch.cuda.empty_cache()        # vLLM checks free memory when an engine is first built
    llm = eng.sampler(p["kind"], ck_t if p["sampler_at"] == "t" else base)
    b_run = generate(llm, S, rows, n, pure.SEED_RUN)
    heads = sampler_sequence_heads(llm, b_run["sequences"], b_run["prompt_len"], cfg["topk"])
    timing["sample_run"] = time.time() - t0; t0 = time.time()
    g_run, tok = gradients(model, b_run, heads, objs, cap=cap)
    timing["grad_run"] = time.time() - t0; t0 = time.time()
    torch.cuda.empty_cache()
    llm_on = eng.sampler("bf16", ck_t)
    b_on = generate(llm_on, S, rows, n, pure.SEED_ON)
    b_on2 = generate(llm_on, S, rows, n, pure.SEED_ON2)
    g_on = gradients(model, b_on, None, ["pg"])[0]["pg"]
    g_on2 = gradients(model, b_on2, None, ["pg"])[0]["pg"]
    timing["on_policy"] = time.time() - t0; t0 = time.time()

    G = {"c": g_run[objs[0]], "o": g_on, "o2": g_on2}
    gram = {(a, b): grad_dot(G[a], G[b]) for a, b in (("c", "c"), ("c", "o"), ("c", "o2"), ("o", "o"), ("o", "o2"),
                                                       ("o2", "o2"))}
    allg = {"c": G["c"], "o": g_on, "o2": g_on2, **{f"run_{g}": g_run[g] for g in objs[1:]}}
    dd, sq = delta_dots(allg, ck_t, ck_prev, base)
    rec = {"run": run, "step": t, "source": p["source"], "method": p["method"], "seed": p["seed"],
           "sampler": f"{p['kind']}@theta_{p['sampler_at']}", "objectives": objs,
           "batch": {"run_reward_mean": float(np.mean(b_run["reward"])), "on_reward_mean": float(np.mean(b_on["reward"])),
                     "on2_reward_mean": float(np.mean(b_on2["reward"])), "run_tokens": tok["n_tokens"],
                     "run_resp_len_mean": float(np.mean([len(s) - q for s, q in zip(b_run["sequences"], b_run["prompt_len"])]))},
           "4a": pure.bias_scalars(gram, {(x, d): dd[(x, d)] for x in ("c", "o", "o2") for d in ("recent", "cum")}, sq),
           "gram": {f"{a},{b}": v for (a, b), v in gram.items()},
           "proj": {f"{g}:{d}": pure.projection(v, sq[d]) for (g, d), v in dd.items()}}
    if p["method"] == "tis" and with_4b:
        rec["4b"] = {"tokens": pure.tis_token_stats(tok["log_ratio"], cap),
                     "proj": {f"{g}:{d}": pure.projection(dd[(k, d)], sq[d])
                              for g, k in (("tis", "c"), ("sc", "run_sc"), ("tis_low", "run_tis_low"), ("sc_low", "run_sc_low"))
                              for d in ("recent", "cum")},
                     "norms": {g: grad_dot(g_run[g], g_run[g]) ** 0.5 for g in objs}}
    else:
        rec["tokens"] = pure.tis_token_stats(tok["log_ratio"], cap) if tok["log_ratio"].size else None
    if check:                                           # check: the on-policy gradient repeats exactly
        # free the step's gradients first: only the Gram scalars are needed (a 1.7B fp32 gradient is 6.9 GB)
        del g_run, g_on, g_on2, G, allg
        torch.cuda.empty_cache()
        g_on_rep = gradients(model, generate(llm_on, S, rows, n, pure.SEED_ON), None, ["pg"])[0]["pg"]
        rec["check"] = {"repeat_rel_diff": abs(grad_dot(g_on_rep, g_on_rep) ** 0.5 - gram[("o", "o")] ** 0.5)
                        / max(gram[("o", "o")] ** 0.5, 1e-30)}
        del g_on_rep
    else:
        del g_run, g_on, g_on2, G, allg
    torch.cuda.empty_cache()
    timing["dots"] = time.time() - t0; t0 = time.time()

    # 4c: the run's sampler on prompts that change with t; lag / engine split of delta
    r4c = pure.rows_4c(t, len(S["table"]), int(cfg4["n_prompts_4c"]), rows)
    torch.cuda.empty_cache()
    llm = eng.sampler(p["kind"], ck_t if p["sampler_at"] == "t" else base)
    b4 = generate(llm, S, r4c, int(cfg4["n_samples_4c"]), pure.SEED_4C)
    h4 = sampler_sequence_heads(llm, b4["sequences"], b4["prompt_len"], cfg["topk"])
    lp_t = sampled_logprobs(model, b4["sequences"], b4["prompt_len"])
    if p["lagged"]:
        lp_old = sampled_logprobs(eng.trainer_model(base, "fa3_liger", slot="old"), b4["sequences"], b4["prompt_len"])
    else:
        lp_old = lp_t
    off = rows_by_sequence(b4["prompt_len"], [len(s) for s in b4["sequences"]])
    rec["4c"] = {**pure.split_means(lp_t, lp_old, h4["next"], off), "rows": r4c[:4] + ["..."]}
    timing["4c"] = time.time() - t0
    rec["timing_s"] = {k: round(v, 1) for k, v in timing.items()}
    return rec
