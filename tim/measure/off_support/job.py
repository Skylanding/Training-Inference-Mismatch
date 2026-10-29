"""Off-support measurement on a GPU: one job per run (`scripts/measure.py off-support --run <run>`, or --run
theta0_06b / theta0_17b / theta0_8b for the base model at step 0), the steps of configs/measure/off_support.json (0.6B 10,
20, 30, 40; 1.7B and 8B also 15), with --config the model's file. Per checkpoint theta (results/off_support/<run>/
step_<t>.json, written last; per-sequence arrays in step_<t>.npz beside it):

1. held-out accuracy: tim.measure.deviation.heldout_accuracy on the resident bf16 sampler refit to theta (64 held-out
   prompts x 8, T = 1, seed 0: the prompts, sampling and scoring of the deviation measurement).
   theta_0's comes from the theta0 job; run - ref and the split are formed in the summary.
2. own samples: the same 64 held-out prompts x n_own_per_prompt (4 -> 256 responses) from the bf16 sampler at theta with the
   training sampling parameters (T = 1, top_p = 1, top_k = -1, at most min(377, 512 - prompt) new tokens, the init's stop
   ids; seed SEED_OWN + j). log p_theta(y) and log p_theta0(y) and the rank of y under theta_0 with the trainers (HF bf16,
   kernel set fa3_liger, packed, as the loop measurement) -> pure.own_stats (D_own, length, truncation, < 64 tokens, off-support share, first
   token outside theta_0's top 20).
3. probe prefixes: theta_0's own samples, 64 held-out prompts x 8, seed 0, built by the rule of the probe set (0.6B: that set,
   data/probes/qwen3_0.6b/probe_set.npz; 1.7B and 8B: the same rule from their own theta_0,
   data/probes/theta0_qwen3_1.7b.npz and data/probes/theta0_qwen3_8b.npz, built on first use). Full-vocabulary KL(p_theta || p_theta0) per position, the stop-token probability
   under both, log p(y) under both -> pure.probe_stats (D_q, max_t KL_t, mean dP(EOS) over the first 64 positions, mu).
   1.7B also on the 0.6B theta_0 probe set, the set its deviation uses ("probe_s1").

GPU memory: the bf16 engine at gmu 0.15, two trainers, two micro-batches of logits.
"""
from __future__ import annotations

import time

import numpy as np

from tim.measure.split.job import sampled_logprobs
from tim.measure.off_support import pure


def own_samples(llm, cfg: dict, n_prompts: int, n: int, seed_base: int) -> dict:
    """n samples per held-out prompt with the training sampling parameters (module doc, item 2)."""
    import os
    from vllm import SamplingParams
    from tim.measure.config import ROOT, init_eos_ids
    from tim.measure.deviation import build_prompts
    TEMPLATE = cfg.get("template") or os.path.join(ROOT, "configs", "chat-templates", "qwen3.jinja")
    from tim.measure.prefixes import MAX_RESPONSE
    from tim.tasks.countdown import build_dataset, compute_reward
    heldout, _ = build_dataset(n_heldout=n_prompts)
    prompts = build_prompts(llm.get_tokenizer(), heldout, open(TEMPLATE).read())
    stop = init_eos_ids(cfg["base_model"])
    params = [SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=-1, stop_token_ids=stop, seed=seed_base + j,
                             max_tokens=min(MAX_RESPONSE, 512 - len(p))) for j, p in enumerate(prompts)]
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
    d = {"sequences": [], "prompt_len": [], "truncated": [], "reward": []}
    for h, p, o in zip(heldout, prompts, outs):
        for c in o.outputs:
            d["sequences"].append(list(p) + list(c.token_ids))
            d["prompt_len"].append(len(p))
            d["truncated"].append(c.finish_reason == "length")
            d["reward"].append(float(compute_reward(c.text, h["numbers"], h["target"])))
    return d


def logprobs_and_rank(model, seqs, plen, micro_batch: int = 16, device: str = "cuda") -> tuple:
    """log p(y) of every response token and the rank of y among the vocabulary (0 = argmax; ties counted in y's favour),
    at the model's weights, no gradient (tim.measure.split.job.sampled_logprobs' loop)."""
    import torch
    from verl.trainer.ppo.score_centering import topk_log_probs_from_logits

    from tim.measure.heads import forward_logits, micro_batches, rows_by_sequence
    off = rows_by_sequence(plen, [len(s) for s in seqs])
    lp, rk = np.zeros(int(off[-1])), np.zeros(int(off[-1]), dtype=np.int64)
    with torch.no_grad():
        for idx in micro_batches(model, seqs, micro_batch):
            per_seq = forward_logits(model, seqs, idx, device)
            for r, i in enumerate(idx):
                P, L = int(plen[i]), len(seqs[i])
                rows = per_seq[r][P - 1:L - 1]
                y = torch.as_tensor(np.asarray(seqs[i][P:L]), device=device)[:, None]
                lp[off[i]:off[i + 1]] = topk_log_probs_from_logits(rows, y).squeeze(-1).double().cpu().numpy()
                rk[off[i]:off[i + 1]] = (rows > rows.gather(-1, y)).sum(-1).cpu().numpy()
            del per_seq
    return lp, rk, off


def kl_eos(model_t, model_0, seqs, plen, stop_ids, micro_batch: int = 16, chunk: int = 2048, device: str = "cuda") -> tuple:
    """Per response row: KL(p_t || p_0) over the full vocabulary, p(stop tokens) under both, log p(y) under both
    (tim.measure.deviation.kl_run_vs_ref's loop: both trainers see the same micro-batches, fp32 log-softmax in chunks)."""
    import torch

    from tim.measure.heads import forward_logits, micro_batches, rows_by_sequence
    if getattr(model_t, "_tim_mode", None) != getattr(model_0, "_tim_mode", None):
        raise ValueError("the two trainers must share the forward layout")
    off = rows_by_sequence(plen, [len(s) for s in seqs])
    n = int(off[-1])
    kl, e_t, e_0, l_t, l_0 = (np.zeros(n) for _ in range(5))
    stop = torch.as_tensor(list(stop_ids), device=device)
    with torch.no_grad():
        for idx in micro_batches(model_t, seqs, micro_batch):
            at, a0 = forward_logits(model_t, seqs, idx, device), forward_logits(model_0, seqs, idx, device)
            for r, i in enumerate(idx):
                P, L = int(plen[i]), len(seqs[i])
                a, b = at[r][P - 1:L - 1], a0[r][P - 1:L - 1]
                y = torch.as_tensor(np.asarray(seqs[i][P:L]), device=device)[:, None]
                parts = []
                for c in range(0, a.shape[0], chunk):
                    la = torch.log_softmax(a[c:c + chunk].float(), -1)
                    lb = torch.log_softmax(b[c:c + chunk].float(), -1)
                    yy = y[c:c + chunk]
                    parts.append(torch.stack([(la.exp() * (la - lb)).sum(-1), la[:, stop].exp().sum(-1), lb[:, stop].exp().sum(-1),
                                              la.gather(-1, yy).squeeze(-1), lb.gather(-1, yy).squeeze(-1)], -1).double().cpu())
                    del la, lb
                x = torch.cat(parts).numpy()
                for k, arr in enumerate((kl, e_t, e_0, l_t, l_0)):
                    arr[off[i]:off[i + 1]] = x[:, k]
            del at, a0
    return kl, e_t, e_0, l_t, l_0, off


def theta0_probe(eng, cfg: dict, out_path: str, n_samples: int = 8, seed: int = 0) -> None:
    """The probe set's rule (scripts/measure.py build-probes, probe branch) on this process's base model: held-out rows
    0-63 x n_samples, T = 1, seed + j, at most min(377, 512 - prompt) new tokens; saved with tim.measure.artifacts."""
    import json
    import os

    import pyarrow.parquet as pq
    from vllm import SamplingParams

    from tim.measure.config import init_eos_ids
    from tim.measure.prefixes import MAX_RESPONSE, _render_prompts
    from tim.measure import artifacts
    from tim.tasks.registry import reward_fn
    llm = eng.sampler("bf16", cfg["base_model"])
    rows = list(range(64))
    table = pq.read_table(cfg["heldout_parquet"]).to_pylist()
    prompts = _render_prompts(llm.get_tokenizer(), [table[r]["prompt"] for r in rows], open(cfg["template"]).read())
    stop = init_eos_ids(cfg["base_model"])
    params = [SamplingParams(n=n_samples, temperature=1.0, top_p=1.0, top_k=-1, stop_token_ids=stop, seed=seed + j,
                             max_tokens=min(MAX_RESPONSE, 512 - len(p))) for j, p in enumerate(prompts)]
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
    reward = reward_fn(cfg.get("task", "countdown"))
    rolls = [(r, p, [list(c.token_ids) for c in o.outputs], [reward(c.text, table[r]["reward_model"]["ground_truth"])
                                                             for c in o.outputs]) for r, p, o in zip(rows, prompts, outs)]
    meta = {"set": "probe", "parquet": os.path.basename(cfg["heldout_parquet"]), "rows": rows, "n_samples": n_samples,
            "seed": seed, "model": cfg["base_model"], "max_response": MAX_RESPONSE, "temperature": 1.0,
            "rule": "scripts/measure.py build-probes (probe set rule) on this base model"}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + f".tmp{os.getpid()}.npz"
    artifacts.save(tmp, artifacts.from_rollouts(rolls), meta)
    os.replace(tmp, out_path)
    print(json.dumps({"built": out_path, "n": len(rolls) * n_samples}), flush=True)


def step_record(eng, run: str, t: int, ck: str, cfg: dict, c7: dict, probes: dict) -> tuple:
    """-> (record, per-sequence arrays). probes {name: tim.measure.artifacts set} (theta_0's samples)."""
    import torch

    from tim.measure.config import init_eos_ids
    from tim.measure.deviation import heldout_accuracy
    p = pure.parse_run(run)
    base = cfg["base_model"]
    timing, rec = {}, {"run": run, "step": t, "model": p["model"], "source": p["source"], "method": p["method"],
                       "seed": p["seed"], "ref": p["ref"], "theta0": p["theta0"]}
    t0 = time.time()
    torch.cuda.empty_cache()                            # vLLM checks free memory when an engine is first built
    llm = eng.sampler("bf16", ck)
    ho = c7["heldout"]
    rec["heldout"] = heldout_accuracy(llm, base, ho["n_prompts"], ho["n_samples"], ho["seed"], cfg=cfg)
    timing["heldout"] = time.time() - t0; t0 = time.time()
    own = own_samples(llm, cfg, ho["n_prompts"], int(c7["n_own_per_prompt"]), pure.SEED_OWN)
    timing["own_samples"] = time.time() - t0; t0 = time.time()

    m_t = eng.trainer_model(ck, "fa3_liger")
    m_0 = eng.trainer_model(base, "fa3_liger", slot="old")
    lp_t = sampled_logprobs(m_t, own["sequences"], own["prompt_len"])
    lp_0, rk_0, off = logprobs_and_rank(m_0, own["sequences"], own["prompt_len"])
    rec["own"] = pure.own_stats(lp_t, lp_0, rk_0, off, own["truncated"], own["reward"])
    arrays = {"own_d": pure.seq_sums(lp_t - lp_0, off), "own_len": np.diff(off),
              "own_d_offsupport": pure.seq_sums(np.where(lp_0 < np.log(pure.P0_FLOOR), lp_t - lp_0, 0.0), off),
              "own_first_out": pure.first_index(rk_0 >= pure.TOP_K0, off), "own_truncated": np.asarray(own["truncated"]),
              "own_reward": np.asarray(own["reward"])}
    timing["own_logprobs"] = time.time() - t0; t0 = time.time()

    stop = init_eos_ids(base)
    for name, s in probes.items():
        kl, e_t, e_0, l_t, l_0, poff = kl_eos(m_t, m_0, s["sequences"], s["prompt_len"], stop, int(c7["trainer_micro_batch"]))
        rec[name] = pure.probe_stats(kl, e_t, e_0, l_t, l_0, poff)
        rec[name]["probe_sha256"] = s["sha256"]
        arrays[f"{name}_sum_kl"], arrays[f"{name}_max_kl"] = pure.seq_sums(kl, poff), pure.seq_max(kl, poff)
        timing[name] = time.time() - t0; t0 = time.time()
    torch.cuda.empty_cache()
    rec["stop_ids"] = list(stop)
    rec["timing_s"] = {k: round(v, 1) for k, v in timing.items()}
    return rec, arrays
