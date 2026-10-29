"""Offline measurements of the paper on saved checkpoints (GPU, except `summarize`).

Every job reads one model config (configs/measure/qwen3_0.6b.json, qwen3_1.7b.json or qwen3_8b.json, --config) and the
checkpoints that scripts/train.sh writes under --ckpt-root (default $TIM_CKPT_ROOT, else outputs/):

    <ckpt-root>/<run>/ckpt/global_step_<t>/actor/huggingface      (step 0 is the base model of the config)

Run names follow scripts/train.sh: <prefix>_<source>_<method>_seed<k> with the internal source and method names
(README.md, "Run names"), e.g. s2_stale64_sc_seed0 (Qwen3-0.6B, stale sampler, score centering) or s2ref_bf16_pg_seed0 (the
on-policy reference). One process builds its engines once (vLLM samplers and HF trainers) and refits them in place to each
checkpoint; a step whose record exists is skipped, so an interrupted job resumes. Records are written last.

    python scripts/measure.py build-probes  --config configs/measure/qwen3_0.6b.json
    python scripts/measure.py replay        --config configs/measure/qwen3_0.6b.json --run s2ref_bf16_pg_seed0 --source stale
    python scripts/measure.py deviation     --config configs/measure/qwen3_0.6b.json --run s2_stale64_sc_seed0
    python scripts/measure.py split         --config configs/measure/qwen3_0.6b.json --run s2_stale64_sc_seed0
    python scripts/measure.py loop          --config configs/measure/qwen3_0.6b.json --run s2_stale64_sc_seed0
    python scripts/measure.py residual-bias-exact --config configs/measure/qwen3_0.6b.json --run s2_stale64_sc_seed0
    python scripts/measure.py off-support   --config configs/measure/qwen3_0.6b.json --run s2_stale64_sc_seed0
    python scripts/measure.py summarize deviation --config configs/measure/qwen3_0.6b.json

replay       each sampler (on_policy, quantized, stale, quantized_stale, eager) on the on-policy reference checkpoints at
             every step 0-40: the sampled-token error, its persistence, the ratio truncation, the per-sequence offset and
             its exact split into policy lag and numerical discrepancy (results/replay_0.6b)
deviation    full-vocabulary KL to the paired on-policy model on the probe prefixes and held-out accuracy, steps 5-40
             (results/deviation_<model>; divide by the noise-floor runs in the summary)
split        residual bias b (4a), ratio tail (4b) and per-token split (4c) on regenerated batches (results/split_<model>)
loop         the residual-bias prediction from the lag weights and the lag by advantage sign (results/loop_<model>)
residual-bias-exact   the same prediction with linearised and exact weights (results/residual_bias_exact_0.6b)
off-support  held-out accuracy, sequence KL of the trainer's own responses to theta_0 and its off-support share
             (results/off_support); --run theta0_06b / theta0_17b / theta0_8b measures the base model at step 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tim.measure import config as mconfig  # noqa: E402

# replay sources: CLI name -> the internal name stored in the records (results/replay_0.6b steps.csv `source` column)
SOURCE_NAMES = {"on_policy": "bf16", "quantized": "fp8", "stale": "stale", "quantized_stale": "fp8_stale", "eager": "eager"}
# internal replay source -> sampler engine kind; stale sources hold theta_0
SAMPLER_KIND = {"bf16": "bf16", "fp8": "fp8", "stale": "bf16", "fp8_stale": "fp8", "eager": "bf16_eager"}
STALE_SOURCES = ("stale", "fp8_stale")
SUMMARIES = ("replay", "deviation", "split", "loop", "residual-bias-exact", "off-support")


def say(msg: str) -> None:
    print(f"[measure {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_steps(spec) -> list:
    if isinstance(spec, list):
        return sorted(set(int(x) for x in spec))
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


class Ctx:
    """Command-line settings shared by the jobs."""

    def __init__(self, a):
        self.cfg = mconfig.load()
        self.raw = mconfig.load(resolve=False)
        self.ckpt_root = Path(a.ckpt_root).resolve()
        self.results = Path(a.results).resolve() if getattr(a, "results", None) else None
        self.work = Path(a.work).resolve()
        self.steps = getattr(a, "steps", None)
        self.probe = mconfig.repo_path(self.raw["probe_set"])
        self.p2 = mconfig.repo_path(self.raw["p2_batch"])

    def ckpt(self, run: str, step: int) -> str:
        """theta_0 = the base model; theta_t = the run's saved hf_model directory. The engines are built from the base
        model and only load the checkpoint's weights, so the checkpoint's own tokenizer and generation files do not matter."""
        if step == 0:
            return self.cfg["base_model"]
        d = self.ckpt_root / run / "ckpt" / f"global_step_{step}" / "actor" / "huggingface"
        if not any(d.glob("*.safetensors")):
            raise FileNotFoundError(f"{d}: no checkpoint (train with scripts/train.sh, OUT_DIR={self.ckpt_root})")
        return str(d)


# ---------------------------------------------------------------------------------------------------------- build
def build_probes(ctx: Ctx, eng) -> None:
    """The fixed sets from theta_0 with the bf16 sampler refit to the base model; sha256 next to them in artifacts.json:
    the probe set (64 held-out prompts x 8 rollouts, seed 0) and the P2 batch (the 64 fixed training rows x 8, seed 1000)."""
    import pyarrow.parquet as pq
    from vllm import SamplingParams

    from tim.measure import artifacts
    from tim.measure.config import init_eos_ids
    from tim.measure.prefixes import MAX_RESPONSE, _render_prompts
    from tim.tasks.registry import reward_fn
    cfg = ctx.cfg
    reward = reward_fn("countdown")
    max_response, model_len = int(cfg.get("max_response", MAX_RESPONSE)), int(cfg.get("max_model_len", 512))
    out = {}
    for name, path, parquet, rows, sec in (
            ("probe", ctx.probe, cfg["heldout_parquet"], list(range(64)), cfg["probe"]),
            ("p2", ctx.p2, cfg["train_parquet"], list(cfg["p2"]["rows"]), cfg["p2"])):
        if not path.exists():
            llm = eng.sampler("bf16", cfg["base_model"])
            table = pq.read_table(parquet).to_pylist()
            prompts = _render_prompts(llm.get_tokenizer(), [table[r]["prompt"] for r in rows], open(cfg["template"]).read())
            stop = init_eos_ids(cfg["base_model"])
            params = [SamplingParams(n=sec["n_samples"], temperature=1.0, top_p=1.0, top_k=-1, stop_token_ids=stop,
                                     seed=sec["seed"] + j, max_tokens=min(max_response, model_len - len(p)))
                      for j, p in enumerate(prompts)]
            outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
            rolls = []
            for r, p, o in zip(rows, prompts, outs):
                gt = table[r]["reward_model"]["ground_truth"]
                rolls.append((r, p, [list(c.token_ids) for c in o.outputs], [reward(c.text, gt) for c in o.outputs]))
            s = artifacts.from_rollouts(rolls)
            meta = {"set": name, "parquet": os.path.basename(parquet), "rows": rows, "n_samples": sec["n_samples"],
                    "seed": sec["seed"], "model": os.path.basename(str(cfg["base_model"]).rstrip("/")),
                    "max_response": max_response, "temperature": 1.0}
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
            artifacts.save(str(tmp), s, meta)
            os.replace(tmp, path)
            rl = np.array([len(x) - int(p) for x, p in zip(s["sequences"], s["prompt_len"])])
            say(f"{name}: {len(s['sequences'])} sequences, {int(rl.sum())} response positions, mean reward "
                f"{np.mean(s['reward']):.3f}; response length mean {rl.mean():.0f} max {rl.max()}")
        out[name] = artifacts.sha256(str(path))
    atomic_json(ctx.probe.parent / "artifacts.json", {"probe_set": {"path": ctx.probe.name, "sha256": out["probe"]},
                                                      "p2_batch": {"path": ctx.p2.name, "sha256": out["p2"]}})


def need_sets(ctx: Ctx) -> None:
    for p in (ctx.probe, ctx.p2):
        if not p.exists():
            raise FileNotFoundError(f"{p}: run `scripts/measure.py build-probes --config <the qwen3_0.6b config>` first")


# --------------------------------------------------------------------------------------------------------- replay
def theta0_cache(ctx: Ctx, eng, probe, p2, kernels: str, kind: str = "bf16") -> Path:
    """Sampler (kind bf16: stale; fp8: quantized and stale) and trainer at theta_0 on both sets: the stale samplers, and the
    trainer at the sampler's weights for the exact split (S4)."""
    from tim.measure import heads
    from tim.measure.formulas import repeat_diff
    cfg = ctx.cfg
    c = ctx.work / (f"theta0_{kernels}" if kind == "bf16" else f"theta0_{kernels}_{kind}")
    if (c / "sampler_repeat.json").exists():
        return c
    k, chunk = cfg["topk"], cfg["sampler_chunk"]
    llm = eng.sampler(kind, cfg["base_model"])
    s1 = heads.sampler_sequence_heads(llm, probe["sequences"], probe["prompt_len"], k, chunk)
    s2 = heads.sampler_sequence_heads(llm, probe["sequences"], probe["prompt_len"], k, chunk)
    s3 = heads.sampler_sequence_heads(llm, probe["sequences"], probe["prompt_len"], k, 1)
    sp = heads.sampler_sequence_heads(llm, p2["sequences"], p2["prompt_len"], k, chunk)
    t1 = heads.trainer_sequence_heads(eng.trainer_model(cfg["base_model"], kernels), probe["sequences"], probe["prompt_len"],
                                      s1["ids"], cfg["trainer_micro_batch"])
    atomic_npz(c / "probe_sampler.npz", ids=s1["ids"], lp=s1["lp"], next=s1["next"])
    atomic_npz(c / "p2_sampler.npz", ids=sp["ids"], lp=sp["lp"], next=sp["next"])
    atomic_npz(c / "probe_trainer.npz", lp=t1["lp"], next=t1["next"])
    atomic_json(c / "sampler_repeat.json", {
        "repeat_same_batch": repeat_diff(s1["ids"], s1["lp"], s1["next"], s2["ids"], s2["lp"], s2["next"]),
        "batch1_vs_batch64": repeat_diff(s1["ids"], s1["lp"], s1["next"], s3["ids"], s3["lp"], s3["next"]),
        "probe_sha256": probe["sha256"], "p2_sha256": p2["sha256"]})
    return c


def replay_step(ctx: Ctx, eng, run: str, src: str, step: int, ckpt: str, probe, p2, root: Path, anchor_every: int) -> dict:
    """One replay step: the sampler `src` and the trainer at the checkpoint on the probe set (and the P2 batch for the
    ratio statistics); S1 lags against the anchors before this step; no gradients."""
    import torch

    from tim.measure import heads
    from tim.measure.engines import KV_BLOCKS, kernels_for_run
    from tim.measure.formulas import dist_stats, eps_stats, ratio_stats, repeat_diff, seq_sums, stale_split, truncation_stats
    cfg = ctx.cfg
    k, chunk = cfg["topk"], cfg["sampler_chunk"]
    kernels = kernels_for_run(run)
    kind, stale = SAMPLER_KIND[src], src in STALE_SOURCES
    res = {"run": run, "source": src, "step": step, "ckpt": f"{run}/global_step_{step}" if step else "theta_0",
           "sampler": kind, "trainer_kernels": kernels,
           "sampler_engine": f"training-matched (engines.TRAIN_ENGINE), KV blocks {KV_BLOCKS}"
                             + (", enforce_eager" if kind == "bf16_eager" else ""),
           "sampler_weights": "theta_0" if stale else "theta_t", "gradients": False,
           "probe_sha256": probe["sha256"], "p2_sha256": p2["sha256"]}
    t0 = time.time()
    if stale:
        c = theta0_cache(ctx, eng, probe, p2, kernels, kind)
        z = {n: np.load(c / f"{n}.npz") for n in ("probe_sampler", "p2_sampler", "probe_trainer")}
        s1 = {x: z["probe_sampler"][x] for x in ("ids", "lp", "next")}
        sp = {x: z["p2_sampler"][x] for x in ("ids", "lp", "next")}
        res["sampler_repeat"] = json.loads((c / "sampler_repeat.json").read_text())
        res["sampler_repeat"]["note"] = "theta_0 sampler, the same for every step"
    else:
        llm = eng.sampler(kind, ckpt)
        s1 = heads.sampler_sequence_heads(llm, probe["sequences"], probe["prompt_len"], k, chunk)
        s2 = heads.sampler_sequence_heads(llm, probe["sequences"], probe["prompt_len"], k, chunk)
        s3 = heads.sampler_sequence_heads(llm, probe["sequences"], probe["prompt_len"], k, 1)
        sp = heads.sampler_sequence_heads(llm, p2["sequences"], p2["prompt_len"], k, chunk)
        res["sampler_repeat"] = {
            "repeat_same_batch": repeat_diff(s1["ids"], s1["lp"], s1["next"], s2["ids"], s2["lp"], s2["next"]),
            "batch1_vs_batch64": repeat_diff(s1["ids"], s1["lp"], s1["next"], s3["ids"], s3["lp"], s3["next"])}
    res["seconds_sampler"] = round(time.time() - t0, 1)
    model = eng.trainer_model(ckpt, kernels)
    t1 = heads.trainer_sequence_heads(model, probe["sequences"], probe["prompt_len"], s1["ids"], cfg["trainer_micro_batch"])
    t2 = heads.trainer_sequence_heads(model, probe["sequences"], probe["prompt_len"], s1["ids"], cfg["trainer_micro_batch"])
    res["trainer_repeat"] = repeat_diff(s1["ids"], t1["lp"], t1["next"], s1["ids"], t2["lp"], t2["next"])
    res["eps"] = eps_stats(s1["lp"], t1["lp"], s1["next"], t1["next"])
    if stale:
        res["stale_split_head"] = stale_split(t1["lp"], z["probe_trainer"]["lp"], s1["lp"])
        res["stale_split_next"] = stale_split(t1["next"], z["probe_trainer"]["next"], s1["next"])
    # S2-S4 on the probe set (delta = log p - log q of the sampled token, rows in sequence order)
    off = heads.rows_by_sequence(probe["prompt_len"], [len(x) for x in probe["sequences"]])
    delta = np.asarray(t1["next"], dtype=np.float64) - s1["next"]
    p_old = z["probe_trainer"]["next"] if stale else t1["next"]       # trainer at the sampler's weights
    res["S2_probe"] = truncation_stats(delta, off, cfg["tis_cap"])
    res["S2_probe"]["tail_mass_mean"] = res["eps"]["q_tail_mass_mean"]
    res["S3"] = {"seq_sum_delta": dist_stats(seq_sums(delta, off))}
    res["S4"] = {"seq_sum_lag": dist_stats(seq_sums(np.asarray(t1["next"], dtype=np.float64) - p_old, off)),
                 "seq_sum_engine": dist_stats(seq_sums(np.asarray(p_old, dtype=np.float64) - s1["next"], off)),
                 "p_old": "trainer at theta_0" if stale else "trainer at theta_t"}
    # the importance ratio on the P2 batch: one trainer forward, no gradients
    tp = heads.trainer_sequence_heads(model, p2["sequences"], p2["prompt_len"], sp["ids"], cfg["trainer_micro_batch"])
    res["ratio"] = ratio_stats(np.asarray(tp["next"], dtype=np.float64) - sp["next"], cfg["tis_cap"])
    # S1: pairs of this step with the anchors before it; then this step's own anchor file; old anchors are deleted
    eps_now, delta_now = t1["lp"] - s1["lp"], t1["next"] - s1["next"]
    res["lags"] = online_lags(ctx, root, run, src, step, s1["ids"], eps_now, delta_now, anchor_every)
    jd = ctx.work / "anchors" / run / src
    if step % anchor_every == 0 and step < max(ctx.steps):
        atomic_npz(jd / f"step_{step}.npz", q_ids=s1["ids"], q_lp=s1["lp"], q_next=s1["next"], p_lp=t1["lp"],
                   p_next=t1["next"])
    last_needed = step - max(cfg["lags"])
    for f in jd.glob("step_*.npz"):
        if int(f.stem.split("_")[1]) <= last_needed:
            f.unlink(missing_ok=True)
    torch.cuda.empty_cache()
    atomic_json(root / run / src / f"step_{step}.json", res)
    return res


def online_lags(ctx: Ctx, root: Path, run: str, src: str, step: int, ids, eps, delta, anchor_every: int) -> dict:
    """Pairs (t, step) with t = step - l an anchor, l in the configured lags; a missing anchor is listed under `missing`."""
    from tim.measure.formulas import eps_lag
    jd = ctx.work / "anchors" / run / src
    out = {"anchor_every": anchor_every, "eps": [], "grad": [], "missing": []}
    for l in ctx.cfg["lags"]:
        t = step - l
        if t < 0 or t % anchor_every:
            continue
        arr = jd / f"step_{t}.npz"
        if not ((root / run / src / f"step_{t}.json").exists() and arr.exists()):
            out["missing"].append({"kind": "eps", "lag": l, "u": t})
            continue
        z = np.load(arr)
        e_t = z["p_lp"].astype(np.float64) - z["q_lp"]
        d_t = z["p_next"].astype(np.float64) - z["q_next"]
        out["eps"].append({"t": t, "u": step, "lag": l, **eps_lag(z["q_ids"], e_t, d_t, ids, eps, delta)})
    return out


def replay_job(ctx: Ctx, eng, run: str, src: str) -> list:
    from tim.measure import artifacts
    need_sets(ctx)
    probe, p2 = artifacts.load(str(ctx.probe)), artifacts.load(str(ctx.p2))
    blk = ctx.cfg["replay"]
    root = ctx.results or mconfig.repo_path(blk["results_dir"])
    ctx.steps = parse_steps(ctx.steps or blk["steps"])
    failed = []
    for step in ctx.steps:
        if (root / run / src / f"step_{step}.json").exists():
            continue
        t0 = time.time()
        try:
            r = replay_step(ctx, eng, run, src, step, ctx.ckpt(run, step), probe, p2, root, int(blk["anchor_every"]))
            say(f"replay {run}:{src} step {step}: {time.time() - t0:.0f}s  eps_abs {r['eps']['eps_abs_mean']:.4g}  "
                f"seq sum delta {r['S3']['seq_sum_delta']['mean']:+.3f}  lag pairs {len(r['lags']['eps'])}")
        except Exception:
            failed.append(step)
            say(f"replay {run}:{src} step {step} FAILED\n{traceback.format_exc()}")
    return failed


# ------------------------------------------------------------------------------------------------------ deviation
def deviation_job(ctx: Ctx, eng, run: str) -> list:
    """KL(pi_run || pi_ref) on the probe prefixes and held-out accuracy of run and reference at the run's saved steps.
    results/deviation_<model>/<run>/step_<t>.json; the reference's held-out results are cached in _ref/<ref>/step_<t>.json."""
    import torch

    from tim.measure import artifacts
    from tim.measure.deviation import heldout_accuracy, kl_run_vs_ref, kl_stats
    cfg = ctx.cfg
    blk = cfg["deviation"]
    root = ctx.results or mconfig.repo_path(blk["results_dir"])
    seed = run.rsplit("_seed", 1)[1]
    ref = blk["reference"].format(seed=seed)
    kernels = "fa3_liger"
    if blk.get("build_probe") and not ctx.probe.exists():
        from tim.measure.off_support import job as offjob
        say(f"deviation: building theta_0's probe set {ctx.probe}")
        offjob.theta0_probe(eng, cfg, str(ctx.probe))
    if not ctx.probe.exists():
        need_sets(ctx)
    probe = artifacts.load(str(ctx.probe))
    failed = []
    for step in parse_steps(ctx.steps or blk["steps"]):
        out = root / run / f"step_{step}.json"
        if out.exists():
            continue
        t0 = time.time()
        try:
            ck_run, ck_ref = ctx.ckpt(run, step), ctx.ckpt(ref, step)
            m_run = eng.trainer_model(ck_run, kernels, slot="run")
            m_ref = eng.trainer_model(ck_ref, kernels, slot="ref")
            kl, nd, off = kl_run_vs_ref(m_run, m_ref, probe["sequences"], probe["prompt_len"], blk["trainer_micro_batch"])
            res = {"run": run, "reference": ref, "step": step, "trainer_kernels": kernels, "probe_sha256": probe["sha256"],
                   "kl": kl_stats(kl, off, nd)}
            ho = blk["heldout"]
            del kl
            torch.cuda.empty_cache()                 # hand the KL step's logits back before vLLM checks free memory
            res["heldout_run"] = heldout_accuracy(eng.sampler("bf16", ck_run), cfg["base_model"], ho["n_prompts"],
                                                  ho["n_samples"], ho["seed"], cfg=cfg)
            ref_cache = root / "_ref" / ref / f"step_{step}.json"
            if ref_cache.exists():
                res["heldout_ref"] = json.loads(ref_cache.read_text())
            else:
                res["heldout_ref"] = heldout_accuracy(eng.sampler("bf16", ck_ref), cfg["base_model"], ho["n_prompts"],
                                                      ho["n_samples"], ho["seed"], cfg=cfg)
                atomic_json(ref_cache, res["heldout_ref"])
            res["heldout_acc_diff"] = res["heldout_run"]["mean_accuracy"] - res["heldout_ref"]["mean_accuracy"]
            res["seconds"] = round(time.time() - t0, 1)
            atomic_json(out, res)
            say(f"deviation {run} step {step}: KL {res['kl']['kl_mean']:.4g}  held-out diff {res['heldout_acc_diff']:+.3f}  "
                f"{res['seconds']:.0f}s")
        except Exception:
            failed.append(step)
            say(f"deviation {run} step {step} FAILED\n{traceback.format_exc()}")
    return failed


# ---------------------------------------------------------------------------------------------------------- split
def split_job(ctx: Ctx, eng, run: str, check: bool = False) -> list:
    """4a / 4b / 4c on one run at the steps of the config's split block (each step also needs the checkpoint 5 steps
    before it). check: the first step only, with the on-policy gradient repeated (results under <results>_check)."""
    from tim.measure.split import job as sj
    cfg = ctx.cfg
    blk = cfg["split"]
    root = ctx.results or mconfig.repo_path(blk["results_dir"])
    if check:
        root = root.with_name(root.name + "_check")
    S = sj._setup(cfg)
    failed = []
    steps = parse_steps(ctx.steps or blk["steps"])
    for t in steps[:1] if check else steps:
        out = root / run / f"step_{t}.json"
        if out.exists():
            continue
        t0 = time.time()
        try:
            rec = sj.step_record(eng, run, t, ctx.ckpt(run, t), ctx.ckpt(run, t - 5), cfg, blk, S, check=check)
            rec["seconds"] = round(time.time() - t0, 1)
            atomic_json(out, rec)
            a = rec["4a"]
            say(f"split {run} step {t}: |b| {a['b_norm']:.4g} noise {a['noise_norm']:.4g} | 4c lag {rec['4c']['lag_mean']:+.4f} "
                f"engine {rec['4c']['engine_mean']:+.4f}  {rec['seconds']:.0f}s")
        except Exception:
            failed.append(t)
            say(f"split {run} step {t} FAILED\n{traceback.format_exc()}")
    return failed


# ----------------------------------------------------------------------------------------------------------- loop
def loop_config() -> dict:
    return json.loads((ROOT / "configs" / "measure" / "loop.json").read_text())


def loop_job(ctx: Ctx, eng, run: str, exact: bool = False) -> list:
    """The loop measurement (6a1 + 6a2) or, with exact, the linearised and exact residual-bias prediction, on one stale
    run (theta_old = theta_0) at the steps of configs/measure/loop.json."""
    from tim.measure.loop import job as lj
    from tim.measure.off_support import pure as p7
    from tim.measure.split import pure as p4
    cfg = ctx.cfg
    c6 = loop_config()
    prefix = p4.parse_run(run)["prefix"]
    model = {"s2": "06b", "s5": "17b"}.get(prefix)
    if model is None or model != p7.model_of_base(cfg["base_model"]):
        raise ValueError(f"{run} does not match the base model of --config ({cfg['base_model']})")
    if exact and prefix != "s2":
        raise ValueError("residual-bias-exact runs on the Qwen3-0.6B s2_* runs")
    root = ctx.results or mconfig.repo_path(c6["exact"]["results_dir"] if exact else c6["results_dir"][prefix])
    S = lj._setup(cfg)
    failed = []
    for t in parse_steps(ctx.steps or c6["steps"]):
        out = root / run / f"step_{t}.json"
        if out.exists():
            continue
        t0 = time.time()
        try:
            ck_t = ctx.ckpt(run, t)
            rec = lj.step_record_exact(eng, run, t, ck_t, cfg, c6, S) if exact else lj.step_record(eng, run, t, ck_t, cfg, c6, S)
            rec["seconds"] = round(time.time() - t0, 1)
            atomic_json(out, rec)
            say(f"{'residual-bias-exact' if exact else 'loop'} {run} step {t}: {rec['seconds']:.0f}s {rec.get('timing_s')}")
        except Exception:
            failed.append(t)
            say(f"loop {run} step {t} FAILED\n{traceback.format_exc()}")
    return failed


# ---------------------------------------------------------------------------------------------------- off-support
def off_support_config() -> dict:
    return json.loads((ROOT / "configs" / "measure" / "off_support.json").read_text())


def off_support_job(ctx: Ctx, eng, run: str) -> list:
    from tim.measure import artifacts
    from tim.measure.off_support import job as oj
    from tim.measure.off_support import pure as p7
    cfg = ctx.cfg
    c7 = off_support_config()
    root = ctx.results or mconfig.repo_path(c7["results_dir"])
    p = p7.parse_run(run)
    if p["model"] != p7.model_of_base(cfg["base_model"]):
        raise ValueError(f"{run} does not match the base model of --config ({cfg['base_model']})")
    probes = {}
    for name, rel in c7["probes"][p["model"]].items():
        path = mconfig.repo_path(rel)
        if not path.exists():
            say(f"off-support: building theta_0's probe set {path}")
            oj.theta0_probe(eng, cfg, str(path))
        probes[name] = artifacts.load(str(path))
    steps = [0] if p["theta0"] else parse_steps(ctx.steps or c7["steps"][p["model"]])
    failed = []
    for t in steps:
        out = root / run / f"step_{t}.json"
        if out.exists():
            continue
        t0 = time.time()
        try:
            ck = cfg["base_model"] if p["theta0"] else ctx.ckpt(run, t)
            rec, arrays = oj.step_record(eng, run, t, ck, cfg, c7, probes)
            rec["seconds"] = round(time.time() - t0, 1)
            atomic_npz(root / run / f"step_{t}.npz", **arrays)
            atomic_json(out, rec)
            o, q = rec["own"], rec["probe"]
            say(f"off-support {run} step {t}: held-out {rec['heldout']['mean_accuracy']:.3f}  D_own {o['d_own']:.4g} "
                f"(off-support {o['share_offsupport']:.3f})  D_q {q['d_q']:.4g}  {rec['seconds']:.0f}s")
        except Exception:
            failed.append(t)
            say(f"off-support {run} step {t} FAILED\n{traceback.format_exc()}")
    return failed


# ------------------------------------------------------------------------------------------------------ summarize
def _has_records(root) -> bool:
    """A results directory holds per-step records (<run>/step_<t>.json); the shipped results hold only the tables."""
    return root is not None and Path(root).is_dir() and any(Path(root).glob("*/*/step_*.json")) or \
        (root is not None and Path(root).is_dir() and any(Path(root).glob("*/step_*.json")))


def summarize(what: str, a) -> None:
    lines = _summarize(what, a)
    for line in lines:
        print(line)


def _summarize(what: str, a) -> list:
    raw = mconfig.load(resolve=False)
    res = Path(a.results).resolve() if a.results else None
    if what == "replay":
        from tim.measure.replay_summary import summarize as f
        root = res or mconfig.repo_path(raw["replay"]["results_dir"])
        if not _has_records(root):
            return [f"no per-step records under {root}; nothing written"]
        lines = f(root, raw)
    elif what == "deviation":
        from tim.measure.deviation_summary import summarize as f
        root = res or mconfig.repo_path(raw["deviation"]["results_dir"])
        name = os.path.basename(str(raw["base_model"]).rstrip("/")).replace("Qwen__", "")
        if not _has_records(root):
            return [f"no per-step records under {root}; nothing written"]
        lines = f(root, raw["deviation"], title=f"Deviation tables, {name}", command="summarize deviation",
                  probe="probe prefixes of theta_0")
    elif what == "split":
        from tim.measure.split.summary import summarize as f
        root = res or mconfig.repo_path(raw["split"]["results_dir"])
        name = os.path.basename(str(raw["base_model"]).rstrip("/"))
        if not _has_records(root):
            return [f"no per-step records under {root}; nothing written"]
        lines = f(root, raw["split"], title=f"Split and residual bias tables, {name}")
    elif what == "loop":
        from tim.measure.loop.summary import summarize as f
        c6 = loop_config()
        roots = [res] if res else [mconfig.repo_path(d) for d in dict.fromkeys(c6["results_dir"].values())]
        lines = [x for root in roots if _has_records(root) for x in f(root, c6)] or ["no per-step records; nothing written"]
    elif what == "residual-bias-exact":
        from tim.measure.loop.summary import summarize_exact as f
        c6 = loop_config()
        root = res or mconfig.repo_path(c6["exact"]["results_dir"])
        if not _has_records(root):
            return [f"no per-step records under {root}; nothing written"]
        lines = f(root, c6["steps"])
    elif what == "off-support":
        from tim.measure.off_support.summary import summarize as f
        c7 = off_support_config()
        root = res or mconfig.repo_path(c7["results_dir"])
        if not _has_records(root):
            return [f"no per-step records under {root}; nothing written"]
        lines = f(root, c7, ROOT)
    else:
        raise ValueError(what)
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, run=True):
        p.add_argument("--config", default="configs/measure/qwen3_0.6b.json", help="the model's measurement config")
        p.add_argument("--ckpt-root", default=os.environ.get("TIM_CKPT_ROOT", str(ROOT / "outputs")),
                       help="OUT_DIR of scripts/train.sh: <root>/<run>/ckpt/global_step_<t>/actor/huggingface")
        p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for this process")
        p.add_argument("--results", default=None, help="results directory (default: the config's)")
        p.add_argument("--work", default=os.environ.get("TIM_MEASURE_WORK", str(ROOT / "outputs" / ".measure_work")),
                       help="scratch directory (theta_0 caches, replay anchors)")
        p.add_argument("--steps", default=None, help="steps, e.g. 0-40 or 5,10 (default: the config's)")
        if run:
            p.add_argument("--run", required=True, action="append", help="run name; repeat for several runs")
    common(sub.add_parser("build-probes", help="build the probe set and the P2 batch from theta_0"), run=False)
    p = sub.add_parser("replay", help="replay the samplers on the on-policy reference checkpoints")
    common(p)
    p.add_argument("--source", required=True, action="append", choices=sorted(set(SOURCE_NAMES) | set(SAMPLER_KIND)))
    common(sub.add_parser("deviation", help="KL to the paired on-policy model and held-out accuracy"))
    p = sub.add_parser("split", help="residual bias (4a), ratio tail (4b) and exact per-token split (4c)")
    common(p)
    p.add_argument("--check", action="store_true", help="first step only, with a repeat of the on-policy gradient")
    common(sub.add_parser("loop", help="residual-bias prediction from the lag weights and the lag by advantage sign"))
    common(sub.add_parser("residual-bias-exact", help="linearised and exact residual-bias prediction (Qwen3-0.6B)"))
    common(sub.add_parser("off-support", help="held-out accuracy, D_own and its off-support share, D_q"))
    s = sub.add_parser("summarize", help="tables (TABLES.md, steps.csv) from the per-step records, CPU only")
    s.add_argument("what", choices=SUMMARIES)
    s.add_argument("--config", default="configs/measure/qwen3_0.6b.json")
    s.add_argument("--results", default=None)
    a = ap.parse_args(argv)

    os.environ[mconfig.ENV] = str(mconfig.repo_path(a.config))
    if a.cmd == "summarize":
        summarize(a.what, a)
        return
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)       # before any CUDA library is imported
    if a.cmd != "replay" and not (a.cmd == "deviation" and "0.6b" in a.config):
        # several objectives keep their graphs: a quarter of the training token budget per micro-batch
        os.environ.setdefault("TIM_MEASURE_PACK_MAX_TOKENS", "4096")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ctx = Ctx(a)
    from tim.measure import engines
    if ctx.cfg.get("sampler"):
        engines.configure(**ctx.cfg["sampler"])
    runs = getattr(a, "run", None) or []
    engines.setup_kernel_env(["fa3_liger"] if a.cmd != "replay" else [engines.kernels_for_run(r) for r in runs])
    eng = engines.shared(ctx.cfg["base_model"], ctx.cfg["topk"])
    failed = {}
    if a.cmd == "build-probes":
        build_probes(ctx, eng)
    for run in runs:
        if a.cmd == "replay":
            for src in a.source:
                failed[f"{run}:{src}"] = replay_job(ctx, eng, run, SOURCE_NAMES.get(src, src))
        elif a.cmd == "deviation":
            failed[run] = deviation_job(ctx, eng, run)
        elif a.cmd == "split":
            failed[run] = split_job(ctx, eng, run, check=a.check)
        elif a.cmd in ("loop", "residual-bias-exact"):
            failed[run] = loop_job(ctx, eng, run, exact=a.cmd == "residual-bias-exact")
        elif a.cmd == "off-support":
            failed[run] = off_support_job(ctx, eng, run)
    bad = {k: v for k, v in failed.items() if v}
    if bad:
        say(f"failed steps (rerun the same command; finished steps are skipped): {bad}")
        sys.exit(1)
    say("done")


if __name__ == "__main__":
    main()
