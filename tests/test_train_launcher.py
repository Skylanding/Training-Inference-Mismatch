"""The method / source matrix of scripts/train.sh (DRY=1 with a stand-in python): hydra overrides, hook modules and the
TIM_* settings each paper configuration passes to verl. Needs bash only."""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FAKE_PY = """#!/bin/sh
if [ "$1" = "-c" ]; then
  case "$2" in *snapshot_download*) echo /tmp/model ;; *init_eos_ids*) echo "[151645,151643]" ;; esac
  exit 0
fi
echo "PY $*"
"""
ENV_KEYS = ("TIM_SYNC_MODE", "TIM_STALENESS", "TIM_CHI2_BUDGET", "TIM_SPLIT_OLD", "TIM_SPLIT_ENGINE", "TIM_BASELINE",
            "TIM_ASYNC", "TIM_ASYNC_OLD", "TIM_ASYNC_ESS_MIN", "VERL_USE_EXTERNAL_MODULES")


def run(tmp_path, method, source, **extra):
    py = tmp_path / "py"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "train.parquet").write_text("")
    # print the exported settings after the (dry) command; the launcher's own `exec`-free tail makes this possible
    probe = "; ".join(f'echo "{k}=${{{k}:-}}"' for k in ENV_KEYS)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "METHOD": method, "SOURCE": source, "DRY": "1",
           "PY": str(py), "OUT_DIR": str(tmp_path / "out"), "TRAIN_FILE": str(data / "train.parquet"),
           "N_GPUS": "2" if source == "async" else "1", **{k: str(v) for k, v in extra.items()}}
    script = f"source {shlex.quote(str(REPO / 'scripts' / 'train.sh'))}; {probe}"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    args = [ln for ln in r.stdout.splitlines() if "=" in ln and not ln.startswith(ENV_KEYS)]
    got = dict(ln.split("=", 1) for ln in r.stdout.splitlines() if ln.startswith(ENV_KEYS))
    return r, args, got


def arg(args, key):
    vals = [a.split("=", 1)[1] for a in args if a.split("=", 1)[0].lstrip("+") == key]
    return vals[-1] if vals else None


@pytest.mark.parametrize("method,source,internal", [
    ("on_policy", "none", "s2_bf16_pg_seed0"), ("pg", "quantized", "s2_fp8_pg_seed0"),
    ("ppo", "stale", "s2_stale64_ppo_seed0"), ("ppo_noclip", "stale", "s2_stale64_pg4_seed0"),
    ("sc", "quantized_stale", "s2_fp8_stale64_sc_seed0"), ("is", "stale", "s2_stale64_tis_seed0"),
    ("is_sc", "stale", "s2_stale64_tis_sc_seed0"), ("is_resync", "stale", "s2_budget_tis_seed0"),
    ("m2po", "stale", "s2_stale64_m2po_seed0"), ("vcpo", "stale", "s2_stale64_vcpo_seed0"),
    ("dlc", "stale", "s2_budget_split_seed0"), ("dlc", "quantized_stale", "s2_fp8_budget_split_seed0"),
    ("dlc_noresync", "stale", "s2_stale64_split_seed0"), ("dlc_async", "async", "s2_async4_split_async_seed0"),
    ("dlc_admission", "async", "s2_async4_split_async_ess_seed0"),
])
def test_run_names(tmp_path, method, source, internal):
    r, args, _ = run(tmp_path, method, source)
    assert r.returncode == 0, r.stderr
    assert arg(args, "trainer.experiment_name") == internal


def test_common_setup(tmp_path):
    r, args, _ = run(tmp_path, "sc", "quantized")
    assert r.returncode == 0, r.stderr
    expect = {"data.train_batch_size": "64", "actor_rollout_ref.rollout.n": "8", "data.max_prompt_length": "135",
              "data.max_response_length": "377", "actor_rollout_ref.rollout.max_model_len": "512",
              "algorithm.adv_estimator": "grpo", "algorithm.norm_adv_by_std_in_grpo": "False",
              "actor_rollout_ref.actor.optim.optimizer": "SGD", "actor_rollout_ref.actor.optim.lr": "1e-2",
              "actor_rollout_ref.actor.optim.weight_decay": "0.0", "actor_rollout_ref.actor.optim.clip_grad": "1e9",
              "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": "0.0", "actor_rollout_ref.rollout.temperature": "1.0",
              "actor_rollout_ref.rollout.top_p": "1.0", "actor_rollout_ref.rollout.top_k": "-1",
              "actor_rollout_ref.actor.strategy": "fsdp2", "actor_rollout_ref.rollout.topk_log_probs": "128",
              "actor_rollout_ref.rollout.quantization": "fp8", "actor_rollout_ref.rollout.seed": "1000",
              "trainer.total_training_steps": "40", "trainer.save_freq": "5", "actor_rollout_ref.actor.ppo_mini_batch_size": "64",
              "actor_rollout_ref.model.use_fused_kernels": "false", "algorithm.rollout_correction.score_centering": "true"}
    for k, v in expect.items():
        assert arg(args, k) == v, (k, arg(args, k))


def test_dlc_with_resync(tmp_path):
    r, args, env = run(tmp_path, "dlc", "stale")
    assert r.returncode == 0, r.stderr
    assert env["TIM_SYNC_MODE"] == "budget" and float(env["TIM_CHI2_BUDGET"]) == 1.0      # ESS < 1/2 <=> chi2 > 1
    assert env["TIM_SPLIT_OLD"] == "ref" and env["TIM_SPLIT_ENGINE"] == "sc"
    assert env["VERL_USE_EXTERNAL_MODULES"].split(",") == ["tim.zmq_tag", "tim.dlc.resync", "tim.dlc.hooks"]
    assert arg(args, "algorithm.rollout_correction.bypass_mode") == "false"
    assert arg(args, "actor_rollout_ref.actor.policy_loss.loss_mode") == "vanilla"
    assert arg(args, "actor_rollout_ref.actor.use_kl_loss") == "True" and arg(args, "actor_rollout_ref.actor.kl_loss_coef") == "0.0"
    assert arg(args, "algorithm.rollout_correction.rollout_is") == "null"
    assert arg(args, "algorithm.rollout_correction.score_centering") == "false"
    assert arg(args, "actor_rollout_ref.rollout.free_cache_engine") == "False"
    _, _, env = run(tmp_path, "dlc", "stale", ESS_MIN="0.25")
    assert float(env["TIM_CHI2_BUDGET"]) == 3.0


def test_dlc_without_resync_and_baselines(tmp_path):
    _, _, env = run(tmp_path, "dlc_noresync", "stale")
    assert env["TIM_SYNC_MODE"] == "stale" and env["TIM_STALENESS"] == "64" and env["TIM_SPLIT_OLD"] == "ref"
    _, args, env = run(tmp_path, "is_resync", "stale")
    assert env["TIM_SYNC_MODE"] == "budget" and env["TIM_SPLIT_OLD"] == "" and arg(args, "algorithm.rollout_correction.rollout_is") == "token"
    _, args, env = run(tmp_path, "m2po", "stale")
    assert env["VERL_USE_EXTERNAL_MODULES"].split(",") == ["tim.zmq_tag", "tim.dlc.resync", "tim.dlc.hooks", "tim.baselines.hooks"]
    assert env["TIM_SPLIT_ENGINE"] == "none" and env["TIM_BASELINE"] == "m2po" and arg(args, "actor_rollout_ref.actor.clip_ratio_c") == "1e9"
    _, args, env = run(tmp_path, "vcpo", "stale")
    assert env["TIM_BASELINE"] == "vcpo" and arg(args, "algorithm.rollout_correction.rollout_is") == "sequence"
    assert arg(args, "algorithm.rollout_correction.rollout_is_threshold") == "8.0"
    assert arg(args, "actor_rollout_ref.actor.loss_agg_mode") == "seq-mean-token-mean"
    _, args, _ = run(tmp_path, "ppo", "stale")
    assert arg(args, "actor_rollout_ref.actor.ppo_mini_batch_size") == "16" and arg(args, "actor_rollout_ref.actor.policy_loss.loss_mode") == "vanilla"
    _, args, _ = run(tmp_path, "ppo_noclip", "quantized")
    assert arg(args, "actor_rollout_ref.actor.ppo_mini_batch_size") == "16" and arg(args, "actor_rollout_ref.actor.policy_loss.loss_mode") == "bypass_mode"


def test_async(tmp_path):
    r, args, env = run(tmp_path, "dlc_admission", "async")
    assert r.returncode == 0, r.stderr
    assert env["TIM_SPLIT_OLD"] == "async" and env["TIM_ASYNC_OLD"] == "version" and env["TIM_ASYNC_ESS_MIN"] == "0.5"
    assert env["VERL_USE_EXTERNAL_MODULES"].split(",") == ["tim.zmq_tag", "tim.dlc.async_hooks", "tim.dlc.hooks"]
    for key, val in (("trainer.v1.trainer_mode", "separate_async"), ("trainer.v1.separate_async.parameter_sync_step", "1"),
                     ("trainer.v1.sampler.max_off_policy_threshold", "4"), ("actor_rollout_ref.hybrid_engine", "False"),
                     ("trainer.n_gpus_per_node", "1")):
        assert arg(args, key) == val, key
    _, _, env = run(tmp_path, "is", "async")
    assert env["TIM_ASYNC_OLD"] == "metrics" and env["TIM_SPLIT_OLD"] == ""


@pytest.mark.parametrize("method,source,msg", [
    ("dlc", "quantized", "needs SOURCE=stale"), ("dlc", "none", "needs SOURCE=stale"),
    ("on_policy", "stale", "needs SOURCE=none"), ("dlc_async", "stale", "needs SOURCE=async"),
    ("ppo", "async", "SOURCE=async runs"), ("nope", "none", "unknown METHOD"), ("sc", "fp4", "SOURCE=fp4"),
])
def test_rejected(tmp_path, method, source, msg):
    r, _, _ = run(tmp_path, method, source)
    assert r.returncode != 0 and msg in r.stderr, r.stderr
