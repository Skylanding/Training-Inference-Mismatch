"""The hooks attach to the installed verl 8718ca30 in a fresh process (VERL_USE_EXTERNAL_MODULES) and report 'applied'.
Needs the training environment (envs/README.md); collected with TIM_GPU_TESTS=1."""
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(env_extra, code, modules="tim.dlc.resync"):
    env = {**os.environ, "PYTHONPATH": ROOT, "VERL_USE_EXTERNAL_MODULES": modules, **env_extra}
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)


def test_stale_manager_patch_applies():
    r = _run({"TIM_SYNC_MODE": "stale", "TIM_STALENESS": "4"},
             "import verl, verl.checkpoint_engine.base as b; from tim._patch import STATE; print(STATE)")
    assert "stale_manager" in r.stdout and "applied" in r.stdout, r.stdout + r.stderr
    assert "not applied" not in r.stderr, r.stderr


def test_resync_patches_apply():
    r = _run({"TIM_SYNC_MODE": "budget", "TIM_CHI2_BUDGET": "1.0"},
             "import verl, verl.checkpoint_engine.base, verl.trainer.ppo.v1.trainer_base; from tim._patch import STATE; "
             "print(STATE)")
    assert "budget_chi2" in r.stdout and "stale_metrics" in r.stdout, r.stdout + r.stderr
    assert "not applied" not in r.stderr, r.stderr


def test_dlc_patches_apply():
    r = _run({"TIM_SYNC_MODE": "budget", "TIM_SPLIT_OLD": "ref", "TIM_SPLIT_ENGINE": "sc", "TIM_SPLIT_MAX_STEP": "40"},
             "import verl, verl.trainer.ppo.v1.trainer_base, verl.workers.engine.fsdp.transformer_impl as t; "
             "from tim._patch import STATE; print(STATE); print(t.FSDPEngineWithLMHead.forward_step.tim_split_sc)",
             modules="tim.dlc.resync,tim.dlc.hooks")
    for key in ("split_old", "split_sc_engine", "split_ref_reload_engine", "budget_chi2"):
        assert key in r.stdout, r.stdout + r.stderr
    assert r.stdout.strip().endswith("True") and "not applied" not in r.stderr, r.stdout + r.stderr


def test_sleep_level1_patch_applies():
    r = _run({"TIM_SYNC_MODE": "stale", "TIM_STALENESS": "4", "TIM_STALE_SLEEP1": "1"},
             "import verl.workers.rollout.vllm_rollout.vllm_async_server as s; print(s.vLLMHttpServer._resolve_sleep_level(None))")
    assert r.stdout.strip().endswith("1"), r.stdout + r.stderr
