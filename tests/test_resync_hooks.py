"""DLC hooks and the resynchronization rule, the torch-free parts: the chi2 of the lag ratio (ESS = 1 / (1 + chi2)), which
key the resynchronization reads, the DLC hook's window check, the reference-reload wiring against a fake FSDPEngine and the
lag metrics. No torch / verl import."""
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tim.dlc import hooks as SH  # noqa: E402
from tim.dlc import resync as H  # noqa: E402
from tim.dlc import resync_config as sc  # noqa: E402


def _finder():
    f = getattr(H, "_FINDER", None)
    if f is None:
        from tim import _patch
        f = _patch._FINDER
    return f


def test_chi2_token_math():
    assert sc.chi2_token([0.0, 0.0, 0.0]) == 0.0                         # r = 1 everywhere
    assert math.isclose(sc.chi2_token([math.log(2), -math.log(2)]), (4 + 0.25) / 2 - 1)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.3, size=(7, 11))                                 # any shape: every token counts once
    assert math.isclose(sc.chi2_token(x), float(np.mean(np.exp(x) ** 2) - 1), rel_tol=1e-12)
    assert math.isclose(sc.chi2_token([100.0]), math.exp(40) - 1)        # clamped at verl's SAFETY_BOUND 20
    assert sc.chi2_token([-100.0]) == pytest.approx(-1.0)
    assert sc.chi2_token(rng.normal(0, 0.5, 10000)) > 0                  # E[r^2] >= E[r]^2: positive for a spread
    with pytest.raises(ValueError):
        sc.chi2_token([])


def test_budget_reads_the_split_chi2_and_skips_the_window():
    assert sc.budget_chi2_key({}) == sc.CHI2_KEY_TIS == "actor/rollout_corr/chi2_token"
    assert sc.budget_chi2_key({"TIM_SPLIT_OLD": "ref"}) == sc.CHI2_KEY_SPLIT == "tim/split_chi2_token"
    assert sc.split_ref_reload({"TIM_SPLIT_OLD": "ref", "TIM_SYNC_MODE": "budget"})
    assert not sc.split_ref_reload({"TIM_SPLIT_OLD": "ref", "TIM_SYNC_MODE": "stale"})
    assert not sc.split_ref_reload({"TIM_SYNC_MODE": "budget"})


def test_split_apply_under_budget_and_stale(monkeypatch):
    for k in ("TIM_STALENESS", "TIM_SPLIT_MAX_STEP"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TIM_SPLIT_OLD", "ref")
    monkeypatch.setenv("TIM_SPLIT_ENGINE", "sc")
    monkeypatch.setenv("TIM_SYNC_MODE", "budget")
    st = SH.apply()                                   # no TIM_STALENESS needed: check_window is skipped
    assert {k: st[k] for k in ("split", "split_engine", "split_ref_reload")} == {"split": "ref", "split_engine": "sc", "split_ref_reload": True}
    monkeypatch.setenv("TIM_SYNC_MODE", "stale")
    monkeypatch.setenv("TIM_STALENESS", "4")
    monkeypatch.setenv("TIM_SPLIT_MAX_STEP", "40")
    with pytest.raises(RuntimeError):
        SH.apply()                                    # stale split keeps its window check
    monkeypatch.setenv("TIM_STALENESS", "64")
    assert SH.apply()["split_ref_reload"] is False
    for name in ("verl.trainer.ppo.v1.trainer_base", "verl.workers.engine.fsdp.transformer_impl"):
        _finder().pending.pop(name, None)             # this test's registrations: nothing imports verl here


def test_chi2_probe_reads_the_split_key(monkeypatch):
    monkeypatch.setenv("TIM_SPLIT_OLD", "ref")
    mod = types.ModuleType("fake_trainer_base")

    class PPOTrainer:
        def _update_actor(self, batch, metrics):
            metrics["actor/pg_loss"] = 0.1
            return batch

    mod.PPOTrainer = PPOTrainer
    assert "tim/split_chi2_token" in H.patch_chi2_probe(mod)
    metrics = {"tim/split_chi2_token": 1.7}            # written by the DLC hook's _compute_old_log_prob, same dict
    PPOTrainer()._update_actor("b", metrics)
    assert H.CHI2["value"] == 1.7
    PPOTrainer()._update_actor("b", {})                # a step without the metric: None, and the decision raises
    assert H.CHI2["value"] is None
    with pytest.raises(RuntimeError):
        sc.should_sync_budget(False, H.CHI2["value"], 1.0)


def test_ref_reload_wiring_on_a_fake_engine(monkeypatch):
    monkeypatch.setattr(SH, "REF_ENGINES", [])
    calls = []
    monkeypatch.setattr(SH, "copy_trainer_to_ref", lambda src, dst: calls.append((src, dst)) or {"tensors": 2, "numel": 9})
    mod = types.ModuleType("fake_transformer_impl")

    class FSDPEngine:
        def __init__(self, forward_only, name):
            self.engine_config = types.SimpleNamespace(forward_only=forward_only)
            self.name, self.module = name, f"module:{name}"

        def initialize(self):
            return "init"

        def get_per_tensor_param(self, layered_summon=False, base_sync_done=False):
            return iter([("w", self.name)]), None

    mod.FSDPEngine = FSDPEngine
    assert SH.patch_ref_reload(mod).startswith("applied") and SH.patch_ref_reload(mod) == "applied"
    actor, ref = FSDPEngine(False, "actor"), FSDPEngine(True, "ref")
    with pytest.raises(RuntimeError, match="0 reference"):
        actor.get_per_tensor_param()                   # no reference engine in the process: stop, never sync silently
    assert actor.initialize() == "init" and ref.initialize() == "init"
    assert SH.REF_ENGINES == [ref]
    gen, peft = actor.get_per_tensor_param(layered_summon=False, base_sync_done=True)
    assert list(gen) == [("w", "actor")] and peft is None
    assert calls == [("module:actor", "module:ref")]   # the trainer's shards into the reference, before the push
    ref.get_per_tensor_param()                         # the reference's own export (never used for a sync): no copy
    assert len(calls) == 1
    FSDPEngine(True, "ref2").initialize()
    with pytest.raises(RuntimeError, match="2 reference"):
        actor.get_per_tensor_param()


def test_lag_metrics_carry_the_split_chi2():
    class Arr(np.ndarray):                     # the torch calls lag_metrics makes, on numpy (no torch needed)
        def float(self):
            return self

        def abs(self):
            return np.abs(self)

    t, old = np.array([0.0, -1.0, 0.5, 0.2]).view(Arr), np.array([0.0, -1.5, 0.5, 0.0]).view(Arr)
    m = SH.lag_metrics(t, old)
    d = np.array([0.0, 0.5, 0.0, 0.2])
    assert m[sc.CHI2_KEY_SPLIT] == pytest.approx(float(np.mean(np.exp(2 * d)) - 1))
    assert m["tim/split_lag_mean"] == pytest.approx(d.mean())
