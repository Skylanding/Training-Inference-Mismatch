"""Weight-synchronization hooks for verl 8718ca30: the stale sampler and the ESS-triggered resynchronization.

Loaded in every process through VERL_USE_EXTERNAL_MODULES (scripts/train.sh does this for the stale sources and for
the corrections with resynchronization):

    export PYTHONPATH=$REPO:$PYTHONPATH
    export VERL_USE_EXTERNAL_MODULES=tim.dlc.resync
    export TIM_SYNC_MODE=stale TIM_STALENESS=64       # or  TIM_SYNC_MODE=budget TIM_CHI2_BUDGET=1.0

Why a patch and not `rollout.checkpoint_manager_class`: that key exists, but only the v0 trainer (ray_trainer.py) and
experimental/separation honour it. Score centering (rollout_topk, score_centering) exists in the v1 trainer only
(trainer/ppo/v1/trainer_base.py), and v1 constructs `CheckpointEngineManager(...)` directly.

 stale  verl.checkpoint_engine.base.CheckpointEngineManager.update_weights (driver process; the v1 sync trainer calls it
        before training and after every optimizer step). The weights are pushed only when `global_steps % (S+1) == 0`
        (and on the first call of the process); otherwise the call returns immediately and the sampler keeps its
        previous weights. The skip happens above the backend, so it works for the colocated "naive" backend and for the
        NCCL/NIXL backends alike.

        Sleep mode (colocated): after every rollout the trainer calls `sleep_replicas()`; vLLM's hybrid sleep uses level
        2 (`vLLMHttpServer._resolve_sleep_level() -> 2`), which discards the weights until the next sync. A skipped sync
        would then wake a sampler without weights. Two ways to keep the old weights:
          (default) `actor_rollout_ref.rollout.free_cache_engine=False`: sleep()/resume() become no-ops and vLLM keeps
                    its weights and KV pool resident (rollout.gpu_memory_utilization ~0.3 so that FSDP fits). On a
                    skipped sync only the prefix cache is cleared, as every real sync does.
          (TIM_STALE_SLEEP1=1) `_resolve_sleep_level` patched to 1: weights are offloaded to pinned CPU memory and come
                    back on wake-up; a skipped sync calls `wake_up_replicas()` instead of the sync.
        The hook raises at the first call if neither is set.

 budget the same update_weights patch, but the decision is `chi2(step s) > TIM_CHI2_BUDGET` (resync_config.
        should_sync_budget; the first call syncs; a missing chi2 after the start raises). With importance sampling the
        chi2 of step s reaches the driver as `actor/rollout_corr/chi2_token` from the actor loss, which
        PPOTrainer._update_actor merges into the step metrics; this module wraps _update_actor and stores the value.
        fit() runs the actor update before on_step_end() -> update_weights(global_steps), so the chi2 of step s is known
        when the end-of-step-s sync is decided, and the weights pushed then sample step s + 1. With DLC the value read is
        `tim/split_chi2_token` (E_q[rho^2] - 1 of the lag ratio), which tim.dlc.hooks writes into the same step-metrics
        dict before the actor update, and tim.dlc.hooks reloads the reference worker at every sync.

Metrics logged with each step: tim/synced, tim/rollout_weights, tim/sampler_age and (budget) tim/chi2_seen.
"""
from __future__ import annotations

import inspect

from tim._patch import STATE, _guard, _say, when_imported
from tim.dlc.resync_config import CHI2_KEY_TIS, SyncSpec, budget_chi2_key, should_sync, should_sync_budget

TAG = "tim.dlc.resync"
CHI2 = {"value": None}      # chi2_token of the step that just trained (budget; driver process only)
CHI2_KEY = CHI2_KEY_TIS     # importance sampling; DLC reads resync_config.CHI2_KEY_SPLIT (budget_chi2_key)


def patch_manager(module, spec: SyncSpec) -> str:
    import asyncio

    from verl.utils.ray_utils import auto_await

    cls = module.CheckpointEngineManager
    if getattr(cls.update_weights, "tim_stale_patch", None):
        return "applied"
    inner = getattr(cls.update_weights, "__wrapped__", None)
    if inner is None or not inspect.iscoroutinefunction(inner):
        return "not applied: CheckpointEngineManager.update_weights is not an auto_await coroutine any more"
    if not hasattr(cls, "wake_up_replicas"):
        return "not applied: CheckpointEngineManager.wake_up_replicas is gone"
    period = spec.refresh_period

    def check_sleep_config(self):
        if self.backend != "naive":
            return   # separated backends never sleep the rollout weights
        for replica in self.replicas:
            free = bool(getattr(replica.config, "free_cache_engine", True))
            if free and not spec.sleep1:
                raise RuntimeError(
                    f"{spec.mode} sync needs the sampler to keep its weights while the trainer runs, but "
                    "rollout.free_cache_engine=True makes vLLM sleep at level 2 (weights discarded). Set "
                    "actor_rollout_ref.rollout.free_cache_engine=False (and gpu_memory_utilization ~0.3), or "
                    "TIM_STALE_SLEEP1=1 to sleep at level 1 (weights offloaded to CPU).")

    @auto_await
    async def update_weights(self, global_steps: int = None):
        first = not self.__dict__.get("_tim_synced_once", False)
        if first:
            check_sleep_config(self)
        prev = self.__dict__.get("_tim_last_sync")      # the sync whose weights sampled the step that just trained
        chi2 = CHI2["value"] if spec.mode == "budget" else None
        if spec.mode == "budget":
            sync = True if global_steps is None else should_sync_budget(first, chi2, spec.chi2_budget)
        else:
            sync = should_sync(global_steps, period, first)
        self.__dict__["_tim_metrics"] = sync_metrics(global_steps, prev, sync, chi2)
        if sync:
            self.__dict__["_tim_synced_once"] = True
            self.__dict__["_tim_last_sync"] = global_steps
            return await inner(self, global_steps)
        if self.backend == "naive":
            if spec.sleep1:
                await self.wake_up_replicas()       # restores weights (level 1) + KV pool + clears the prefix cache
            else:
                await asyncio.gather(*[r.clear_kv_cache() for r in self.replicas])
        return self.__dict__["_tim_metrics"]

    update_weights.tim_stale_patch = period
    cls.update_weights = update_weights
    if spec.mode == "budget":
        return f"applied (chi2 budget={spec.chi2_budget}, sleep1={spec.sleep1})"
    return f"applied (staleness={spec.staleness}, refresh_period={period}, sleep1={spec.sleep1})"


def sync_metrics(global_steps, prev_sync, synced: bool, chi2=None) -> dict:
    """Logged with step s (update_weights(s) runs at the end of step s, after s optimizer updates):
    tim/synced           1 if this call pushes the trainer weights (they sample step s + 1), else 0
    tim/rollout_weights  the global step of the sync whose weights sampled step s (-1: none before)
    tim/sampler_age      optimizer updates between those weights and the trainer that computed step s's loss
                         (s - 1 - rollout_weights; 0 = on-policy)
    tim/chi2_seen        budget only: the chi2_token the decision used"""
    out = {"tim/synced": 1.0 if synced else 0.0}
    if chi2 is not None:
        out["tim/chi2_seen"] = float(chi2)
    if global_steps is not None and global_steps > 0 and prev_sync is not None:
        out["tim/rollout_weights"] = float(prev_sync)
        out["tim/sampler_age"] = float(global_steps - 1 - prev_sync)
    return out


def patch_trainer_metrics(module) -> str:
    """The v1 sync trainer drops update_weights' return value (trainer_sync.py on_step_end); its per-step metrics take
    weight-sync stats only through `_consume_sync_metrics`. Wrap it to also pop the stats the patched update_weights
    left on the checkpoint manager."""
    cls = module.PPOTrainer
    original = cls._consume_sync_metrics
    if getattr(original, "tim_patch", None):
        return "applied"

    def _consume_sync_metrics(self):
        out = dict(original(self) or {})
        manager = getattr(self, "checkpoint_manager", None)
        if manager is not None:
            out.update(manager.__dict__.pop("_tim_metrics", None) or {})
        return out

    _consume_sync_metrics.tim_patch = True
    cls._consume_sync_metrics = _consume_sync_metrics
    return "applied (tim/* sync metrics merged into the step metrics)"


def patch_chi2_probe(module) -> str:
    """budget: remember the step's chi2_token right after the actor update, for the update_weights of the same step.
    The key is verl's IS chi2 (METHOD=tis) or the lag chi2 of tim.dlc.hooks (DLC; resync_config.budget_chi2_key)."""
    cls = module.PPOTrainer
    update_actor = cls._update_actor
    if getattr(update_actor, "tim_patch", None):
        return "applied"
    key = budget_chi2_key()

    def _update_actor(self, batch, metrics):
        CHI2["value"] = None
        batch = update_actor(self, batch, metrics)
        if key in metrics:
            CHI2["value"] = float(metrics[key])
        return batch

    _update_actor.tim_patch = True
    cls._update_actor = _update_actor
    return f"applied (reads {key} after the actor update)"


def patch_sleep_level(module) -> str:
    cls = module.vLLMHttpServer
    if not hasattr(cls, "_resolve_sleep_level"):
        return "not applied: vLLMHttpServer._resolve_sleep_level is gone"
    cls._resolve_sleep_level = lambda self: 1
    return "applied (vLLM hybrid sleep level forced to 1: weights are offloaded, not discarded)"


def apply() -> dict:
    spec = SyncSpec.from_env()
    if not spec.active:
        return {"mode": "off"}
    when_imported("verl.checkpoint_engine.base", _guard("stale_manager", lambda m: patch_manager(m, spec)))
    when_imported("verl.trainer.ppo.v1.trainer_base", _guard("stale_metrics", patch_trainer_metrics))
    if spec.mode == "budget":
        when_imported("verl.trainer.ppo.v1.trainer_base", _guard("budget_chi2", patch_chi2_probe))
    if spec.sleep1:
        when_imported("verl.workers.rollout.vllm_rollout.vllm_async_server", _guard("sleep_level1", patch_sleep_level))
    return {"mode": spec.mode}


try:
    _spec = SyncSpec.from_env()
    STATE.update(apply())
    if _spec.active:
        _say(f"{TAG} armed: {_spec}")
except Exception as _error:  # bad env must stop the run, not silently train a different source
    raise RuntimeError(f"{TAG}: invalid TIM_* sync settings: {_error}") from _error
