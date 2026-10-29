"""Weight-synchronization settings of the stale sampler and of the ESS-triggered resynchronization (pure python, no
torch or verl import).

One string names the synchronization rule of a run (the internal SOURCE of scripts/train.sh):

  bf16 | fp8      verl defaults (on-policy sampler, or the quantized sampler set by rollout.quantization=fp8); no hook
  stale<S>        stale sampler with staleness S, refresh period S + 1: the sampler receives the trainer's weights when
                  `step % (S + 1) == 0`. stale64 keeps the initial weights theta_0 for a whole 40-step training.
  budget          resynchronization: the sampler receives the trainer's weights after a step whose chi2 of the ratio,
                  E_q[r^2] - 1, exceeds TIM_CHI2_BUDGET (default 1.0), and keeps its weights otherwise.
                  chi2 > 1 is the same condition as an effective sample size 1 / E_q[r^2] below one half.
                  With importance sampling (METHOD=tis) the ratio is verl's p_theta / q (`actor/rollout_corr/chi2_token`).
                  With DLC (METHOD=split) the ratio is the lag ratio p_theta / p_theta_old on the sampled tokens
                  (`tim/split_chi2_token`, written by tim.dlc.hooks), and every synchronization also loads the trainer's
                  weights into the reference worker, so that p_theta_old follows the sampler's weights.

The hook module (tim.dlc.resync) reads the same settings from environment variables, which Ray workers inherit from
the launching shell; `SyncSpec.to_env()` / `SyncSpec.from_env()` are the two directions.

Refresh rule in verl's v1 sync trainer (verl 8718ca30, trainer_base.py / trainer_sync.py): `update_weights(global_steps)`
is called once before training with global_steps = the resumed step (0 on a fresh run) and once at the end of step s
(1-indexed) with global_steps = s, i.e. after s optimizer updates; the weights it pushes sample step s + 1. A refresh at
0-indexed step t with t % (S+1) == 0 after t updates is therefore the verl call with global_steps == t.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

ENV_MODE = "TIM_SYNC_MODE"            # "stale" | "budget" | "" (off)
ENV_STALENESS = "TIM_STALENESS"       # int S >= 0
ENV_CHI2_BUDGET = "TIM_CHI2_BUDGET"   # budget only: float, sync after a step iff its chi2_token > budget
ENV_SLEEP1 = "TIM_STALE_SLEEP1"       # "1": keep weights across sleep (vLLM sleep level 1) instead of free_cache_engine=False
ENV_SPLIT_OLD = "TIM_SPLIT_OLD"       # "ref": DLC (tim.dlc.hooks), p_theta_old from the reference worker
CHI2_KEY_TIS = "actor/rollout_corr/chi2_token"   # verl's, rollout_is on (rollout_corr_helper.py compute_rollout_corr_metrics)
CHI2_KEY_SPLIT = "tim/split_chi2_token"          # tim.dlc.hooks' (_compute_old_log_prob), the lag ratio p_theta / p_theta_old
CHI2_LOG_RATIO_BOUND = 20.0           # verl rollout_corr_helper.SAFETY_BOUND: log ratios are clamped to +-20 before exp

_STALE_RE = re.compile(r"^stale(\d+)$")


@dataclass(frozen=True)
class SyncSpec:
    mode: str = ""                    # "", "stale", "budget"
    staleness: int = 0
    sleep1: bool = False
    chi2_budget: float = 1.0

    @property
    def refresh_period(self) -> int:
        return self.staleness + 1

    @property
    def active(self) -> bool:
        return self.mode in ("stale", "budget")

    def to_env(self) -> dict:
        if not self.active:
            return {}
        env = {ENV_MODE: self.mode, ENV_SLEEP1: "1" if self.sleep1 else "0"}
        if self.mode == "stale":
            env[ENV_STALENESS] = str(self.staleness)
        else:
            env[ENV_CHI2_BUDGET] = repr(self.chi2_budget)
        return env

    @staticmethod
    def from_env(env=None) -> "SyncSpec":
        env = os.environ if env is None else env
        mode = env.get(ENV_MODE, "")
        if mode == "":
            return SyncSpec()
        if mode == "stale":
            s = int(env[ENV_STALENESS])
            if s < 0:
                raise ValueError(f"{ENV_STALENESS} must be >= 0, got {s}")
            return SyncSpec("stale", staleness=s, sleep1=env.get(ENV_SLEEP1, "0") == "1")
        if mode == "budget":
            return SyncSpec("budget", chi2_budget=float(env.get(ENV_CHI2_BUDGET, "1.0")),
                            sleep1=env.get(ENV_SLEEP1, "0") == "1")
        raise ValueError(f"{ENV_MODE}={mode!r}: expected '', 'stale' or 'budget'")


def parse_source(source: str, sleep1: bool = False, chi2_budget: float = 1.0) -> SyncSpec:
    """'stale64' -> SyncSpec(stale, staleness=64); 'budget' -> the resynchronization rule; 'bf16'/'fp8' -> inactive."""
    if source in ("bf16", "fp8"):
        return SyncSpec()
    if source in ("budget", "fp8_budget"):
        return SyncSpec("budget", chi2_budget=chi2_budget, sleep1=sleep1)
    m = _STALE_RE.match(source.removeprefix("fp8_"))
    if m:
        return SyncSpec("stale", staleness=int(m.group(1)), sleep1=sleep1)
    raise ValueError(f"unknown SOURCE {source!r} (bf16|fp8|stale<S>|fp8_stale<S>|budget|fp8_budget)")


def should_sync(global_steps: Optional[int], refresh_period: int, first_call: bool) -> bool:
    """Whether `CheckpointEngineManager.update_weights(global_steps)` must push weights.

    Always on the first call of a process (a fresh vLLM engine is loaded with dummy weights, and after a resume the old
    sampler weights are gone, so the staleness cycle restarts there), when global_steps is None (an internal wake-up
    that is not a training sync), or when global_steps is a multiple of refresh_period.
    """
    if first_call or global_steps is None:
        return True
    return global_steps % refresh_period == 0


def should_sync_budget(first: bool, chi2: Optional[float], budget: float) -> bool:
    """Resynchronization rule: the first call of the process always syncs (no step has run yet); afterwards sync iff
    the step that just trained had chi2_token > budget. A missing chi2 after the start raises, so a run never silently
    trains a different configuration."""
    if first:
        return True
    if chi2 is None:
        raise RuntimeError(f"TIM_SYNC_MODE=budget: no {CHI2_KEY_TIS} / {CHI2_KEY_SPLIT} seen for this step (needs "
                           "METHOD=tis, i.e. rollout_is on, or METHOD=split, and the actor update before update_weights)")
    return chi2 > budget


def chi2_budget_from_ess(ess_min: float) -> float:
    """The chi2 budget equivalent to an effective-sample-size threshold: 1 / E[r^2] < ess_min  <=>  chi2 > 1/ess_min - 1."""
    if not 0.0 < ess_min <= 1.0:
        raise ValueError(f"ESS threshold must be in (0, 1], got {ess_min}")
    return 1.0 / ess_min - 1.0


def budget_chi2_key(env=None) -> str:
    """The step metric the resynchronization reads: the lag chi2 of tim.dlc.hooks for DLC, else verl's IS chi2."""
    env = os.environ if env is None else env
    return CHI2_KEY_SPLIT if env.get(ENV_SPLIT_OLD, "") == "ref" else CHI2_KEY_TIS


def split_ref_reload(env=None) -> bool:
    """DLC with resynchronization: every sync also loads the trainer weights into the reference worker (p_theta_old =
    the sampler's weights), so the run may sync after step 0 and tim.dlc.hooks.check_window does not apply."""
    env = os.environ if env is None else env
    return env.get(ENV_SPLIT_OLD, "") == "ref" and env.get(ENV_MODE, "") == "budget"


def chi2_token(log_ratio) -> float:
    """E_q[r^2] - 1 over the sampled response tokens (all rows, token mean), r = exp(log_ratio) with the log ratio
    clamped to +-20: verl's chi2_token (rollout_corr_helper.py, `chi2_token = masked_mean(rho^2) - 1`) for any ratio.
    Importance sampling: log_ratio = log p_theta - log q; DLC: log p_theta - log p_theta_old (the lag ratio). numpy."""
    import numpy as np
    x = np.asarray(log_ratio, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("chi2_token of an empty token set")
    x = np.clip(x, -CHI2_LOG_RATIO_BOUND, CHI2_LOG_RATIO_BOUND)
    return float(np.mean(np.exp(2.0 * x)) - 1.0)


def sampler_age(global_steps: int, refresh_period: int) -> int:
    """Steps since the last sync of the sampler weights that sample step global_steps + 1."""
    return global_steps % refresh_period


def stale_sync_step(ckpt_step: int, refresh_period: int) -> int:
    """Step of the sync whose weights are the stale sampler at checkpoint `ckpt_step` (sampling step ckpt_step + 1)."""
    return (ckpt_step // refresh_period) * refresh_period
