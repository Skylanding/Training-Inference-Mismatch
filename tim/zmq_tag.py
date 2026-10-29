"""Give each run its own colocate weight-sync ZMQ socket when several verl runs share a node.

verl main names the socket `ipc:///tmp/rl-colocate-zmq-{ray job id}-replica-{r}-rank-{k}.sock`, with the job id taken
from `ray.get_runtime_context().get_job_id()` on both sides (vllm_rollout.py, the trainer; vllm_async_server.py, which
exports it to the vLLM worker as VERL_RAY_JOB_ID). Every run starts its own Ray cluster, whose first job id is always
01000000, so concurrent runs on one node collide: `ZMQError: Address already in use` at the first weight sync.
This module swaps the module-level `ray` of exactly those two modules for a proxy whose
runtime context returns `<job id>-<TIM_RUN_TAG>`; the rest of Ray is untouched. Keep the tag short: IPC paths are
limited to 107 characters.

Loaded through VERL_USE_EXTERNAL_MODULES; active only when TIM_RUN_TAG is set.
"""
from __future__ import annotations

import os
import sys

from tim._patch import when_imported

TAG = os.environ.get("TIM_RUN_TAG", "")
TARGETS = ("verl.workers.rollout.vllm_rollout.vllm_rollout", "verl.workers.rollout.vllm_rollout.vllm_async_server")


class _Context:
    """Wraps a RuntimeContext; only get_job_id changes. Private names are never forwarded (unpickled copies have no
    `_inner`, and forwarding it would recurse)."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def get_job_id(self):
        return f"{object.__getattribute__(self, '_inner').get_job_id()}-{TAG}"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_inner"), name)


def _tagged_ray(real):
    """A real module object named "ray" with ray's attributes and a tagged get_runtime_context. Being a module, it is
    pickled by name (cloudpickle), so a pickled function from the patched modules unpickles with the real ray instead
    of a half-built proxy (a proxy class instead recurses in __getattr__ on unpickling)."""
    import types

    m = types.ModuleType("ray")
    m.__dict__.update(real.__dict__)
    m.get_runtime_context = lambda: _Context(real.get_runtime_context())
    m.__tim_zmq_tag__ = True
    return m


def _patch(module) -> None:
    real = getattr(module, "ray", None)
    if real is None:
        state = "nothing to patch"
    elif getattr(real, "__tim_zmq_tag__", False):
        state = "applied"
    else:
        module.ray = _tagged_ray(real)
        state = "applied"
    print(f"[zmq_tag pid={os.getpid()}] {module.__name__}: {state} (tag {TAG})", file=sys.stderr, flush=True)


if TAG:
    for _name in TARGETS:
        when_imported(_name, _patch)
