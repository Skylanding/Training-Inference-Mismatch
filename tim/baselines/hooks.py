"""Driver hooks of the two published baselines, M2PO and VCPO (docs/BASELINES.md), on verl 8718ca30.

Math in tim/baselines/weights.py (tested against the official code). Selected by TIM_BASELINE (scripts/train.sh sets it
for METHOD=m2po / vcpo) and loaded through VERL_USE_EXTERNAL_MODULES after tim.dlc.hooks (M2PO, which needs
p_theta_old = the sampler's weights from the reference worker) or alone (VCPO).

Where the masks are applied. In trainer_base.step the advantages are computed after `_compute_old_log_prob`
(`_compute_advantage` follows it), so the M2PO mask is computed here, in a wrapper of `PPOTrainer._update_actor`, right
before the update, from:
  - log p_theta_t: the trainer's own old-log-prob forward, stashed by tim.dlc.hooks as `tim_logp_theta_t` before it
    overwrites old_log_probs with the reference (TIM_SPLIT_ENGINE=none);
  - log p_theta_0: old_log_probs after tim.dlc.hooks (the reference = the stale sampler's weights);
  - advantages (written by `_compute_advantage`).
The 0/1 mask is written as `rollout_is_weights` (verl's loss multiplies it in when the field is present,
workers/utils/losses.py L92). With one update per batch the ratio at the update is r(theta_t), so this equals the
official PPO clip at the M2PO bounds.

VCPO (decoupled mode, pi_old = theta_t, verl's sequence TIS at threshold 8 writes rollout_is_weights in
`_compute_advantage`): the ESS ratio of the UNtruncated sequence weights w_i = exp(sum_t log p_theta_t - log q) scales
rollout_is_weights by s = sqrt(min(1, rho / rho_on)) (SGD without momentum or clipping: scaling the gradient = scaling
the lr). The optimal off-policy baseline (TIM_VCPO_OPOB=1, default) needs ||grad sum_t log p_theta(y_t)||^2 of
every trajectory on its own: a worker RPC `tim_score_norms` (patched onto ActorRolloutRefWorker, dispatched like
update_actor) gathers the DP shards so that every rank runs the SAME trajectory in the same order (FSDP's reduce-scatter
then averages identical gradients: the trajectory's own gradient, no unsharded buffer), one forward + backward per
trajectory with loss sum_t log p over its response, reads the FSDP-aware total grad norm, zeroes the gradients, and
returns its own shard's norms through TransferQueue (`tim_score_norm_sq`). The driver then overwrites `advantages`
with the per-trajectory OPOB advantage (broadcast to the tokens), computed in log space (W_i = wbar_i^2 n_i w_i^2 with
w_i = exp(sum log ratio) overflows; the group baseline is a W-weighted mean, so a common factor per group cancels).
"""
from __future__ import annotations

import math
import os

import torch

from tim._patch import STATE, _say, when_imported
from tim.baselines.weights import EPS, M2PO_BUDGET, VCPO_TIS_C, m2po_keep_mask, vcpo_ess_ratio, vcpo_lr_scale

ENV = "TIM_BASELINE"            # "m2po" | "vcpo"
STASH = "tim_logp_theta_t"      # TransferQueue field written by tim.dlc.hooks under TIM_SPLIT_ENGINE=none
NORMS = "tim_score_norm_sq"     # TransferQueue field written by the OPOB pre-pass (one value per trajectory)
RPC = "tim_score_norms"


def baseline() -> str:
    mode = os.environ.get(ENV, "")
    if mode not in ("m2po", "vcpo"):
        raise ValueError(f"{ENV}={mode!r}: m2po or vcpo")
    return mode


def _padded(data, fields):
    return data.select(*fields).to_padded_tensor()


def m2po_weights(data) -> tuple[torch.Tensor, dict]:
    """data: TensorDict (nested, from TransferQueue) with STASH, old_log_probs, advantages, response_mask."""
    p = _padded(data, [STASH, "old_log_probs", "advantages", "response_mask"])
    keep, stats = m2po_keep_mask(p[STASH].float(), p["old_log_probs"].float(), p["advantages"].float(),
                                 p["response_mask"], budget=float(os.environ.get("TIM_M2PO_BUDGET", M2PO_BUDGET)))
    return keep.float(), stats


def vcpo_scale(data) -> tuple[float, dict]:
    """ESS ratio of the untruncated sequence weights, computed in log space (the ratio is invariant to a common factor,
    so subtracting the max log weight changes nothing but avoids exp overflow over ~377 tokens)."""
    p = _padded(data, ["old_log_probs", "rollout_log_probs", "response_mask"])
    m = p["response_mask"].double()
    log_w = ((p["old_log_probs"].double() - p["rollout_log_probs"].double()) * m).sum(-1)
    w = torch.exp(log_w - log_w.max())
    rho = vcpo_ess_ratio(w)
    s = vcpo_lr_scale(rho, rho_on=float(os.environ.get("TIM_VCPO_RHO_ON", 1.0)))
    return s, {"vcpo/ess_ratio": rho, "vcpo/lr_scale": s, "vcpo/log_w_mean": float(log_w.mean()),
               "vcpo/log_w_max": float(log_w.max())}


def opob_advantages(rewards, group_ids, log_w, score_norm_sq, c: float = VCPO_TIS_C) -> tuple[torch.Tensor, dict]:
    """weights.vcpo_opob_advantages in log space: W_i = wbar_i^2 * n_i * w_i^2 with log w_i given; per group the
    baseline sum W R / (sum W + 1e-8) is computed as a softmax-weighted mean (the 1e-8 is dropped: it matters only when
    every W of a group is ~0, where the official code falls back to b ~ 0 and this gives the plain mean)."""
    r, lw = rewards.double(), log_w.double()
    lwbar = torch.clamp(lw, max=math.log(c))
    logW = 2 * lwbar + torch.log(score_norm_sq.double().clamp(min=1e-300)) + 2 * lw
    adv, baselines = torch.empty_like(r), []
    for g in torch.unique(group_ids):
        idx = group_ids == g
        a = torch.softmax(logW[idx], 0)
        b = (a * r[idx]).sum()
        sd = r[idx].std(unbiased=False)
        adv[idx] = (r[idx] - b) / sd if float(sd) > 1e-8 else r[idx] - b
        baselines.append(float(b))
    return adv, {"vcpo/opob_baseline_mean": sum(baselines) / len(baselines),
                 "vcpo/opob_norm_sq_mean": float(score_norm_sq.double().mean())}


def opob_driver(trainer, batch, tq, to_nested) -> dict:
    batch.extra_info.update({"temperature": trainer.config.actor_rollout_ref.rollout.temperature,
                             "calculate_entropy": False, "score_centering": False, "distillation_use_topk": False})
    trainer.actor_rollout_wg.__getattribute__(RPC)(batch)
    data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                           select_fields=[NORMS, "uid", "rm_scores", "old_log_probs", "rollout_log_probs", "response_mask"])
    p = _padded(data, ["rm_scores", "old_log_probs", "rollout_log_probs", "response_mask"])
    m = p["response_mask"].double()
    rewards = (p["rm_scores"].double() * m).sum(-1)
    log_w = ((p["old_log_probs"].double() - p["rollout_log_probs"].double()) * m).sum(-1)
    uids = [str(u) for u in (data["uid"].tolist() if hasattr(data["uid"], "tolist") else data["uid"])]
    index = {u: k for k, u in enumerate(dict.fromkeys(uids))}
    group_ids = torch.tensor([index[u] for u in uids])
    norms = data[NORMS].reshape(-1).double()
    adv, stats = opob_advantages(rewards, group_ids, log_w, norms)
    tok = (adv.float()[:, None] * p["response_mask"].float())
    from tensordict import TensorDict
    tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id,
                    fields=TensorDict({"advantages": to_nested(tok, data["response_mask"])}, batch_size=len(batch)))
    stats["vcpo/opob_norm_sq_min"] = float(norms.min())
    return stats


def patch_trainer(module) -> str:
    cls = module.PPOTrainer
    original = cls._update_actor
    if getattr(original, "tim_baseline", None):
        return "applied"
    mode = baseline()
    opob = mode == "vcpo" and os.environ.get("TIM_VCPO_OPOB", "1") == "1"
    tq = module.tq
    to_nested = module.response_to_nested

    def _update_actor(self, batch, metrics):
        if mode == "m2po":
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                                   select_fields=[STASH, "old_log_probs", "advantages", "response_mask"])
            keep, stats = m2po_weights(data)
            out = {"rollout_is_weights": to_nested(keep, data["response_mask"])}
        else:
            opob_stats = opob_driver(self, batch, tq, to_nested) if opob else {}
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                                   select_fields=["old_log_probs", "rollout_log_probs", "response_mask", "rollout_is_weights"])
            s, stats = vcpo_scale(data)
            stats.update(opob_stats)
            w = _padded(data, ["rollout_is_weights"])["rollout_is_weights"].float() * s
            out = {"rollout_is_weights": to_nested(w, data["response_mask"])}
        from tensordict import TensorDict
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=TensorDict(out, batch_size=len(batch)))
        metrics.update({f"tim/{k}": v for k, v in stats.items()})
        if not STATE.get("baseline_said"):
            STATE["baseline_said"] = True
            _say(f"{mode}: first update, " + ", ".join(f"{k}={v:.4g}" for k, v in stats.items()))
        return original(self, batch, metrics)

    _update_actor.tim_baseline = mode
    cls._update_actor = _update_actor
    return f"applied ({mode}: rollout_is_weights rewritten before the actor update)"


def patch_worker(module) -> str:
    """ActorRolloutRefWorker.tim_score_norms: the OPOB pre-pass (module docstring). Registered with update_actor's
    dispatch so the driver's worker group binds it and TransferQueue carries its input and output."""
    from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import no_padding_2_padding
    cls = module.ActorRolloutRefWorker
    if hasattr(cls, RPC):
        return "applied"

    def score_loss(model_output=None, data=None, dp_group=None, **_):
        lp = no_padding_2_padding(model_output["log_probs"], data)
        mask = data["response_mask"]
        mask = mask.to_padded_tensor(0) if getattr(mask, "is_nested", False) else mask
        return (lp * mask.to(lp.dtype)).sum(), {}

    def tim_score_norms(self, data):
        import torch.distributed as dist
        from torch.distributed.tensor import DTensor
        from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
        eng = self.actor.engine
        module.maybe_fix_3d_position_ids(data) if hasattr(module, "maybe_fix_3d_position_ids") else None
        n_local = data.shape[0]
        group, dp, rank = eng.get_data_parallel_group(), eng.get_data_parallel_size(), eng.get_data_parallel_rank()
        shards = [None] * dp
        dist.all_gather_object(shards, data.cpu(), group=group)
        defaults = dict(use_remove_padding=self.actor.model_config.get("use_remove_padding", False),
                        use_dynamic_bsz=False, max_token_len_per_gpu=self.actor.engine_config.max_token_len_per_gpu,
                        micro_batch_size_per_gpu=1, use_fused_kernels=self.actor.engine_config.use_fused_kernels)
        temperature = tu.get(data, "temperature", default=1.0)
        mine = []
        with eng.train_mode(disable_auto_offload=True):
            eng.optimizer_zero_grad()
            for r, shard in enumerate(shards):
                for i in range(shard.shape[0]):
                    mb = tu.index_select_tensor_dict(shard, [i]).to(torch.cuda.current_device())
                    if "loss_mask" not in mb.keys():
                        mb["loss_mask"] = mb["response_mask"]
                    tu.assign_non_tensor(mb, **defaults, disable_auto_offload=True, temperature=temperature,
                                         calculate_entropy=False, score_centering=False,
                                         global_token_num=[int(mb["input_ids"].offsets().diff()[0])])
                    eng.forward_backward_batch(mb, loss_function=score_loss, forward_only=False)
                    norm = fsdp2_clip_grad_norm_(eng.module.parameters(), max_norm=float("inf"))
                    if isinstance(norm, DTensor):
                        norm = norm.full_tensor()
                    eng.optimizer_zero_grad()
                    if r == rank:
                        mine.append(float(norm) ** 2)
        assert len(mine) == n_local, (len(mine), n_local)
        out = tu.get_tensordict({NORMS: torch.tensor(mine, dtype=torch.float32)})
        return out if eng.is_mp_src_rank_with_outputs() else None

    tim_score_norms = register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))(tim_score_norms)
    setattr(cls, RPC, tim_score_norms)
    return "applied (OPOB pre-pass: one forward + backward per trajectory, all DP ranks in lockstep)"


def _strict(module) -> None:
    """A failure raises: a baseline run without its weights would train plain decoupled PG."""
    STATE["baseline"] = patch_trainer(module)
    _say(f"baseline: {STATE['baseline']}")


def apply() -> dict:
    if not os.environ.get(ENV):
        return {"baseline": "off"}
    when_imported("verl.trainer.ppo.v1.trainer_base", _strict)
    if baseline() == "vcpo" and os.environ.get("TIM_VCPO_OPOB", "1") == "1":
        def _worker(m):
            STATE["baseline_worker"] = patch_worker(m)
            _say(f"baseline worker: {STATE['baseline_worker']}")
        when_imported("verl.workers.engine_workers", _worker)
    return {"baseline": baseline()}


STATE.update(apply())
