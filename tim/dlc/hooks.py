"""DLC (discrepancy-lag correction) on verl 8718ca30: the clip of the lag ratio against the sampler's weights and the
centering term of the numerical discrepancy.

The ratio p_theta / q is split exactly at the trainer evaluated at the sampler's weights, p_theta_old:
    p_theta / q = (p_theta / p_theta_old) * (p_theta_old / q)
                   lag ratio rho: PPO clip    discrepancy factor: centering term C_t (or verl's token IS weight)

verl's three-policy ("decoupled") rollout correction (algorithm.rollout_correction.bypass_mode=false) has this loss
shape: PPO clip on pi_theta / pi_old and IS weights pi_old / pi_rollout (docs/algo/rollout_corr.md, "Decoupled Mode").
In the v1 sync trainer, however, pi_old is the trainer at its current weights theta_t (trainer_base.py
`_compute_old_log_prob`, computed once per batch), so with one optimizer step per batch the PPO ratio is 1 and the
whole lag sits in the IS weight. This module makes pi_old the sampler's weights:

    TIM_SPLIT_OLD=ref    PPOTrainer._compute_old_log_prob (driver) is wrapped: after the original theta_t forward (kept for
                         verl's entropy and rollout-vs-actor metrics), the reference policy's log-probs are computed and
                         written as old_log_probs. The reference policy starts as the initial model theta_0
                         (model.path), the stale sampler's weights. Without resynchronization this is true for the whole
                         training as long as the stale sampler never syncs after step 0 (refresh period > total steps;
                         checked against TIM_SPLIT_MAX_STEP). The launcher enables the reference worker with
                         actor.use_kl_loss=True, kl_loss_coef=0.
                         Also written: tim/split_lag_* and tim/split_chi2_token = E_q[rho^2] - 1 of the lag ratio
                         rho = p_theta_t / p_theta_old on the sampled tokens (resync_config.chi2_token; verl's chi2_token
                         definition), the trigger of the resynchronization (1 / E_q[rho^2] < 1/2 <=> chi2 > 1).

    TIM_SYNC_MODE=budget (DLC with resynchronization): the run syncs whenever the step's tim/split_chi2_token exceeds
                         TIM_CHI2_BUDGET (tim.dlc.resync), so pi_old must follow the sampler.
                         Engine patch (worker processes, `patch_ref_reload`): FSDPEngine.initialize records the
                         forward-only (reference) engine of the process; FSDPEngine.get_per_tensor_param on the trainer
                         engine, which engine_workers.ActorRolloutRefWorker.update_weights calls for exactly the syncs
                         the driver pushes (naive backend; the reference lives in the same worker, role
                         actor_rollout_ref), first copies the trainer's FSDP2 shards into the reference's
                         (`copy_trainer_to_ref`: same mesh and placements, local shard to local shard, cast to the
                         reference's dtype and device, after resharding both so no unsharded root parameter keeps the
                         old values). The reference then has the weights the sampler gets in the same call.

    TIM_SPLIT_OLD=async  (DLC under continuous asynchrony, SOURCE=async) pi_old = theta_k(t), the weights version that
                         generated each token, and log p_theta_k(t) at the sampler's heads come from tim.dlc.async_hooks
                         (the separate-async trainer's own _compute_old_log_prob is replaced there); this module adds only
                         the centering engine patch below and the _update_actor check (patch_sc_check). No reference
                         worker.

    TIM_SPLIT_ENGINE=sc  DLC: the discrepancy factor is removed by the centering term C_t (tim.dlc.centering has the math;
                         docs/DLC_LOSS.md). verl's score centering exists for bypass + REINFORCE only and its config checks
                         refuse anything else, so the config keeps score_centering=false (rollout_is=null,
                         rollout.topk_log_probs=128: the sampler's heads are recorded whenever topk_log_probs > 0) and
                         three patches do the work:
                           driver   _compute_old_log_prob marks the reference call (`tim_ref_topk`), _update_actor checks
                                    that the correction was applied (its metrics came back) and stops the run otherwise;
                           engine   FSDPEngineWithLMHead.prepare_model_outputs: on the marked reference forward, log p_old
                                    at the sampler's head ids, returned as model output `ref_topk_log_probs` ([B, j, k],
                                    the layout of rollout_topk_log_probs), which the worker writes to TransferQueue;
                           engine   FSDPEngineWithLMHead.forward_step: on a training step the micro-batch gets verl's
                                    score_centering flag (the engine then calls the loss function as logits processor
                                    and keeps the logits for its backward) and the loss function is wrapped: as processor
                                    it returns the centering term for both advantage signs, as loss it is verl's
                                    ppo_loss (vanilla PPO clip on p_theta / p_theta_old, no IS weight) plus A * C_t.
    TIM_SPLIT_ENGINE=tis the discrepancy factor p_theta_old / q is verl's token IS weight (rollout_is=token, clamp 2),
                         computed on the driver; nothing else is patched. Not used in the paper.
    TIM_SPLIT_ENGINE=none no discrepancy factor; log p_theta_t is kept for the M2PO mask (tim.baselines.hooks).

Loaded through VERL_USE_EXTERNAL_MODULES (scripts/train.sh, METHOD=dlc / dlc_noresync / dlc_async / dlc_admission / m2po).
"""
from __future__ import annotations

import os

from tim._patch import STATE, _guard, _say, _strict, when_imported
from tim.dlc.resync_config import CHI2_KEY_SPLIT, chi2_token, split_ref_reload

ENV_OLD = "TIM_SPLIT_OLD"            # "ref" | "" (off)
ENV_SKIP_KL_REF = "TIM_SPLIT_SKIP_KL_REF"   # "1": skip verl's second reference forward on the KL path (kl_loss_coef = 0)
ENV_MAX_STEP = "TIM_SPLIT_MAX_STEP"  # total training steps of the run
ENV_ENGINE = "TIM_SPLIT_ENGINE"      # "tis" (default: verl's token IS weight) | "sc" (centering term C_t, DLC)
REF_FLAG = "tim_ref_topk"            # micro-batch flag of the reference forward that also returns the head log-probs
REF_FIELD = "ref_topk_log_probs"     # TransferQueue field: log p_theta_0 at rollout_topk_ids, [B, j, k]
SC_PARTS = ("pos", "neg", "q_mass", "p0_mass", "p_mass", "clip_pos", "clip_neg")   # tim.dlc.centering

SC_METRIC = "actor/split_sc_abs_mean"   # the worker's metric the driver requires at every step
REF_ENGINES = []                     # DLC with resynchronization: the forward-only (reference) FSDP engine(s) of this process


def engine_factor() -> str:
    mode = os.environ.get(ENV_ENGINE, "") or "tis"
    if mode not in ("tis", "sc", "none"):   # none: no discrepancy factor (M2PO, tim.baselines.hooks)
        raise ValueError(f"{ENV_ENGINE}={mode!r}: tis, sc or none")
    return mode


def check_window(staleness: int, max_step: int) -> None:
    """Without resynchronization, theta_0 is the sampler's weights at every step only if the first re-sync of the stale
    sampler (at global step staleness + 1) never happens within the run."""
    if staleness + 1 <= max_step:
        raise RuntimeError(f"split with pi_old = theta_0 needs no re-sync within the run: staleness {staleness} + 1 <= "
                           f"{max_step} steps (use the resynchronization, TIM_SYNC_MODE=budget, which reloads the reference at every sync)")


def _flat(t):
    """Response tokens of all rows as one 1-D tensor. Both fields are jagged nested tensors (padding.response_from_nested)
    built separately, so their raggedness differs and `a - b` on them raises; their .values() line up token for token."""
    return t.values() if getattr(t, "is_nested", False) else t.reshape(-1)


def lag_metrics(old_theta_t, old_theta_0) -> dict:
    """The policy lag log p_theta_t - log p_theta_old on the sampled tokens (~0 right after a sync: bf16 noise only;
    theta_old = pi_old, the sampler's weights), and its chi2 (E_q[rho^2] - 1, the resynchronization trigger of DLC).
    A failure is printed once, never raised (a resynchronizing run then stops at the sync decision, which finds no chi2)."""
    try:
        d = (_flat(old_theta_t).float() - _flat(old_theta_0).float())
        out = {"tim/split_lag_abs_mean": float(d.abs().mean()), "tim/split_lag_mean": float(d.mean()),
               "tim/split_lag_abs_max": float(d.abs().max())}
    except Exception as error:
        if not STATE.get("split_lag_warned"):
            STATE["split_lag_warned"] = True
            _say(f"split lag metrics failed: {type(error).__name__}: {error}")
        return {}
    try:
        out[CHI2_KEY_SPLIT] = chi2_token(d.detach().double().cpu().numpy() if hasattr(d, "detach") else d)
    except Exception as error:
        if not STATE.get("split_chi2_warned"):
            STATE["split_chi2_warned"] = True
            _say(f"split chi2 failed: {type(error).__name__}: {error}")
    return out


def patch_trainer(module) -> str:
    cls = module.PPOTrainer
    original = cls._compute_old_log_prob
    if getattr(original, "tim_split", None):
        return "applied"
    engine = engine_factor()
    if not hasattr(cls, "_compute_ref_log_prob"):
        return "not applied: PPOTrainer._compute_ref_log_prob is gone"
    tq = module.tq

    def _compute_old_log_prob(self, batch, metrics):
        batch = original(self, batch, metrics)
        rc = self.config.algorithm.get("rollout_correction", None)
        if rc and rc.get("bypass_mode", False):
            raise RuntimeError("TIM_SPLIT_OLD=ref needs the decoupled mode (algorithm.rollout_correction.bypass_mode=false)")
        if not self.use_reference_policy:
            raise RuntimeError("TIM_SPLIT_OLD=ref needs the reference worker (actor.use_kl_loss=True, kl_loss_coef=0)")
        if engine == "sc":
            batch.extra_info[REF_FLAG] = True      # this reference forward also returns log p_0 at the sampler's heads
        try:
            batch = ref_original(self, batch, metrics)
            self.__dict__["_tim_ref_done"] = True   # verl's own reference call of this step may reuse it (SKIP_KL_REF)
        finally:
            if engine == "sc":
                batch.extra_info.pop(REF_FLAG, None)   # verl's own reference call later in the step: log-probs only
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                               select_fields=["ref_log_prob", "old_log_probs"])
        metrics.update(lag_metrics(data["old_log_probs"], data["ref_log_prob"]))
        keep = ["old_log_probs"]
        if engine == "none":   # M2PO's mask needs log p_theta_t after the advantages exist (tim.baselines.hooks.STASH)
            data["tim_logp_theta_t"] = data["old_log_probs"]
            keep.append("tim_logp_theta_t")
        data["old_log_probs"] = data.pop("ref_log_prob")
        return tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select(*keep))

    ref_original = cls._compute_ref_log_prob
    skip_kl_ref = os.environ.get(ENV_SKIP_KL_REF, "0") == "1"

    def _compute_ref_log_prob(self, batch, metrics):
        """TIM_SPLIT_SKIP_KL_REF=1 (optional, off in the paper's runs): verl's KL path asks for the reference log-probs a
        second time in the step; with kl_loss_coef = 0 they never enter the loss, and the hook's call above already
        stored ref_log_prob in the batch, so the second forward can be skipped."""
        done = self.__dict__.pop("_tim_ref_done", False)
        if skip_kl_ref and done:
            if float(self.config.actor_rollout_ref.actor.get("kl_loss_coef", 0.0)) != 0.0:
                raise RuntimeError(f"{ENV_SKIP_KL_REF}=1 needs kl_loss_coef=0")
            return batch
        return ref_original(self, batch, metrics)

    _compute_old_log_prob.tim_split = "ref"
    cls._compute_old_log_prob = _compute_old_log_prob
    cls._compute_ref_log_prob = _compute_ref_log_prob
    if engine == "none":
        return "applied (pi_old = reference policy theta_0 = the stale sampler's weights; no discrepancy factor, log p_theta_t kept)"
    if engine != "sc":
        return "applied (pi_old = reference policy theta_0 = the stale sampler's weights; discrepancy factor: token IS)"

    cls._update_actor = sc_update_check(cls._update_actor)
    return "applied (pi_old = reference policy theta_0 = the stale sampler's weights; discrepancy factor: centering term)"


def sc_update_check(update_actor):
    """PPOTrainer._update_actor wrapped for DLC's centering: refuse verl's IS weight / score centering in the config,
    require the worker's centering metric after the update (else the engine patch did not run), mirror it under tim/."""
    def _update_actor(self, batch, metrics):
        rc = self.config.algorithm.get("rollout_correction", None) or {}
        if rc.get("rollout_is", None) is not None or rc.get("score_centering", False):
            raise RuntimeError(f"{ENV_ENGINE}=sc needs rollout_is=null and score_centering=false in the config (the "
                               "centering term replaces the IS weight; verl's own score centering is bypass-only)")
        batch = update_actor(self, batch, metrics)
        if SC_METRIC not in metrics:   # the engine patch did not run in the actor worker: plain PPO clip was trained
            raise RuntimeError(f"{ENV_ENGINE}=sc: the actor update returned no {SC_METRIC}; the DLC centering "
                               "patch of FSDPEngineWithLMHead was not applied in the worker process")
        for key in list(metrics):      # the same numbers under tim/ (the worker's metrics all get the actor/ prefix)
            if key.startswith("actor/split_sc_"):
                metrics["tim/" + key[len("actor/"):]] = metrics[key]
        return batch

    _update_actor.tim_split_sc_check = True
    return _update_actor


def patch_sc_check(module) -> str:
    """TIM_SPLIT_OLD=async (DLC under continuous asynchrony): pi_old and the head log-probs at the generating version
    come from tim.dlc.async_hooks; this module adds only the centering (engine patch) and this check."""
    cls = module.PPOTrainer
    if getattr(cls._update_actor, "tim_split_sc_check", None):
        return "applied"
    cls._update_actor = sc_update_check(cls._update_actor)
    return "applied (centering check of the actor update; pi_old from tim.dlc.async_hooks)"


# ---------------------------------------------------------------------------------------------------------------------
# centering term (worker processes): reference head log-probs, logits processor, loss
# ---------------------------------------------------------------------------------------------------------------------
def ref_head_log_probs(engine, output, output_args, micro_batch):
    """log p_theta_old at the sampler's head ids for every row of the packed micro-batch, as a nested [B, j, k] tensor
    (rollout_topk_log_probs' layout: row i of a sequence predicts its token i + 1, verl score_centering.pad_rollout_topk).
    Read from the model output before verl's own post-processing; the logits are not modified."""
    import torch
    from verl.trainer.ppo.score_centering import topk_log_probs_from_logits
    from verl.utils import tensordict_utils as tu

    if not tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True):
        raise NotImplementedError("DLC centering needs use_remove_padding=True (packed logits)")
    if tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False):
        raise NotImplementedError("DLC centering needs the logits: use_fused_kernels=False")
    if getattr(engine, "use_ulysses_sp", False):
        raise NotImplementedError("DLC centering is not built for Ulysses sequence parallelism")
    if "rollout_topk_ids" not in micro_batch.keys():
        raise RuntimeError("DLC centering: no rollout_topk_ids in the batch (rollout.topk_log_probs must be > 0)")
    ids = micro_batch["rollout_topk_ids"].values().long()
    logits = output.logits.squeeze(0)
    if not output_args.get("temperature_is_one", False):
        logits = logits / output_args["temperature_rmpad"].unsqueeze(-1).clamp(min=1e-8).to(logits.dtype)
    pad = output_args.get("pad_size", 0) or 0
    if pad:
        logits = logits[:-pad]
    if logits.shape[0] != ids.shape[0]:
        raise RuntimeError(f"DLC centering: {logits.shape[0]} logits rows vs {ids.shape[0]} head rows")
    head = topk_log_probs_from_logits(logits, ids)
    return torch.nested.nested_tensor_from_jagged(head, micro_batch["input_ids"].offsets())


def split_sc_loss(loss_function):
    """verl's actor loss (functools.partial(ppo_loss, config=actor_config)) with DLC's centering term: the engine calls it
    with `student_logits` as the logits processor and with `model_output` as the loss."""
    config = getattr(loss_function, "keywords", {}).get("config", None)
    if getattr(getattr(loss_function, "func", None), "__name__", "") != "ppo_loss" or config is None:
        raise RuntimeError(f"DLC centering wraps verl's ppo_loss only, got {loss_function!r}")
    if config.policy_loss.get("loss_mode", "vanilla") != "vanilla":
        raise RuntimeError("DLC centering needs actor.policy_loss.loss_mode=vanilla (the clip indicator is its)")
    parts_of = SC_PARTS

    def processor(student_logits, data):
        import torch
        from verl.trainer.ppo.score_centering import topk_log_probs_from_logits
        from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size

        from tim.dlc.centering import split_sc_correction

        if get_ulysses_sequence_parallel_world_size() > 1:
            raise NotImplementedError("DLC centering is not built for Ulysses sequence parallelism")
        missing = [k for k in ("rollout_topk_ids", "rollout_topk_log_probs", REF_FIELD) if k not in data.keys()]
        if missing:
            raise RuntimeError(f"DLC centering: the micro-batch has no {missing} (rollout.topk_log_probs > 0 and "
                               f"the reference forward marked {REF_FLAG} are needed)")
        ids = data["rollout_topk_ids"].values().long()
        low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
        high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
        train = topk_log_probs_from_logits(student_logits.squeeze(0), ids)
        clip = dict(clip_low=low, clip_high=high, clip_c=config.get("clip_ratio_c", 3.0))
        logq, logp0 = data["rollout_topk_log_probs"].values(), data[REF_FIELD].values()
        out = split_sc_correction(train, logq, logp0, **clip)
        return {f"tim_sc_{k}": out[k].unsqueeze(0) for k in parts_of}

    def loss(model_output=None, data=None, dp_group=None, student_logits=None, **_):
        if student_logits is not None:
            return processor(student_logits, data)
        from verl.trainer.ppo.core_algos import agg_loss
        from verl.utils.metric import AggregationType, Metric
        from verl.utils.torch_functional import masked_mean
        from verl.workers.utils.padding import no_padding_2_padding

        from tim.dlc.centering import pick

        if "tim_sc_pos" not in model_output:
            raise RuntimeError("DLC centering: the logits processor did not run (score_centering micro-batch flag)")
        if "rollout_is_weights" in data.keys():
            raise RuntimeError("DLC centering replaces the IS weight of the discrepancy factor: rollout_is must be null")
        parts = {k: no_padding_2_padding(model_output.pop(f"tim_sc_{k}"), data) for k in parts_of}   # (bsz, response)
        total, metrics = loss_function(model_output=model_output, data=data, dp_group=dp_group)       # PPO clip, vanilla
        padded = data.select("response_mask", "advantages").to_padded_tensor()
        mask, advantages = padded["response_mask"].to(bool), padded["advantages"]
        sc = pick(advantages, parts["pos"], parts["neg"])
        extra = agg_loss(loss_mat=advantages * sc, loss_mask=mask, loss_agg_mode=config.loss_agg_mode,
                         **config.global_batch_info)          # ppo_loss just filled global_batch_info for this batch
        total = total + extra
        metrics["actor/split_sc_loss"] = Metric(value=extra.detach(), aggregation=metrics["actor/pg_loss"].aggregation)
        with_no_grad = {
            "actor/sc_correction": sc.detach(),
            "actor/split_sc_abs_mean": sc.detach().abs(),
            "actor/split_sc_q_mass": parts["q_mass"], "actor/split_sc_p0_mass": parts["p0_mass"],
            "actor/split_sc_p_mass": parts["p_mass"],
            "actor/split_sc_head_clipfrac": pick(advantages, parts["clip_pos"], parts["clip_neg"]),
        }
        metrics.update(Metric.from_dict({k: masked_mean(v, mask).item() for k, v in with_no_grad.items()},
                                        aggregation=AggregationType.MEAN))
        return total, metrics

    loss.tim_split_sc = True
    return loss


def patch_engine(module) -> str:
    cls = module.FSDPEngineWithLMHead
    outputs, step = cls.prepare_model_outputs, cls.forward_step
    if getattr(step, "tim_split_sc", None):
        return "applied"
    from verl.utils import tensordict_utils as tu

    def prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
        head = None
        if tu.get_non_tensor_data(data=micro_batch, key=REF_FLAG, default=False):
            head = ref_head_log_probs(self, output, output_args, micro_batch)
        model_output = outputs(self, output=output, output_args=output_args, micro_batch=micro_batch,
                               logits_processor_func=logits_processor_func)
        if head is not None:
            model_output[REF_FIELD] = head
        return model_output

    def forward_step(self, micro_batch, loss_function, forward_only):
        if loss_function is not None and not forward_only and not getattr(loss_function, "tim_split_sc", False):
            tu.assign_non_tensor(micro_batch, score_centering=True)
            loss_function = split_sc_loss(loss_function)
        return step(self, micro_batch, loss_function=loss_function, forward_only=forward_only)

    forward_step.tim_split_sc = True
    cls.prepare_model_outputs, cls.forward_step = prepare_model_outputs, forward_step
    return "applied (reference head log-probs; DLC centering term in the actor loss)"


# ---------------------------------------------------------------------------------------------------------------------
# DLC with resynchronization: the reference worker takes the trainer weights at every sync (worker processes)
# ---------------------------------------------------------------------------------------------------------------------
def copy_trainer_to_ref(src, dst) -> dict:
    """Copy the trainer module's parameters into the reference module in place, FSDP2 local shard to local shard.

    Both modules come from the same FSDPEngine._build_model_optimizer on the same ranks (create_device_mesh), so names,
    meshes and placements agree; anything else raises. Both are resharded first: FSDP2 keeps the root group unsharded
    after a no-grad forward (the reference's last log-prob call), and then named_parameters() returns the unsharded
    tensors and the next forward would reuse them. Values are cast to the reference's dtype and device (fp32 shards on
    the GPU -> the reference's fp32 shards, CPU-pinned under its CPUOffloadPolicy). Returns {tensors, numel}."""
    import torch
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    for module in (src, dst):
        if not isinstance(module, FSDPModule):
            raise RuntimeError(f"split ref reload needs FSDP2 (actor.strategy=fsdp2), got {type(module).__name__}")
        for m in module.modules():
            if isinstance(m, FSDPModule):
                m.reshard()
    s, d = dict(src.named_parameters()), dict(dst.named_parameters())
    if s.keys() != d.keys():
        raise RuntimeError(f"split ref reload: parameter names differ, e.g. {sorted(set(s) ^ set(d))[:5]}")
    numel = 0
    with torch.no_grad():
        for name, p in d.items():
            q = s[name]
            if isinstance(p, DTensor) != isinstance(q, DTensor):
                raise RuntimeError(f"split ref reload: {name} is a DTensor on one side only (unsharded module?)")
            if isinstance(p, DTensor):
                if p.placements != q.placements or p.device_mesh.mesh.tolist() != q.device_mesh.mesh.tolist():
                    raise RuntimeError(f"split ref reload: {name} sharded differently ({q.placements} on "
                                       f"{q.device_mesh} vs {p.placements} on {p.device_mesh})")
                p, q = p.to_local(), q.to_local()      # no grad: the shard itself (a view of FSDP2's sharded storage)
            if p.shape != q.shape:
                raise RuntimeError(f"split ref reload: {name} local shapes {tuple(q.shape)} vs {tuple(p.shape)}")
            p.copy_(q)
            numel += p.numel()
    return {"tensors": len(d), "numel": numel}


def reload_ref(trainer_engine) -> dict:
    """The reference engine of this process takes the trainer engine's weights (one reference per process, or raise)."""
    import time
    if len(REF_ENGINES) != 1:
        raise RuntimeError(f"DLC resynchronization: {len(REF_ENGINES)} reference (forward-only) engines in this worker, expected "
                           "1 (actor.use_kl_loss=True puts the reference in the actor_rollout_ref worker)")
    t0 = time.time()
    out = copy_trainer_to_ref(trainer_engine.module, REF_ENGINES[0].module)
    STATE["split_ref_reloads"] = STATE.get("split_ref_reloads", 0) + 1
    _say(f"split ref reload #{STATE['split_ref_reloads']}: {out['tensors']} tensors, {out['numel']} elements, "
         f"{time.time() - t0:.1f}s")
    return out


def patch_ref_reload(module) -> str:
    cls = module.FSDPEngine
    initialize, per_tensor = cls.initialize, cls.get_per_tensor_param
    if getattr(per_tensor, "tim_ref_reload", None):
        return "applied"

    def init(self, *args, **kwargs):
        out = initialize(self, *args, **kwargs)
        if self.engine_config.forward_only:
            REF_ENGINES.append(self)
        return out

    def get_per_tensor_param(self, *args, **kwargs):
        if not self.engine_config.forward_only:    # the trainer's export for a sync the driver decided to push
            reload_ref(self)
        return per_tensor(self, *args, **kwargs)

    get_per_tensor_param.tim_ref_reload = True
    cls.initialize, cls.get_per_tensor_param = init, get_per_tensor_param
    return "applied (every weight sync also loads the trainer weights into the reference engine: pi_old = sampler)"


def apply() -> dict:
    mode = os.environ.get(ENV_OLD, "")
    if not mode:
        return {"split": "off"}
    if mode == "async":   # DLC under continuous asynchrony: pi_old per token from tim.dlc.async_hooks (no reference)
        engine = engine_factor()
        if engine != "sc":
            raise ValueError(f"{ENV_OLD}=async needs {ENV_ENGINE}=sc (the centering term), got {engine!r}")
        when_imported("verl.trainer.ppo.v1.trainer_base", _strict("split_sc_check", patch_sc_check))
        when_imported("verl.workers.engine.fsdp.transformer_impl", _guard("split_sc_engine", patch_engine))
        return {"split": mode, "split_engine": engine, "split_ref_reload": False}
    if mode != "ref":
        raise ValueError(f"{ENV_OLD}={mode!r}: ref or async")
    engine = engine_factor()
    reload = split_ref_reload()
    if not reload:
        check_window(int(os.environ["TIM_STALENESS"]), int(os.environ[ENV_MAX_STEP]))
    when_imported("verl.trainer.ppo.v1.trainer_base", _guard("split_old", patch_trainer))
    if engine == "sc":
        when_imported("verl.workers.engine.fsdp.transformer_impl", _guard("split_sc_engine", patch_engine))
    if reload:
        when_imported("verl.workers.engine.fsdp.transformer_impl", _strict("split_ref_reload_engine", patch_ref_reload))
    return {"split": mode, "split_engine": engine, "split_ref_reload": reload}


try:
    STATE.update(apply())
    if STATE.get("split", "off") != "off":
        _say(f"DLC armed: pi_old {os.environ.get(ENV_OLD)}, discrepancy factor {STATE.get('split_engine')} "
             f"(staleness {os.environ.get('TIM_STALENESS')}, steps {os.environ.get(ENV_MAX_STEP)}, "
             f"ref reload at every sync: {STATE.get('split_ref_reload')})")
except Exception as _error:  # bad env must stop the run, not silently train a different method
    raise RuntimeError(f"tim.dlc.hooks: invalid TIM_SPLIT_* settings: {_error}") from _error
