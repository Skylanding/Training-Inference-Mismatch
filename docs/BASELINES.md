# M2PO and VCPO on verl 8718ca30

The method-specific parts (ratio, mask, weights, advantage baseline, learning-rate scale, loss aggregation) follow the
official code; the shared training configuration (optimizer, learning rate, batch, steps, no gradient clipping, no
KL) is the same as for every other correction. The math is in `tim/baselines/weights.py` and tested against the
official definitions in `tests/test_baseline_weights.py`; the verl glue is `tim/baselines/hooks.py`.

- M2PO: Zheng, Zhao, Chen, "Prosperity before Collapse" (arXiv 2510.01161), github Infini-AI-Lab/M2PO @ af54a3e.
- VCPO: Huang, Zhang, Hu, Yang, Han, "Stable Asynchrony" (arXiv 2602.17616 v2), github mit-han-lab/vcpo @ c7f030f.

## M2PO (`METHOD=m2po`)

- **Ratio.** r = p_θ / p_θ_old, with θ_old the weights that generated the data, recomputed on the trainer (the official
  code does not use the sampler's log-probs). Under the stale sampler θ_old = θ_0, supplied by the reference worker
  through the same path as DLC (`TIM_SPLIT_OLD=ref` in `tim/dlc/hooks.py`), with `TIM_SPLIT_ENGINE=none`: no importance
  weight and no centering for the discrepancy factor. Under continuous asynchrony θ_old is the generating version of
  each token (`tim/dlc/async_hooks.py`).
- **Mask.** The advantages are computed after `_compute_old_log_prob`, so the mask is computed in a wrapper of
  `PPOTrainer._update_actor`, right before the update, from log p_θ_t (verl's own old-log-prob forward, stashed as
  `tim_logp_theta_t`), log p_θ_old and the advantages: `m2po_keep_mask(..., budget=0.04)`. Trust-region tokens are
  (A > 0 and r > 1) or (A < 0 and r < 1); if the mean of (log r)² over them exceeds 0.04, a single cap c on (log r)² is
  solved so that the capped mean equals 0.04, and the PPO clip with bounds [e^{−√c}, e^{√c}] applies. The 0/1 mask is
  written as `rollout_is_weights`, which verl's vanilla loss multiplies in. With one update per batch the ratio at the
  update is r(θ_t), so the mask gives the same gradient as the official clip at that ratio. The official code solves the
  cap per micro-batch; the paper describes it per batch, and the hook computes it on the whole batch.
- **Configuration.** `bypass_mode=false` (decoupled), `loss_mode=vanilla`, `clip_ratio_low = clip_ratio_high =
  clip_ratio_c = 1e9` (no PPO clip besides the mask), `rollout_is=null`, `loss_agg_mode=token-mean` (masked tokens stay
  in the denominator, as in the official code), reference worker on (`use_kl_loss=True`, `kl_loss_coef=0`).
- **Metrics.** `tim/m2po/{M2, M2_after, tau, tr_frac, masked_frac}` per step.

## VCPO (`METHOD=vcpo`)

All three parts are used (the official scripts set both `ess_scaling` and `grad_baselining`):

- **Sequence-level truncated IS.** In decoupled mode verl's π_old is the trainer before the update, θ_t;
  `rollout_is=sequence` with `rollout_is_threshold=8.0` gives w̄_i = min(exp Σ_t (log p_θ_t − log q), 8), as the official
  code with `skip_recompute_old_log_prob`. `loss_mode=vanilla` with the three clip parameters at 1e9 (no PPO clip) and
  `loss_agg_mode=seq-mean-token-mean`, as in the official code.
- **ESS learning-rate scale.** On the driver, before `_update_actor`, the untruncated weights
  w_i = exp Σ_t (log p_θ_t − log q) give the ESS ratio ρ = (Σ w)² / (Σ w² + 1e-8) / B and the scale
  s = sqrt(min(1, ρ / ρ_on)) with ρ_on = 1.0 (official Countdown scripts). With SGD without momentum or gradient
  clipping, scaling the learning rate by s equals scaling the gradient by s, so `rollout_is_weights` is multiplied by s.
  The ratio is computed in log space (a common factor cancels). Metrics `tim/vcpo/{ess_ratio, lr_scale}`.
- **Optimal off-policy baseline (OPOB).** It needs the norm of each response's own score gradient
  ‖∇_θ Σ_t log p_θ(y_t)‖². A worker RPC (`tim_score_norms`, registered on the actor worker with the dispatch of
  `update_actor`) runs one forward and backward per response (micro-batch of one, loss Σ_t log p), reads the
  FSDP-aware gradient norm and zeroes the gradients. All data-parallel ranks process the same response in the same
  round, so the reduced shard gradient is that response's gradient and no unsharded buffer is needed. The driver then
  computes the per-group baseline b_g = Σ_i W_i R_i / Σ_i W_i with W_i = w̄_i² ‖g_i‖² w_i² (the official ‖g_i‖ includes
  the truncated weight) and overwrites `advantages` with (R_i − b_g) / σ_g, broadcast to the tokens. Computed in log
  space, since w_i overflows over a few hundred tokens.
- **Cost.** The OPOB pre-pass adds one per-response forward and backward per step, about four times the step time of
  the other corrections.
