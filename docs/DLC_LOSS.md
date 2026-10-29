# The DLC loss and its implementation in verl

This note derives the centering term of DLC (paper, Sec. 7 and App. L) in the form the code computes it, and lists
where each piece lives in verl 8718ca30. Code: `tim/dlc/centering.py` (math), `tim/dlc/hooks.py` (verl patches).

## Notation

At a response position t:

- q: the sampler's next-token distribution, reported on its top-128 head H;
- p = p_θ: the trainer at its current weights;
- p_old = p_θ_old: the trainer at the sampler's weights θ_old, computed with the trainer's numerics
  (for the stale sampler without resynchronization, θ_old = θ_0 for the whole 40-step training);
- a: the sampled token, A its advantage, s_v = ∇ log p(v) the score of token v;
- ρ_v = p(v) / p_old(v): the lag ratio; m_v: the PPO clip indicator of ρ_v (1 where the clipped loss has a gradient).

The sampled-token error splits exactly at p_old into the policy lag and the numerical discrepancy,

    log p(a) − log q(a) = [log p(a) − log p_old(a)] + [log p_old(a) − log q(a)] = ℓ_t + e_t.

## The bias that the clip leaves

DLC trains PPO's clipped surrogate on the lag ratio, with no importance weight for the discrepancy factor
p_old / q. The gradient on the sampled token is A · m_a · ρ_a · s_a. If a were drawn from p_old, the unclipped
expectation would be Σ_v p_old(v) ρ_v s_v = Σ_v p(v) s_v = 0. Since a is drawn from q, the expectation at a prefix with
a constant advantage is

    C = Σ_v q(v) m_v ρ_v s_v,

the drift of the numerical discrepancy for this estimator. DLC subtracts C at each sampled token, as score centering
subtracts the sampler's mean score.

## C from the head only

The sampler reports q only on H. As in verl's score centering, the tail of q is taken as the tail of p_old scaled by
the ratio of the tail masses, q(v) = κ p_old(v) for v ∉ H with κ = (1 − Σ_H q) / (1 − Σ_H p_old), and the clip is
inactive on the tail. Then q m ρ = κ p on the tail, and Σ_v p(v) s_v = 0 turns the tail sum into minus the head sum:

    C = Σ_{v∈H} [ q(v) m_v ρ_v − κ p(v) ] s_v.

The loss adds A · Σ_{v∈H} sg[ q(v) m_v ρ_v − κ p(v) ] · log p(v) (sg = stop gradient), which is how verl's
`score_centering_correction` enters the REINFORCE loss, with the weight w(p/q) replaced by m ρ and verl's tail
coefficient replaced by κ (named `rho0` in the code).

Two limits:

- no discrepancy (q = p_old): κ = 1 and the bracket is p_old ρ − p = 0, so DLC is PPO with the sampler's weights as
  reference;
- no lag (θ = θ_old, ρ = 1, m = 1): the bracket is q − κ p, which is verl's score centering.

## Differences to verl's score centering

| | verl score centering (bypass + REINFORCE) | DLC (decoupled + PPO clip) |
|---|---|---|
| head weight | w(p/q), truncated for importance sampling | m_v ρ_v, ρ relative to p_old |
| tail coefficient | relative to the current p | κ, relative to p_old |
| extra input | none | log p_old on the 128 head ids (the reference forward returns an N × 128 tensor) |
| importance weight of the discrepancy factor | optional | none (`rollout_is=null`) |

The clip indicator follows verl's `compute_policy_loss_vanilla`: for A > 0, m = [ρ ≤ 1 + ε]; for A < 0,
m = [1 − ε ≤ ρ < c], with ε = 0.2 and the dual-clip bound c = 3.

## Implementation (`METHOD=dlc` / `dlc_noresync` in `scripts/train.sh`, `TIM_SPLIT_ENGINE=sc`)

The installed verl is not modified; every change is a patch applied through `VERL_USE_EXTERNAL_MODULES`.

| Step | What the code does | Process |
|---|---|---|
| configuration | `bypass_mode=false`, `loss_mode=vanilla`, `rollout_is=null`, `rollout.topk_log_probs=128`, fused log-prob kernels off; `score_centering` stays false in the config (verl records the sampler's head whenever `topk_log_probs > 0`) | launcher |
| p_old | `PPOTrainer._compute_old_log_prob` is wrapped: after verl's own θ_t forward, the reference worker's log-probs are written as `old_log_probs`. The reference worker starts at θ_0 (`use_kl_loss=True`, `kl_loss_coef=0`) | driver |
| head of p_old | `FSDPEngineWithLMHead.prepare_model_outputs` is patched: on the marked reference forward it computes log p_old at `rollout_topk_ids` from the logits and returns `ref_topk_log_probs` ([B, j, k], the layout of `rollout_topk_log_probs`) | reference worker |
| centering term | `FSDPEngineWithLMHead.forward_step` is patched: a training micro-batch gets verl's `score_centering` flag (the engine then calls the loss function as logits processor and keeps the logits for the backward), and the loss function is wrapped. As processor it returns C for both advantage signs; as loss it returns verl's `ppo_loss` plus `agg_loss(A · C)`, with C taken per token by the sign of A | actor worker |
| guard | `PPOTrainer._update_actor` is wrapped: an update without `actor/split_sc_abs_mean` (the worker patch did not run) or with `rollout_is` set stops the run | driver |
| resynchronization | the lag chi2 E_q[ρ²] − 1 on the sampled tokens is logged as `tim/split_chi2_token`; `tim.dlc.resync` pushes the trainer's weights to the sampler after a step whose chi2 exceeds 1/ESS_MIN − 1 = 1 (ESS < 1/2), and every push also copies the trainer's FSDP2 shards into the reference worker, so p_old follows the sampler | driver, actor worker |

Per-step metrics: `actor/split_sc_abs_mean` (|C|), `actor/sc_correction`, `actor/split_sc_loss`,
`actor/split_sc_{q,p0,p}_mass` (head masses), `actor/split_sc_head_clipfrac` (the sampler's head mass on clipped
tokens), `tim/split_lag_{mean,abs_mean,abs_max}`, `tim/split_chi2_token`, `tim/synced`; the driver mirrors the
`actor/split_sc_*` metrics under `tim/`.

## Details of the implementation

1. The clip indicator depends on the sign of A, and the logits processor does not see per-row advantages, so C is
   computed for both signs and the loss picks one per token.
2. With q = p_old the term vanishes only when no head token is clipped. In general it is the expectation under q of
   the clipped estimator, including the part the clip itself introduces, Σ_H p (m − 1) s; DLC centers the whole
   clipped estimator.
3. log ρ is clamped to ±20 as in the vanilla loss; the two tail masses of κ have a floor of 1e-6 as in verl.
4. The term is added after `ppo_loss` returns rather than passed through its `policy_loss_fn`, which does not accept
   it; the result is the same.
5. Ulysses sequence parallelism, fused log-prob kernels and padded (non-packed) micro-batches are not supported and
   raise.
6. verl's KL path runs the reference forward a second time in each step. With `kl_loss_coef = 0` this output does not
   enter the loss; it accounts for about 19 of the 54 seconds of the forward at the sampler's weights in `tab:cost`.
   `TIM_SPLIT_SKIP_KL_REF=1` skips it (off in the paper's runs).
7. Right after a synchronization the lag is not exactly zero: θ_t = θ_old, but the actor and the reference worker run
   two separate bf16 forwards with different micro-batch layouts, so the first step shows a lag of about 0.01 nats and a
   clip fraction of about 4e-4.
