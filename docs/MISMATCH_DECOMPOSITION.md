# Decomposition of the training-inference mismatch

This note organizes the mismatch by the layer that can change it, which is the structure of the paper's analysis and of
DLC. The infrastructure decides how far the sampler is from the trainer and which information the trainer receives;
the algorithm builds an estimator from the samples it is given; and an interface connects the two in both directions.
An error can often be reduced by either layer, at a cost in throughput on the infrastructure side and in variance on
the algorithm side.

## 0. Target and estimator

The target is the on-policy gradient of the trainer at its weights θ with its own numerics,

    G = E_{y∼p} [ Σ_t A(y) s_t ],   s_t = ∇ log p(y_t | h_t),   p = p_θ,   A = R − b.

The estimator uses samples from the sampler q. The sampled-token error is δ_t = log p(y_t|h_t) − log q(y_t|h_t), and
any factor that moves the expectation of the estimator away from G is a mismatch.

## 1. Infrastructure layer: directions of the mismatch

The infrastructure returns a batch of responses, the sampler's q on its head H, and (under asynchrony) the weights
version k(t) of each token. The directions (paper, `tab:directions`):

| Direction | What differs between sampler and trainer | Grows with the updates | Controlled by | In the paper |
|---|---|---|---|---|
| weights version | the weights that generate each token | yes | infrastructure and algorithm | studied with the stale sampler |
| engine numerics | the computation at equal weights | no | infrastructure and algorithm | studied with the quantized sampler |
| sampling and reporting | the distribution sampled and the one reported | no | infrastructure | removed by the configuration |
| sequence identity | token sequence and context | no | infrastructure | removed by the configuration |
| response selection | which responses enter the batch | no | infrastructure | not studied |

The weights version gives the policy lag ℓ, and the engine numerics give the numerical discrepancy e. In the paper's
configuration these are the only two terms that reach the algorithm, δ_t = ℓ_t + e_t, and the trainer's forward pass
at the sampler's weights splits δ_t into the two exactly.

## 2. Algorithm layer: exact bias of the estimator

Take score centering as the example. Let c_t(h) = Σ_v q(v|h) s(v|h) be the sampler's mean score and ĉ_t its head
estimate (q replaced by κ p outside H). For π ∈ {p, q}, μ^π_t is the distribution of the prefix h_t under π, and
a^π_t(h, v) = E_π[A | h_t = h, y_t = v] is the expected advantage after choosing v, with π generating the rest.

At a prefix, Σ_v q_v a_v (s_v − c) = Cov_q(a, s), and the subtracted ā^q c is the drift, which centering removes
exactly. The target at the same prefix is Cov_p(a, s), since E_p[s] = 0. Replacing the prefix distribution, the
continuation, the current token and the head estimate one at a time gives the bias as the sum of four terms (paper,
Eq. `eq:exact`):

- **(I) prefix term** Σ_t (E_{μ^q_t} − E_{μ^p_t}) Cov_q(a^q_t, s_t). Its likelihood ratio is e^{−S_t} with
  S_t = Σ_{j<t} δ_j; to first order it is the A_t S_t term of the paper.
- **(II) continuation term** Σ_t E_{μ^p_t} [Cov_q(a^q_t, s_t) − Cov_q(a^p_t, s_t)]. After y_t the response continues
  under q, with the product of e^{−δ_j} over j > t as likelihood ratio. Together with (I) it is the distribution mismatch
  of the whole trajectory except y_t.
- **(III) token term** Σ_t E_{μ^p_t} [Cov_q(a^p_t, s_t) − Cov_p(a^p_t, s_t)]. Substituting q_v = p_v e^{−δ_v} gives
  exactly E_p[(e^{−δ_v} − 1)(a_v − ā^p) s_v] − (ā^q − ā^p) c; to first order −E_p[(A − Ā) δ s]. It vanishes when the
  current token does not affect the advantage.
- **(IV) tail term** Σ_t E_{μ^q_t} [ā^q_t (c_t − ĉ_t)], with c_t − ĉ_t = Σ_{v∉H} (q_v − κ p_v) s_v. It vanishes when
  q is proportional to p on the tail.

A constant baseline b is orthogonal to the score under p and under the centered q, so it enters none of the terms.

**How δ = ℓ + e enters.** Terms (I) to (III) are set by δ. The part from e has a fixed size, while the part from ℓ grows
with the accumulated update Δ and returns into Δ through the update, which closes the loop
Δ_{k+1} = Δ_k + η (g_k + b_k). The lag is correlated with the advantage (tokens with negative advantage carry the larger
lag under score centering), so the token term of ℓ is a systematic bias. Term (IV) is set by the infrastructure's H and
does not grow with Δ.

**Corrections, term by term.**

- drift: centering, exactly.
- (III): a ratio on the current token. For ℓ, the clip of the lag ratio against the generating weights (DLC) reduces the
  first-order token term to the truncation error inside the validity range of the ratio. For e, an untruncated
  p_old / q weight is possible; with a light-tailed e the truncation rarely acts and centering becomes a control
  variate.
- (I) and (II): the variance of trajectory-level weights grows with the length, so the algorithm keeps S_t small instead,
  which requires the interface to reset the version gap.
- (IV): the algorithm cannot shrink it alone; the interface asks the infrastructure for a larger H.

## 3. Interface between the layers

**From the infrastructure to the algorithm.** q on H and the version k(t). Without k(t) the trainer cannot compute ℓ
and the clip has no reference; with a small H, term (IV) stays in the gradient.

**From the algorithm to the infrastructure.** When the effective sample size 1/E_q[ρ²] of the lag ratio falls below
one half, the algorithm triggers a resynchronization under periodic synchronization, or defers the response and has it
regenerated with newer weights under continuous asynchrony (ESS admission). This changes the version gap of the
infrastructure and resets the ℓ part of (I) to (III) together with S_t.

The loop passes through the interface: the version gap produces ℓ, ℓ enters the update through the residual bias of
the algorithm, and the update widens the version gap, until the algorithm resets it through the interface. The tail of
the ratio sets the time of the reset: on Qwen3-1.7B the ESS of importance sampling falls below one half at step 11,
three steps before the Gaussian approximation predicts.

## 4. The argument in order

1. RL training samples with an inference engine and computes gradients with a trainer; the infrastructure sets their
   difference. Configuration-type errors are removed in the infrastructure, and e and ℓ reach the algorithm.
2. One extra forward pass of the trainer at the sampler's weights, with the trainer's numerics, splits δ_t exactly into
   e_t and ℓ_t. The discrepancy e does not grow with Δ; the lag ℓ does (paper, Sec. 4).
3. The bias of the algorithm is the drift plus four terms. Centering removes the drift exactly; for e it leaves the
   fixed-size terms (I) to (IV); for ℓ the token and prefix terms grow with Δ and feed back into Δ, forming the loop
   (Sec. 5).
4. Ratio weighting removes the token term to first order, but only inside the validity range of the ratio. The tail of
   the ratio grows heavy earlier than a Gaussian lag predicts, and the crossing of the ESS below one half marks the end
   of the range (Sec. 6).
5. DLC follows the layers: centering for e and the clip against the generating weights for ℓ in the algorithm, and a
   resynchronization through the interface when the ESS falls below one half, which resets ℓ and S_t (Sec. 7).
6. Continuous asynchrony is another distribution of version gaps in the infrastructure. The algorithm replaces the
   denominator of the clip by the per-token version, and the interface replaces the global resynchronization by ESS
   admission (App. C, `docs/ASYNC.md`).
