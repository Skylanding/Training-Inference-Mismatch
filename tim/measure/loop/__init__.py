"""Training loop of the policy lag, measured offline on the stale-sampler checkpoints (no training).

6a1  the residual-bias link: the residual bias b = g_corr - g_on predicted from the per-token lag weights (linearised), with
     the token and the prefix term separately
6a2  where the lag sits: the lag split by the sign of the advantage (and of A - Abar) and the largest-lag tokens
exact  the same prediction with the exact weights (prefix A (1 - e^{S_t}), score centering token A (1 - e^{delta_t}),
       importance sampling token -A max(e^{delta_t} - 2, 0)), side by side with the linearised one

`pure` is numpy only; `job` needs a GPU (scripts/measure.py loop / residual-bias-exact); `summary` writes the tables."""
