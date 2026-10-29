"""Offline split and residual bias on trained checkpoints (no training).

4a  residual bias b = g_corr - g_on of a run's correction on a regenerated batch, against the on-policy gradient noise
4b  importance-ratio tail: share of sampled tokens with p/q below 0.5 or above the truncation, and their delta mass
4c  exact per-token split of the sampled-token error delta = log p_theta - log q into the policy lag
    l = log p_theta - log p_theta_old and the numerical discrepancy e = log p_theta_old - log q

`pure` is numpy only; `job` needs a GPU (scripts/measure.py split); `summary` writes TABLES.md and steps.csv.
"""
