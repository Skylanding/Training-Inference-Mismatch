"""Mass off the support of the initial policy, measured offline on saved checkpoints (no training): held-out accuracy,
the sequence KL of the trainer's own responses to theta_0 and its off-support share (tokens with p_theta0(y) < 1e-3), and
the KL to theta_0 on theta_0's own samples. `pure` is numpy only; `job` needs a GPU (scripts/measure.py off-support);
`summary` writes results/off_support/TABLES.md and steps.csv."""
