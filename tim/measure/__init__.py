"""Offline measurements on saved checkpoints (CLI: scripts/measure.py; configs: configs/measure/).

config       the model's measurement settings (configs/measure/<model>.json)
artifacts    the fixed probe set and P2 batch (sequences packed as int32 + offsets, sha256 recorded)
engines      vLLM samplers (bf16, quantized fp8, eager) and HF trainers, built once per process and refit in place
heads        sampler and trainer log-probs on the sampler's top-128 head and on the sampled token
formulas     pure-numpy statistics of the replay (sampled-token error, ratio truncation, sequence offset, exact split)
tail_model   the head + rescaled-tail model of the sampler (as verl's score centering)
deviation    full-vocabulary KL to the paired on-policy model on the probe prefixes, held-out accuracy
split        residual bias b (4a), ratio tail (4b), exact per-token split into lag and discrepancy (4c)
loop         residual-bias prediction from the lag weights (linearised and exact) and the lag by advantage sign
off_support  the trainer's mass off the support of theta_0 on its own responses
ess          report of DLC with ESS-triggered resynchronization from the training logs
"""
