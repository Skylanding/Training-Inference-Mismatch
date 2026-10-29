"""Code release of "On the Origin and Evolution of Training-Inference Mismatch in LLM Reinforcement Learning".

tim.dlc        discrepancy-lag correction (DLC): verl hooks for the lag-ratio clip, the centering term and the
               ESS-triggered resynchronization, plus the continuous-asynchrony variant
tim.baselines  M2PO and VCPO on the same verl stack
tim.tasks      the Countdown task and its +-1 reward
tim.measure    offline measurements on saved checkpoints (deviation, exact split, residual bias, loop, off-support share)
"""
