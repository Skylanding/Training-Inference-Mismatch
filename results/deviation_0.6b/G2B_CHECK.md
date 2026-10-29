# Order check at step 40

Generated during the study from `results/deviation_0.6b` (the checking script is not part of this release). Outcome: KL to the same-seed reference, mean over seeds [0, 1, 2]. Noise floor (bf16 PG re-runs): 0.03219 [0.01367, 0.05457, 0.02833]; ordered only if the difference exceeds 2 x floor = 0.06438. Held-out accuracy floor (mean |diff| of the re-runs): 0.008464.
**15 of 20 predicted relations are consistent with the observed ordering.**


## fp8: engine_error, recommended sc

| group | KL mean [per seed] | KL / floor | held-out acc diff mean [per seed] |
|---|---|---|---|
| PG | 0.4103 [0.3331, 0.3963, 0.5015] | 12.7 | -0.202 [-0.232, -0.195, -0.178] |
| TIS | 0.0238 [0.02863, 0.02012, 0.02264] | 0.739 | -0.034 [-0.055, -0.023, -0.023] |
| SC | 0.0256 [0.03276, 0.01751, 0.02652] | 0.795 | -0.001 [+0.004, -0.008, +0.002] |
| TIS+SC | 0.03567 [0.03877, 0.03919, 0.02905] | 1.11 | -0.006 [-0.006, -0.006, -0.006] |

| predicted | observed (mean) | consistent | per seed | by held-out accuracy (|diff|, information) |
|---|---|---|---|---|
| SC ~ TIS+SC | SC ~ TIS+SC (0.0256 vs 0.0357) | yes | ~ ~ ~ | ~ (0.001 vs 0.006) |
| TIS+SC < TIS | TIS+SC ~ TIS (0.0357 vs 0.0238) | **no** | ~ ~ ~ | < (0.006 vs 0.034) |
| SC < TIS | SC ~ TIS (0.0256 vs 0.0238) | **no** | ~ ~ ~ | < (0.001 vs 0.034) |
| TIS < PG | TIS < PG (0.0238 vs 0.41) | yes | < < < | < (0.034 vs 0.202) |

## stale64: policy_lag, recommended split (engine factor sc)

| group | KL mean [per seed] | KL / floor | held-out acc diff mean [per seed] |
|---|---|---|---|
| PG | 11.09 [14.42, 10.18, 8.677] | 345 | -0.464 [-0.465, -0.467, -0.459] |
| TIS | 0.145 [0.1515, 0.1278, 0.1558] | 4.51 | -0.257 [-0.295, -0.215, -0.260] |
| SC | 1.023 [1.132, 0.9404, 0.9963] | 31.8 | -0.343 [-0.350, -0.295, -0.385] |
| TIS+SC | 0.06768 [0.04542, 0.08036, 0.07726] | 2.1 | -0.145 [-0.162, -0.129, -0.145] |
| split + engine factor | 0.06232 [0.06502, 0.05658, 0.06536] | 1.94 | -0.148 [-0.131, -0.143, -0.170] |

| predicted | observed (mean) | consistent | per seed | by held-out accuracy (|diff|, information) |
|---|---|---|---|---|
| split + engine factor < TIS+SC | split + engine factor ~ TIS+SC (0.0623 vs 0.0677) | **no** | ~ ~ ~ | ~ (0.148 vs 0.145) |
| TIS+SC < SC | TIS+SC < SC (0.0677 vs 1.02) | yes | < < < | < (0.145 vs 0.343) |
| TIS+SC < TIS | TIS+SC < TIS (0.0677 vs 0.145) | yes | < ~ < | < (0.145 vs 0.257) |
| SC < PG | SC < PG (1.02 vs 11.1) | yes | < < < | < (0.343 vs 0.464) |
| TIS < PG | TIS < PG (0.145 vs 11.1) | yes | < < < | < (0.257 vs 0.464) |

## fp8_stale64: policy_lag, recommended split (engine factor sc)

| group | KL mean [per seed] | KL / floor | held-out acc diff mean [per seed] |
|---|---|---|---|
| PG | 3.441 [2.637, 3.834, 3.852] | 107 | -0.463 [-0.455, -0.467, -0.467] |
| TIS | 0.1256 [0.08982, 0.1368, 0.1502] | 3.9 | -0.195 [-0.182, -0.193, -0.209] |
| SC | 0.9057 [0.7389, 0.9948, 0.9836] | 28.1 | -0.281 [-0.262, -0.324, -0.258] |
| TIS+SC | 0.08421 [0.09305, 0.08913, 0.07043] | 2.62 | -0.161 [-0.188, -0.146, -0.148] |
| split + engine factor | 0.06997 [0.05915, 0.0806, 0.07017] | 2.17 | -0.162 [-0.158, -0.178, -0.150] |

| predicted | observed (mean) | consistent | per seed | by held-out accuracy (|diff|, information) |
|---|---|---|---|---|
| split + engine factor < TIS+SC | split + engine factor ~ TIS+SC (0.07 vs 0.0842) | **no** | < ~ ~ | ~ (0.162 vs 0.161) |
| TIS+SC < SC | TIS+SC < SC (0.0842 vs 0.906) | yes | < < < | < (0.161 vs 0.281) |
| TIS+SC < TIS | TIS+SC ~ TIS (0.0842 vs 0.126) | **no** | ~ ~ < | < (0.161 vs 0.195) |
| SC < PG | SC < PG (0.906 vs 3.44) | yes | < < < | < (0.281 vs 0.463) |
| TIS < PG | TIS < PG (0.126 vs 3.44) | yes | < < < | < (0.195 vs 0.463) |

## eager: noise, recommended none

| group | KL mean [per seed] | KL / floor | held-out acc diff mean [per seed] |
|---|---|---|---|
| PG | 0.01958 [0.01467, 0.02617, 0.01788] | 0.608 | +0.016 [+0.021, +0.006, +0.021] |
| TIS | 0.01965 [0.02772, 0.01335, 0.01789] | 0.611 | -0.003 [-0.002, +0.020, -0.025] |
| SC | 0.02081 [0.02705, 0.01343, 0.02196] | 0.646 | +0.005 [+0.000, +0.008, +0.008] |
| TIS+SC | 0.02391 [0.02999, 0.01878, 0.02297] | 0.743 | +0.007 [+0.035, -0.020, +0.006] |

| predicted | observed (mean) | consistent | per seed | by held-out accuracy (|diff|, information) |
|---|---|---|---|---|
| PG ~ TIS | PG ~ TIS (0.0196 vs 0.0197) | yes | ~ ~ ~ | ~ (0.016 vs 0.003) |
| PG ~ SC | PG ~ SC (0.0196 vs 0.0208) | yes | ~ ~ ~ | ~ (0.016 vs 0.005) |
| PG ~ TIS+SC | PG ~ TIS+SC (0.0196 vs 0.0239) | yes | ~ ~ ~ | ~ (0.016 vs 0.007) |
| TIS ~ SC | TIS ~ SC (0.0197 vs 0.0208) | yes | ~ ~ ~ | ~ (0.003 vs 0.005) |
| TIS ~ TIS+SC | TIS ~ TIS+SC (0.0197 vs 0.0239) | yes | ~ ~ ~ | ~ (0.003 vs 0.007) |
| SC ~ TIS+SC | SC ~ TIS+SC (0.0208 vs 0.0239) | yes | ~ ~ ~ | ~ (0.005 vs 0.007) |

