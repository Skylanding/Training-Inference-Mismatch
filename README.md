# On the Origin and Evolution of Training-Inference Mismatch in LLM Reinforcement Learning

Code, run configurations, measurement scripts and result tables of the paper (Yu Li, Yufa Zhou, Tian Lan).

In RL for language models an inference engine samples the responses and a separate trainer computes the gradients,
so the training data are off-policy. Evaluating the trainer at the sampler's weights splits every sampled-token error
exactly, log p_θ(y_t) − log q(y_t) = ℓ_t + e_t, into the **policy lag** ℓ_t = log p_θ − log p_θ_old, produced by the
trainer's own updates since the sampler's weights θ_old, and the **numerical discrepancy** e_t = log p_θ_old − log q of
the engine at equal weights. The numerical discrepancy stays approximately constant during training and biases each
update by a nearly fixed amount, which centering the score removes to leading order. The policy lag grows with the
trainer's movement and feeds back through the residual bias of each correction into later updates; importance weighting
cuts this loop only until the effective sample size (ESS) of the lag ratio falls below one half. The
**discrepancy-lag correction (DLC)** follows this split: it centers the discrepancy (a centering term C_t on the
sampler's top-128 head), clips the lag ratio ρ_t = p_θ / p_θ_old with PPO's clip against the sampler's weights, and
resynchronizes the sampler with the trainer when the ESS 1/E_q[ρ_t²] of the lag ratio falls below one half.


## Repository layout

```
tim/                    the package (verl hooks, task, offline measurements)
  dlc/                  DLC: hooks.py (lag-ratio clip against the sampler's weights + centering term), centering.py
                        (the math of C_t), resync.py / resync_config.py (stale sampler and ESS-triggered
                        resynchronization), async_hooks.py / async_versions.py (continuous asynchrony, ESS admission)
  baselines/            M2PO and VCPO (hooks.py, weights.py)
  tasks/                Countdown task and its +-1 reward
  measure/              offline measurements on saved checkpoints
  _patch.py, zmq_tag.py shared hook machinery; one weight-sync socket per run when several runs share a node
scripts/                train.sh (training launcher), run_list.sh, measure.py (offline measurements), report_*.py
                        (tables from measurement records and training logs), report_cost.py, build_countdown_data.py, convert_ckpt.py
configs/runs/           run lists: one line per training of the paper, grouped by table / figure
configs/measure/        settings of the offline measurements
configs/chat-templates/ Qwen3 chat template
results/                the measurement tables, training logs and figure data behind the paper
envs/                   locked environment (uv) and the FlashAttention-3 build
docs/                   notes on the DLC loss, the mismatch decomposition, the baselines and continuous asynchrony
tests/                  CPU tests; tests/gpu needs the training environment (TIM_GPU_TESTS=1)
```

## Installation

The trainings and measurements need Linux, NVIDIA H100 (or B200) GPUs and the locked environment of `envs/`
(verl `8718ca30`, vLLM 0.29.0, torch 2.13.0+cu129, transformers 5.12.1, Python 3.12):

```bash
bash envs/verl-main-cu129/build_env.sh                         # -> .venv-tim
PY=.venv-tim/bin/python FA3_PATH=$PWD/.fa3 bash envs/fa3/build_fa3.sh
export PATH=$PWD/.venv-tim/bin:$PATH FA3_PATH=$PWD/.fa3
```

The table scripts and the CPU tests need only Python 3.9+ with numpy and pytest
(torch for some tests). See `envs/README.md` for versions and notes.

## Data

```bash
python scripts/build_countdown_data.py        # data/countdown/{train,heldout}.parquet: 19,936 training and 64 test problems
```

The Countdown generator and reward are ported from martin-marek/score-centering (MIT License, kept in
`tim/tasks/countdown.py`); `tests/test_countdown.py` checks them against the original.

## Training

`scripts/train.sh` runs one training of the paper's setup on verl (Countdown, 64 prompts with eight samples per step,
SGD at learning rate 1e-2 without momentum, warmup or weight decay, no gradient clipping, GRPO advantages without
std normalization, rewards ±1, temperature 1 without truncation, FSDP2 with BF16 compute and FP32 master weights,
40 steps, a checkpoint every five steps). `METHOD` and `SOURCE` take the paper's names:

| `METHOD` | Paper | `SOURCE` | Paper |
|---|---|---|---|
| `on_policy` | on-policy training (reference and noise floor) | `none` | BF16 vLLM sampler (on-policy) |
| `pg` | policy gradient | `quantized` | FP8 weights (128x128 blocks), FP8 activations and KV cache |
| `ppo` | PPO with batch reference (4 updates per batch) | `stale` | BF16 sampler synchronized every 64 steps |
| `ppo_noclip` | PPO without clip | `quantized_stale` | the stale weights, quantized |
| `sc` | score centering | `async` | continuous asynchrony, maximum version gap 4 (two GPUs) |
| `is` | importance sampling, ratio capped at 2 | | |
| `is_sc` | importance sampling with centering | | |
| `is_resync` | importance sampling with resynchronization | | |
| `m2po`, `vcpo` | M2PO, VCPO | | |
| `dlc` | DLC | | |
| `dlc_noresync` | DLC without resynchronization | | |
| `dlc_async`, `dlc_admission` | DLC under continuous asynchrony, and with ESS admission | | |

```bash
# one configuration
MODEL=Qwen/Qwen3-1.7B METHOD=dlc SOURCE=stale SEED=0 RUN_PREFIX=s7d bash scripts/train.sh
# every training of a table (one line per training, run one after the other)
bash scripts/run_list.sh configs/runs/qwen3_1.7b_stale.txt
```

Other variables: `SEED` (vLLM seed 1000 + SEED), `STEPS`, `SAVE_FREQ`, `LR`, `N_GPUS` (8 for Qwen3-8B, 2 for
`SOURCE=async`), `ESS_MIN` (0.5), `KERNELS` (`fa3_liger`, or `sdpa` without FA3), `OUT_DIR` (default `outputs/`),
`DRY=1` to print the verl command. A training writes `OUT_DIR/<run>/ckpt/global_step_<t>` (verl checkpoints with an
HF copy of the weights) and `OUT_DIR/<run>/logs/metrics.<attempt>.jsonl` (verl's step metrics, which include the
`tim/*` metrics of the hooks, e.g. `tim/split_chi2_token`, `tim/synced`, `tim/split_lag_mean`).

### Run names

Run names follow `<prefix>_<source>_<method>_seed<k>` with the internal source and method names, which the measurement
scripts parse:

| Internal name | Paper | | Prefix | Trainings |
|---|---|---|---|---|
| `bf16`, `fp8`, `stale64`, `fp8_stale64`, `async4` | the five sources above | | `s2ref`, `s5ref`, `s8ref` | on-policy reference of Qwen3-0.6B / 1.7B / 8B |
| `budget`, `fp8_budget` | stale source with resynchronization | | `s2`, `s5`, `s8b` | trainings on Qwen3-0.6B / 1.7B / 8B |
| `pg`, `ppo`, `pg4`, `sc`, `tis`, `tis_sc` | pg, ppo, ppo_noclip, sc, is, is_sc | | `s6b` | Qwen3-0.6B at learning rate 5e-3 |
| `split` + `stale64` / `budget` | DLC without / with resynchronization | | `s6c`, `s7d` | Qwen3-1.7B, IS with resync and DLC w/o resync / DLC |
| `split_async`, `split_async_ess` | dlc_async, dlc_admission | | `s8a`, `s8a6`, `s9b` | M2PO / VCPO on the small models, continuous asynchrony |

## Offline measurements

`scripts/measure.py` measures saved checkpoints with a vLLM sampler refit to each checkpoint (BF16, FP8, the stale
θ_0, or eager mode) and an HF trainer with the training kernels. It reads checkpoints from `--ckpt-root` (default
`outputs/`, the layout `train.sh` writes), writes one JSON record per run and step under `results/`, skips steps whose
record exists, and summarizes the records into the shipped tables. Settings are in `configs/measure/`
(`qwen3_0.6b.json`, `qwen3_1.7b.json`, `qwen3_8b.json`, `loop.json`, `off_support.json`).

```bash
C06=configs/measure/qwen3_0.6b.json; C17=configs/measure/qwen3_1.7b.json
python scripts/measure.py build-probes --config $C06                      # probe prefixes and fixed batch from θ_0
# deviation: full-vocabulary KL to the paired on-policy model on the probe prefixes, and test accuracy
python scripts/measure.py deviation --config $C06 --run s2_stale64_sc_seed0
python scripts/measure.py deviation --config $C17 --run s7d_budget_split_seed0
python scripts/measure.py deviation --config configs/measure/qwen3_8b.json --run s8b_budget_split_seed0
# replay of each sampler on the on-policy reference checkpoints: sequence offset, exact split ℓ + e
python scripts/measure.py replay --config $C06 --run s2ref_bf16_pg_seed0 --source quantized --source stale --source quantized_stale
# residual bias b, ratio tail and the per-token split on the trained runs
python scripts/measure.py split --config $C06 --run s2_stale64_tis_seed0
python scripts/measure.py split --config $C17 --run s5_stale64_sc_seed0
# links of the loop, and the residual bias with exact weights
python scripts/measure.py loop --config $C06 --run s2_stale64_sc_seed0
python scripts/measure.py residual-bias-exact --config $C06 --run s2_stale64_sc_seed0
# off-support share of the trainer's own responses (θ_0 first)
python scripts/measure.py off-support --config $C17 --run theta0_17b --run s6c_stale64_split_seed0
# tables (CPU)
python scripts/measure.py summarize deviation --config $C06
```

Report scripts (pure Python) build the remaining tables from measurement records and training logs:
`report_train_logs.py` and `report_resync_deviation.py` (ESS, synchronizations and deviation per step of the
resynchronized trainings), `report_dlc_resync.py` (DLC with resynchronization, and `check-reload <metrics.jsonl>` for
the reference reload of a run), `report_loop_fit.py`, `report_loop_gain.py`, `report_scale_8b.py`, `report_async.py`.
They refuse to write when they find no input records, so running them on the shipped results leaves the shipped files
unchanged. The per-step JSON records behind the tables (except `results/loop_0.6b`) and the probe sets are not shipped;
`build-probes` and the jobs above regenerate them.

## Step-cost table

```bash
python scripts/report_cost.py                      # tab:cost from the timing_s/* keys of the training logs
```

## Paper items

| Paper | Trainings (`configs/runs/`) | Measurement / script | Result files |
|---|---|---|---|
| Tab. `source`, `dev`; Fig. `source` a, b | `qwen3_0.6b_reference.txt`, `qwen3_0.6b_sources.txt` | `measure.py deviation`, `summarize deviation`; accuracy also `off-support` | `results/deviation_0.6b/{TABLES.md,steps.csv,G2B_CHECK.md}`, `results/off_support/` |
| Fig. `source` c | `qwen3_0.6b_reference.txt` | `measure.py replay`, `summarize replay` | `results/replay_0.6b/steps.csv` |
| Fig. `source` d, e, f | `qwen3_0.6b_sources.txt`, `qwen3_1.7b_stale.txt` | `measure.py split`, `loop` | `results/split_0.6b/TABLES.md`, `results/split_1.7b/TABLES.md`, `results/loop_0.6b/` |
| Fig. `loop` a–d; Tab. `exact` | `qwen3_0.6b_sources.txt` | `measure.py loop`, `residual-bias-exact`, `report_loop_fit.py` | `results/loop_0.6b/`, `results/residual_bias_exact_0.6b/`, `results/loop_fit_0.6b/FIT.md` |
| Fig. `loop` e, f | `qwen3_0.6b_lr_sweep.txt` | `measure.py split --results results/loop_gain_0.6b`, `report_loop_gain.py` | `results/loop_gain_0.6b/compare.csv` |
| Fig. `range` a | `qwen3_0.6b_sources.txt` | `measure.py split` (ratio tail) | `results/split_0.6b/TABLES.md` |
| Fig. `range` b, c | `qwen3_1.7b_stale.txt` | training logs; `measure.py deviation` | `results/train_logs_1.7b/`, `results/deviation_1.7b/` |
| Tab. `main` | `qwen3_0.6b_*.txt`, `qwen3_1.7b_stale.txt` | `measure.py deviation`, `off-support`, `report_resync_deviation.py` | `results/deviation_0.6b/`, `results/deviation_1.7b/`, `results/train_logs_resync_1.7b/deviation.csv`, `results/off_support/` |
| Fig. `slc` a–d | `qwen3_1.7b_stale.txt` | `report_train_logs.py`, `report_dlc_resync.py`, `measure.py off-support` | `results/train_logs_resync_1.7b/`, `results/dlc_resync_1.7b/`, `results/off_support/steps.csv` |
| Tab. `cost` | `qwen3_1.7b_stale.txt` | `scripts/report_cost.py` | `results/train_logs_1.7b/*.jsonl`, `results/train_logs_resync_1.7b/*.jsonl` |
| Tab. `8b` | `qwen3_8b.txt` (also `baselines_small_models.txt`) | `measure.py deviation --config configs/measure/qwen3_8b.json`, `off-support`, `report_scale_8b.py` | `results/scale_8b/TABLES.md` |
| Tab. `async` | `async.txt` | `measure.py deviation --config $C17`, `off-support`, `report_async.py` | `results/async_1.7b/{TABLES.md,steps.csv}` |

## Tests

```bash
python -m pytest tests -q                         # CPU; tests that need torch, transformers or verl skip without them
TIM_GPU_TESTS=1 python -m pytest tests/gpu -q     # in the training environment
```

## Citation

```bibtex
@article{li2026tim,
  title   = {On the Origin and Evolution of Training-Inference Mismatch in LLM Reinforcement Learning},
  author  = {Li, Yu and Zhou, Yufa and Lan, Tian},
  journal = {arXiv preprint arXiv:TODO},
  year    = {2026}
}
```
