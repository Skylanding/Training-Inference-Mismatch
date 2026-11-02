# On the Origin and Evolution of Training-Inference Mismatch in LLM Reinforcement Learning

<p align="center">
  <a href="https://arxiv.org/abs/2610.40945"><img src="https://img.shields.io/badge/arXiv-2610.40945-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/papers/2610.40945"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Paper-2610.40945-ffd21e.svg" alt="Hugging Face Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/volcengine/verl/tree/8718ca30"><img src="https://img.shields.io/badge/verl-8718ca30-4b8bbe.svg" alt="verl 8718ca30"></a>
  <a href="https://github.com/vllm-project/vllm/releases/tag/v0.29.0"><img src="https://img.shields.io/badge/vLLM-0.29.0-30a2ff.svg" alt="vLLM 0.29.0"></a>
</p>

Evaluating the trainer at the sampler's weights splits each sampled-token error into a **numerical discrepancy** of
the engine and a **policy lag** of the trainer. The **discrepancy-lag correction (DLC)** centers the discrepancy, clips
the lag ratio against the sampler's weights and resynchronizes the sampler when the effective sample size of the lag
ratio falls below one half.

## Layout

```
tim/dlc/        DLC hooks for verl: lag-ratio clip, centering term, ESS-triggered resynchronization, asynchrony
tim/baselines/  M2PO and VCPO
tim/tasks/      Countdown task and reward
tim/measure/    offline measurements (deviation, exact split, residual bias, loop, ESS, off-support share)
scripts/        train.sh, measure.py, report_*.py (tables), build_countdown_data.py
configs/        run lists per table, measurement settings, chat template
results/        measurement tables and training logs behind the paper
docs/           notes on the DLC loss, the decomposition, the baselines and asynchrony
```

## Setup

Linux with H100 or B200 GPUs; verl `8718ca30`, vLLM 0.29.0, torch 2.13.0, Python 3.12 (see `envs/README.md`).

```bash
bash envs/verl-main-cu129/build_env.sh
PY=.venv-tim/bin/python FA3_PATH=$PWD/.fa3 bash envs/fa3/build_fa3.sh
python scripts/build_countdown_data.py
```

## Training

```bash
MODEL=Qwen/Qwen3-1.7B METHOD=dlc SOURCE=stale SEED=0 RUN_PREFIX=s7d bash scripts/train.sh
bash scripts/run_list.sh configs/runs/qwen3_1.7b_stale.txt      # every training of a table
```

- `METHOD`: `on_policy`, `pg`, `ppo`, `ppo_noclip`, `sc`, `is`, `is_sc`, `is_resync`, `m2po`, `vcpo`, `dlc`,
  `dlc_noresync`, `dlc_async`, `dlc_admission`.
- `SOURCE`: `none`, `quantized`, `stale`, `quantized_stale`, `async`.
- `DRY=1` prints the verl command; the header of `scripts/train.sh` lists the other variables.
- Run names take the form `<prefix>_<source>_<method>_seed<k>` with internal names (`stale64`, `budget`, `split`, ...),
  which the run lists use and the measurement scripts parse.

## Measurements and tables

```bash
python scripts/measure.py deviation --config configs/measure/qwen3_1.7b.json --run s7d_budget_split_seed0
python scripts/measure.py --help                                  # replay, split, loop, off-support, summarize
python scripts/report_cost.py                                     # step-cost table
```

| Paper | Run list (`configs/runs/`) | Results (`results/`) |
|---|---|---|
| Sources on Qwen3-0.6B | `qwen3_0.6b_reference.txt`, `qwen3_0.6b_sources.txt` | `deviation_0.6b`, `replay_0.6b`, `split_0.6b` |
| Loop | `qwen3_0.6b_sources.txt`, `qwen3_0.6b_lr_sweep.txt` | `loop_0.6b`, `loop_fit_0.6b`, `loop_gain_0.6b`, `residual_bias_exact_0.6b` |
| Validity range, DLC on Qwen3-1.7B | `qwen3_1.7b_stale.txt` | `deviation_1.7b`, `split_1.7b`, `train_logs_*1.7b`, `dlc_resync_1.7b`, `off_support` |
| Qwen3-8B, asynchrony | `qwen3_8b.txt`, `async.txt` | `scale_8b`, `async_1.7b` |

## Tests

```bash
python -m pytest tests -q
TIM_GPU_TESTS=1 python -m pytest tests/gpu -q
```

## Citation

```bibtex
@article{li2026tim,
  title   = {On the Origin and Evolution of Training-Inference Mismatch in LLM Reinforcement Learning},
  author  = {Li, Yu and Zhou, Yufa and Lan, Tian},
  journal = {arXiv preprint arXiv:2610.40945},
  year    = {2026}
}
```

The Countdown generator and reward are ported from martin-marek/score-centering (MIT License).
