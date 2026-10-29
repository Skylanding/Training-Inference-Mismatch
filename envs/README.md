# Environment

All trainings and offline measurements of the paper ran in one environment, locked in
`verl-main-cu129/uv.lock` (254 packages, Linux x86_64, Python 3.12), on single nodes of NVIDIA H100 / B200 GPUs.
The FP8 sampler needs H100 or B200; the FA3 build below targets H100 (sm_90a), which the default trainer kernels use.

| Package | Version |
|---|---|
| Python | 3.12 |
| verl | git `8718ca30a3f002f93b7c4fd99b9b2506718681bc` (0.10.0.dev0, `verl[verl-core]`); includes the score-centering PR #8010 and the FP8 tied-embedding fix #8064 |
| vLLM | 0.29.0+cu129 (GitHub release wheel) |
| torch / torchvision / torchaudio | 2.13.0+cu129 / 0.28.0+cu129 / 2.11.0+cu129 |
| transformers | 5.12.1 |
| ray | 2.59.0 |
| tensordict | 0.10.0 |
| TransferQueue | 0.1.11 |
| liger-kernel | >= 0.8.2 |
| flashinfer-python | 0.6.18 (pinned by vLLM) |
| nvidia-nccl-cu12 | 2.29.7 |
| FlashAttention-3 | built from source for H100 (`fa3/build_fa3.sh`) |

## Build

```bash
bash envs/verl-main-cu129/build_env.sh          # uv sync --frozen into .venv-tim, then verl's score-centering CPU tests
PY=.venv-tim/bin/python FA3_PATH=$PWD/.fa3 bash envs/fa3/build_fa3.sh
FA3_PATH=$PWD/.fa3 .venv-tim/bin/python envs/fa3/check_fa3.py   # prints ATTENTION=flash_attention_3
```

`scripts/train.sh` and `scripts/measure.py` take `FA3_PATH` from the environment and put it on `PYTHONPATH`
(Ray workers inherit it). Without FA3, `KERNELS=sdpa` trains with SDPA on padded micro-batches; DLC, the async source
and the replay of the reference runs need the default `KERNELS=fa3_liger`.

Notes on the lock:

- verl's `fsdp` / `vllm` / `cpu` extras are not used, because they resolve to CUDA 13 wheels and pull flash-attn and
  cupy-cuda13x; the FSDP dependencies are listed directly in `pyproject.toml`.
- flash-attn (FA2), flash-attn-4 and the CUDA 13 artefacts are excluded by overrides. Score centering and DLC need
  `use_fused_kernels=False`, so FA2's fused kernels are not needed; the trainer uses FA3 through transformers.
- cu129 wheels run on drivers of the CUDA 12 line (tested with 535.129), with CUDA minor-version compatibility.
- `datasets>=4.0.0` is pinned because an unconstrained resolve picks a version incompatible with pyarrow 25.
- The vLLM sampler uses the `TRITON_ATTN` backend; `scripts/train.sh` sets `VLLM_USE_DEEP_GEMM=0` and
  `VLLM_USE_FLASHINFER_SAMPLER=0` as in the paper's runs.
