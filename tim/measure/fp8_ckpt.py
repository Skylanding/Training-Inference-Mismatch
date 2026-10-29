"""Block-fp8 quantization config of verl's quantized sampler (rollout.quantization=fp8).

verl starts vLLM with load_format=dummy plus an fp8 block config (FP8_BLOCK_QUANT_KWARGS in
verl/workers/rollout/vllm_rollout/vllm_async_server.py::_apply_quantization @ 8718ca30: activation_scheme dynamic, fmt
e4m3, weight_block_size [128, 128]) and quantizes each linear weight at every weight sync. The replay sampler of
tim.measure.engines is built with the same config and refit through verl's quantized-weight loader.
"""
from __future__ import annotations

BLOCK = (128, 128)
LINEAR_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def is_linear_weight(name: str, ndim: int) -> bool:
    return ndim == 2 and name.endswith(".weight") and any(f".{p}." in name for p in LINEAR_PROJ)


def quantization_config(num_hidden_layers: int) -> dict:
    """Copied from verl _apply_quantization (FP8_BLOCK_QUANT_KWARGS)."""
    return {
        "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": list(BLOCK),
        "ignored_layers": [f"model.layers.{i}.mlp.gate" for i in range(num_hidden_layers)],
    }
