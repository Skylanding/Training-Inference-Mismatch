"""Engines of a measurement process: built once per process, refit in place to each checkpoint. GPU only.

Samplers (vLLM 0.29, engine core in its own subprocess):
  bf16        dtype bf16, load_format=dummy, then refit to each checkpoint (the reference and the stale sampler);
  fp8         the quantized sampler as verl `rollout.quantization=fp8` builds it (vllm_async_server._apply_quantization @
              8718ca30): quantization=fp8 with the block-fp8 hf_overrides (tim.measure.fp8_ckpt.quantization_config),
              kv_cache_dtype=fp8, load_format=dummy, verl's colocate worker extension (whose constructor applies verl's vLLM
              quant patches inside the engine subprocess when VERL_VLLM_FP8_QUANT_ENABLED=1);
  bf16_eager  bf16 with enforce_eager=True (no CUDA graphs, no torch.compile);
  all         the training sampler's engine arguments (TRAIN_ENGINE) with the KV size pinned (KV_BLOCKS).
Refit = the steps of verl's `vLLMColocateWorkerExtension.update_weights_from_ipc` / `_update_weights` with the weights read
from the checkpoint instead of the IPC bucket: split buffers, drop tied-embedding aliases, then
  fp8:  prepare_quanted_weights_for_loading -> load_quanted_weights (quantizes with verl quant_weights) -> buffers ->
        process_quanted_weights_after_loading;
  bf16: model.load_weights -> buffers -> vLLM process_weights_after_loading.
Trainer: HF bf16 with the run's kernel set (kernels_for_run): `sdpa` = sdpa on right-padded micro-batches; `fa3_liger` =
FlashAttention-3 on packed sequences in verl's dynamic micro-batches plus Liger with Triton RoPE, as the trainings
(scripts/train.sh KERNELS=fa3_liger). One resident model per (kernel set, slot), refit by load_state_dict from the
checkpoint (fp32 safetensors cast to bf16, as FSDP casts the fp32 master weights to its compute dtype).
"""
from __future__ import annotations

import json
import os


SAMPLER_GMU = float(os.environ.get("TIM_MEASURE_SAMPLER_GMU", "0.15"))   # per engine; 0.6B weights + KV for 64 x 512 tokens
FA3_PATH = os.environ.get("FA3_PATH", "")   # optional: a directory with a FlashAttention-3 build to put on sys.path


# The training sampler's engine arguments (verl main @ 8718ca30 rollout.yaml defaults, which scripts/train.sh does not override,
# plus its max_model_len=512, TP 1 and TRITON_ATTN; vllm_async_server.py builds the engine from them):
# max_num_batched_tokens 8192, max_num_seqs 256, chunked prefill and prefix caching on, enforce_eager off with
# cudagraph_mode FULL_AND_PIECEWISE (verl's setdefault), load_format dummy, processed log-probs, verl's worker extension.
# gpu_memory_utilization (0.5 in training, colocated with the FSDP trainer) only sets the KV size, which therefore depends
# on free memory at build; here the KV size is pinned (num_gpu_blocks_override) so results cannot depend on build order.
KV_BLOCKS = int(os.environ.get("TIM_MEASURE_KV_BLOCKS", "2310"))      # x 16 tokens = 36,960 >= 64 x 512 of one probe chunk
TRAIN_ENGINE = dict(dtype="bfloat16", max_model_len=512, max_num_batched_tokens=8192, max_num_seqs=256,
                    enable_chunked_prefill=True, enable_prefix_caching=True, enforce_eager=False,
                    compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"}, logprobs_mode="processed_logprobs",
                    generation_config="vllm", attention_backend="TRITON_ATTN", seed=0)


REFIT_BUCKET_GB = 0.0    # > 0: bf16 refit in buckets of this size (Qwen3-8B: its checkpoint is one 32.8 GB file)


def configure(max_model_len: int = None, kv_blocks: int = None, gmu: float = None, refit_bucket_gb: float = None,
              **_ignored) -> None:
    """A model's sampler sizes (the `sampler` block of configs/measure/<model>.json): the training sampler's
    max_model_len, the pinned KV size, the memory share of each engine (Qwen3-8B: its 16 GB bf16 weights plus the pinned
    KV) and refit_bucket_gb (the bf16 refit then holds one bucket of the fp32 checkpoint on the GPU instead of all of it).
    Call before any engine is built."""
    global KV_BLOCKS, SAMPLER_GMU, REFIT_BUCKET_GB
    if max_model_len is not None:
        TRAIN_ENGINE["max_model_len"] = int(max_model_len)
    if kv_blocks is not None:
        KV_BLOCKS = int(kv_blocks)
    if gmu is not None:
        SAMPLER_GMU = float(gmu)
    if refit_bucket_gb is not None:
        REFIT_BUCKET_GB = float(refit_bucket_gb)


def engine_kwargs(model: str, kind: str, *, refit: bool, gmu: float = None, **overrides) -> dict:
    """vLLM LLM(...) arguments of a replay sampler. refit=True: the training sampler (dummy weights, verl's worker extension,
    weights loaded later by the refit); refit=False: a natively loaded engine of the same configuration (`model` = a real
    checkpoint; for fp8 an offline block-fp8 checkpoint, tim.measure.fp8_ckpt). `overrides` change one argument."""
    kw = dict(TRAIN_ENGINE, model=model, tokenizer=model, max_logprobs=128,
              gpu_memory_utilization=SAMPLER_GMU if gmu is None else gmu, num_gpu_blocks_override=KV_BLOCKS)
    if refit:
        kw.update(load_format="dummy", worker_extension_cls=EXTENSION)
    if kind == "fp8":
        kw["kv_cache_dtype"] = "fp8"
        if refit:
            n_layers = json.load(open(os.path.join(model, "config.json")))["num_hidden_layers"]
            from tim.measure.fp8_ckpt import quantization_config
            kw.update(quantization="fp8", hf_overrides={"quantization_config": quantization_config(n_layers)})
    elif kind != "bf16":
        raise ValueError(kind)
    kw.update(overrides)
    return kw


def build_engine(model: str, kind: str, *, refit: bool, gmu: float = None, **overrides):
    """A new engine; it stays alive for the rest of the process and is refit in place."""
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    if kind == "fp8" and refit:
        os.environ["VERL_VLLM_FP8_QUANT_ENABLED"] = "1"            # read by the extension in the engine subprocess
    from vllm import LLM
    return LLM(**engine_kwargs(model, kind, refit=refit, gmu=gmu, **overrides))


REF_PREFIXES = ("s2ref_", "s5ref_", "s8ref_", "s9b_")


def kernels_for_run(run: str) -> str:
    """The trainer kernel set of a reference run: every training of the paper ran with scripts/train.sh KERNELS=fa3_liger
    (fused log-prob kernels off), so the reference prefixes s2ref_, s5ref_, s8ref_ (and the continuous-asynchrony runs
    s9b_) map to "fa3_liger". The deviation, split, loop and off-support jobs always use "fa3_liger"; "sdpa" is the
    fallback for other run names (sdpa attention on right-padded micro-batches)."""
    return "fa3_liger" if run.startswith(REF_PREFIXES) else "sdpa"


def setup_kernel_env(kernel_sets) -> None:
    """Before transformers / liger are imported: the optional FA3 build on sys.path (env FA3_PATH) and Liger's Triton
    RoPE, as scripts/train.sh exports them for KERNELS=fa3_liger."""
    import sys
    if "fa3_liger" in set(kernel_sets):
        if FA3_PATH and FA3_PATH not in sys.path:
            sys.path.insert(0, FA3_PATH)
            os.environ["PYTHONPATH"] = FA3_PATH + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
        os.environ.setdefault("LIGER_KERNEL_IMPL_ROPE", "nvidia-triton")


EXTENSION = "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"


def _refit_worker(worker, path: str, quantized: bool, bucket_gb: float = 0.0):
    """Runs inside the vLLM worker (collective_rpc): load the checkpoint at `path` into the live model, verl's refit steps.
    bucket_gb > 0 (bf16 only): the same steps on buckets of at most bucket_gb of checkpoint tensors (read one
    tensor at a time), as verl's refit loads bucket by bucket, so the GPU holds one bucket rather than the whole fp32
    checkpoint (Qwen3-8B: 32.8 GB); process_weights_after_loading once at the end."""
    import torch
    from safetensors import safe_open
    from verl.workers.rollout.vllm_rollout.weight_update_utils import (apply_buffer_updates, drop_tied_alias_updates,
                                                                       split_buffer_updates)
    runner = worker.model_runner
    model = runner.model
    if bucket_gb > 0 and not quantized:
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        loaded, cap = set(), int(bucket_gb * 2 ** 30)

        def load(weights):
            params, buffers, named_buffers = split_buffer_updates(model, weights)
            params = drop_tied_alias_updates(model, params)
            loaded.update(model.load_weights(params) or [])
            apply_buffer_updates(model, buffers, named_buffers=named_buffers)
            weights.clear()
            del params
            torch.cuda.empty_cache()

        with torch.no_grad():
            for fn in sorted(x for x in os.listdir(path) if x.endswith(".safetensors")):
                with safe_open(os.path.join(path, fn), framework="pt", device="cuda") as f:
                    bucket, size = [], 0
                    for k in f.keys():
                        t = f.get_tensor(k)
                        bucket.append((k, t))
                        size += t.numel() * t.element_size()
                        if size >= cap:
                            load(bucket)
                            size = 0
                    if bucket:
                        load(bucket)
            process_weights_after_loading(model, runner.vllm_config.model_config, torch.device("cuda"))
        torch.cuda.empty_cache()
        return {"n_loaded": len(loaded), "n_params": sum(1 for _ in model.named_parameters())}
    weights = []
    for fn in sorted(x for x in os.listdir(path) if x.endswith(".safetensors")):
        with safe_open(os.path.join(path, fn), framework="pt", device="cuda") as f:
            weights.extend((k, f.get_tensor(k)) for k in f.keys())
    params, buffers, named_buffers = split_buffer_updates(model, weights)
    params = drop_tied_alias_updates(model, params)
    with torch.no_grad():
        if quantized:
            from verl.utils.vllm.vllm_quant_utils import (load_quanted_weights, prepare_quanted_weights_for_loading,
                                                          process_quanted_weights_after_loading)
            state = prepare_quanted_weights_for_loading(model)
            loaded = load_quanted_weights(params, runner)
            apply_buffer_updates(model, buffers, named_buffers=named_buffers)
            process_quanted_weights_after_loading(model, state)
        else:
            from vllm.model_executor.model_loader.utils import process_weights_after_loading
            loaded = model.load_weights(params)
            apply_buffer_updates(model, buffers, named_buffers=named_buffers)
            process_weights_after_loading(model, runner.vllm_config.model_config, torch.device("cuda"))
    del weights, params
    torch.cuda.empty_cache()
    return {"n_loaded": len(loaded or []), "n_params": sum(1 for _ in model.named_parameters())}


class Engines:
    """Lazily built, kept for the whole process: samplers by kind ('bf16', 'fp8') and the HF trainer, each refit to a path on demand."""

    def __init__(self, base_model: str, topk: int = 128):
        self.base, self.topk = base_model, topk
        self.samplers, self.sampler_at = {}, {}
        self.trainers, self.trainer_at = {}, {}
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")   # collective_rpc with a function (cloudpickle)

    def _build(self, kind: str):
        """bf16 | fp8 | bf16_eager (the execution-mode source: vLLM bf16 with enforce_eager=True, i.e. neither
        torch.compile nor CUDA graphs; everything else as the training sampler)."""
        if kind == "bf16_eager":
            return build_engine(self.base, "bf16", refit=True, enforce_eager=True, compilation_config=None)
        return build_engine(self.base, kind, refit=True)

    def sampler(self, kind: str, path: str):
        """The `kind` sampler holding the weights at `path` (refit only when they differ from what it holds)."""
        if kind not in self.samplers:
            self.samplers[kind] = self._build(kind)
            self.sampler_at[kind] = None
        llm = self.samplers[kind]
        if self.sampler_at[kind] != path:
            res = llm.collective_rpc(_refit_worker, args=(path, kind.startswith("fp8")) + ((REFIT_BUCKET_GB,) if REFIT_BUCKET_GB > 0 else ()))
            if not res or res[0]["n_loaded"] == 0:
                raise RuntimeError(f"{kind} refit to {path} loaded nothing: {res}")
            llm.reset_prefix_cache()
            self.sampler_at[kind] = path
        return llm

    def _build_trainer(self, kernels: str):
        import torch
        if kernels == "sdpa":
            from tim.measure.forward import load_trainer
            model = load_trainer(self.base)                 # sdpa, padded micro-batches
            model._tim_mode = "padded"
            return model
        if kernels != "fa3_liger":
            raise ValueError(kernels)
        setup_kernel_env(["fa3_liger"])
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(self.base, dtype=torch.bfloat16,
                                                     attn_implementation="flash_attention_3").to("cuda").eval()
        # as verl's FSDP engine with use_liger (verl/workers/engine/fsdp/transformer_impl.py)
        from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
        _apply_liger_kernel_to_instance(model=model, fused_linear_cross_entropy=False, swiglu=True)
        # packed sequences in verl's dynamic micro-batches, like training's remove padding (heads.forward_logits);
        # TIM_MEASURE_TRAINER_PACK=0: one sequence per forward instead
        model._tim_mode = "single" if os.environ.get("TIM_MEASURE_TRAINER_PACK", "1") == "0" else "packed"
        return model

    def trainer_model(self, path: str, kernels: str = "sdpa", slot: str = "main"):
        """HF bf16 trainer with the run's kernel set (sdpa | fa3_liger) holding the weights at `path`; one per (kernel set,
        slot), resident (the deviation keeps the run and its reference side by side in slots "run" and "ref")."""
        import torch
        from safetensors.torch import load_file
        key = kernels if slot == "main" else (kernels, slot)
        if key not in self.trainers:
            self.trainers[key], self.trainer_at[key] = self._build_trainer(kernels), self.base
        model = self.trainers[key]
        if self.trainer_at[key] != path:
            sd = {}
            for fn in sorted(x for x in os.listdir(path) if x.endswith(".safetensors")):
                sd.update({k: v.to(torch.bfloat16) for k, v in load_file(os.path.join(path, fn)).items()})
            missing, unexpected = model.load_state_dict(sd, strict=False)
            tied = getattr(model.config, "tie_word_embeddings", False)
            if unexpected or [m for m in missing if not (tied and m == "lm_head.weight")]:
                raise RuntimeError(f"trainer refit {path}: missing {missing[:5]} unexpected {unexpected[:5]}")
            self.trainer_at[key] = path
        model.zero_grad(set_to_none=True)
        return model


_SHARED = None


def shared(base_model: str, topk: int = 128) -> Engines:
    """The process's one Engines (every job of a measure.py call and the GPU tests use the same engines)."""
    global _SHARED
    if _SHARED is None:
        _SHARED = Engines(base_model, topk)
    return _SHARED
