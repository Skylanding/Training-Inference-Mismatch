"""Measurement engines on a GPU (TIM_GPU_TESTS=1 and TIM_BASE_MODEL = an HF dir of Qwen3-0.6B).

1. The FlatLogprobs heads of tim.measure.heads equal verl's own extract_prompt_logprobs on the same engine and sequence.
2. The bf16 sampler built as in training (dummy weights + verl's refit steps) stays as close to the HF trainer as a natively
   loaded engine of the same configuration (a broken load would make eps orders of magnitude larger).
"""
import os
import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not os.environ.get("TIM_BASE_MODEL"), reason="TIM_BASE_MODEL not set")]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")

PROMPT = "Using the numbers [3, 11, 60, 63], create an equation that equals 54. Show your work in <think> </think> tags."
ANSWER = " Let me try 60 - ((3 + 63) / 11). <answer>60 - (3 + 63) / 11</answer>"
REF_GMU, REF_KV_BLOCKS = 0.06, 256


def _base():
    return os.environ["TIM_BASE_MODEL"]


def _eng():
    from tim.measure.engines import shared
    return shared(_base())


def _seqs():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_base())
    p = tok.encode(PROMPT, add_special_tokens=False)
    r = tok.encode(ANSWER, add_special_tokens=False)
    return [p + r, p + r[:10]], [len(p), len(p)]


def test_flat_heads_equal_verl_extraction():
    from vllm import SamplingParams
    from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs

    from tim.measure.heads import sampler_sequence_heads
    seqs, plen = _seqs()
    llm = _eng().sampler("bf16", _base())
    ours = sampler_sequence_heads(llm, seqs, plen, topk=128, chunk=2)
    res = llm.generate([{"prompt_token_ids": seqs[0]}], SamplingParams(max_tokens=1, prompt_logprobs=128), use_tqdm=False)[0]
    d = {}
    extract_prompt_logprobs(res, 128, d)
    ids = np.asarray(d["prompt_ids"])[plen[0] - 1:len(seqs[0]) - 1]
    lp = np.asarray(d["prompt_logprobs"], dtype=np.float32)[plen[0] - 1:len(seqs[0]) - 1]
    R = len(seqs[0]) - plen[0]
    assert np.array_equal(ours["ids"][:R], ids)
    assert np.array_equal(ours["lp"][:R], lp)


def _eps_vs_trainer(heads_, seqs, plen):
    from tim.measure.heads import trainer_sequence_heads
    t = trainer_sequence_heads(_eng().trainer_model(_base()), seqs, plen, heads_["ids"], micro_batch=2)
    return float(np.abs(t["lp"] - heads_["lp"]).mean())


def test_bf16_refit_close_to_native_load():
    from tim.measure.engines import build_engine
    from tim.measure.heads import sampler_sequence_heads
    seqs, plen = _seqs()
    refit_llm = _eng().sampler("bf16", _base())
    native_llm = build_engine(_base(), "bf16", refit=False, gmu=REF_GMU, num_gpu_blocks_override=REF_KV_BLOCKS)
    for llm in (refit_llm, native_llm):                     # one warm-up call per engine (CUDA-graph capture)
        sampler_sequence_heads(llm, seqs, plen, topk=128, chunk=2)
    ref = sampler_sequence_heads(refit_llm, seqs, plen, topk=128, chunk=2)
    nat = sampler_sequence_heads(native_llm, seqs, plen, topk=128, chunk=2)
    e_ref, e_nat = _eps_vs_trainer(ref, seqs, plen), _eps_vs_trainer(nat, seqs, plen)
    print(f"eps vs HF trainer: refit {e_ref:.3g}, native {e_nat:.3g}")
    assert e_ref < 1.5 * e_nat + 1e-4, "refit sampler is far from the trainer: broken load"
