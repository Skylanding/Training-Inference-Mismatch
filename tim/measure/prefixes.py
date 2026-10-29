"""Token sequences of the fixed probe sets and prompt rendering (numpy; the renderer needs a tokenizer).

The probe set is built once from rollouts of the initial model theta_0 (`scripts/measure.py build-probes`); every
response position of a probe sequence is a probe prefix. Prompts are rows of data/countdown/{train,heldout}.parquet,
rendered with configs/chat-templates/qwen3.jinja and enable_thinking=False, as in training.
"""
from __future__ import annotations

import numpy as np

MAX_RESPONSE = 377   # data.max_response_length of the Countdown trainings


def pack(seqs: list) -> tuple:
    """list of int sequences -> (concat int32, offsets int64 (n+1))."""
    lens = np.array([len(s) for s in seqs], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(lens)])
    flat = np.concatenate([np.asarray(s, dtype=np.int32) for s in seqs]) if seqs else np.zeros(0, np.int32)
    return flat, off


def unpack(flat: np.ndarray, off: np.ndarray) -> list:
    return [flat[off[i]:off[i + 1]].tolist() for i in range(len(off) - 1)]


def _render_prompts(tokenizer, messages_list, template: str) -> list:
    """Chat rendering of the trainings (qwen3.jinja, enable_thinking=False) -> token ids."""
    out = []
    for m in messages_list:
        text = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                             chat_template=template)
        out.append(tokenizer.encode(text, add_special_tokens=False))
    return out
