"""The two fixed sequence sets of the replay, built once from theta_0 and never regenerated (pure numpy).

probe set : 64 held-out prompts x 8 rollouts at theta_0; every response position is a probe prefix.
P2 batch  : 64 fixed training prompts x 8 rollouts at theta_0, with +-1 rewards and group-centred advantages
            (the fixed training batch; its rows are also the batch rows of the split and loop measurements).

File = npz: tokens (int32, all sequences concatenated), offsets (int64, n+1), prompt_len, reward, advantage, row, group,
meta (json string). The sha256 of the file goes into every record measured on it.
Response positions: sequence i contributes rows for its tokens prompt_len[i] .. len-1, in sequence order; `response_index`
gives the (sequence, position) of each flattened row, so per-prefix arrays of all checkpoints line up exactly.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from tim.measure.prefixes import pack, unpack
from tim.measure.formulas import centred_advantages


def from_rollouts(rollouts: list) -> dict:
    """[(row, prompt_ids, [response_ids...], [reward...])] (tim.measure.scripts/measure.py build-probes) -> set dict."""
    d = {"sequences": [], "prompt_len": [], "reward": [], "row": [], "group": []}
    for gi, (row, prompt, resp, rew) in enumerate(rollouts):
        for rs, r in zip(resp, rew):
            d["sequences"].append(list(prompt) + list(rs))
            d["prompt_len"].append(len(prompt))
            d["reward"].append(float(r))
            d["row"].append(int(row))
            d["group"].append(gi)
    d["advantage"] = centred_advantages(d["reward"], d["group"]).tolist()
    return d


def save(path: str, s: dict, meta: dict) -> str:
    tokens, offsets = pack(s["sequences"])
    with open(path, "wb") as f:
        np.savez(f, tokens=tokens, offsets=offsets, prompt_len=np.asarray(s["prompt_len"], dtype=np.int64),
                 reward=np.asarray(s["reward"], dtype=np.float64), advantage=np.asarray(s["advantage"], dtype=np.float64),
                 row=np.asarray(s["row"], dtype=np.int64), group=np.asarray(s["group"], dtype=np.int64),
                 meta=np.asarray(json.dumps(meta, sort_keys=True)))
    return sha256(path)


def load(path: str) -> dict:
    z = np.load(path)
    return {"sequences": unpack(z["tokens"], z["offsets"]), "prompt_len": z["prompt_len"], "reward": z["reward"],
            "advantage": z["advantage"], "row": z["row"], "group": z["group"], "meta": json.loads(str(z["meta"])),
            "sha256": sha256(path)}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def response_index(s: dict) -> tuple:
    """(seq (n,), pos (n,)) of every response position, in the flattened row order used by all per-prefix arrays."""
    seq, pos = [], []
    for i, (x, p) in enumerate(zip(s["sequences"], s["prompt_len"])):
        L = len(x)
        seq.extend([i] * (L - int(p)))
        pos.extend(range(int(p), L))
    return np.asarray(seq, dtype=np.int64), np.asarray(pos, dtype=np.int64)


def next_tokens(s: dict) -> np.ndarray:
    """The token at each response position (the sampled token y_t predicted by that row), flattened."""
    return np.concatenate([np.asarray(x[int(p):], dtype=np.int64) for x, p in zip(s["sequences"], s["prompt_len"])])
