"""Settings of the offline measurements (pure python): one JSON file per model under configs/measure/.

The file is the one named by an explicit path, else by the environment variable TIM_MEASURE_CONFIG (set by
`scripts/measure.py --config`), else configs/measure/qwen3_0.6b.json. Relative paths in it are relative to the repository
root. `base_model` is a Hugging Face model id (resolved to the local snapshot with huggingface_hub) or a local directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(ROOT, "configs", "measure", "qwen3_0.6b.json")
ENV = "TIM_MEASURE_CONFIG"
_PATH_KEYS = ("heldout_parquet", "train_parquet", "template")


def path() -> str:
    p = os.environ.get(ENV) or PATH
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def resolve_model(name: str) -> str:
    """A local model directory: `name` itself if it is a directory, else the Hugging Face snapshot of the model id."""
    if os.path.isdir(name):
        return name
    from huggingface_hub import snapshot_download
    return snapshot_download(name)


def load(path_: str = None, resolve: bool = True) -> dict:
    cfg = json.load(open(path_ or path()))
    for k in _PATH_KEYS:
        if k in cfg and not os.path.isabs(cfg[k]):
            cfg[k] = os.path.join(ROOT, cfg[k])
    if resolve and "base_model" in cfg:
        cfg["base_model"] = resolve_model(cfg["base_model"])
    return cfg


def repo_path(rel: str) -> Path:
    """A path from a config value, relative to the repository root unless absolute."""
    p = Path(rel)
    return p if p.is_absolute() else Path(ROOT) / p


def _eos_ids(declared) -> list:
    if isinstance(declared, bool):
        return []
    if isinstance(declared, int):
        return [declared]
    if isinstance(declared, list):
        return [v for v in declared if isinstance(v, int) and not isinstance(v, bool)]
    return []


def init_eos_ids(init) -> list:
    """The initial model's full eos declaration (generation_config.json first, then config.json), de-duplicated: the stop
    tokens of every generation in the measurements (Qwen3: <|im_end|> and <|endoftext|>)."""
    init = Path(init)
    out = []
    for name in ("generation_config.json", "config.json"):
        f = init / name
        if f.is_file():
            for v in _eos_ids(json.loads(f.read_text()).get("eos_token_id")):
                if v not in out:
                    out.append(v)
    return out
