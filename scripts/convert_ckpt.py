"""Verl checkpoint -> HF directory with the initial model's generation facts restored.

Qwen3 declares eos [<|im_end|>, <|endoftext|>] (generation_config.json); a verl-saved huggingface/ dir carries the
tokenizer's single eos in config.json, so generation of a converted model may not stop at <|im_end|>. `restore_from_init`
puts the initial model's generation_config, tokenizer, chat template and eos ids back.

  python scripts/convert_ckpt.py SRC DEST --init Qwen/Qwen3-0.6B [--step N] [--force-merge]

SRC: a verl actor dir (global_step_N/actor), global_step_N, or the checkpoint root (latest_checkpointed_iteration.txt).
If actor/huggingface/ already holds weights (scripts/train.sh saves hf_model), that dir is copied and fixed: no merger
run. Otherwise (or --force-merge) `python -m verl.model_merger merge --backend fsdp` runs on CPU. DEST is never
overwritten. --init is a local HF directory or a Hugging Face model id (resolved with huggingface_hub.snapshot_download).
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tim.ckpt_metadata import copy_missing_metadata  # noqa: E402

STEP_DIRECTORY = re.compile(r"global_step_([0-9]+)")


class CheckpointError(Exception):
    pass


def read_object(path, error=CheckpointError):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise error(f"{path}: missing") from None
    except (OSError, ValueError) as cause:
        raise error(f"{path}: unreadable JSON ({cause})") from cause
    if not isinstance(value, dict):
        raise error(f"{path}: not a JSON object")
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _unique(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def eos_ids(declared):
    """An `eos_token_id` declaration, an id or a list of ids, as a list."""
    if isinstance(declared, bool):
        return []
    if isinstance(declared, int):
        return [declared]
    if isinstance(declared, list):
        return [v for v in declared if isinstance(v, int) and not isinstance(v, bool)]
    return []


def resolve_model(model):
    """A local HF directory as is; otherwise a Hugging Face model id, downloaded (or found in the cache) as a snapshot."""
    if Path(model).is_dir():
        return str(model)
    from huggingface_hub import snapshot_download
    return snapshot_download(str(model))


def init_eos_ids(init):
    """The init's full eos declaration (generation_config.json first, then config.json), de-duplicated."""
    init = Path(resolve_model(init))
    generation = init / "generation_config.json"
    declared = eos_ids(read_object(generation).get("eos_token_id")) if generation.is_file() else []
    return _unique(declared + eos_ids(read_object(init / "config.json").get("eos_token_id")))


def _checkpoint_root(path):
    return path.is_dir() and ((path / "latest_checkpointed_iteration.txt").is_file()
                              or any(item.is_dir() for item in path.glob("global_step_*")))


def resolve_actor(source, steps=None):
    """The FSDP actor directory and its step, from the actor, its global_step_N, or the checkpoint root."""
    source = Path(source)
    if not source.is_dir():
        raise CheckpointError(f"{source}: not a directory")
    if _checkpoint_root(source):
        if steps is None:
            tracker = source / "latest_checkpointed_iteration.txt"
            if not tracker.is_file():
                raise CheckpointError(f"{source}: global_step_* dirs but no latest_checkpointed_iteration.txt; use --step")
            steps = int(tracker.read_text().strip())
        source = source / f"global_step_{steps}"
        if not source.is_dir():
            raise CheckpointError(f"{source}: no such checkpoint")
    named = STEP_DIRECTORY.fullmatch(source.name) or STEP_DIRECTORY.fullmatch(source.parent.name)
    found = int(named.group(1)) if named else None
    if steps is not None and found is not None and steps != found:
        raise CheckpointError(f"--step {steps} disagrees with {source}, which is step {found}")
    actor = source / "actor" if (source / "actor").is_dir() else source
    return actor, steps if steps is not None else found


def run_merger(python, actor, target):
    """`verl.model_merger` in the training interpreter, with no GPU visible so a busy card is never touched
    (verl 8718ca30: model_merger/__main__.py has the `merge` subcommand with --backend/--local_dir/--target_dir, and the
    base parser has --trust-remote-code and --use_cpu_initialization)."""
    argv = [python, "-m", "verl.model_merger", "merge", "--backend", "fsdp", "--local_dir", str(actor),
            "--target_dir", str(target), "--trust-remote-code", "--use_cpu_initialization"]
    try:
        completed = subprocess.run(argv, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    except OSError as error:
        raise CheckpointError(f"cannot start the merger with {python} ({error})") from error
    if completed.returncode != 0:
        raise CheckpointError(f"the Verl merger exited with status {completed.returncode}: {shlex.join(argv)}")


def restore_from_init(directory, init):
    """The init's generation config, tokenizer and chat template; config.json declares the init's eos ids.

    The merger writes a generation config without the sampling fields and derives both eos
    declarations from the tokenizer's single eos id; the init's own declarations replace them,
    generation_config.json first, then config.json.
    """
    directory, init = Path(directory), Path(resolve_model(init))
    actions = []
    generation = init / "generation_config.json"
    if generation.is_file():
        shutil.copy2(generation, directory / "generation_config.json")
        actions.append("generation_config.json: the init's, replacing the merger's")
    actions += [f"{name}: copied from the init" for name in copy_missing_metadata(str(directory), str(init))]
    declared = eos_ids(read_object(generation).get("eos_token_id")) if generation.is_file() else []
    declared = _unique(declared + eos_ids(read_object(init / "config.json").get("eos_token_id")))
    config_path = directory / "config.json"
    config = read_object(config_path)
    wanted = declared if len(declared) > 1 else (declared[0] if declared else None)
    if wanted is not None and config.get("eos_token_id") != wanted:
        actions.append(f"config.json: eos_token_id {config.get('eos_token_id')!r} -> {wanted!r}, the init's")
        config["eos_token_id"] = wanted
        write_json(config_path, config)
    return actions


def has_weights(directory):
    d = Path(directory)
    return (d / "config.json").is_file() and any(d.glob("*.safetensors"))


def convert(source, destination, init, *, steps=None, python=sys.executable, force_merge=False):
    """hf_model dir copied (or FSDP shards merged), then restore_from_init. Never overwrites. Returns actions."""
    destination = Path(destination)
    init = resolve_model(init)
    if destination.exists():
        raise CheckpointError(f"{destination} exists; conversion never overwrites")
    if not (Path(init) / "config.json").is_file():
        raise CheckpointError(f"{init}: the init has no config.json")
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise CheckpointError(f"{staging} is left from an earlier conversion; remove it and retry")
    actor, step = resolve_actor(source, steps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if has_weights(actor / "huggingface") and not force_merge:
            shutil.copytree(actor / "huggingface", staging)
            actions = [f"step {step}: copied {actor}/huggingface (verl hf_model); no merger run"]
        else:
            if not any(actor.glob("model_world_size_*_rank_*.pt")):
                raise CheckpointError(f"{actor}: no hf_model weights and no FSDP shards to merge")
            run_merger(python, actor, staging)
            if not has_weights(staging):
                raise CheckpointError(f"the merger wrote no config.json or *.safetensors into {staging}")
            actions = [f"step {step}: merged {actor} with verl.model_merger"]
        actions += restore_from_init(staging, init)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return actions


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dest")
    ap.add_argument("--init", required=True, help="the initial model: local HF dir or Hugging Face id")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--force-merge", action="store_true", help="run verl.model_merger even if huggingface/ has weights")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    for line in convert(a.src, a.dest, a.init, steps=a.step, python=a.python, force_merge=a.force_merge):
        print(line)


if __name__ == "__main__":
    main()
