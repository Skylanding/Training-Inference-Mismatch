"""Task-specific pieces the offline measurements need: by task name, a function (response text, the parquet row's
reward_model.ground_truth) -> reward in {+1, -1}, as the task's verl reward returns it."""
from __future__ import annotations

import json


def reward_fn(task: str):
    if task == "countdown":
        from tim.tasks.countdown import compute_reward

        def f(text, ground_truth):
            gt = json.loads(ground_truth)
            return 2.0 * compute_reward(text, [int(x) for x in gt["numbers"]], gt["target"]) - 1.0
        return f
    raise ValueError(f"unknown task {task!r}: this release ships countdown only")


def answer_fn(task: str):
    """(response text) -> whether the response states a final answer in the task's format (a closed answer tag), the
    held-out record's frac_answer_tag."""
    if task == "countdown":
        from tim.tasks.countdown import extract_answer

        return lambda text: extract_answer(text) is not None
    raise ValueError(f"unknown task {task!r}: this release ships countdown only")
