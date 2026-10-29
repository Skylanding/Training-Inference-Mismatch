"""verl custom reward function for Countdown.

verl main (8718ca30) calls ``compute_score(data_source=, solution_str=,
ground_truth=, extra_info=)`` from
verl/experimental/reward_loop/reward_manager/naive.py (and the legacy
verl/workers/reward_manager/naive.py), where ``solution_str`` is the decoded
response with special tokens skipped. A dict return must contain "score";
every key is also logged as reward_extra_info.

Return value: score = 2*correct - 1 (+1 / -1, not 0/1),
acc = correct (0.0 / 1.0), usable as a validation metric.
Hydra: reward.custom_reward_function.path=<this file>
       reward.custom_reward_function.name=compute_score
"""
import json
import os
import sys

# make `tim` importable when verl loads this file by path
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tim.tasks.countdown import compute_reward  # noqa: E402


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    correct = compute_reward(solution_str, [int(n) for n in gt["numbers"]], gt["target"])
    return {"score": 2.0 * correct - 1.0, "acc": float(correct)}
