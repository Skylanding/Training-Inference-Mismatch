"""The Countdown task against the reference implementation (tests/fixtures/official_countdown.py): problem generation,
prompt text and reward. Pure python. Run: python -m pytest tests -q"""
import importlib.util
import json
import os
import random
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tim.tasks import countdown as mine  # noqa: E402
from tim.tasks.countdown_reward import compute_score  # noqa: E402


def _load_official():
    sys.modules.setdefault("verifiers", types.ModuleType("verifiers"))  # only used by env glue
    spec = importlib.util.spec_from_file_location("official_countdown", os.path.join(ROOT, "tests/fixtures/official_countdown.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


official = _load_official()


def test_generate_first5_and_prompt_identical():
    r1, r2 = random.Random(0), random.Random(0)
    for _ in range(5):
        a, b = official.generate(r1), mine.generate(r2)
        assert a == b
        assert official._messages(a) == mine.messages(b)
    assert mine.SYSTEM_PROMPT == official.SYSTEM_PROMPT
    assert mine.instruction([1, 2, 3, 4], 5) == official.instruction([1, 2, 3, 4], 5)
    assert mine.instruction([1, 2, 3, 4], 5).startswith(" Using")


def test_generate_long_stream_identical():
    r1, r2 = random.Random(0), random.Random(0)
    assert all(official.generate(r1) == mine.generate(r2) for _ in range(2000))


EDGE = {"correct": ("<answer>3 * 7 + 5</answer>", 1.0), "wrong value": ("<answer>3 + 5 + 7</answer>", 0.0),
        "reuses number": ("<answer>3 * 7 + 5 + 5</answer>", 0.0), "unknown number": ("<answer>3 * 7 + 9</answer>", 0.0),
        "no tag": ("the answer is 3*7+5", 0.0), "bare target in answer": ("<think>3*7+5</think><answer>26</answer>", 0.0),
        "pow": ("<answer>3 ** 7 + 5</answer>", 0.0), "floordiv": ("<answer>3 // 7 + 5</answer>", 0.0)}


def test_reward_edge_cases():
    nums, tgt = [3, 5, 7], 26
    for name, (c, want) in EDGE.items():
        assert official.compute_reward(c, nums, tgt) == want, name
        assert mine.compute_reward(c, nums, tgt) == want, name
        gt = json.dumps({"numbers": nums, "target": tgt})
        out = compute_score("countdown", c, gt)
        assert out["score"] == 2 * want - 1 and out["acc"] == want, name


def test_reward_matches_official_on_random_completions():
    rng = random.Random(1)
    for _ in range(300):
        p = official.generate(rng)
        for c in (f"<answer>{p['solution']}</answer>", f"<answer> {p['solution']} + 1 </answer>",
                  f"x<answer>{p['solution']}</answer>y<answer>{p['numbers'][0]}</answer>"):
            assert official.compute_reward(c, p["numbers"], p["target"]) == mine.compute_reward(c, p["numbers"], p["target"])


def test_heldout_first64_disjoint_from_train():
    heldout, train = mine.build_dataset(seed=0, size=20000, n_heldout=64)
    full = [official.generate(r) for r in [random.Random(0)] for _ in range(20000)]
    assert len(heldout) == 64 and len(train) == 19936
    assert heldout == full[:64] and train == full[64:]
    key = lambda p: (tuple(p["numbers"]), p["target"])
    assert not ({key(p) for p in heldout} & {key(p) for p in train})  # no identical problem in both (holds for seed 0)
    try:
        import pyarrow.parquet as pq
        d = os.path.join(ROOT, "data/countdown")
        h = pq.read_table(os.path.join(d, "heldout.parquet")).to_pylist()
        t = pq.read_table(os.path.join(d, "train.parquet")).to_pylist()
    except Exception:
        return
    assert [r["extra_info"]["index"] for r in h] == list(range(64))
    assert [r["extra_info"]["index"] for r in t] == list(range(64, 20000))
    for r, p in zip(h + t, full):
        assert json.loads(r["reward_model"]["ground_truth"]) == {"numbers": p["numbers"], "target": p["target"]}
        assert r["prompt"] == mine.messages(p)


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("PASS", k)
