"""Write data/countdown/{train,heldout}.parquet in verl's RL parquet format
(schema mirrors verl examples/data_preprocess/gsm8k.py at 8718ca30:
data_source, prompt (chat list), ability, reward_model{style, ground_truth},
extra_info). Pure python + pyarrow; no torch.

heldout = first 64 problems of random.Random(0); train = remaining 19,936.
ground_truth is a JSON string {"numbers": [...], "target": N}.
"""
import argparse
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tim.tasks.countdown import build_dataset, messages  # noqa: E402

DATA_SOURCE = "countdown"


def to_rows(problems, split, offset=0):
    return [{
        "data_source": DATA_SOURCE,
        "prompt": messages(p),
        "ability": "math",
        "reward_model": {"style": "rule",
                         "ground_truth": json.dumps({"numbers": p["numbers"], "target": p["target"]})},
        "extra_info": {"split": split, "index": offset + i, "solution": p["solution"]},
    } for i, p in enumerate(problems)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "countdown"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=20000)
    ap.add_argument("--n_heldout", type=int, default=64)
    a = ap.parse_args()
    heldout, train = build_dataset(a.seed, a.size, a.n_heldout)
    os.makedirs(a.out_dir, exist_ok=True)
    # index = position in the full 20k dataset (heldout 0..63, train 64..19999)
    pq.write_table(pa.Table.from_pylist(to_rows(train, "train", a.n_heldout)), os.path.join(a.out_dir, "train.parquet"))
    pq.write_table(pa.Table.from_pylist(to_rows(heldout, "heldout", 0)), os.path.join(a.out_dir, "heldout.parquet"))
    print(f"wrote {len(train)} train, {len(heldout)} heldout to {a.out_dir}")


if __name__ == "__main__":
    main()
