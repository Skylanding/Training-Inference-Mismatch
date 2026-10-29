"""Wall-clock time per training step on Qwen3-1.7B under a stale sampler (the step-cost table of the appendix).

    python scripts/report_cost.py [--results results]

Reads the verl step metrics of the training logs (one JSON line per step, {"step": s, "data": {...}}) in
results/train_logs_1.7b and results/train_logs_resync_1.7b. A step logged twice (a resumed attempt) keeps its last line.
Columns, each averaged over steps 1-40 and then over the three runs of a correction:
  generation                       timing_s/gen
  forward at the sampler's weights timing_s/old_log_prob + timing_s/ref (for DLC the hook computes log p at the sampler's
                                   weights inside old_log_prob; timing_s/ref is verl's second reference forward on the KL
                                   path, whose output does not enter the loss at kl_loss_coef = 0; absent = 0)
  update                           timing_s/update_actor
  step                             timing_s/step
  copies in 40 steps               steps 1-40 with tim/synced = 1 (the initial copy before step 1 is not counted), mean
                                   over the runs; on-policy training copies after every step (40, no sync hook)
Also printed: the longest weight copy of a resynchronizing run (timing_s/update_weights on its synced steps) and the
mean copy time per step of on-policy training.
"""
import argparse
import json
import math
from pathlib import Path

ROWS = [  # (label, log directory, run prefix, paper values: generation, forward, update, step, copies)
    ("On-policy training", "train_logs_1.7b", "s5_bf16_pg", (6.5, 0.1, 37.1, 50.4, 40)),
    ("Score centering", "train_logs_1.7b", "s5_stale64_sc", (14.7, 0.1, 42.1, 62.2, 0)),
    ("Importance sampling", "train_logs_1.7b", "s5_stale64_tis", (6.7, 0.1, 40.8, 53.2, 0)),
    ("Importance sampling with resynchronization", "train_logs_resync_1.7b", "s6c_budget_tis", (7.0, 0.1, 39.5, 52.7, 3.7)),
    ("DLC w/o resynchronization", "train_logs_resync_1.7b", "s6c_stale64_split", (14.5, 56.6, 41.7, 118.2, 0)),
    ("DLC", "train_logs_resync_1.7b", "s7d_budget_split", (13.7, 54.2, 40.1, 113.2, 1)),
]
STEPS = range(1, 41)


def load(path):
    by_step = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_step[int(r["step"])] = r["data"]          # the last line of a step wins
    return by_step


def mean(xs):
    return sum(xs) / len(xs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    a = ap.parse_args()
    print("| Correction | Generation | Forward at the sampler's weights | Update | Step | Copies in 40 steps | runs |")
    print("|---|---|---|---|---|---|---|")
    notes = []
    for label, d, prefix, paper in ROWS:
        runs = [load(f) for f in sorted((a.results / d).glob(f"{prefix}_seed*.jsonl"))]
        if not runs:
            print(f"| {label} | no logs under {d} |")
            continue
        for r in runs:
            missing = [s for s in STEPS if s not in r]
            if missing:
                raise SystemExit(f"{label}: steps {missing} missing in a log")

        def col(f):
            return mean([mean([f(r[s]) for s in STEPS]) for r in runs])
        gen = col(lambda x: x["timing_s/gen"])
        fwd = col(lambda x: x.get("timing_s/old_log_prob", 0.0) + x.get("timing_s/ref", 0.0))
        upd = col(lambda x: x["timing_s/update_actor"])
        step = col(lambda x: x["timing_s/step"])
        if prefix.endswith("bf16_pg"):
            copies = float(len(STEPS))
            notes.append(f"on-policy copy time per step: {col(lambda x: x['timing_s/update_weights']):.2f} s")
        else:
            per_run = [sum(1 for s in STEPS if r[s].get("tim/synced", 0.0) == 1.0) for r in runs]
            copies = mean(per_run)
            if any(per_run):
                longest = max(r[s]["timing_s/update_weights"] for r in runs for s in STEPS if r[s].get("tim/synced") == 1.0)
                notes.append(f"{label}: copies per run {per_run}, longest copy {longest:.2f} s")
        got = (gen, fwd, upd, step, copies)
        cells = [f"{g:.1f} ({p})" if not math.isclose(round(g, 1), p) else f"{g:.1f}" for g, p in zip(got, paper)]
        print(f"| {label} | " + " | ".join(cells) + f" | {len(runs)} |")
    print("\nCells show the value computed here; a value in brackets is the paper's where the two differ after rounding.")
    for n in notes:
        print(n)


if __name__ == "__main__":
    main()
