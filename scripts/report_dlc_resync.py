#!/usr/bin/env python3
"""Report of DLC with ESS-triggered resynchronization on Qwen3-1.7B, pure python + numpy:

    python3 scripts/report_dlc_resync.py                        # tables -> results/dlc_resync_1.7b/TABLES.md + steps.csv
    python3 scripts/report_dlc_resync.py check-reload <metrics.jsonl> [--every-step]
                                                                # the reference-reload check of one DLC run (exit 1 on failure)

The report refreshes the training-log and deviation CSVs first (scripts/report_train_logs.py ->
results/train_logs_resync_1.7b/steps.csv, scripts/report_resync_deviation.py -> results/train_logs_resync_1.7b/deviation.csv),
reads the off-support records (results/off_support/<run>/step_<t>.json, through tim.measure.off_support.summary.flat) and
puts DLC (s7d_budget_split) next to DLC without resynchronization (s6c_stale64_split) and importance sampling with
resynchronization (s6c_budget_tis) (tim/measure/ess.py).
"""
import argparse
import contextlib
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tim.measure import ess as pure  # noqa: E402


def smoke(path: Path, every_step: bool) -> int:
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    ok, msgs = pure.check_reload(lines, every_step_synced=every_step, min_checked=2 if every_step else 1)
    for m in msgs:
        print(m)
    print(f"RELOAD_CHECK {'PASS' if ok else 'FAIL'} {path}")
    return 0 if ok else 1


def report(out_dir: Path) -> None:
    import report_resync_deviation
    import report_train_logs
    from tim.measure.off_support.summary import flat
    with contextlib.redirect_stdout(io.StringIO()):       # their own tables are not part of this report
        report_train_logs.main()
        report_resync_deviation.main()
    c7 = json.loads((ROOT / "configs" / "measure" / "off_support.json").read_text())
    recs = [json.loads(f.read_text()) for f in sorted((ROOT / c7["results_dir"]).glob("*/step_*.json"))]
    if not recs:
        raise SystemExit(f"no off-support records under {ROOT / c7['results_dir']} (run scripts/measure.py off-support "
                         "first); nothing written")
    md, rows = pure.report(report_train_logs.rows(), report_resync_deviation.rows(), flat(recs, c7, ROOT) if recs else [])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "TABLES.md").write_text("\n".join(md) + "\n")
    if rows:
        with open(out_dir / "steps.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n".join(md))
    print(f"-> {out_dir / 'TABLES.md'}, {out_dir / 'steps.csv'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("check-reload", help="check the reference reload in one run's metrics.jsonl")
    s.add_argument("metrics", type=Path)
    s.add_argument("--every-step", action="store_true", help="the run synced at every step (TIM_CHI2_BUDGET=-1)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "dlc_resync_1.7b")
    a = ap.parse_args(argv)
    if a.cmd == "check-reload":
        return smoke(a.metrics, a.every_step)
    report(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
