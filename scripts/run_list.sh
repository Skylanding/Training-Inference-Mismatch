#!/usr/bin/env bash
# Train every configuration of a run list (configs/runs/*.txt) with scripts/train.sh, one after the other.
# Each non-comment line is a set of environment assignments for scripts/train.sh.
#   bash scripts/run_list.sh configs/runs/qwen3_0.6b_sources.txt          # train
#   DRY=1 bash scripts/run_list.sh configs/runs/qwen3_0.6b_sources.txt    # print each command only
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LIST=${1:?usage: run_list.sh <configs/runs/file.txt> [extra hydra overrides ...]}
shift
grep -vE '^\s*(#|$)' "$LIST" | while read -r line; do
  echo "== $line"
  # shellcheck disable=SC2086
  env $line bash "$REPO/scripts/train.sh" "$@" < /dev/null
done
