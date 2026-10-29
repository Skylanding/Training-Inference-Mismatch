#!/usr/bin/env bash
# Build the training / measurement environment from the lock file and run verl's score-centering CPU tests.
#   bash envs/verl-main-cu129/build_env.sh                 # -> .venv-tim in the repository root
#   TARGET_DIR=/path/to/env bash envs/verl-main-cu129/build_env.sh
# Needs uv, Linux x86_64, Python 3.12, and network access to PyPI, download.pytorch.org and github.com.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${TARGET_DIR:-$(cd "$HERE/../.." && pwd)/.venv-tim}"
VERL_COMMIT=8718ca30a3f002f93b7c4fd99b9b2506718681bc

echo "== NVIDIA kernel driver =="
if [ -r /proc/driver/nvidia/version ]; then cat /proc/driver/nvidia/version; else echo "/proc/driver/nvidia/version not present"; fi

echo "== uv sync --frozen -> $TARGET_DIR =="
mkdir -p "$(dirname "$TARGET_DIR")"
export UV_PROJECT_ENVIRONMENT="$TARGET_DIR"
(cd "$HERE" && uv sync --frozen)
PY="$TARGET_DIR/bin/python"

echo "== torch / vllm versions =="
"$PY" -c "import torch,vllm;print(torch.__version__, torch.version.cuda, vllm.__version__)"

echo "== clone verl @ $VERL_COMMIT =="
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git clone --quiet https://github.com/verl-project/verl "$WORK/verl"
git -C "$WORK/verl" checkout --quiet "$VERL_COMMIT"

echo "== verl CPU tests of score centering (PR #8010) =="
cd "$WORK/verl"
"$PY" -m pytest -q -rsxX -p no:cacheprovider \
  tests/trainer/ppo/test_score_centering_on_cpu.py \
  tests/workers/test_score_centering_wiring_on_cpu.py \
  tests/experimental/agent_loop/test_agent_loop_extra_fields_schema_on_cpu.py \
  tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py \
  tests/trainer/ppo/v1/test_trainer_base_on_cpu.py
echo "environment ready: $PY"
