#!/usr/bin/env bash
# Build FlashAttention-3 (Hopper, sm_90a) for torch 2.13 + CUDA 12.9 into a target directory used through PYTHONPATH.
#   PY=.venv-tim/bin/python FA3_PATH=$PWD/.fa3 bash envs/fa3/build_fa3.sh
# Source: Dao-AILab/flash-attention at commit 14c3779 (hopper/, with the cutlass submodule). With torch >= 2.9 its
# hopper/setup.py compiles flash_api_stable.cpp against torch's stable C ABI (aoti_torch_*), so the same source builds
# for torch 2.13 unchanged. The build keeps only what the trainer uses: bf16, head_dim 128, causal, dense + varlen
# forward and backward, PackGQA, sm90a (Qwen3: head_dim 128, no sliding window). Needs nvcc >= 12.6 and ~3 min on 32 cores.
set -euo pipefail
PY=${PY:?set PY to the environment python (torch 2.13)}
FA3_PATH=${FA3_PATH:?set FA3_PATH to the install directory}
FA3_COMMIT=${FA3_COMMIT:-14c3779}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
[ -e "$FA3_PATH" ] && { echo "$FA3_PATH exists; choose a new directory"; exit 2; }
"$PY" -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.version.cuda); sys.exit(0 if torch.__version__.startswith('2.13') else 1)"

git clone --quiet https://github.com/Dao-AILab/flash-attention "$WORK/flash-attention"
git -C "$WORK/flash-attention" checkout --quiet "$FA3_COMMIT"
git -C "$WORK/flash-attention" submodule update --init --quiet csrc/cutlass

export FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE
export MAX_JOBS=${MAX_JOBS:-32} NVCC_THREADS=${NVCC_THREADS:-2} CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

cd "$WORK/flash-attention/hopper"
if command -v uv >/dev/null; then
  uv pip install --python "$PY" --target "$WORK/out" . --no-build-isolation --no-deps
else
  "$PY" -m pip install --target "$WORK/out" . --no-build-isolation --no-deps
fi
# the stable-ABI path: the extension calls libtorch only through aoti_torch_* (no at::/c10:: C++ symbols)
SO=$(ls "$WORK"/out/flash_attn_3/_C*.so | head -1)
n_aoti=$(nm -D --undefined-only "$SO" | grep -c "aoti_torch_" || true)
n_cxx=$(nm -D --undefined-only "$SO" | grep -c -E "_ZN2at|_ZN3c10|_ZN5torch" || true)
echo "aoti_torch_* symbols: $n_aoti, libtorch C++ symbols: $n_cxx"
[ "$n_aoti" -ge 1 ] && [ "$n_cxx" = 0 ] || { echo "not a stable-ABI build"; exit 3; }
cp -a "$WORK/out" "$FA3_PATH"
echo "FA3 installed in $FA3_PATH; use it with FA3_PATH=$FA3_PATH bash scripts/train.sh ... (check: python envs/fa3/check_fa3.py)"
