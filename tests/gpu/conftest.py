"""GPU tests: not collected unless TIM_GPU_TESTS=1, so a plain `pytest tests` on a CPU machine never imports torch, vLLM or
verl from here.

    TIM_GPU_TESTS=1 TIM_BASE_MODEL=<HF dir of Qwen3-0.6B> python -m pytest tests/gpu -q -m gpu
"""
import os

collect_ignore_glob = [] if os.environ.get("TIM_GPU_TESTS") == "1" else ["test_*.py"]
