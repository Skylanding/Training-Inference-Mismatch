"""HF trainer forward module for the offline measurements (GPU only; imports torch / transformers).

`load_trainer` gives the trainer's numerics of the `sdpa` kernel set: HF bf16, sdpa attention, no fused kernels, right-padded
micro-batches as the FSDP trainer runs them without remove-padding. The `fa3_liger` kernel set of the paper's trainings is
built in tim.measure.engines.
"""
from __future__ import annotations


def load_trainer(ckpt_dir: str, device: str = "cuda"):
    """HF bf16 trainer forward module (sdpa attention, no fused kernels)."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=torch.bfloat16, attn_implementation="sdpa")
    return model.to(device).eval()
