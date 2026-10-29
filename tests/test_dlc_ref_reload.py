"""The reference reload of DLC with resynchronization (tim.dlc.hooks.copy_trainer_to_ref), on two FSDP2 copies of a tiny
Qwen3 built as verl's FSDPEngine builds the trainer and the reference (fully_shard per decoder layer, then the root; the
reference forward-only). After a training step moves the trainer, the lag log p_trainer - log p_ref on sampled tokens
is large; after the copy it is back to 0 (the unit-level form of "after a resynchronization the next step's
tim/split_lag_abs_mean is back at the recompute noise"), including the root group that FSDP2 leaves unsharded after the
reference's no-grad forward.

CPU torch (gloo, one rank):
  CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_dlc_ref_reload.py
TIM_RELOAD_DEVICE=cuda runs the same on one GPU with NCCL and the reference under CPUOffloadPolicy (verl's reference
setting).
"""
import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from tim.dlc.hooks import copy_trainer_to_ref  # noqa: E402

DEVICE = os.environ.get("TIM_RELOAD_DEVICE", "cpu")


@pytest.fixture(scope="module")
def mesh():
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    if not dist.is_initialized():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        dist.init_process_group("nccl" if DEVICE == "cuda" else "gloo", init_method=f"tcp://127.0.0.1:{port}",
                                rank=0, world_size=1)
    if DEVICE == "cuda":
        torch.cuda.set_device(0)
    yield init_device_mesh(DEVICE, (1,))


def _model(seed: int):
    cfg = transformers.Qwen3Config(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
                                   max_position_embeddings=64)
    torch.manual_seed(seed)
    return transformers.Qwen3ForCausalLM(cfg).float()


def _shard(model, mesh, ref: bool):
    """verl's apply_fsdp2 order (decoder layers, then the root); the reference with CPUOffloadPolicy on a GPU."""
    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
    kw = {"mesh": mesh, "reshard_after_forward": True}
    if ref and DEVICE == "cuda":
        kw["offload_policy"] = CPUOffloadPolicy(pin_memory=True)
    elif DEVICE == "cuda":
        model.to("cuda")
    try:
        for layer in model.model.layers:
            fully_shard(layer, **kw)
        fully_shard(model, **kw)
    except Exception as error:      # FSDP2 without a CPU backend in this torch: the smoke wave is the check then
        pytest.skip(f"fully_shard on {DEVICE}: {type(error).__name__}: {error}")
    return model


def _logp(model, ids):
    with torch.no_grad():
        logits = model(input_ids=ids.to(DEVICE)).logits.float()
    return logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:].to(DEVICE).unsqueeze(-1)).squeeze(-1).cpu()


def test_reload_puts_the_lag_back_to_zero(mesh):
    actor = _shard(_model(0), mesh, ref=False)
    ref = _shard(_model(0), mesh, ref=True)                 # theta_0 in both, as at the run's start
    ids = torch.randint(0, 96, (4, 24), generator=torch.Generator().manual_seed(1))
    step1 = (_logp(actor, ids) - _logp(ref, ids)).abs().mean().item()
    assert step1 < 1e-5                                     # same weights: no lag (bf16 noise in a real run)

    opt = torch.optim.SGD(actor.parameters(), lr=0.5)       # a few "training steps" move the trainer
    for _ in range(3):
        loss = -_logp_grad(actor, ids).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    lag = (_logp(actor, ids) - _logp(ref, ids)).abs().mean().item()
    assert lag > 1e-3, lag                                  # pi_old = theta_0 now lags the trainer

    _logp(ref, ids)                                         # the reference's last forward leaves its root unsharded
    out = copy_trainer_to_ref(actor, ref)
    assert out["tensors"] == len(dict(ref.named_parameters())) and out["numel"] > 0
    after = (_logp(actor, ids) - _logp(ref, ids)).abs().mean().item()
    assert after < 1e-5, (step1, lag, after)                # back to the step-1 level
    actor.reshard(); ref.reshard()                          # the roots again (named_parameters -> the sharded DTensors)
    full = lambda t: (t.full_tensor() if hasattr(t, "full_tensor") else t).detach().cpu().float()
    for (n, p), (m, q) in zip(sorted(actor.named_parameters()), sorted(ref.named_parameters())):
        assert n == m and torch.equal(full(p), full(q)), n


def test_reload_refuses_a_mismatched_module(mesh):
    actor = _shard(_model(0), mesh, ref=False)
    with pytest.raises(RuntimeError, match="FSDP2"):
        copy_trainer_to_ref(actor, _model(0))               # an unsharded module (FSDP1 / no FSDP) is not handled


def _logp_grad(model, ids):
    logits = model(input_ids=ids.to(DEVICE)).logits.float()
    return logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:].to(DEVICE).unsqueeze(-1)).squeeze(-1)
