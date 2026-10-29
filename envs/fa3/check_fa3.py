"""Smoke test of the FA3 build: causal varlen forward + backward in bf16 in a child process, compared with SDPA.

    FA3_PATH=/path/to/fa3 python envs/fa3/check_fa3.py [--heads 32 --kv-heads 8 --head-dim 128]

Prints `ATTENTION=flash_attention_3` and exits 0, or `FA3 not active: <reason>` + `ATTENTION=flash_attention_2`, exit 1.
The check runs in a child process (it starts CUDA and binds a flash kernel for the whole process), so the parent stays
clean.
"""
import argparse
import os
import subprocess
import sys

TARGET = os.environ.get('FA3_PATH', '')
CHECK_S = 300


def unavailable(heads: int, kv_heads: int, size: int) -> str | None:
    """Why FA3 cannot run this attention here; None when it can (the check runs in a process of its own)."""
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(p for p in (TARGET, env.get('PYTHONPATH', '')) if p)
    try:
        done = subprocess.run([sys.executable, __file__, '--check', str(heads), str(kv_heads), str(size)],
                              capture_output=True, text=True, timeout=CHECK_S, env=env)
    except subprocess.TimeoutExpired:
        return f"its check on the first GPU did not end within {CHECK_S} s"
    sys.stdout.write(done.stdout)
    if done.returncode == 0:
        return None
    lines = done.stderr.strip().splitlines()
    return lines[-1] if lines else f'its check exited with code {done.returncode}'


def check(heads: int, kv_heads: int, size: int) -> None:
    """Raise unless FA3 runs a causal varlen forward and backward of this attention in bf16 on the first GPU."""
    try:
        import flash_attn_interface
    except ImportError as error:
        raise RuntimeError(f'{error}: flash_attn_3 is not on PYTHONPATH or does not match torch') from error
    import torch
    from transformers.utils import is_flash_attn_3_available
    print('torch', torch.__version__, 'cuda', torch.version.cuda, 'flash_attn_interface', flash_attn_interface.__file__)
    if not torch.cuda.is_available():
        raise RuntimeError('no GPU is visible')
    print('device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
    if not is_flash_attn_3_available():
        raise RuntimeError('transformers does not find the flash_attn_3 package (needs dist name flash-attn-3)')
    shapes = [(heads, size)] + [(kv_heads, size)] * 2
    q, k, v = (torch.randn(128, *shape, dtype=torch.bfloat16, device='cuda', requires_grad=True) for shape in shapes)
    edges = torch.tensor([0, 64, 128], dtype=torch.int32, device='cuda')        # two sequences of 64 tokens
    out = flash_attn_interface.flash_attn_varlen_func(q, k, v, edges, edges, 64, 64, causal=True)
    out = out[0] if isinstance(out, tuple) else out
    out.float().sum().backward()
    torch.cuda.synchronize()
    grads = [t.grad for t in (q, k, v)]
    if any(g is None or not torch.isfinite(g.float()).all() for g in grads):
        raise RuntimeError('FA3 backward produced missing or non-finite gradients')
    # numeric agreement with SDPA (GQA expanded), first sequence only
    rep = heads // kv_heads
    qs, ks, vs = (t[:64].detach().float().transpose(0, 1) for t in (q, k, v))
    ks, vs = ks.repeat_interleave(rep, dim=0), vs.repeat_interleave(rep, dim=0)
    ref = torch.nn.functional.scaled_dot_product_attention(qs, ks, vs, is_causal=True).transpose(0, 1)
    err = (out[:64].float() - ref).abs().max().item()
    print(f'max |FA3 - SDPA| on seq 0: {err:.4f}')
    if err > 0.05:
        raise RuntimeError(f'FA3 forward disagrees with SDPA by {err:.3f}')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--check']:
        try:
            check(*map(int, sys.argv[2:5]))
        except Exception as error:                      # any failure means FA2, and says why in one line
            text = str(error).strip() if isinstance(error, RuntimeError) else f'{type(error).__name__}: {error}'
            sys.exit(text.splitlines()[0] if text else type(error).__name__)
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument('--heads', type=int, default=32)
        ap.add_argument('--kv-heads', type=int, default=8)
        ap.add_argument('--head-dim', type=int, default=128)
        a = ap.parse_args()
        reason = unavailable(a.heads, a.kv_heads, a.head_dim)
        if reason:
            print(f'FA3 not active: {reason}', file=sys.stderr)
            print('ATTENTION=flash_attention_2')
            sys.exit(1)
        print('ATTENTION=flash_attention_3')
