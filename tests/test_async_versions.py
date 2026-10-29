"""CPU tests of DLC under continuous asynchrony: tim/dlc/async_versions.py (per-token versions, per-version selection /
merge, ESS per response, version-gap statistics) and the rollout-side patches of tim/dlc/async_hooks.py on stand-in
classes (no verl, no vLLM).
    CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_async_versions.py
"""
import asyncio
import math
import types

import pytest

torch = pytest.importorskip("torch")

from tim.dlc import async_hooks  # noqa: E402
from tim.dlc.async_versions import (MISSING, abs_diff_by_gap, defer_responses, drop_token_mask,  # noqa: E402
                                    ess_per_response, full_row_versions, gap_stats, merge_by_version,
                                    per_response_mean, segment_token_versions, tag_token_consistency,
                                    versions_to_evict)


# ---------------------------------------------------------------------------------------------------------------------
# per-token versions
# ---------------------------------------------------------------------------------------------------------------------
def test_segment_token_versions():
    assert segment_token_versions([(3, 2), (4, 0), (5, 3)], 5) == [3, 3, 5, 5, 5]
    assert segment_token_versions([(None, 2), (1, 1)], 3) == [MISSING, MISSING, 1]
    assert segment_token_versions([(0, 4)], 2) == [0, 0]                     # cut to the client's count
    assert segment_token_versions([(0, 1)], 3) == [0, MISSING, MISSING]      # padded: the trainer refuses MISSING
    assert segment_token_versions([], 0) == []


def test_versions_to_evict():
    assert versions_to_evict({0, 1, 2, 3, 4, 5}, 5, 5) == [0]
    assert versions_to_evict({3, 4, 5}, 5, 5) == []
    assert versions_to_evict({1, 5, 7}, 5, 2) == [1, 7]
    with pytest.raises(ValueError):
        versions_to_evict({1}, 1, 0)


# ---------------------------------------------------------------------------------------------------------------------
# selection / merge, against verl's own index conventions
# ---------------------------------------------------------------------------------------------------------------------
def _verl_response_from_nested(values, seq_offsets, resp_lens):
    """verl.workers.utils.padding.response_from_nested on values/offsets: rows seq_end - R - 1 .. seq_end - 2."""
    out = []
    for r, end in zip(resp_lens.tolist(), seq_offsets[1:].tolist()):
        out.append(values[end - r - 1:end - 1])
    return torch.cat(out)


def _batch():
    # 3 sequences: prompt lengths 4, 2, 3; responses 3, 5, 1
    prompt = torch.tensor([4, 2, 3])
    resp = torch.tensor([3, 5, 1])
    seq_offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(prompt + resp, 0)])
    versions = torch.tensor([7, 7, 8,   5, 5, 6, 7, 8,   8])          # cur = 8
    return prompt, resp, seq_offsets, versions


def test_full_row_versions_matches_response_rows():
    prompt, resp, offs, versions = _batch()
    rows = full_row_versions(offs, resp, versions)
    assert rows.shape[0] == int(offs[-1])
    # the rows response_from_nested reads are exactly the versioned rows, in order
    assert torch.equal(_verl_response_from_nested(rows, offs, resp), versions)
    assert int((rows != MISSING).sum()) == int(resp.sum())
    # verl pad_rollout_topk: response token r of a sequence is predicted by row prompt_length - 1 + r
    start = int(offs[1]) + int(prompt[1]) - 1
    assert torch.equal(rows[start:start + 5], versions[3:8])


def test_full_row_versions_rejects_bad_shapes():
    _, resp, offs, versions = _batch()
    with pytest.raises(ValueError):
        full_row_versions(offs, resp, versions[:-1])


def test_merge_full_rows_equals_merge_of_responses():
    """Merging the full-sequence log-probs by row version and then extracting the response (verl's old_log_probs path)
    equals merging the response-level values by token version; for every token the value is its version's."""
    _, resp, offs, versions = _batch()
    n = int(offs[-1])
    g = torch.Generator().manual_seed(0)
    per_full = {k: torch.randn(n, generator=g) for k in (5, 6, 7, 8)}
    cur = 8
    rows = full_row_versions(offs, resp, versions)
    merged_full = merge_by_version(per_full[cur], rows, {k: v for k, v in per_full.items() if k != cur})
    a = _verl_response_from_nested(merged_full, offs, resp)
    per_resp = {k: _verl_response_from_nested(v, offs, resp) for k, v in per_full.items()}
    b = merge_by_version(per_resp[cur], versions, {k: v for k, v in per_resp.items() if k != cur})
    assert torch.equal(a, b)
    for t, k in enumerate(versions.tolist()):
        assert b[t] == per_resp[k][t]


def test_merge_heads_2d():
    _, resp, offs, versions = _batch()
    n, k = int(offs[-1]), 4
    base = torch.zeros(n, k)
    per = {5: torch.full((n, k), 5.0), 6: torch.full((n, k), 6.0), 7: torch.full((n, k), 7.0)}
    rows = full_row_versions(offs, resp, versions)
    out = merge_by_version(base, rows, per)
    got = _verl_response_from_nested(out, offs, resp)[:, 0]
    want = torch.tensor([7.0, 7, 0, 5, 5, 6, 7, 0, 0])               # version 8 = current -> base
    assert torch.equal(got, want)
    # prompt rows and the last row of every sequence keep the base value
    assert float(out[rows == MISSING].abs().sum()) == 0.0
    with pytest.raises(ValueError):
        merge_by_version(base, rows, {5: torch.zeros(n, k + 1)})


# ---------------------------------------------------------------------------------------------------------------------
# ESS admission
# ---------------------------------------------------------------------------------------------------------------------
def test_ess_per_response():
    lens = torch.tensor([2, 3, 2])
    log_ratio = torch.tensor([0.0, math.log(2.0),   0.0, 0.0, 0.0,   5.0, -5.0])
    mask = torch.tensor([1, 1,   1, 1, 1,   0, 0])
    ess = ess_per_response(log_ratio, mask, lens)
    assert ess[0] == pytest.approx(1 / 2.5)           # mean rho^2 = (1 + 4) / 2
    assert ess[1] == pytest.approx(1.0)
    assert ess[2] == pytest.approx(1.0)               # no valid token: never deferred
    defer = defer_responses(ess, 0.5, torch.tensor([True, True, True]))
    assert defer.tolist() == [True, False, False]
    assert defer_responses(ess, 0.5, torch.tensor([False, True, True])).tolist() == [False, False, False]
    new = drop_token_mask(mask, lens, defer)
    assert new.tolist() == [0, 0, 1, 1, 1, 0, 0] and new.dtype == mask.dtype


def test_ess_clamps_huge_ratios():
    ess = ess_per_response(torch.tensor([200.0]), torch.tensor([1]), torch.tensor([1]))
    assert float(ess[0]) == pytest.approx(math.exp(-40.0))


def test_per_response_mean_empty():
    out = per_response_mean(torch.tensor([1.0, 3.0]), torch.tensor([1, 1]), torch.tensor([2, 0]))
    assert float(out[0]) == 2.0 and math.isnan(float(out[1]))


# ---------------------------------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------------------------------
def test_gap_stats():
    _, resp, _, versions = _batch()
    mask = torch.ones_like(versions)
    s = gap_stats(versions, mask, resp, cur=8)
    gaps = 8 - versions
    assert s["tim/async_tok_gap_mean"] == pytest.approx(float(gaps.double().mean()))
    assert s["tim/async_tok_gap_max"] == 3.0
    fr = [s[f"tim/async_tok_gap_frac_g{g}"] for g in range(8)] + [s["tim/async_tok_gap_frac_g8plus"]]
    assert sum(fr) == pytest.approx(1.0)
    assert s["tim/async_tok_gap_frac_g0"] == pytest.approx(3 / 9)
    assert s["tim/async_n_versions"] == 4.0
    assert s["tim/async_resp_versions_mean"] == pytest.approx((2 + 4 + 1) / 3)
    assert s["tim/async_resp_multi_version_frac"] == pytest.approx(2 / 3)
    # masked tokens do not count
    m2 = mask.clone()
    m2[3:8] = 0
    s2 = gap_stats(versions, m2, resp, cur=8)
    assert s2["tim/async_n_versions"] == 2.0 and s2["tim/async_tokens"] == 4.0
    assert gap_stats(versions, torch.zeros_like(versions), resp, 8) == {"tim/async_tokens": 0.0}


def test_abs_diff_by_gap():
    versions = torch.tensor([3, 3, 2, 1])
    a, b = torch.tensor([1.0, 1.0, 2.0, 4.0]), torch.tensor([0.5, 1.5, 1.0, 1.0])
    out = abs_diff_by_gap(a, b, versions, torch.tensor([1, 1, 1, 0]), cur=3, name="x")
    assert out["tim/async_x_abs_mean_fresh"] == pytest.approx(0.5)
    assert out["tim/async_x_abs_mean_stale"] == pytest.approx(1.0)      # the masked gap-2 token is out
    assert out["tim/async_x_abs_mean_all"] == pytest.approx(2.0 / 3)


def test_tag_token_consistency():
    lens = torch.tensor([2, 2])
    versions = torch.tensor([1, 2, 3, 3])
    out = tag_token_consistency([1, 3], [2, 4], versions, torch.ones(4, dtype=torch.long), lens)
    assert out["tim/async_tag_token_mismatch_frac"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------------------------------------------------
# rollout-side patches on stand-in classes
# ---------------------------------------------------------------------------------------------------------------------
class _Out:
    def __init__(self, token_ids, stop_reason, extra=None):
        self.token_ids, self.stop_reason = list(token_ids), stop_reason
        self.extra_fields = dict(extra or {})


def _fake_client_module(script):
    """LLMServerClient.generate returns the scripted segments; FullyAsyncLLMServerClient.generate is verl's resume loop
    in miniature (resubmit while aborted, concatenate tokens)."""
    class LLMServerClient:
        async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
            await asyncio.sleep(0)
            return script[request_id].pop(0)

    class FullyAsyncLLMServerClient(LLMServerClient):
        async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
            final = _Out([], None)
            while True:
                out = await super().generate(request_id, prompt_ids=prompt_ids + final.token_ids,
                                             sampling_params=sampling_params)
                final.token_ids.extend(out.token_ids)
                if out.stop_reason != "aborted":
                    break
            return final

    return types.SimpleNamespace(LLMServerClient=LLMServerClient, FullyAsyncLLMServerClient=FullyAsyncLLMServerClient)


def test_client_patch_records_segment_versions():
    K = async_hooks.ADMIT_KEY
    script = {"a": [_Out([1, 2], "aborted", {K: 4}), _Out([], "aborted", {K: None}), _Out([3, 4, 5], "stop", {K: 6})],
              "b": [_Out([9], "aborted", {K: 5}), _Out([8, 7], "stop", {K: 5})]}
    mod = _fake_client_module(script)
    assert async_hooks.patch_client(mod).startswith("applied")
    assert async_hooks.patch_client(mod) == "applied"
    client = mod.FullyAsyncLLMServerClient()

    async def run():
        return await asyncio.gather(client.generate("a", prompt_ids=[0], sampling_params={}),
                                    client.generate("b", prompt_ids=[0], sampling_params={}))
    a, b = asyncio.run(run())
    assert a.extra_fields[async_hooks.TOKENS_KEY] == [4, 4, 6, 6, 6]
    assert b.extra_fields[async_hooks.TOKENS_KEY] == [5, 5, 5]
    # the plain client outside the resume loop passes the segment through and records nothing
    script["c"] = [_Out([1], "stop", {K: 2})]
    plain = asyncio.run(mod.LLMServerClient.generate(client, "c", prompt_ids=[], sampling_params={}))
    assert async_hooks.TOKENS_KEY not in plain.extra_fields and plain.extra_fields[K] == 2


def test_server_patch_version_at_admission():
    """Two concurrent requests; the weights version moves between one's admission and its return (an aborted segment
    returning after set_global_steps): each segment keeps the version it was admitted under."""
    order = []

    class vLLMHttpServer:
        def __init__(self):
            self.global_steps = 3

        async def _park_until_admitted(self, request_id):
            await asyncio.sleep(0)
            return None

        async def generate(self, prompt_ids, sampling_params, request_id):
            rejected = await self._park_until_admitted(request_id)
            if rejected is not None:
                return rejected
            order.append(request_id)
            await asyncio.sleep(0.01 if request_id == "slow" else 0)
            return _Out([1], "aborted" if request_id == "slow" else "stop", {"global_steps": self.global_steps})

    mod = types.SimpleNamespace(vLLMHttpServer=vLLMHttpServer)
    assert async_hooks.patch_server(mod).startswith("applied")
    server = vLLMHttpServer()

    async def run():
        slow = asyncio.ensure_future(server.generate([0], {}, "slow"))
        await asyncio.sleep(0.002)
        server.global_steps = 4                # set_global_steps after the push, while "slow" is still returning
        fast = await server.generate([0], {}, "fast")
        return await slow, fast
    slow, fast = asyncio.run(run())
    assert slow.extra_fields[async_hooks.ADMIT_KEY] == 3 and slow.extra_fields["global_steps"] == 4
    assert fast.extra_fields[async_hooks.ADMIT_KEY] == 4


def test_output_patch_moves_versions_to_field():
    class AgentLoopOutput:
        def __init__(self, n, versions):
            self.n, self.versions = n, versions

        def as_dict(self):
            return {"responses": torch.zeros(self.n, dtype=torch.long),
                    "extra_fields": {async_hooks.TOKENS_KEY: self.versions, "other": 1}}

    mod = types.SimpleNamespace(AgentLoopOutput=AgentLoopOutput)
    async_hooks.patch_output(mod)
    out = AgentLoopOutput(3, [2, 2, 3, 3]).as_dict()
    assert out[async_hooks.FIELD].tolist() == [2, 2, 3] and out[async_hooks.FIELD].dtype == torch.int64
    assert async_hooks.TOKENS_KEY not in out["extra_fields"] and out["extra_fields"]["other"] == 1
    assert AgentLoopOutput(3, [1]).as_dict()[async_hooks.FIELD].tolist() == [1, MISSING, MISSING]
