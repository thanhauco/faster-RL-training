"""The core claim: KV-streams replay reproduces inference log-probs exactly."""

import pytest
import torch

from kvstreams.envs import build_env
from kvstreams.training import (
    build_sequences,
    kvstream_sequence,
    measure_mismatch,
    window_sequences,
)

EXACT = 1e-9


def run(engine, n=3, seed=0, env="recall", **env_kwargs):
    env_kwargs = env_kwargs or {"num_distractors": 6}
    gen = torch.Generator().manual_seed(seed)
    return [engine.rollout(build_env(env, seed=seed + i, **env_kwargs), gen) for i in range(n)]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(block_size=16, pad=True, commit=False),  # vLLM-style with padding
        dict(block_size=16, pad=True, commit=True),
        dict(block_size=1, pad=False, commit=True),  # SGLang-style
        dict(block_size=8, pad=True, commit=True, policy="markovian"),
        dict(block_size=16, pad=True, commit=True, policy="summary"),
    ],
)
def test_kvstream_replay_is_exact(make_engine, kwargs):
    rs = run(make_engine("kvstream", record_full=True, **kwargs))
    assert all(r.stats.compactions > 0 for r in rs)
    rep = measure_mismatch(make_engine().model, rs, "kvstream")
    assert rep.max_abs_logprob_diff < EXACT
    assert abs(rep.exact_kl) < EXACT


def test_naive_trainers_do_not_match_kvstream_inference(make_engine):
    rs = run(make_engine("kvstream"))
    model = make_engine().model
    assert measure_mismatch(model, rs, "full").max_abs_logprob_diff > 1e-3
    assert measure_mismatch(model, rs, "windows").max_abs_logprob_diff > 1e-3


def test_reprefill_engine_matches_window_replay(make_engine):
    rs = run(make_engine("reprefill"))
    model = make_engine().model
    assert measure_mismatch(model, rs, "windows").max_abs_logprob_diff < EXACT
    assert measure_mismatch(model, rs, "kvstream").max_abs_logprob_diff > 1e-3


def test_unpadded_partial_block_recompute_causes_mismatch(make_engine):
    rs = run(make_engine("kvstream", pad=False, commit=False), n=4)
    assert sum(r.stats.stale_recomputed_tokens for r in rs) > 0
    assert measure_mismatch(make_engine().model, rs, "kvstream").max_abs_logprob_diff > 1e-6


def test_replay_mask_semantics(make_engine):
    (r,) = run(make_engine("kvstream"), n=1)
    seq = kvstream_sequence(r)
    ev = r.log.events[0]
    s, e = ev.evicted[0]
    # Tokens forwarded before the event saw the span; tokens after did not.
    assert seq.mask[ev.time - 1, s:e].all()
    assert not seq.mask[ev.time :, s:e].any()
    # Positions are the original stream indices.
    assert torch.equal(seq.positions, torch.arange(r.length))
    # Diagonal is always visible.
    assert seq.mask.diagonal().all()


def test_kvstream_replay_trains_each_token_once(make_engine):
    (r,) = run(make_engine("kvstream"), n=1)
    kv = build_sequences(r, "kvstream")
    win = window_sequences(r)
    assert sum(s.length for s in kv) == r.length
    assert sum(s.length for s in win) > r.length
    # Every policy token is trained exactly once in both layouts.
    assert sum(int(s.loss_mask.sum()) for s in win) == r.num_loss_tokens


def test_log_roundtrip(make_engine):
    from kvstreams.compaction import CompactionLog

    (r,) = run(make_engine("kvstream"), n=1)
    log2 = CompactionLog.from_json(r.log.to_json())
    assert torch.equal(log2.evicted_at(r.length), r.log.evicted_at(r.length))
