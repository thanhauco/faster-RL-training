import torch

from kvstreams.envs import build_env


def run(engine, n=3, seed=0, tf=False, env="recall", **env_kwargs):
    env_kwargs = env_kwargs or {"num_distractors": 6}
    gen = torch.Generator().manual_seed(seed)
    return [
        engine.rollout(build_env(env, seed=seed + i, **env_kwargs), gen, teacher_forcing=tf)
        for i in range(n)
    ]


def test_kvstream_processes_every_token_once(make_engine):
    for r in run(make_engine("kvstream")):
        assert r.stats.compactions > 0
        assert r.stats.tokens_forwarded == r.length
        assert r.stats.reprefill_tokens == 0


def test_reprefill_reprocesses_retained_tokens(make_engine):
    for r in run(make_engine("reprefill")):
        assert r.stats.compactions > 0
        assert r.stats.reprefill_tokens > 0
        assert r.stats.tokens_forwarded == r.length + r.stats.reprefill_tokens


def test_padded_turns_are_block_aligned(make_engine):
    for r in run(make_engine("kvstream", block_size=16, pad=True)):
        for seg in r.segments:
            assert seg.start % 16 == 0 and seg.end % 16 == 0
        for ev in r.log:
            for s, e in ev.evicted:
                assert s % 16 == 0 and e % 16 == 0


def test_turn_eviction_removes_whole_turns_with_tags(make_engine, tok):
    for r in run(make_engine("kvstream")):
        evicted = [seg for seg in r.segments if not seg.alive]
        assert evicted
        for seg in evicted:
            assert r.tokens[seg.start] == tok.role_id("user")
            assert not seg.partially_evicted


def test_overflow_is_reported_when_policy_cannot_fit(make_engine):
    # keep_last=2 retains more than a 160-token budget can hold for these turns.
    rs = run(make_engine("kvstream", budget=160, keep_last=2))
    assert sum(r.stats.budget_overflows for r in rs) > 0


def test_live_cache_respects_budget(make_engine):
    eng = make_engine("kvstream", budget=160, keep_last=1)
    for r in run(eng):
        assert r.stats.budget_overflows == 0
        assert r.stats.peak_live_tokens <= 160


def test_prompt_is_never_evicted(make_engine):
    for r in run(make_engine("kvstream", policy="markovian")):
        assert r.segments[0].alive and r.segments[0].protected


def test_teacher_forcing_follows_oracle(make_engine):
    eng = make_engine("kvstream")
    (r,) = run(eng, n=1, tf=True)
    env = build_env("recall", seed=0, num_distractors=6)
    env.reset()
    assert r.reward == 1.0
    assert r.completions[-1] == env.name
    assert all(c == "ok" for c in r.completions[:-1])


def test_engines_agree_on_compaction_under_teacher_forcing(make_engine):
    a = run(make_engine("kvstream"), n=2, tf=True)
    b = run(make_engine("reprefill"), n=2, tf=True)
    for x, y in zip(a, b):
        assert x.tokens == y.tokens
        assert [e.evicted for e in x.log] == [e.evicted for e in y.log]


def test_partial_block_recompute_is_flagged(make_engine):
    rs = run(make_engine("kvstream", pad=False, commit=False))
    assert sum(r.stats.recomputed_tokens for r in rs) > 0
    assert sum(r.stats.stale_recomputed_tokens for r in rs) > 0


def test_padding_removes_partial_block_recompute(make_engine):
    rs = run(make_engine("kvstream", pad=True, commit=False))
    assert sum(r.stats.recomputed_tokens for r in rs) == 0


def test_sglang_style_single_token_blocks(make_engine):
    for r in run(make_engine("kvstream", block_size=1, pad=False)):
        assert r.stats.padding_tokens == 0
        assert r.stats.compactions > 0


def test_summary_policy_inserts_summary_turn(make_engine):
    (r,) = run(make_engine("kvstream", policy="summary", budget=128), n=1, tf=True)
    kinds = [s.kind for s in r.segments]
    assert "summary" in kinds
    assert r.reward == 1.0


def test_token_window_policy_runs(make_engine):
    (r,) = run(
        make_engine("kvstream", policy="token_window", block_size=1, pad=False, window=64), n=1
    )
    assert r.stats.compactions > 0
