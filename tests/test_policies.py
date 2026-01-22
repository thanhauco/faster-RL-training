import pytest

from kvstreams.compaction import (
    KeepRecentTurns,
    MarkovianPolicy,
    Segment,
    SummaryPolicy,
    TokenSlidingWindow,
    build_policy,
)


def segs():
    out = [Segment(0, "prompt", 0, 32, protected=True)]
    for i in range(1, 6):
        out.append(Segment(i, "turn", 32 + 16 * (i - 1), 32 + 16 * i))
    return out


def test_keep_recent_evicts_oldest_until_fit():
    plan = KeepRecentTurns(keep_last=2).plan(segs(), live_tokens=112, budget=100, required=130)
    assert plan.segments == [1, 2]
    assert plan.spans == [(32, 64)]


def test_keep_recent_never_touches_prompt_or_recent():
    plan = KeepRecentTurns(keep_last=2).plan(segs(), 112, 10, 10_000)
    assert plan.segments == [1, 2, 3]


def test_markovian_keeps_only_last_turn():
    plan = MarkovianPolicy(keep_last=1).plan(segs(), 112, 100, 101)
    assert plan.segments == [1, 2, 3, 4]
    assert plan.spans == [(32, 96)]


def test_summary_policy_requests_summary():
    plan = SummaryPolicy(keep_last=1).plan(segs(), 112, 100, 101)
    assert plan.summary_request and plan.segments == [1, 2, 3, 4]


def test_evicted_segments_are_skipped():
    s = segs()
    s[1].evicted_at = 50
    plan = MarkovianPolicy().plan(s, 96, 50, 100)
    assert plan.segments == [2, 3, 4]


def test_token_window_ignores_boundaries_but_keeps_last_turn():
    plan = TokenSlidingWindow(window=8).plan(segs(), 112, 50, 100)
    assert plan.spans == [(32, 96)]  # capped at the start of the latest turn
    plan = TokenSlidingWindow(window=40).plan(segs(), 112, 50, 100)
    assert plan.spans == [(32, 72)]  # cuts turn 3 in half


@pytest.mark.parametrize("cls", [KeepRecentTurns, MarkovianPolicy, SummaryPolicy])
def test_keep_last_must_be_positive(cls):
    with pytest.raises(ValueError):
        cls(keep_last=0)


def test_build_policy():
    assert isinstance(build_policy("markovian"), MarkovianPolicy)
    with pytest.raises(ValueError):
        build_policy("nope")
