"""Compaction policies.

A policy decides *what* to forget; KV-streams only changes *how* the forgetting
is executed (in place on the live cache instead of by re-prefilling). Policies
therefore return spans of the rollout stream and are shared by both engines.

The authors found that token-level eviction produced degenerate text (likely
because it removes message-boundary tokens the chat format depends on), so the
default policies evict complete turns, boundary tags included.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..utils.spans import Span, merge_spans
from .segments import Segment


@dataclass
class CompactionPlan:
    spans: list[Span]
    segments: list[int] = field(default_factory=list)
    summary_request: Optional[str] = None
    reason: str = ""

    @property
    def empty(self) -> bool:
        return not self.spans and self.summary_request is None


def _evictable(segments: list[Segment], keep_last: int) -> list[Segment]:
    """Alive, unprotected segments, excluding the ``keep_last`` most recent ones."""
    alive = [s for s in segments if s.alive and not s.partially_evicted]
    recent = {s.index for s in alive[-keep_last:]} if keep_last > 0 else set()
    return [s for s in alive if not s.protected and s.index not in recent]


class CompactionPolicy(ABC):
    name = "base"
    granularity = "turn"

    @abstractmethod
    def plan(
        self, segments: list[Segment], live_tokens: int, budget: int, required: int
    ) -> CompactionPlan:
        """Return what to evict.

        Args:
            segments: all segments of the rollout so far, in stream order.
            live_tokens: entries currently in the KV cache.
            budget: the soft context budget.
            required: live entries needed for the next turn (live + incoming + reserve).
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class KeepRecentTurns(CompactionPolicy):
    """Keep the prompt and the most recent turns; drop the oldest turns until it fits."""

    name = "keep_recent"

    def __init__(self, keep_last: int = 2) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1 (the latest turn must stay live)")
        self.keep_last = keep_last

    def plan(self, segments, live_tokens, budget, required) -> CompactionPlan:
        overflow = required - budget
        chosen: list[Segment] = []
        freed = 0
        for seg in _evictable(segments, self.keep_last):
            if freed >= overflow:
                break
            chosen.append(seg)
            freed += seg.length
        return CompactionPlan(
            spans=merge_spans(s.span for s in chosen),
            segments=[s.index for s in chosen],
            reason=f"overflow={overflow}",
        )

    def __repr__(self) -> str:
        return f"KeepRecentTurns(keep_last={self.keep_last})"


class MarkovianPolicy(CompactionPolicy):
    """Markovian-Thinker style reset: keep the prompt and only the last ``keep_last`` turns.

    The model must carry whatever state it needs in its most recent turn(s).
    """

    name = "markovian"

    def __init__(self, keep_last: int = 1) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.keep_last = keep_last

    def plan(self, segments, live_tokens, budget, required) -> CompactionPlan:
        chosen = _evictable(segments, self.keep_last)
        return CompactionPlan(
            spans=merge_spans(s.span for s in chosen),
            segments=[s.index for s in chosen],
            reason="markovian reset",
        )

    def __repr__(self) -> str:
        return f"MarkovianPolicy(keep_last={self.keep_last})"


class SummaryPolicy(CompactionPolicy):
    """Ask the model for a summary turn, then drop everything older except the last turn(s)."""

    name = "summary"

    def __init__(self, keep_last: int = 1, request: str = "summarize what matters") -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.keep_last = keep_last
        self.request = request

    def plan(self, segments, live_tokens, budget, required) -> CompactionPlan:
        chosen = _evictable(segments, self.keep_last)
        return CompactionPlan(
            spans=merge_spans(s.span for s in chosen),
            segments=[s.index for s in chosen],
            summary_request=self.request,
            reason="summarize then evict",
        )

    def __repr__(self) -> str:
        return f"SummaryPolicy(keep_last={self.keep_last})"


class TokenSlidingWindow(CompactionPolicy):
    """Token-level eviction: keep the prompt plus the last ``window`` stream tokens.

    Ignores message boundaries. Included for comparison: it can cut a turn in half
    and drop ``<|user|>``/``<|end|>`` tags, which in the original experiments led to
    degenerate text. With ``block_size > 1`` the request is shrunk to whole blocks.
    """

    name = "token_window"
    granularity = "token"

    def __init__(self, window: int = 128) -> None:
        self.window = window

    def plan(self, segments, live_tokens, budget, required) -> CompactionPlan:
        if not segments:
            return CompactionPlan(spans=[])
        prompt_end = max((s.end for s in segments if s.protected), default=0)
        stream_end = segments[-1].end
        # Always keep the most recent turn intact: it may hold uncommitted tokens.
        cut = min(stream_end - self.window, segments[-1].start)
        if cut <= prompt_end:
            return CompactionPlan(spans=[])
        touched = [s.index for s in segments if s.start < cut and s.end > prompt_end and s.alive]
        return CompactionPlan(spans=[(prompt_end, cut)], segments=touched, reason="sliding window")

    def __repr__(self) -> str:
        return f"TokenSlidingWindow(window={self.window})"


POLICIES = {
    "keep_recent": KeepRecentTurns,
    "markovian": MarkovianPolicy,
    "summary": SummaryPolicy,
    "token_window": TokenSlidingWindow,
}


def build_policy(name: str, **kwargs) -> CompactionPolicy:
    try:
        cls = POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}") from exc
    return cls(**kwargs)
