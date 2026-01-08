"""Segments: contiguous stream spans that compaction policies reason about."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class Segment:
    """A contiguous ``[start, end)`` span of the rollout stream.

    kind:
        ``"prompt"`` (system prompt, protected), ``"turn"`` (user message +
        assistant completion + padding) or ``"summary"`` (a turn produced by the
        summary policy).
    evicted_at:
        Stream length at the compaction that removed the whole segment, or None
        while (at least partly) alive.
    """

    index: int
    kind: str
    start: int
    end: int
    protected: bool = False
    evicted_at: Optional[int] = None
    partially_evicted: bool = False

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def alive(self) -> bool:
        return self.evicted_at is None

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    def to_dict(self) -> dict:
        return asdict(self)
