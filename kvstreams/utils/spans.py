"""Helpers for half-open integer spans ``[start, end)`` over the rollout stream."""

from __future__ import annotations

from typing import Iterable

Span = tuple[int, int]


def merge_spans(spans: Iterable[Span]) -> list[Span]:
    """Sort and merge overlapping or adjacent spans, dropping empty ones."""
    out: list[list[int]] = []
    for s, e in sorted((int(s), int(e)) for s, e in spans if e > s):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def positions_to_spans(positions: Iterable[int]) -> list[Span]:
    """Compress integer positions into merged spans."""
    return merge_spans((p, p + 1) for p in positions)


def span_length(spans: Iterable[Span]) -> int:
    return sum(e - s for s, e in spans)


def in_spans(position: int, spans: Iterable[Span]) -> bool:
    return any(s <= position < e for s, e in spans)
