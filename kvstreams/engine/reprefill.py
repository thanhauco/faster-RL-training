"""Re-prefill baseline: rebuild a shorter sequence and run it through the model again.

This is what most agentic-RL systems do today. After every compaction the KV cache
is thrown away, the retained tokens are re-numbered from position 0 and every one
of them is prefilled again, so a token that survives ``k`` compactions is
processed ``k + 1`` times.
"""

from __future__ import annotations

from ..utils.spans import Span, positions_to_spans
from .base import BaseEngine, _Run


class RePrefillEngine(BaseEngine):
    name = "reprefill"

    def _positions(self, run: _Run, idx: list[int]) -> list[int]:
        start = len(run.context)
        return list(range(start, start + len(idx)))

    def _evict(self, run: _Run, spans: list[Span]) -> list[Span]:
        current = run.context + run.pending
        drop = [i for i in current if any(s <= i < e for s, e in spans)]
        if not drop:
            return []
        dropped = set(drop)
        retained = [i for i in current if i not in dropped]
        self.cache.free_sequence(run.seq_id)
        self.cache.add_sequence(run.seq_id)
        run.context, run.pending = [], []
        if retained:
            self._forward(run, retained, "reprefill")
        self._uncommit_tail(run)
        return positions_to_spans(drop)
