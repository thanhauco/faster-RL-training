"""KV-streams engine: compaction happens in place on the live paged KV cache.

* Positions never reset. A token's RoPE position is its stream index, so retained
  keys keep their original rotation and new queries/keys continue counting from
  the original position counter even though physical storage got shorter.
* Eviction removes whole KV blocks and joins the remaining block references; no
  retained token is ever processed twice.
"""

from __future__ import annotations

from ..utils.spans import Span
from .base import BaseEngine, _Run


class KVStreamEngine(BaseEngine):
    name = "kvstream"

    def _positions(self, run: _Run, idx: list[int]) -> list[int]:
        return list(idx)

    def _evict(self, run: _Run, spans: list[Span]) -> list[Span]:
        evicted = self.cache.evict_positions(run.seq_id, spans)
        if evicted:
            run.context = [i for i in run.context if not any(s <= i < e for s, e in evicted)]
            self.cache.check_invariants(run.seq_id)
        return evicted
