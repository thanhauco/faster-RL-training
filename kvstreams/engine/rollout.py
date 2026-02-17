"""Rollout records produced by the engines and consumed by the trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from ..compaction.events import CompactionLog
from ..compaction.segments import Segment


@dataclass
class RolloutStats:
    tokens_forwarded: int = 0  # every token pushed through the model, for any reason
    prefill_tokens: int = 0  # prompt / observation / padding chunks
    decode_tokens: int = 0  # sampled tokens fed back one at a time
    reprefill_tokens: int = 0  # retained tokens re-processed after a compaction
    recomputed_tokens: int = 0  # uncommitted partial-block tokens recomputed next turn
    stale_recomputed_tokens: int = 0  # ... recomputed after a compaction changed their context
    padding_tokens: int = 0
    compactions: int = 0
    evicted_tokens: int = 0
    peak_live_tokens: int = 0
    budget_overflows: int = 0
    wall_time: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Rollout:
    """One episode.

    ``tokens`` is the full *stream*: every token ever appended, in order, including
    tokens that were later compacted away. Stream index == logical (RoPE) position
    for KV-streams rollouts. ``loss_mask[t]`` is True for tokens produced by the
    policy (sampled or teacher-forced) and ``inf_logprobs[t]`` is the log-probability
    the inference engine assigned to ``tokens[t]`` when it was produced.
    """

    engine: str
    tokens: list[int]
    loss_mask: list[bool]
    inf_logprobs: list[float]
    segments: list[Segment]
    log: CompactionLog
    stats: RolloutStats
    reward: float = 0.0
    completions: list[str] = field(default_factory=list)
    block_size: int = 16
    temperature: float = 1.0
    full_logprobs: dict[int, torch.Tensor] = field(default_factory=dict)
    env_seed: Optional[int] = None

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def num_loss_tokens(self) -> int:
        return sum(self.loss_mask)

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "tokens": self.tokens,
            "loss_mask": self.loss_mask,
            "inf_logprobs": self.inf_logprobs,
            "segments": [s.to_dict() for s in self.segments],
            "events": [e.to_dict() for e in self.log],
            "stats": self.stats.to_dict(),
            "reward": self.reward,
            "completions": self.completions,
            "block_size": self.block_size,
            "temperature": self.temperature,
            "env_seed": self.env_seed,
        }
