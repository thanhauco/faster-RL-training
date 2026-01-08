"""The compaction log: everything the trainer needs to replay generation.

Each event records *when* compaction happened (the stream length ``time``) and
*which* stream spans were removed from the live KV cache. Every token with stream
index ``>= time`` was forwarded after the event and therefore never saw the
evicted spans. Every token with index ``< time`` formed its KV state while those
spans were still visible. That is all the information a single masked forward
pass needs to reproduce generation exactly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import torch

from ..utils.spans import Span, span_length

NEVER = torch.iinfo(torch.long).max


@dataclass
class CompactionEvent:
    time: int
    evicted: list[Span]
    requested: list[Span] = field(default_factory=list)
    live_before: int = 0
    live_after: int = 0
    policy: str = ""
    segments: list[int] = field(default_factory=list)

    @property
    def num_evicted(self) -> int:
        return span_length(self.evicted)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evicted"] = [list(s) for s in self.evicted]
        d["requested"] = [list(s) for s in self.requested]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CompactionEvent":
        d = dict(d)
        d["evicted"] = [tuple(s) for s in d["evicted"]]
        d["requested"] = [tuple(s) for s in d.get("requested", [])]
        return cls(**d)


class CompactionLog:
    def __init__(self, events: list[CompactionEvent] | None = None) -> None:
        self.events: list[CompactionEvent] = list(events or [])

    def append(self, event: CompactionEvent) -> None:
        if self.events and event.time < self.events[-1].time:
            raise ValueError("compaction events must be appended in time order")
        self.events.append(event)

    def __iter__(self):
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def evicted_at(self, length: int, device=None) -> torch.Tensor:
        """``[length]`` tensor: stream time at which each token was evicted (NEVER if kept)."""
        out = torch.full((length,), NEVER, dtype=torch.long, device=device)
        for ev in self.events:
            for s, e in ev.evicted:
                e = min(e, length)
                if s < e:
                    out[s:e] = torch.clamp(out[s:e], max=ev.time)
        return out

    def total_evicted(self) -> int:
        return sum(ev.num_evicted for ev in self.events)

    def to_json(self) -> str:
        return json.dumps([ev.to_dict() for ev in self.events])

    @classmethod
    def from_json(cls, text: str) -> "CompactionLog":
        return cls([CompactionEvent.from_dict(d) for d in json.loads(text)])
