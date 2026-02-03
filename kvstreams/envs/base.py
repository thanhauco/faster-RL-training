"""Multi-turn text environments.

An episode starts with a system prompt and a first user message. After every
assistant completion the environment returns the next user message or finishes
with a reward. ``oracle()`` returns the ideal completion for the current turn and
is used for teacher forcing (SFT warm-up and deterministic benchmarks).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvStart:
    system: str
    user: str


@dataclass
class EnvStep:
    user: Optional[str]
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


class Env(ABC):
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    @abstractmethod
    def reset(self) -> EnvStart: ...

    @abstractmethod
    def step(self, completion: str) -> EnvStep: ...

    @abstractmethod
    def oracle(self) -> str:
        """Ideal assistant completion for the current turn."""

    def summary_oracle(self) -> str:
        """Ideal answer to a summary request (used by the summary policy under teacher forcing)."""
        return ""
