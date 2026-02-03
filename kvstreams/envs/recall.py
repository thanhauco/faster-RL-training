"""Assignment recall: can the agent answer after the assignment turn was evicted?

Turn 1 assigns a task to a person, several distractor turns follow, and the final
turn asks who owns the task. With a small context budget the assignment turn is
compacted away long before the question arrives. Under KV-streams the retained
entries were formed while the assignment was still visible, so traces of it can
survive in them; the original authors found RL can learn to exploit this and
reach 100% recall when enough cache remains.
"""

from __future__ import annotations

import string

from .base import Env, EnvStart, EnvStep

NAMES = ["ana", "bob", "cai", "dev", "eli", "fay", "gus", "hal"]
TASKS = [f"t{i}" for i in range(10)]


class AssignmentRecallEnv(Env):
    def __init__(self, seed: int = 0, num_distractors: int = 4, filler_len: int = 24) -> None:
        super().__init__(seed)
        self.num_distractors = num_distractors
        self.filler_len = filler_len
        self.name = ""
        self.task = ""
        self.turn = 0

    def _filler(self) -> str:
        letters = string.ascii_lowercase + "   "
        return "".join(self.rng.choice(letters) for _ in range(self.filler_len)).strip() or "x"

    def reset(self) -> EnvStart:
        self.name = self.rng.choice(NAMES)
        self.task = self.rng.choice(TASKS)
        self.turn = 0
        return EnvStart(
            system="track task owners. reply ok to notes.",
            user=f"assign {self.task} to {self.name}",
        )

    @property
    def _final_turn(self) -> int:
        return self.num_distractors + 1

    def oracle(self) -> str:
        return self.name if self.turn == self._final_turn else "ok"

    def summary_oracle(self) -> str:
        return f"{self.task}={self.name}"

    def step(self, completion: str) -> EnvStep:
        if self.turn == self._final_turn:
            correct = completion.strip() == self.name
            return EnvStep(user=None, reward=float(correct), done=True, info={"answer": self.name})
        self.turn += 1
        if self.turn == self._final_turn:
            return EnvStep(user=f"who has {self.task}?")
        return EnvStep(user=f"note: {self._filler()}")
