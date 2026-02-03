"""Running sum: a Markovian task whose whole state lives in the latest turn.

Each user turn gives a number and the assistant must reply with the running sum
modulo 100. Because the answer to turn ``k`` only depends on the answer to turn
``k-1`` and the new number, a Markovian policy that keeps only the last turn
loses nothing.
"""

from __future__ import annotations

from .base import Env, EnvStart, EnvStep


class ChainSumEnv(Env):
    def __init__(self, seed: int = 0, num_steps: int = 6, max_value: int = 9) -> None:
        super().__init__(seed)
        self.num_steps = num_steps
        self.max_value = max_value
        self.total = 0
        self.turn = 0
        self.correct = 0

    def reset(self) -> EnvStart:
        self.total = self.rng.randint(0, self.max_value)
        self.turn = 0
        self.correct = 0
        return EnvStart(system="keep a running sum mod 100.", user=f"start {self.total}")

    def oracle(self) -> str:
        return str(self.total)

    def summary_oracle(self) -> str:
        return f"sum {self.total}"

    def step(self, completion: str) -> EnvStep:
        self.correct += int(completion.strip() == str(self.total))
        self.turn += 1
        if self.turn >= self.num_steps:
            return EnvStep(user=None, reward=self.correct / self.num_steps, done=True)
        x = self.rng.randint(1, self.max_value)
        self.total = (self.total + x) % 100
        return EnvStep(user=f"add {x}")
