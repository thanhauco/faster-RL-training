"""Supervised warm-up on teacher-forced rollouts.

Teacher-forced rollouts go through the same engine, so they carry real
compaction logs, and the loss is computed with the same KV-streams replay mask
used by RL. The model therefore learns the chat format *under* compaction.
"""

from __future__ import annotations

from typing import Callable

import torch

from ..envs.base import Env
from .logprobs import token_logprobs
from .replay import build_sequences, collate


def sft_loss(model, batch) -> torch.Tensor:
    lp = token_logprobs(model, batch)
    mask = batch.loss_mask.to(lp.dtype)
    return -(lp * mask).sum() / mask.sum().clamp(min=1)


class SFTTrainer:
    def __init__(
        self,
        model,
        engine,
        env_factory: Callable[[int], Env],
        lr: float = 3e-3,
        batch_size: int = 8,
        seed: int = 0,
        replay: str = "auto",
    ) -> None:
        self.model = model
        self.engine = engine
        self.env_factory = env_factory
        self.batch_size = batch_size
        self.seed = seed
        self.replay = replay
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    def step(self, step: int) -> float:
        base = 10_000_000 + self.seed * 100_003 + step * self.batch_size
        rollouts = [
            self.engine.rollout(self.env_factory(base + i), teacher_forcing=True)
            for i in range(self.batch_size)
        ]
        seqs = [s for r in rollouts for s in build_sequences(r, self.replay)]
        batch = collate(seqs, self.engine.tok.pad_id).to(self.model.device)
        self.model.train()
        loss = sft_loss(self.model, batch)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    def train(self, steps: int, log_every: int = 10, log_fn=print) -> list[float]:
        losses = []
        for step in range(steps):
            losses.append(self.step(step))
            if log_fn and (step % log_every == 0 or step == steps - 1):
                log_fn(f"[sft] step {step:4d}  loss {losses[-1]:.4f}")
        return losses
