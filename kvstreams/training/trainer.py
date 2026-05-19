"""On-policy RL loop: rollouts with compaction, exact replay, GRPO update."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch

from ..envs.base import Env
from .grpo import group_advantages, policy_loss
from .logprobs import token_logprobs
from .mismatch import mismatch_report
from .replay import build_sequences, collate


@dataclass
class RLConfig:
    steps: int = 50
    tasks_per_step: int = 4
    group_size: int = 4
    lr: float = 1e-3
    ppo_epochs: int = 1
    minibatch_size: int = 8
    clip_eps: float = 0.2
    tis_cap: float = 2.0
    grad_clip: float = 1.0
    replay: str = "auto"
    seed: int = 0
    log_every: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


class RLTrainer:
    def __init__(
        self,
        model,
        engine,
        env_factory: Callable[[int], Env],
        cfg: Optional[RLConfig] = None,
        log_fn: Optional[Callable[[str], None]] = print,
    ) -> None:
        self.model = model
        self.engine = engine
        self.env_factory = env_factory
        self.cfg = cfg or RLConfig()
        self.log_fn = log_fn
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=0.0)
        self.gen = torch.Generator(device="cpu").manual_seed(self.cfg.seed)
        self.history: list[dict] = []

    def collect(self, step: int):
        c = self.cfg
        rollouts = []
        for task in range(c.tasks_per_step):
            env_seed = c.seed * 1_000_003 + step * c.tasks_per_step + task
            for _ in range(c.group_size):
                rollouts.append(self.engine.rollout(self.env_factory(env_seed), generator=self.gen))
        return rollouts

    def step(self, step: int) -> dict:
        c = self.cfg
        t0 = time.perf_counter()
        rollouts = self.collect(step)
        t_rollout = time.perf_counter() - t0

        rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float64)
        adv = group_advantages(rewards, c.group_size)
        items = [(s, float(a)) for r, a in zip(rollouts, adv) for s in build_sequences(r, c.replay)]

        t1 = time.perf_counter()
        dev = self.model.device
        pad_id = self.engine.tok.pad_id
        minibatches = []
        for i in range(0, len(items), c.minibatch_size):
            chunk = items[i : i + c.minibatch_size]
            batch = collate([s for s, _ in chunk], pad_id).to(dev)
            adv_tok = torch.tensor([a for _, a in chunk], device=dev)[:, None].expand_as(
                batch.loss_mask
            )
            minibatches.append((batch, adv_tok))

        # Old log-probs from the trainer, and train/inference mismatch on this batch.
        self.model.eval()
        olds, infs, trains = [], [], []
        with torch.no_grad():
            for batch, _ in minibatches:
                lp = token_logprobs(self.model, batch, self.engine.sampling.temperature)
                olds.append(lp)
                infs.append(batch.inf_logprobs[batch.loss_mask].cpu())
                trains.append(lp[batch.loss_mask].cpu())
        inf_all = torch.cat(infs)
        mm = mismatch_report(inf_all, torch.cat(trains), torch.ones_like(inf_all, dtype=torch.bool))

        self.model.train()
        losses, clip_fracs, tis = [], [], []
        for _ in range(c.ppo_epochs):
            for (batch, adv_tok), old in zip(minibatches, olds):
                lp = token_logprobs(self.model, batch, self.engine.sampling.temperature)
                loss, m = policy_loss(
                    lp, old, batch.inf_logprobs, adv_tok, batch.loss_mask, c.clip_eps, c.tis_cap
                )
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
                self.opt.step()
                losses.append(loss.item())
                clip_fracs.append(m["clip_frac"])
                tis.append(m["tis_mean"])
        t_train = time.perf_counter() - t1

        n = len(rollouts)
        metrics = {
            "step": step,
            "reward_mean": float(rewards.mean()),
            "loss": sum(losses) / max(len(losses), 1),
            "mismatch_kl": mm.kl_k3,
            "max_abs_logprob_diff": mm.max_abs_logprob_diff,
            "clip_frac": sum(clip_fracs) / max(len(clip_fracs), 1),
            "tis_mean": sum(tis) / max(len(tis), 1),
            "stream_tokens": sum(r.length for r in rollouts) / n,
            "tokens_forwarded": sum(r.stats.tokens_forwarded for r in rollouts) / n,
            "compactions": sum(r.stats.compactions for r in rollouts) / n,
            "trainer_tokens": sum(int(b.lengths.sum()) for b, _ in minibatches),
            "rollout_time": t_rollout,
            "train_time": t_train,
        }
        self.history.append(metrics)
        return metrics

    def train(self, steps: Optional[int] = None) -> list[dict]:
        steps = self.cfg.steps if steps is None else steps
        for step in range(steps):
            m = self.step(step)
            if self.log_fn and (step % self.cfg.log_every == 0 or step == steps - 1):
                self.log_fn(
                    f"[rl] step {step:4d}  reward {m['reward_mean']:.3f}  loss {m['loss']:+.4f}  "
                    f"mismatch_kl {m['mismatch_kl']:.2e}  compactions {m['compactions']:.1f}  "
                    f"fwd/rollout {m['tokens_forwarded']:.0f}  t_roll {m['rollout_time']:.1f}s"
                )
        return self.history
