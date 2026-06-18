"""KV-streams vs re-prefill: how much work does compaction cost?

Both engines run the *same* teacher-forced episodes (so the token streams and
compaction decisions are identical) and we count every token pushed through the
model, the wall-clock time, and the number of tokens the trainer has to process.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from .config import CacheConfig, ModelConfig, RolloutConfig, SamplingConfig
from .envs import build_env
from .factory import build_stack
from .training.replay import build_sequences


@dataclass
class BenchResult:
    engine: str
    budget: int
    rollouts: int
    stream_tokens: float
    tokens_forwarded: float
    reprefill_tokens: float
    compactions: float
    forward_ratio: float  # tokens_forwarded / stream_tokens (1.0 == every token processed once)
    trainer_tokens: float
    rollout_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_benchmark(
    budgets: list[int],
    num_rollouts: int = 4,
    env: str = "recall",
    env_kwargs: Optional[dict] = None,
    policy: str = "keep_recent",
    policy_kwargs: Optional[dict] = None,
    model_cfg: Optional[ModelConfig] = None,
    block_size: int = 16,
    seed: int = 0,
    device: str = "cpu",
) -> list[BenchResult]:
    env_kwargs = env_kwargs or {"num_distractors": 16, "filler_len": 48}
    model_cfg = model_cfg or ModelConfig(d_model=256, n_layers=4, n_heads=8, n_kv_heads=4, d_ff=768)
    results = []
    base = build_stack(model_cfg=model_cfg, seed=seed, device=device)
    for budget in budgets:
        for engine in ("reprefill", "kvstream"):
            stack = build_stack(
                engine=engine,
                policy=policy,
                policy_kwargs=policy_kwargs,
                model=base.model,
                cache_cfg=CacheConfig(block_size=block_size, num_blocks=8192),
                rollout_cfg=RolloutConfig(context_budget=budget, max_turns=128),
                sampling=SamplingConfig(max_new_tokens=16),
                device=device,
            )
            rollouts = [
                stack.engine.rollout(
                    build_env(env, seed=seed + i, **env_kwargs), teacher_forcing=True
                )
                for i in range(num_rollouts)
            ]
            n = len(rollouts)
            stream = sum(r.length for r in rollouts) / n
            fwd = sum(r.stats.tokens_forwarded for r in rollouts) / n
            trainer = sum(s.length for r in rollouts for s in build_sequences(r)) / n
            results.append(
                BenchResult(
                    engine=engine,
                    budget=budget,
                    rollouts=n,
                    stream_tokens=stream,
                    tokens_forwarded=fwd,
                    reprefill_tokens=sum(r.stats.reprefill_tokens for r in rollouts) / n,
                    compactions=sum(r.stats.compactions for r in rollouts) / n,
                    forward_ratio=fwd / stream,
                    trainer_tokens=trainer,
                    rollout_seconds=sum(r.stats.wall_time for r in rollouts) / n,
                )
            )
    return results


def format_table(results: list[BenchResult]) -> str:
    header = (
        "| budget | engine | stream tok | forwarded tok | fwd ratio | compactions "
        "| trainer tok | rollout s | speedup |\n"
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header]
    by_budget: dict[int, dict[str, BenchResult]] = {}
    for r in results:
        by_budget.setdefault(r.budget, {})[r.engine] = r
    for budget, d in by_budget.items():
        ref = d.get("reprefill")
        for name in ("reprefill", "kvstream"):
            r = d.get(name)
            if r is None:
                continue
            speed = (
                f"{ref.rollout_seconds / r.rollout_seconds:.2f}x"
                if ref and r.rollout_seconds
                else "-"
            )
            rows.append(
                f"| {budget} | {name} | {r.stream_tokens:.0f} | {r.tokens_forwarded:.0f} "
                f"| {r.forward_ratio:.2f} | {r.compactions:.1f} | {r.trainer_tokens:.0f} "
                f"| {r.rollout_seconds:.3f} | {speed} |"
            )
    return "\n".join(rows)
