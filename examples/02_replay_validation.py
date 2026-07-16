"""Generate compacted rollouts and check that the trainer reproduces them.

Compares three ways to compute training log-probs for the same KV-streams
rollouts: the KV-streams replay mask, a naive full-context pass, and rebuilding
the compacted windows (what a re-prefill trainer would do).
"""

import torch

from kvstreams.config import CacheConfig, ModelConfig, RolloutConfig, SamplingConfig
from kvstreams.envs import build_env
from kvstreams.factory import build_stack
from kvstreams.training import measure_mismatch

stack = build_stack(
    engine="kvstream",
    policy="keep_recent",
    policy_kwargs={"keep_last": 2},
    model_cfg=ModelConfig(init_std=0.1),
    cache_cfg=CacheConfig(block_size=16, pad_to_block=True),
    rollout_cfg=RolloutConfig(context_budget=160, record_full_logprobs=True),
    sampling=SamplingConfig(max_new_tokens=12),
    dtype="float64",
)
gen = torch.Generator().manual_seed(0)
rollouts = [stack.engine.rollout(build_env("recall", seed=i), gen) for i in range(4)]
r = rollouts[0]
print(f"stream length {r.length}, compactions {r.stats.compactions}, live peak {r.stats.peak_live_tokens}")
for ev in r.log:
    print(f"  t={ev.time:4d} evicted {ev.evicted} ({ev.live_before} -> {ev.live_after} live)")

for mode in ("kvstream", "full", "windows"):
    rep = measure_mismatch(stack.model, rollouts, mode)
    print(f"{mode:9s} exact KL {rep.exact_kl:+.2e}   max |dlogp| {rep.max_abs_logprob_diff:.2e}")
