"""SFT warm-up followed by GRPO on the running-sum task with Markovian compaction."""

from kvstreams.config import CacheConfig, ModelConfig, RolloutConfig, SamplingConfig
from kvstreams.envs import build_env
from kvstreams.factory import build_stack
from kvstreams.training import RLConfig, RLTrainer, SFTTrainer

stack = build_stack(
    engine="kvstream",
    policy="markovian",
    policy_kwargs={"keep_last": 1},
    model_cfg=ModelConfig(d_model=64, n_layers=2, d_ff=256),
    cache_cfg=CacheConfig(block_size=16),
    rollout_cfg=RolloutConfig(context_budget=80),
    sampling=SamplingConfig(max_new_tokens=6),
)


def env_factory(seed):
    return build_env("chainsum", seed=seed, num_steps=5)


SFTTrainer(stack.model, stack.engine, env_factory, lr=3e-3).train(300, log_every=50)
RLTrainer(stack.model, stack.engine, env_factory, RLConfig(steps=20, lr=5e-4, log_every=5)).train()
