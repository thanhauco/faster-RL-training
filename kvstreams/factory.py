"""One-call construction of model + tokenizer + engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .compaction.policies import build_policy
from .config import CacheConfig, ModelConfig, RolloutConfig, SamplingConfig
from .engine import BaseEngine, build_engine
from .model import TinyLM
from .tokenizer import CharTokenizer

DTYPES = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16}


@dataclass
class Stack:
    model: TinyLM
    tokenizer: CharTokenizer
    engine: BaseEngine


def build_stack(
    engine: str = "kvstream",
    policy: str = "keep_recent",
    policy_kwargs: Optional[dict] = None,
    model_cfg: Optional[ModelConfig] = None,
    cache_cfg: Optional[CacheConfig] = None,
    rollout_cfg: Optional[RolloutConfig] = None,
    sampling: Optional[SamplingConfig] = None,
    model: Optional[TinyLM] = None,
    dtype: str = "float32",
    device: str = "cpu",
    seed: int = 0,
) -> Stack:
    tok = CharTokenizer()
    if model is None:
        torch.manual_seed(seed)
        model_cfg = model_cfg or ModelConfig()
        model_cfg.vocab_size = tok.vocab_size
        model = TinyLM(model_cfg)
    model = model.to(device=device, dtype=DTYPES[dtype])
    eng = build_engine(
        engine,
        model,
        tok,
        build_policy(policy, **(policy_kwargs or {})),
        cache_cfg or CacheConfig(),
        rollout_cfg or RolloutConfig(),
        sampling or SamplingConfig(),
    )
    return Stack(model, tok, eng)
