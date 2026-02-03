from .base import Env, EnvStart, EnvStep
from .chain_sum import ChainSumEnv
from .recall import AssignmentRecallEnv

ENVS = {"recall": AssignmentRecallEnv, "chainsum": ChainSumEnv}


def build_env(name: str, seed: int = 0, **kwargs) -> Env:
    try:
        cls = ENVS[name]
    except KeyError as exc:
        raise ValueError(f"unknown env {name!r}; choose from {sorted(ENVS)}") from exc
    return cls(seed=seed, **kwargs)


__all__ = ["AssignmentRecallEnv", "ChainSumEnv", "ENVS", "Env", "EnvStart", "EnvStep", "build_env"]
