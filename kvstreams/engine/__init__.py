from .base import BaseEngine
from .kv_stream import KVStreamEngine
from .reprefill import RePrefillEngine
from .rollout import Rollout, RolloutStats

ENGINES = {"kvstream": KVStreamEngine, "reprefill": RePrefillEngine}


def build_engine(name: str, *args, **kwargs) -> BaseEngine:
    try:
        cls = ENGINES[name]
    except KeyError as exc:
        raise ValueError(f"unknown engine {name!r}; choose from {sorted(ENGINES)}") from exc
    return cls(*args, **kwargs)


__all__ = [
    "BaseEngine",
    "ENGINES",
    "KVStreamEngine",
    "RePrefillEngine",
    "Rollout",
    "RolloutStats",
    "build_engine",
]
