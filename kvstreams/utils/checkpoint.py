"""Minimal checkpointing for the toy model."""

from __future__ import annotations

from pathlib import Path

import torch

from ..config import ModelConfig
from ..model import TinyLM


def save_checkpoint(path: str | Path, model: TinyLM, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> tuple[TinyLM, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = TinyLM(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    extra = {k: v for k, v in ckpt.items() if k not in ("config", "state_dict")}
    return model, extra
