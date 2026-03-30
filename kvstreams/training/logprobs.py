"""Trainer-side log-probabilities."""

from __future__ import annotations

import torch

from .replay import Batch


def token_logprobs(model, batch: Batch, temperature: float = 1.0, return_full: bool = False):
    """Log-probability of ``tokens[:, t]`` given the replayed context of ``t - 1``.

    Returns ``[B, T]`` (position 0 is 0). With ``return_full`` also returns the
    ``[B, T, V]`` log-softmax rows aligned the same way (row ``t`` is the
    distribution that produced token ``t``).
    """
    logits = model(batch.tokens, batch.positions, batch.mask)
    logp = torch.log_softmax(logits / temperature, dim=-1)
    out = torch.zeros(batch.tokens.shape, dtype=logp.dtype, device=logp.device)
    out[:, 1:] = logp[:, :-1].gather(-1, batch.tokens[:, 1:, None]).squeeze(-1)
    if not return_full:
        return out
    full = torch.zeros_like(logp)
    full[:, 1:] = logp[:, :-1]
    return out, full
