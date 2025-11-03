"""Rotary position embeddings driven by explicit *logical* positions.

KV-streams separates where a key lives in memory (its physical slot in the paged
cache) from the position it was rotated with (its logical position). After a
compaction a key can move from physical slot 1000 to slot 500, but it was rotated
for position 1000 and must keep being treated as position 1000. New queries and
keys continue counting from the original position counter.

Because RoPE is relative (``<R(m) q, R(n) k>`` depends only on ``m - n``), keeping
the original rotation is exactly what preserves the attention pattern the
retained entries had before compaction. Callers therefore always pass positions
explicitly instead of deriving them from ``arange(seq_len)``.
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
        inv_freq = 1.0 / (theta**exponent)
        self.register_buffer("inv_freq", inv_freq.float(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` with shape ``positions.shape + (head_dim,)``."""
        freqs = positions.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``[..., T, H, D]`` with ``cos``/``sin`` of shape ``[..., T, D]``."""
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    return x * cos + rotate_half(x) * sin
