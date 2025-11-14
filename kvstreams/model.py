"""A small Llama-style decoder with two forward paths.

* :meth:`TinyLM.forward` – full-sequence forward with an arbitrary boolean
  attention mask and explicit position ids. The trainer uses it to replay a
  KV-streams rollout in a single pass.
* :meth:`TinyLM.forward_cached` – incremental forward through a
  :class:`~kvstreams.paged_cache.PagedKVCache`. Engines use it for prefill and
  decode. Keys are rotated with their logical position *before* they are written
  to the cache and are never re-rotated.

Both paths share the same attention math, so a replay with the right mask and
positions reproduces inference log-probabilities up to floating point rounding.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .rope import RotaryEmbedding, apply_rope

KVHook = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        kv_hook: Optional[KVHook] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv_hook is not None:
            # Write the freshly rotated keys to the cache and read back every live entry.
            k, v = kv_hook(k, v)
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * (self.head_dim**-0.5)
        scores = scores.masked_fill(~mask[:, None, :, :], float("-inf"))
        probs = scores.softmax(dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", probs, v).reshape(B, T, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, mask, kv_hook: Optional[KVHook] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, mask, kv_hook)
        return x + self.mlp(self.mlp_norm(x))


def causal_mask(T: int, device=None) -> torch.Tensor:
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    @property
    def device(self) -> torch.device:
        return self.embed.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embed.weight.dtype

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        tokens: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-sequence forward.

        Args:
            tokens: ``[B, T]`` token ids.
            positions: ``[B, T]`` logical positions (defaults to ``arange(T)``).
            mask: ``[B, T, T]`` boolean, ``mask[b, q, k]`` is True when query ``q`` may
                attend to key ``k``. Defaults to causal.
        Returns:
            ``[B, T, vocab]`` logits.
        """
        B, T = tokens.shape
        if positions is None:
            positions = torch.arange(T, device=tokens.device).expand(B, T)
        if mask is None:
            mask = causal_mask(T, tokens.device).expand(B, T, T)
        x = self.embed(tokens)
        cos, sin = self.rope(positions)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def forward_cached(self, tokens: torch.Tensor, positions: torch.Tensor, cache, seq_id: int):
        """Incremental forward of one sequence through a paged KV cache.

        New tokens attend to every live cache entry plus causally to each other.
        Their keys/values are rotated with ``positions`` and appended to the cache.

        Args:
            tokens: ``[T]`` token ids.
            positions: ``[T]`` logical positions.
        Returns:
            ``[T, vocab]`` logits.
        """
        T = tokens.numel()
        n_past = cache.num_tokens(seq_id)
        slots = cache.append_slots(seq_id, positions)
        mask = torch.ones(1, T, n_past + T, dtype=torch.bool, device=tokens.device)
        mask[0, :, n_past:] = causal_mask(T, tokens.device)

        x = self.embed(tokens[None])
        cos, sin = self.rope(positions[None])
        for i, layer in enumerate(self.layers):

            def hook(k, v, layer_idx=i):
                cache.write(layer_idx, slots, k[0], v[0])
                k_all, v_all = cache.gather(seq_id, layer_idx)
                return k_all[None], v_all[None]

            x = layer(x, cos, sin, mask, kv_hook=hook)
        return self.lm_head(self.norm(x))[0]
