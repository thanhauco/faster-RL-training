"""Configuration dataclasses shared across the model, cache, engines and trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    """A small Llama-style decoder (RMSNorm, RoPE, GQA, SwiGLU)."""

    vocab_size: int = 104  # CharTokenizer: 8 specials + 95 printable ASCII + newline
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 128
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CacheConfig:
    """Paged KV cache layout.

    block_size:
        Tokens per physical block. 16 mirrors vLLM; 1 mirrors SGLang's token-level
        radix cache.
    num_blocks:
        Physical blocks in the pool (shared by all sequences).
    pad_to_block:
        Pad every completion (and the system prompt) with ``<pad>`` tokens up to a
        multiple of ``block_size`` so that every turn starts and ends on a block
        boundary and can be removed as whole blocks.
    commit_partial_blocks:
        If False, emulate vLLM's prefix cache: a trailing, partially filled block is
        *not* committed at the end of a turn and its tokens are recomputed when the
        next turn arrives. Combined with compaction this causes a train/inference
        mismatch, which ``pad_to_block=True`` avoids.
    """

    block_size: int = 16
    num_blocks: int = 4096
    pad_to_block: bool = True
    commit_partial_blocks: bool = True

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    max_new_tokens: int = 24
    greedy: bool = False


@dataclass
class RolloutConfig:
    """Rollout-level limits.

    context_budget:
        Soft limit on live KV entries. Before each new turn, if the live cache plus
        the incoming observation and the generation reserve would exceed this, the
        compaction policy is invoked.
    """

    context_budget: int = 256
    max_turns: int = 32
    record_full_logprobs: bool = False
