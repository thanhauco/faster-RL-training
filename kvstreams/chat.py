"""Chat template used by the engines.

Layout of a rollout stream (``P`` = ``<pad>``)::

    <bos><|system|>...<|end|>PPPP            <- prompt segment (protected)
    <|user|>...<|end|><|assistant|>...<|end|>PP  <- turn 1
    <|user|>...<|end|><|assistant|>...<|end|>P   <- turn 2
    ...

With ``pad_to_block`` every segment occupies whole KV blocks, so turn-level
eviction maps onto whole-block eviction. Each turn carries its own boundary tags,
so removing a complete turn never leaves a dangling ``<|user|>`` or ``<|end|>``.
"""

from __future__ import annotations

from .tokenizer import CharTokenizer


class ChatTemplate:
    def __init__(self, tokenizer: CharTokenizer) -> None:
        self.tok = tokenizer

    def system(self, text: str) -> list[int]:
        t = self.tok
        return [t.bos_id, t.role_id("system"), *t.encode(text), t.end_id]

    def user_turn(self, text: str) -> list[int]:
        """User message followed by the assistant header the model completes."""
        t = self.tok
        return [t.role_id("user"), *t.encode(text), t.end_id, t.role_id("assistant")]

    def assistant_end(self) -> list[int]:
        return [self.tok.end_id]

    def padding(self, stream_len: int, block_size: int) -> list[int]:
        n = (-stream_len) % block_size
        return [self.tok.pad_id] * n
