"""A tiny character-level tokenizer with chat special tokens.

Character-level keeps the project dependency free and makes it easy to reason
about block alignment (one character == one token).
"""

from __future__ import annotations

SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]

# Tokens that delimit chat messages. Evicting these mid-turn is what the
# KV-streams authors suspect broke token-level eviction.
BOUNDARY_TOKENS = {"<bos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"}


class CharTokenizer:
    def __init__(self) -> None:
        chars = [chr(i) for i in range(32, 127)] + ["\n"]
        self.itos: list[str] = list(SPECIAL_TOKENS) + chars
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(self.itos)}
        self.special_ids = {self.stoi[s] for s in SPECIAL_TOKENS}
        self.boundary_ids = {self.stoi[s] for s in BOUNDARY_TOKENS}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def end_id(self) -> int:
        return self.stoi["<|end|>"]

    def role_id(self, role: str) -> int:
        return self.stoi[f"<|{role}|>"]

    def encode(self, text: str) -> list[int]:
        unk = self.unk_id
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids, skip_special: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if skip_special and i in self.special_ids:
                continue
            out.append(self.itos[i])
        return "".join(out)

    def is_boundary(self, token_id: int) -> bool:
        return int(token_id) in self.boundary_ids
