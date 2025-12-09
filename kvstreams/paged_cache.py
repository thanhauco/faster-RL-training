"""A paged KV cache with block tables, in the style of vLLM's PagedAttention.

Physical storage is a pool of fixed-size blocks. Each sequence owns a *block
table*: an ordered list of physical block ids. Live token ``i`` of a sequence
lives in block ``table[i // block_size]`` at offset ``i % block_size``.

KV-streams compaction removes whole blocks from a block table and joins the
remaining references. The attention kernel walks the table densely, so the table
must never contain holes (a freed block id left in the table would be read as if
it held valid entries). Removing complete blocks keeps the invariant that every
block except the last one is full.

Every physical slot also stores the *logical position* of the token it holds.
Physical index and logical position diverge as soon as anything is evicted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from .config import CacheConfig, ModelConfig
from .utils.spans import Span, positions_to_spans


class OutOfBlocksError(RuntimeError):
    pass


class BlockAllocator:
    """LIFO free-list allocator over ``num_blocks`` physical blocks."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, -1, -1))
        self._used: set[int] = set()

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocksError(f"all {self.num_blocks} KV blocks are in use")
        block = self._free.pop()
        self._used.add(block)
        return block

    def free(self, block: int) -> None:
        if block not in self._used:
            raise ValueError(f"double free of block {block}")
        self._used.remove(block)
        self._free.append(block)

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return len(self._used)


@dataclass
class SlotMapping:
    """Physical (block, offset) for each token written in one forward call."""

    blocks: torch.Tensor
    offsets: torch.Tensor


@dataclass
class CacheStats:
    written_tokens: int = 0
    evicted_tokens: int = 0
    evicted_blocks: int = 0
    truncated_tokens: int = 0
    peak_blocks: int = 0
    eviction_calls: int = 0
    shrunk_tokens: int = 0  # requested for eviction but kept because the block was shared


class PagedKVCache:
    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 4096,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        shape = (n_layers, num_blocks, block_size, n_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v = torch.zeros(shape, dtype=dtype, device=self.device)
        self.slot_pos = torch.full(
            (num_blocks, block_size), -1, dtype=torch.long, device=self.device
        )
        self.allocator = BlockAllocator(num_blocks)
        self._tables: dict[int, list[int]] = {}
        self._lengths: dict[int, int] = {}
        self.stats = CacheStats()

    @classmethod
    def from_config(
        cls, model_cfg: ModelConfig, cache_cfg: CacheConfig, device=None, dtype=torch.float32
    ):
        return cls(
            n_layers=model_cfg.n_layers,
            n_kv_heads=model_cfg.n_kv_heads,
            head_dim=model_cfg.head_dim,
            block_size=cache_cfg.block_size,
            num_blocks=cache_cfg.num_blocks,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------ sequences
    def add_sequence(self, seq_id: int) -> None:
        if seq_id in self._tables:
            raise ValueError(f"sequence {seq_id} already exists")
        self._tables[seq_id] = []
        self._lengths[seq_id] = 0

    def free_sequence(self, seq_id: int) -> None:
        for block in self._tables.pop(seq_id, []):
            self.slot_pos[block] = -1
            self.allocator.free(block)
        self._lengths.pop(seq_id, None)

    def has_sequence(self, seq_id: int) -> bool:
        return seq_id in self._tables

    def num_tokens(self, seq_id: int) -> int:
        return self._lengths[seq_id]

    def block_table(self, seq_id: int) -> list[int]:
        return list(self._tables[seq_id])

    # ------------------------------------------------------------------ writes
    def append_slots(self, seq_id: int, positions: torch.Tensor) -> SlotMapping:
        """Reserve slots for ``len(positions)`` new tokens, allocating blocks on demand."""
        table = self._tables[seq_id]
        n = self._lengths[seq_id]
        T = int(positions.numel())
        blocks, offsets = [], []
        for i in range(T):
            b, o = divmod(n + i, self.block_size)
            if b == len(table):
                table.append(self.allocator.allocate())
            blocks.append(table[b])
            offsets.append(o)
        mapping = SlotMapping(
            torch.tensor(blocks, dtype=torch.long, device=self.device),
            torch.tensor(offsets, dtype=torch.long, device=self.device),
        )
        self.slot_pos[mapping.blocks, mapping.offsets] = positions.to(self.device, torch.long)
        self._lengths[seq_id] = n + T
        self.stats.written_tokens += T
        self.stats.peak_blocks = max(self.stats.peak_blocks, self.allocator.num_used)
        return mapping

    def write(self, layer: int, slots: SlotMapping, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store already-rotated keys and values (``[T, n_kv_heads, head_dim]``)."""
        self.k[layer, slots.blocks, slots.offsets] = k.to(self.k.dtype)
        self.v[layer, slots.blocks, slots.offsets] = v.to(self.v.dtype)

    # ------------------------------------------------------------------ reads
    def _table_tensor(self, seq_id: int) -> torch.Tensor:
        return torch.tensor(self._tables[seq_id], dtype=torch.long, device=self.device)

    def gather(self, seq_id: int, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the live keys/values of a sequence in block-table order."""
        n = self._lengths[seq_id]
        table = self._table_tensor(seq_id)
        k = self.k[layer, table].reshape(-1, self.n_kv_heads, self.head_dim)[:n]
        v = self.v[layer, table].reshape(-1, self.n_kv_heads, self.head_dim)[:n]
        return k, v

    def positions(self, seq_id: int) -> torch.Tensor:
        """Logical positions of the live entries, in physical (table) order."""
        n = self._lengths[seq_id]
        return self.slot_pos[self._table_tensor(seq_id)].reshape(-1)[:n]

    # ------------------------------------------------------------------ eviction
    def evict_blocks(self, seq_id: int, table_indices: Iterable[int]) -> int:
        """Remove whole blocks (by index into the block table) and join the rest.

        Only full blocks can be removed; the trailing partial block is still being
        written to. Returns the number of tokens removed.
        """
        idxs = sorted(set(int(i) for i in table_indices))
        if not idxs:
            return 0
        table = self._tables[seq_id]
        n = self._lengths[seq_id]
        n_full = n // self.block_size
        for i in idxs:
            if not 0 <= i < n_full:
                raise ValueError(
                    f"block index {i} is not a full block (sequence has {n_full} full blocks)"
                )
        drop = set(idxs)
        for i in idxs:
            self.slot_pos[table[i]] = -1
            self.allocator.free(table[i])
        self._tables[seq_id] = [b for j, b in enumerate(table) if j not in drop]
        removed = len(idxs) * self.block_size
        self._lengths[seq_id] = n - removed
        self.stats.evicted_blocks += len(idxs)
        self.stats.evicted_tokens += removed
        self.stats.eviction_calls += 1
        return removed

    def evict_positions(self, seq_id: int, spans: Iterable[Span]) -> list[Span]:
        """Evict every full block whose tokens all fall inside ``spans``.

        ``spans`` are logical-position ranges. Blocks that are only partially
        covered are kept (the request is "shrunk" to block granularity). Returns
        the logical spans that were actually removed.
        """
        spans = list(spans)
        n = self._lengths[seq_id]
        if not spans or n == 0:
            return []
        pos = self.positions(seq_id)
        covered = torch.zeros(n, dtype=torch.bool, device=self.device)
        for s, e in spans:
            covered |= (pos >= s) & (pos < e)
        bs = self.block_size
        n_full = n // bs
        drop = [j for j in range(n_full) if bool(covered[j * bs : (j + 1) * bs].all())]
        removed_pos = [int(p) for j in drop for p in pos[j * bs : (j + 1) * bs].tolist()]
        self.stats.shrunk_tokens += int(covered.sum()) - len(removed_pos)
        self.evict_blocks(seq_id, drop)
        return positions_to_spans(removed_pos)

    def truncate(self, seq_id: int, n_tokens: int) -> None:
        """Drop the last ``n_tokens`` live entries (used to un-commit a partial block)."""
        if n_tokens <= 0:
            return
        n = self._lengths[seq_id]
        if n_tokens > n:
            raise ValueError("cannot truncate more tokens than are live")
        new_len = n - n_tokens
        table = self._tables[seq_id]
        keep_blocks = -(-new_len // self.block_size)
        for i in range(new_len, n):
            b, o = divmod(i, self.block_size)
            self.slot_pos[table[b], o] = -1
        for block in table[keep_blocks:]:
            self.allocator.free(block)
        del table[keep_blocks:]
        self._lengths[seq_id] = new_len
        self.stats.truncated_tokens += n_tokens

    # ------------------------------------------------------------------ checks
    def check_invariants(self, seq_id: int) -> None:
        table = self._tables[seq_id]
        n = self._lengths[seq_id]
        if len(set(table)) != len(table):
            raise AssertionError("block table references a block twice")
        if len(table) != -(-n // self.block_size):
            raise AssertionError("block table length does not match token count")
        pos = self.positions(seq_id)
        if bool((pos < 0).any()):
            raise AssertionError("live slot without a logical position (hole in block table)")
        if pos.numel() > 1 and not bool((pos[1:] > pos[:-1]).all()):
            raise AssertionError("logical positions are not strictly increasing")
