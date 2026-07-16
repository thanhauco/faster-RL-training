"""Physical slot vs logical position after an in-place compaction.

Fills a paged cache with 48 tokens in 16-token blocks, evicts the middle block,
and prints the block table and the logical position held by each physical slot.
"""

import torch

from kvstreams.paged_cache import PagedKVCache

cache = PagedKVCache(n_layers=1, n_kv_heads=1, head_dim=2, block_size=16, num_blocks=8)
cache.add_sequence(0)
cache.append_slots(0, torch.arange(48))
print("before: block table", cache.block_table(0))
print("        positions   ", cache.positions(0).tolist())

evicted = cache.evict_positions(0, [(16, 32)])
print(f"\nevicted logical spans {evicted}")
print("after:  block table", cache.block_table(0))
pos = cache.positions(0).tolist()
print("        positions   ", pos)
print(f"\nphysical slot 16 now holds logical position {pos[16]}.")
print("Its key was rotated for position 32 and keeps that rotation; the next token")
print(f"gets logical position 48 even though only {cache.num_tokens(0)} entries are live.")
