# Block tables, whole-block eviction and padding

## Why only whole blocks

vLLM stores KV in fixed 16-token blocks and gives the attention kernel a block
table per sequence. The kernel reads the table densely: live token `i` is at
`table[i // 16]`, offset `i % 16`. A freed block left inside the table, or a
half-empty block in the middle, would make the kernel read memory that does not
belong to this sequence.

`PagedKVCache.evict_blocks` therefore only accepts *full* blocks, removes them,
and joins the remaining references. The invariant "every block except the last
is full" holds before and after every compaction (`check_invariants`).

`evict_positions(spans)` maps a policy's stream spans onto blocks and evicts only
the blocks that are covered entirely. Partially covered blocks are kept, and the
log records what was *actually* evicted, so replay stays exact even when a policy
asks for something unaligned.

## The partial-block problem

A turn rarely ends exactly at a block boundary. In the vLLM setup described by
the authors, a trailing partial block is not committed to the prefix cache; its
tokens are recomputed when the next turn arrives. If a compaction happens in
between, the recomputed tokens attend to a *different* context than the one they
were sampled under. A single trainer pass can only give each token one hidden
state, so the mismatch cannot be replayed away.

`CacheConfig(commit_partial_blocks=False)` emulates this behaviour:
`_uncommit_tail` truncates the trailing partial block at the end of every turn
and the engine re-forwards those tokens next turn. `RolloutStats` counts
`recomputed_tokens` and `stale_recomputed_tokens` (recomputed after a compaction).

## The fix: pad completions to the block size

With `CacheConfig(pad_to_block=True)` the engine appends `<pad>` tokens after
every completion (and after the system prompt) until the stream length is a
multiple of `block_size`. Every generated token is committed before the next
turn or compaction, every turn occupies whole blocks, and turn-level eviction
is whole-block eviction. Padding tokens are model inputs but never loss tokens.

SGLang-style caches with `block_size=1` need no padding at all.

Measured with `kvstreams validate` (float64, 8 rollouts):

| configuration | max \|Δ log p\| |
|---|---:|
| bs=16, padded | 3.6e-15 |
| bs=1, no padding | 2.7e-15 |
| bs=16, no padding, partial-block recompute | 3.3e-02 |
