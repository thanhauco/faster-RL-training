# Logical positions vs physical slots

RoPE rotates each key by its position *before* the key is written to the cache:

```
k_cached = R(pos_k) · W_k x
score(q at m, k at n) = <R(m) q, R(n) k> = <q, R(n - m) k>
```

The score depends only on `n - m`. After compaction a key may move from physical
slot 1000 to physical slot 500. Its rotation still says "position 1000", and
that is correct: the token *was* at position 1000 relative to everything that
was generated after it.

KV-streams therefore keeps two things separate:

* **physical index**: where the entry lives in the block table (changes on every compaction);
* **logical position**: what it was rotated with (never changes).

New queries and keys keep counting from the original position counter (the
stream length), not from the number of live entries.

`PagedKVCache.slot_pos` stores the logical position next to every physical slot,
so the separation is visible in tests and in
`examples/01_logical_vs_physical_positions.py`:

```
evicted logical spans [(16, 32)]
after:  block table [0, 2]
        positions   [0 … 15, 32 … 47]
physical slot 16 now holds logical position 32.
```

## What goes wrong if you renumber

`tests/test_rope.py::test_wrong_position_after_compaction_changes_scores` and
`tests/test_model.py::test_renumbering_positions_breaks_equivalence` show that
using physical indices as positions changes the attention scores. Re-prefill
avoids that inconsistency by re-rotating everything, which is exactly the
compute KV-streams saves.

## Range

Positions grow with the total stream length rather than the live context, so
long RL rollouts reach positions beyond the context budget. The model has to be
usable at those positions: pick a RoPE base / context extension that covers the
maximum *stream* length you expect, not just the budget.
