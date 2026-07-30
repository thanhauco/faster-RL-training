# Design

This repository is a small, dependency-light reimplementation of the **KV-streams**
idea for agentic RL: compact the context *in place* on the live KV cache instead
of rebuilding a shorter sequence and re-prefilling it, then make the trainer
replay the exact same history.

```
            ┌──────────────┐  plan (spans)   ┌───────────────────────┐
 segments ─►│ Compaction   ├────────────────►│ Engine                │
            │ policy       │                 │  KVStreamEngine       │──► Rollout
            └──────────────┘                 │  RePrefillEngine      │     tokens, loss_mask,
                                             │      │                │     inf_logprobs,
                                             │      ▼                │     segments,
                                             │  PagedKVCache         │     CompactionLog
                                             │  (block tables)       │
                                             └───────────────────────┘
                                                        │
                                                        ▼
                                  replay.kvstream_sequence(rollout)
                                  positions = arange(T)
                                  mask[t, s] = s <= t  and  t < evicted_at[s]
                                                        │
                                                        ▼
                                  TinyLM.forward(tokens, positions, mask)  ──► GRPO / SFT
```

## Modules

| module | role |
|---|---|
| `kvstreams/rope.py` | RoPE that takes explicit logical positions |
| `kvstreams/model.py` | Llama-style decoder; full masked forward + paged incremental forward |
| `kvstreams/paged_cache.py` | block allocator, per-sequence block tables, whole-block eviction |
| `kvstreams/chat.py` | chat template; every message carries its boundary tags |
| `kvstreams/compaction/` | segments, policies (keep-recent, Markovian, summary, token window), compaction log |
| `kvstreams/engine/` | rollout loop; `KVStreamEngine` (in place) and `RePrefillEngine` (baseline) |
| `kvstreams/training/` | replay masks, trainer log-probs, mismatch KL, GRPO loss, SFT and RL loops |
| `kvstreams/envs/` | toy multi-turn tasks: assignment recall and running sum |
| `kvstreams/benchmark.py` | forwarded-token and wall-clock comparison of the two engines |

## The rollout stream

A rollout is a single append-only *stream* of tokens. Every token ever produced
or observed gets the next stream index, and under KV-streams that index is also
its RoPE position. Compaction never rewrites the stream; it only removes KV
entries from the cache and appends a `CompactionEvent(time, evicted_spans)` to
the log. `time` is the stream length when compaction happened.

That gives a simple invariant the trainer relies on:

> token `t` attended to token `s` during generation **iff** `s <= t` and `s` had
> not been evicted before `t` was forwarded, i.e. `t < evicted_at[s]`.

## Turn-level segments

The engine tracks the stream as segments: a protected prompt segment, then one
segment per turn (`<|user|> … <|end|><|assistant|> … <|end|>` + padding).
Policies return stream spans, usually whole segments, so a compaction never
leaves a dangling role tag behind. Token-level eviction is available
(`TokenSlidingWindow`) to show the contrast, but the original authors reported
that it produced degenerate text without substantial SFT.

## What the engines do differently

| | `RePrefillEngine` | `KVStreamEngine` |
|---|---|---|
| on compaction | free the cache, renumber retained tokens from 0, prefill them again | `evict_positions` on the live cache, join the block table |
| RoPE position of new tokens | index in the rebuilt context | stream index (never reset) |
| tokens forwarded per rollout | `T + Σ retained_k` | `T` |
| trainer layout | one sequence per compaction window (retained tokens duplicated) | one sequence, one masked pass |

Both engines share the same policy, segments and log, so under teacher forcing
they make identical compaction decisions and produce identical streams.
