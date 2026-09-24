# faster-RL-training

**KV-streams in PyTorch: compact agent context in place on the live KV cache, and
train on exactly what generation saw.**

Agentic RL produces long rollouts. When a rollout hits its context budget, a
compaction policy drops older turns and keeps the prompt, recent turns or a
summary. Most systems do this with **re-prefill**: build the shorter sequence
and run every retained token through the model again. A token that survives `k`
compactions gets processed `k + 1` times.

**KV-streams** removes the discarded turns directly from the live paged KV cache
and continues generating from the surviving entries. The trainer then replays
the recorded compaction history in a single masked forward pass. The original
work reports that a Markovian-Thinker setup on SWE-bench Verified peaked in
about 32 hours with KV-streams versus about 60 hours with re-prefill, with
comparable final scores.

This repo is a compact, tested reimplementation of the mechanism on a small
Llama-style model, intended for study and as a reference for porting the idea
to vLLM or SGLang.

```
re-prefill   [P|t1|t2|t3]  → compact →  [P|t2|t3] re-run all retained ─► generate t4
KV-streams   [P|t1|t2|t3]  → drop t1's blocks from the block table    ─► generate t4
                                         (t2, t3 keep their KV and RoPE positions)
```

## What's implemented

| Problem | What this repo does | Where |
|---|---|---|
| RoPE already rotated each cached key with its original position | Physical slot and logical position are kept separate. Retained keys keep their rotation and new tokens continue the original position counter. | [`rope.py`](kvstreams/rope.py), [`engine/kv_stream.py`](kvstreams/engine/kv_stream.py) |
| vLLM stores KV in 16-token blocks behind a block table | Only whole blocks are evicted, and the remaining references are joined, so the table never has holes. | [`paged_cache.py`](kvstreams/paged_cache.py) |
| Turns end mid-block; the uncommitted tail is recomputed after compaction against different context | Completions are padded to a multiple of the block size, so every token is committed. Single-token blocks (SGLang-style) need no padding. The unpadded failure mode can be switched on to measure it. | [`config.py`](kvstreams/config.py), [`engine/base.py`](kvstreams/engine/base.py) |
| Token-level eviction drops boundary tags and degrades text | Default policies evict complete turns including their tags. A token sliding window is included for comparison. | [`compaction/policies.py`](kvstreams/compaction/policies.py) |
| The trainer must see the same KV states as generation | Every compaction event is logged. Replay uses `mask[t,s] = s≤t ∧ t<evicted_at[s]` with original positions, so it matches inference in one pass. | [`training/replay.py`](kvstreams/training/replay.py) |
| Proving it | Mismatch KL against inference log-probs (k3 estimator + exact KL), plus a re-prefill baseline engine | [`training/mismatch.py`](kvstreams/training/mismatch.py), [`engine/reprefill.py`](kvstreams/engine/reprefill.py) |

Also included: keep-recent, Markovian and summary compaction policies, GRPO with
truncated importance sampling, SFT warm-up under compaction, two toy multi-turn
environments (assignment recall, running sum), a benchmark and a CLI.

## Quick start

```bash
git clone https://github.com/thanhauco/faster-RL-training.git
cd faster-RL-training
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

```bash
kvstreams validate      # train/inference mismatch for six replay strategies
kvstreams bench         # KV-streams vs re-prefill: forwarded tokens and time
kvstreams train --env chainsum --policy markovian --keep-last 1 --turns 5 --budget 80
```

## Results (toy model, CPU)

Replay exactness, float64, 8 sampled rollouts with 3 compactions each:

| replay strategy for the same KV-streams rollouts | max \|Δ log p\| vs inference |
|---|---:|
| **KV-streams mask** (bs=16, padded) | **3.6e-15** |
| **KV-streams mask** (bs=1, SGLang-style) | **2.7e-15** |
| KV-streams mask, no padding, partial-block recompute | 3.3e-02 |
| full rollout, ignore compaction | 1.19 |
| rebuild compacted windows (re-prefill trainer) | 0.86 |

Rollout cost, keep-recent policy, 24-turn episodes (2032 stream tokens):

| budget | re-prefill tokens forwarded | KV-streams tokens forwarded | rollout speedup |
|---:|---:|---:|---:|
| 256 | 6816 (3.35×) | 2032 (1.00×) | 1.87× |
| 512 | 9024 (4.44×) | 2032 (1.00×) | 2.02× |
| 1024 | 15024 (7.39×) | 2032 (1.00×) | 3.39× |

Full tables and caveats: [`docs/results.md`](docs/results.md).

## Using the library

```python
import torch
from kvstreams.config import CacheConfig, RolloutConfig
from kvstreams.envs import build_env
from kvstreams.factory import build_stack
from kvstreams.training import build_sequences, collate, measure_mismatch, token_logprobs

stack = build_stack(
    engine="kvstream",                    # or "reprefill"
    policy="keep_recent", policy_kwargs={"keep_last": 2},
    cache_cfg=CacheConfig(block_size=16, pad_to_block=True),
    rollout_cfg=RolloutConfig(context_budget=160),
)
rollout = stack.engine.rollout(build_env("recall", seed=0), torch.Generator().manual_seed(0))
print(rollout.stats.compactions, [ev.evicted for ev in rollout.log])

batch = collate(build_sequences(rollout))          # one masked sequence per rollout
logp = token_logprobs(stack.model, batch)          # == rollout.inf_logprobs on policy tokens
print(measure_mismatch(stack.model, [rollout]).kl_k3)
```

## Repository layout

```
kvstreams/
  rope.py              RoPE with explicit logical positions
  model.py             TinyLM: masked full forward + paged incremental forward
  paged_cache.py       block allocator, block tables, whole-block eviction
  chat.py, tokenizer.py
  compaction/          segments, policies, compaction log
  engine/              base loop, KVStreamEngine, RePrefillEngine, Rollout
  training/            replay, logprobs, mismatch, grpo, sft, trainer
  envs/                assignment recall, running sum
  benchmark.py, cli.py, factory.py
docs/                  design, positions, blocks & padding, replay, results
examples/              runnable walkthroughs
tests/                 62 tests covering cache, RoPE, policies, engines, replay, training
```

## Scope and caveats

* KV-streams does not decide what to forget. It makes existing compaction
  policies cheaper to run and to train, while keeping the KV state they retain.
* Retained entries were computed while the evicted turns were still visible, so
  some influence of those turns persists. That can help (recall after eviction)
  and is part of what the policy learns to use.
* The speedup is mainly for RL training, where one rollout is compacted many
  times. At production inference, batching hides some prefill cost, and keeping
  caches for inactive users costs serving capacity.
* Positions grow with the total stream length, so the model must handle RoPE
  positions beyond the live context budget.
* This is a reference implementation on a toy model and a pure-PyTorch paged
  cache. It is not a vLLM or SGLang patch, and the SWE-bench numbers above are
  the original authors', not reproduced here.

## License

MIT
