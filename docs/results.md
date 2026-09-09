# Results on the toy setup

All numbers come from this repository on an Apple M5 CPU, PyTorch 2.x. They
measure the mechanism, not agent quality. The SWE-bench Verified results
(peak at ~32 h with KV-streams versus ~60 h with re-prefill, at comparable final
scores) are from the original KV-streams work and are **not** reproduced here.

## Replay exactness — `kvstreams validate`

Random-init model (`init_std=0.1`, 87k params, float64), assignment-recall env,
`keep_recent(keep_last=2)`, budget 160, 8 sampled rollouts.

| case | compactions | KL (k3) | exact KL | max \|Δ log p\| |
|---|---:|---:|---:|---:|
| KV-streams replay (vLLM-style, bs=16, padded) | 3.0 | 0.00e+00 | -4.84e-18 | 3.55e-15 |
| KV-streams replay (SGLang-style, bs=1) | 3.0 | 0.00e+00 | -1.22e-19 | 2.66e-15 |
| KV-streams replay, no padding + partial-block recompute | 2.8 | 7.94e-06 | 7.43e-06 | 3.31e-02 |
| naive trainer: full rollout, no compaction | 3.0 | 1.66e-02 | 1.57e-02 | 1.19e+00 |
| naive trainer: rebuild compacted windows | 3.0 | 1.85e-02 | 1.63e-02 | 8.55e-01 |
| re-prefill engine + window trainer | 3.0 | 0.00e+00 | -1.47e-18 | 2.66e-15 |

Negative exact-KL values are floating-point rounding around zero.

## Rollout cost — `kvstreams bench`

Teacher-forced episodes (identical streams for both engines), assignment recall
with 24 distractor turns of 64 characters, 4 rollouts, model `d=256, L=4`.

`keep_recent(keep_last=2)` evicts only as much as it needs, so it compacts
almost every turn:

| budget | engine | stream tok | forwarded tok | fwd ratio | compactions | trainer tok | rollout s | speedup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | reprefill | 2032 | 6816 | 3.35 | 23.0 | 6816 | 0.219 | 1.00x |
| 256 | kvstream | 2032 | 2032 | 1.00 | 23.0 | 2032 | 0.117 | 1.87x |
| 512 | reprefill | 2032 | 9024 | 4.44 | 19.0 | 9024 | 0.265 | 1.00x |
| 512 | kvstream | 2032 | 2032 | 1.00 | 19.0 | 2032 | 0.131 | 2.02x |
| 1024 | reprefill | 2032 | 15024 | 7.39 | 14.0 | 15024 | 0.531 | 1.00x |
| 1024 | kvstream | 2032 | 2032 | 1.00 | 14.0 | 2032 | 0.156 | 3.39x |

`markovian(keep_last=1)` resets aggressively, so it compacts less often and
re-prefills less:

| budget | engine | stream tok | forwarded tok | fwd ratio | compactions | trainer tok | rollout s | speedup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | reprefill | 2032 | 4848 | 2.39 | 22.0 | 4848 | 0.177 | 1.00x |
| 256 | kvstream | 2032 | 2032 | 1.00 | 22.0 | 2032 | 0.128 | 1.38x |
| 512 | reprefill | 2032 | 2672 | 1.31 | 5.0 | 2672 | 0.138 | 1.00x |
| 512 | kvstream | 2032 | 2032 | 1.00 | 5.0 | 2032 | 0.124 | 1.11x |
| 1024 | reprefill | 2032 | 2288 | 1.13 | 2.0 | 2288 | 0.144 | 1.00x |
| 1024 | kvstream | 2032 | 2032 | 1.00 | 2.0 | 2032 | 0.139 | 1.03x |

Takeaways:

* KV-streams forwards each stream token exactly once (`fwd ratio = 1.00`), no
  matter how often the rollout is compacted.
* The saving grows with how much context is retained and how often compaction
  fires. Frequent, light compaction (keep-recent) benefits most.
* The trainer gets the same saving: one masked pass over `T` tokens instead of
  one pass per window.
* Wall-clock numbers from a CPU toy model are only indicative: at real model
  sizes prefill dominates far more. At production inference, batching
  hides part of the prefill cost, and keeping caches alive for inactive users
  reduces serving capacity. The win is mainly for RL training, where one rollout
  is compacted many times.
