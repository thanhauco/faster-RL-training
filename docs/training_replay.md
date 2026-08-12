# Training on KV-streams rollouts

Once inference compacts in place, the trainer has to follow the same process.
The two usual options both compute different KV states from the ones used
during generation:

1. **Full rollout, causal mask.** Later tokens can see turns that inference had
   already removed.
2. **Rebuild the compacted sequence** (what a re-prefill trainer does). Retained
   tokens are recomputed without the evicted context and with renumbered
   positions, but during KV-streams generation they were computed *with* that
   context and *at* their original positions.

## Replay mask

`kvstreams.training.replay.kvstream_sequence` builds one sequence for the whole
stream:

```python
positions = arange(T)                       # logical positions
evicted_at = log.evicted_at(T)              # NEVER for tokens that were kept
mask[t, s] = (s <= t) & (t < evicted_at[s])
```

* Retained tokens keep the states they formed before compaction: rows `t < τ`
  can still see spans evicted at `τ`.
* Later tokens are blocked from attending to spans that inference had already
  removed: rows `t ≥ τ` cannot.

Because every layer's state for token `t` only depends on the states of the
tokens it attended to, applying this mask in every layer reproduces generation
by induction. Each token is processed once, so the trainer cost is `T` tokens
instead of `T + Σ retained_k`.

## Measuring the gap

`measure_mismatch(model, rollouts, mode)` compares the trainer's log-probs with
the ones recorded at sampling time and reports `KL(π_inference || π_trainer)` with
the k3 estimator, the exact full-vocabulary KL (when the engine recorded
distributions) and the max absolute log-prob difference.

| replay of KV-streams rollouts | exact KL | max \|Δ log p\| |
|---|---:|---:|
| KV-streams mask | ~1e-18 | 3.6e-15 |
| full rollout | 1.6e-02 | 1.19 |
| rebuilt windows | 1.6e-02 | 0.86 |

(float64 toy model, `kvstreams validate`). In float32 the RL loop reports a
mismatch KL around 1e-13. The original authors reported a mismatch KL near 1e-3
in their bf16 vLLM validation runs, where kernels differ between inference and
training.

## Inside the RL step

`RLTrainer.step`:

1. collects `tasks_per_step × group_size` rollouts with the engine;
2. computes group-normalized advantages (GRPO);
3. replays each rollout (`replay="auto"` picks the KV-streams mask for
   `KVStreamEngine` and windows for `RePrefillEngine`);
4. computes "old" log-probs and logs the train/inference mismatch;
5. optimizes a PPO-clipped surrogate weighted by a truncated importance ratio
   `min(π_old / π_inf, cap)`, which stays ≈ 1 with exact replay.

`SFTTrainer` uses the same replay on teacher-forced rollouts, so the warm-up
already trains the model under compaction.

## What survives eviction

The retained KV entries were created while the removed turns were still
visible. Future tokens cannot attend to those turns directly, but some of their
influence persists in the entries that were kept. The assignment-recall
environment tests this directly: the assignment turn is evicted before the
question arrives. The original authors found RL could recover the assignment
and reach 100% recall when enough cache remained. The tiny model here doesn't
learn it reliably in a CPU-scale run, but the environment and loop are there to
try it at larger scale.
