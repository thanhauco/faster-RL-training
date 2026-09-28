# Changelog

## v0.2.0 — 2026-09-28

Tooling, benchmarks and documentation on top of the core library.

### Added
- `kvstreams` CLI with `validate`, `bench` and `train` commands.
- `benchmark.py`: KV-streams vs re-prefill on identical teacher-forced episodes
  (tokens forwarded, compactions, trainer tokens, rollout time).
- `factory.build_stack` for one-call model + tokenizer + engine construction.
- Checkpoint save/load, seeding and JSONL metric logging utilities.
- Four runnable examples: logical vs physical positions, replay validation,
  engine comparison, SFT + GRPO on the running-sum task.
- Docs: design overview, RoPE positions, block tables and padding, training
  replay, measured results.
- GitHub Actions CI (ruff, pytest, replay validation) on Python 3.10–3.12.

### Results (toy model, CPU)
- KV-streams replay matches inference log-probs to ~3e-15 (float64).
- KV-streams forwards each stream token once; re-prefill forwards 3.4–7.4× as
  many with the keep-recent policy, for 1.9–3.4× faster rollouts.

## v0.1.0 — 2026-05-19

First release of the core mechanism.

### Added
- RoPE driven by explicit logical positions.
- `TinyLM`: Llama-style decoder (RMSNorm, GQA, SwiGLU) with a masked
  full-sequence forward and a paged incremental forward.
- `PagedKVCache`: block allocator, per-sequence block tables, whole-block
  eviction that joins the remaining references, logical position per slot.
- Turn-level compaction policies (keep-recent, Markovian, summary) plus a
  token-level sliding window for comparison; compaction event log.
- `KVStreamEngine` (in-place compaction, completion padding to the block size,
  vLLM partial-block recompute emulation) and `RePrefillEngine` baseline.
- Trainer replay: KV-streams visibility mask, full-context and window layouts.
- Train/inference mismatch metrics (k3 and exact KL).
- GRPO loss with truncated importance sampling, SFT warm-up and RL trainer.
- Toy environments: assignment recall and running sum.
