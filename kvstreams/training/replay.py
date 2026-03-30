"""Turn a rollout plus its compaction log into training sequences.

KV-streams replay (:func:`kvstream_sequence`)
    One sequence covering the whole stream, logical positions ``0..T-1`` and the
    visibility mask::

        mask[t, s] = (s <= t) and (t < evicted_at[s])

    Token ``s`` evicted at stream time ``tau`` is visible to every token forwarded
    before ``tau`` and hidden from every token forwarded afterwards. Retained tokens
    therefore keep the states they formed *before* compaction (they could see the
    later-evicted spans), while later tokens are blocked from attending to spans
    inference had already removed. This reproduces generation exactly in a single
    forward pass with no duplicated tokens.

Naive alternatives, kept for comparison:

* :func:`full_context_sequence` – trains on the full rollout as if nothing had
  been compacted.
* :func:`window_sequences` – rebuilds the shorter sequence after every compaction
  (re-numbered positions, causal mask). This matches the re-prefill engine
  exactly but *not* KV-streams inference, and it duplicates retained tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..engine.rollout import Rollout


@dataclass
class TrainSequence:
    tokens: torch.Tensor  # [T] long
    positions: torch.Tensor  # [T] long
    mask: torch.Tensor  # [T, T] bool, mask[q, k]
    loss_mask: torch.Tensor  # [T] bool, token t is a policy token (predicted from t-1)
    inf_logprobs: torch.Tensor  # [T] float64
    stream_index: torch.Tensor  # [T] long, index into rollout.tokens

    @property
    def length(self) -> int:
        return int(self.tokens.numel())


def _tensors(rollout: Rollout):
    tokens = torch.tensor(rollout.tokens, dtype=torch.long)
    loss = torch.tensor(rollout.loss_mask, dtype=torch.bool)
    inf = torch.tensor(rollout.inf_logprobs, dtype=torch.float64)
    return tokens, loss, inf


def kvstream_sequence(rollout: Rollout) -> TrainSequence:
    tokens, loss, inf = _tensors(rollout)
    T = tokens.numel()
    idx = torch.arange(T)
    evicted_at = rollout.log.evicted_at(T)
    q = idx[:, None]
    k = idx[None, :]
    mask = (k <= q) & (q < evicted_at[None, :])
    return TrainSequence(tokens, idx.clone(), mask, loss, inf, idx.clone())


def full_context_sequence(rollout: Rollout) -> TrainSequence:
    tokens, loss, inf = _tensors(rollout)
    T = tokens.numel()
    idx = torch.arange(T)
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    return TrainSequence(tokens, idx.clone(), mask, loss, inf, idx.clone())


def window_sequences(rollout: Rollout) -> list[TrainSequence]:
    tokens, loss, inf = _tensors(rollout)
    T = tokens.numel()
    evicted_at = rollout.log.evicted_at(T)
    times = sorted({ev.time for ev in rollout.log if ev.evicted})
    bounds = [0, *times, T]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a >= b:
            continue
        ctx = [s for s in range(a) if int(evicted_at[s]) > a]
        idx = torch.tensor(ctx + list(range(a, b)), dtype=torch.long)
        n = idx.numel()
        window_loss = loss[idx].clone()
        window_loss[: len(ctx)] = False
        out.append(
            TrainSequence(
                tokens=tokens[idx],
                positions=torch.arange(n),
                mask=torch.ones(n, n, dtype=torch.bool).tril(),
                loss_mask=window_loss,
                inf_logprobs=inf[idx],
                stream_index=idx,
            )
        )
    return out


def build_sequences(rollout: Rollout, mode: str = "auto") -> list[TrainSequence]:
    """``mode``: ``auto`` (match the engine), ``kvstream``, ``windows`` or ``full``."""
    if mode == "auto":
        mode = "kvstream" if rollout.engine == "kvstream" else "windows"
    if mode == "kvstream":
        return [kvstream_sequence(rollout)]
    if mode == "windows":
        return window_sequences(rollout)
    if mode == "full":
        return [full_context_sequence(rollout)]
    raise ValueError(f"unknown replay mode {mode!r}")


@dataclass
class Batch:
    tokens: torch.Tensor  # [B, T]
    positions: torch.Tensor  # [B, T]
    mask: torch.Tensor  # [B, T, T]
    loss_mask: torch.Tensor  # [B, T]
    inf_logprobs: torch.Tensor  # [B, T]
    lengths: torch.Tensor  # [B]

    def to(self, device) -> "Batch":
        return Batch(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))

    @property
    def num_tokens(self) -> int:
        return int(self.lengths.sum())


def collate(seqs: list[TrainSequence], pad_id: int = 0) -> Batch:
    B = len(seqs)
    T = max(s.length for s in seqs)
    tokens = torch.full((B, T), pad_id, dtype=torch.long)
    positions = torch.zeros(B, T, dtype=torch.long)
    # Padding rows attend to themselves only so softmax stays finite.
    mask = torch.eye(T, dtype=torch.bool).repeat(B, 1, 1)
    loss_mask = torch.zeros(B, T, dtype=torch.bool)
    inf = torch.zeros(B, T, dtype=torch.float64)
    for b, s in enumerate(seqs):
        n = s.length
        tokens[b, :n] = s.tokens
        positions[b, :n] = s.positions
        mask[b, :n, :n] = s.mask
        loss_mask[b, :n] = s.loss_mask
        inf[b, :n] = s.inf_logprobs
    lengths = torch.tensor([s.length for s in seqs], dtype=torch.long)
    return Batch(tokens, positions, mask, loss_mask, inf, lengths)
