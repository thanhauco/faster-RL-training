"""Train/inference mismatch metrics.

If the trainer does not see the same KV states as generation, the policy being
optimised is not the policy that produced the samples. We measure the gap as
``KL(pi_inference || pi_trainer)`` on the sampled tokens.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch

from ..engine.rollout import Rollout
from .logprobs import token_logprobs
from .replay import build_sequences, collate


@dataclass
class MismatchReport:
    kl_k3: float  # unbiased, non-negative estimator E_inf[r - 1 - log r], r = p_train / p_inf
    kl_k1: float  # E_inf[log p_inf - log p_train]
    exact_kl: Optional[float]  # full-vocabulary KL when inference distributions were recorded
    max_abs_logprob_diff: float
    mean_abs_logprob_diff: float
    num_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def mismatch_report(
    inf_logp: torch.Tensor,
    train_logp: torch.Tensor,
    mask: torch.Tensor,
    inf_full: Optional[torch.Tensor] = None,
    train_full: Optional[torch.Tensor] = None,
) -> MismatchReport:
    inf_logp = inf_logp[mask].double()
    train_logp = train_logp[mask].double()
    n = int(inf_logp.numel())
    if n == 0:
        return MismatchReport(0.0, 0.0, None, 0.0, 0.0, 0)
    log_r = train_logp - inf_logp
    k3 = (log_r.exp() - 1 - log_r).mean()
    k1 = (-log_r).mean()
    exact = None
    if inf_full is not None and train_full is not None:
        p = inf_full.double().exp()
        exact = float((p * (inf_full.double() - train_full.double())).sum(-1).mean())
    diff = log_r.abs()
    return MismatchReport(float(k3), float(k1), exact, float(diff.max()), float(diff.mean()), n)


@torch.no_grad()
def measure_mismatch(
    model, rollouts: list[Rollout], mode: str = "auto", batch_size: int = 8
) -> MismatchReport:
    """Replay ``rollouts`` with the given mode and compare against inference log-probs."""
    was_training = model.training
    model.eval()
    dev = model.device
    infs, trains, inf_rows, train_rows = [], [], [], []
    want_full = all(r.full_logprobs for r in rollouts if r.num_loss_tokens)
    items = [(r, s) for r in rollouts for s in build_sequences(r, mode)]
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        batch = collate([s for _, s in chunk]).to(dev)
        temperature = chunk[0][0].temperature
        lp, full = token_logprobs(model, batch, temperature, return_full=True)
        for b, (r, s) in enumerate(chunk):
            m = batch.loss_mask[b, : s.length]
            infs.append(batch.inf_logprobs[b, : s.length][m].cpu())
            trains.append(lp[b, : s.length][m].double().cpu())
            if want_full:
                for j in torch.nonzero(m).flatten().tolist():
                    inf_rows.append(r.full_logprobs[int(s.stream_index[j])])
                    train_rows.append(full[b, j].cpu())
    model.train(was_training)
    inf = torch.cat(infs) if infs else torch.zeros(0)
    train = torch.cat(trains) if trains else torch.zeros(0)
    ones = torch.ones_like(inf, dtype=torch.bool)
    if want_full and inf_rows:
        return mismatch_report(inf, train, ones, torch.stack(inf_rows), torch.stack(train_rows))
    return mismatch_report(inf, train, ones)
