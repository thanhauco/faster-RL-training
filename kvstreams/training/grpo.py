"""GRPO-style policy-gradient loss with truncated importance sampling.

Advantages are rewards normalised within each group of rollouts that share the
same task. The surrogate is PPO-clipped against the trainer's own "old"
log-probs, and a truncated importance weight ``min(pi_old / pi_inf, cap)``
corrects for whatever train/inference mismatch remains. With KV-streams replay
that weight stays ~1.
"""

from __future__ import annotations

import torch


def group_advantages(
    rewards: torch.Tensor, group_size: int, normalize_std: bool = True, eps: float = 1e-6
):
    if rewards.numel() % group_size:
        raise ValueError("number of rewards must be a multiple of group_size")
    r = rewards.view(-1, group_size).double()
    adv = r - r.mean(dim=1, keepdim=True)
    if normalize_std:
        adv = adv / (r.std(dim=1, keepdim=True, unbiased=False) + eps)
    return adv.view(-1)


def policy_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    inf_logp: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    clip_eps: float = 0.2,
    tis_cap: float = 2.0,
):
    """Token-mean clipped surrogate.

    All tensors are ``[B, T]``; ``advantages`` is broadcast per token.
    """
    mask = loss_mask.to(logp.dtype)
    adv = advantages.to(logp.dtype)
    ratio = torch.exp(logp - old_logp.detach())
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
    tis = torch.exp(old_logp.detach() - inf_logp.to(logp.dtype)).clamp(max=tis_cap)
    per_token = -torch.minimum(surr1, surr2) * tis
    denom = mask.sum().clamp(min=1)
    loss = (per_token * mask).sum() / denom
    with torch.no_grad():
        clipped = ((ratio - 1).abs() > clip_eps).to(logp.dtype)
        metrics = {
            "clip_frac": float((clipped * mask).sum() / denom),
            "tis_mean": float((tis * mask).sum() / denom),
        }
    return loss, metrics
