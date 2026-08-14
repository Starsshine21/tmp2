from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PPOStats:
    loss: torch.Tensor
    ratio_mean: float
    ratio_std: float
    ratio_min: float
    ratio_max: float
    clip_fraction: float


def full_chain_ais_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: torch.Tensor | float,
    log_ratio_clip: float = 20.0,
) -> PPOStats:
    """OGPO AIS objective with one joint likelihood ratio per flow chain."""
    assert new_log_probs.shape == old_log_probs.shape
    assert new_log_probs.ndim == 2
    assert advantages.shape == new_log_probs.shape[:1]
    log_ratio = (new_log_probs - old_log_probs.detach()).sum(dim=1)
    ratio = torch.exp(log_ratio.clamp(-log_ratio_clip, log_ratio_clip))
    eps = torch.as_tensor(clip_eps, dtype=ratio.dtype, device=ratio.device)
    if eps.ndim > 1 or (eps.ndim == 1 and eps.shape != ratio.shape):
        raise ValueError("chain clip must be scalar or have one value per trajectory")
    eps = eps.expand_as(ratio)
    clipped_ratio = torch.maximum(torch.minimum(ratio, 1.0 + eps), 1.0 - eps)
    advantage = advantages.detach()
    objective = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    return PPOStats(
        loss=-objective.mean(),
        ratio_mean=float(ratio.detach().mean().item()),
        ratio_std=float(ratio.detach().std(unbiased=False).item()),
        ratio_min=float(ratio.detach().min().item()),
        ratio_max=float(ratio.detach().max().item()),
        clip_fraction=float(ratio.ne(clipped_ratio).float().mean().item()),
    )


def full_chain_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: torch.Tensor | float,
    timestep_weights: torch.Tensor | None = None,
    log_ratio_clip: float = 20.0,
) -> PPOStats:
    """Full-chain OGPO objective.

    new_log_probs/old_log_probs: [N, K]
    advantages: [N], broadcast over all stochastic transitions.
    """
    assert new_log_probs.shape == old_log_probs.shape
    assert advantages.shape == new_log_probs.shape[:1]
    log_ratio = (new_log_probs - old_log_probs.detach()).clamp(-log_ratio_clip, log_ratio_clip)
    ratio = torch.exp(log_ratio)
    adv = advantages.detach().unsqueeze(-1)
    unclipped = ratio * adv
    eps = torch.as_tensor(clip_eps, dtype=ratio.dtype, device=ratio.device)
    if eps.ndim == 1:
        eps = eps.unsqueeze(-1)
    eps = eps.expand_as(ratio)
    clipped_ratio = torch.maximum(torch.minimum(ratio, 1.0 + eps), 1.0 - eps)
    clipped = clipped_ratio * adv
    objective = torch.minimum(unclipped, clipped)
    if timestep_weights is not None:
        assert timestep_weights.shape == objective.shape
        objective = objective * timestep_weights.detach()
    loss = -objective.mean()
    clip_fraction = (ratio.ne(clipped_ratio)).float().mean()
    return PPOStats(
        loss=loss,
        ratio_mean=float(ratio.detach().mean().item()),
        ratio_std=float(ratio.detach().std(unbiased=False).item()),
        ratio_min=float(ratio.detach().min().item()),
        ratio_max=float(ratio.detach().max().item()),
        clip_fraction=float(clip_fraction.item()),
    )
