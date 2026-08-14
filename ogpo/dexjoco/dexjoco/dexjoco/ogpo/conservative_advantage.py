from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AdvantageStats:
    positive_consensus_ratio: float
    negative_consensus_ratio: float
    zero_ratio: float
    sign_agreement_ratio: float
    mean_conservative_magnitude: float


def sign_consensus_advantage(
    q_values: torch.Tensor,
    value_baselines: torch.Tensor,
    *,
    positive_margin: float = 0.0,
    negative_margin: float = 0.0,
) -> tuple[torch.Tensor, AdvantageStats]:
    """Two-sided conservative advantage.

    q_values: [M, B, G] or [M, N]
    value_baselines: [M, B] for grouped candidates or [M, N]
    """
    assert q_values.ndim in (2, 3)
    if q_values.ndim == 3:
        assert value_baselines.ndim == 2
        raw = q_values - value_baselines.unsqueeze(-1)
    else:
        assert value_baselines.shape == q_values.shape
        raw = q_values - value_baselines

    min_adv = raw.min(dim=0).values
    max_adv = raw.max(dim=0).values
    pos = min_adv > float(positive_margin)
    neg = max_adv < -float(negative_margin)
    cons = torch.zeros_like(min_adv)
    cons = torch.where(pos, min_adv, cons)
    cons = torch.where(neg, max_adv, cons)
    total = cons.numel()
    stats = AdvantageStats(
        positive_consensus_ratio=float(pos.float().mean().item()),
        negative_consensus_ratio=float(neg.float().mean().item()),
        zero_ratio=float((~(pos | neg)).float().mean().item()),
        sign_agreement_ratio=float((pos | neg).float().mean().item()),
        mean_conservative_magnitude=float(cons.abs().sum().item() / max(1, int((pos | neg).sum().item()))),
    )
    assert total > 0
    return cons, stats


def lcb_advantage(
    q_values: torch.Tensor,
    value_baselines: torch.Tensor,
    *,
    kappa: float,
    calibrated_scale: float = 1.0,
) -> torch.Tensor:
    assert q_values.ndim == 3
    q_mean = q_values.mean(dim=0)
    q_std = q_values.std(dim=0, unbiased=False) * float(calibrated_scale)
    return q_mean - float(kappa) * q_std - value_baselines.mean(dim=0).unsqueeze(-1)


def group_normalized_advantage(q_values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """GRPO-style ablation only, not the default method."""
    q_mean = q_values.mean(dim=0)
    centered = q_mean - q_mean.mean(dim=-1, keepdim=True)
    scale = q_mean.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
    return centered / scale


def scheduled_lambda_abs(
    step: int,
    *,
    start: float = 1.0,
    end: float = 1.0,
    warmup_steps: int = 0,
) -> float:
    if not 0.0 <= float(start) <= 1.0 or not 0.0 <= float(end) <= 1.0:
        raise ValueError("absolute-advantage mixture weights must be in [0, 1]")
    if int(warmup_steps) <= 0:
        return float(end)
    progress = min(max(int(step), 0) / int(warmup_steps), 1.0)
    return float(start) + progress * (float(end) - float(start))


class RunningMAD:
    """EMA running robust scale for conservative advantages."""

    def __init__(self, momentum: float = 0.95, eps: float = 1e-6, initial: float = 1.0):
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.value = float(initial)

    def update(self, values: torch.Tensor, *, ignore_zero: bool = True) -> float:
        flat = values.detach().reshape(-1)
        if ignore_zero:
            flat = flat[flat != 0]
        if flat.numel() == 0:
            return self.value
        median = torch.median(flat)
        mad = 1.4826 * torch.median(torch.abs(flat - median))
        new_value = float(mad.clamp_min(self.eps).item())
        self.value = self.momentum * self.value + (1.0 - self.momentum) * new_value
        return self.value

    def normalize(self, values: torch.Tensor, clip: float) -> torch.Tensor:
        return (values / (self.value + self.eps)).clamp(-float(clip), float(clip))
