from __future__ import annotations

from dataclasses import dataclass

import torch

from .distributional_value import (
    adaptive_alpha,
    categorical_entropy,
    categorical_projection,
    categorical_quantile,
)


@dataclass(frozen=True)
class DIVLStats:
    entropy: torch.Tensor
    alpha: torch.Tensor
    quantile_value: torch.Tensor


def divl_projection_targets(q_target_data: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Project per-member replay-action Q targets into Z_m(s) targets."""
    assert q_target_data.ndim == 2  # [M, B]
    return categorical_projection(q_target_data, support)


def divl_quantile_values(
    probs: torch.Tensor,
    support: torch.Tensor,
    *,
    alpha_min: float,
    alpha_max: float,
    entropy_temperature: float = 1.0,
    alpha_mode: str = "linear",
    use_adaptive_quantile: bool = True,
    interpolate_quantile: bool = True,
) -> DIVLStats:
    """Compute adaptive replay-value quantile V_m(s) from Z_m(s)."""
    assert probs.ndim == 3  # [M, B, atoms]
    entropy = categorical_entropy(probs, normalized=True)
    if use_adaptive_quantile:
        alpha = adaptive_alpha(
            entropy,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            temperature=entropy_temperature,
            mode=alpha_mode,
        )
    else:
        alpha = torch.full_like(entropy, float(alpha_max))
    quantile = categorical_quantile(
        probs,
        support.to(probs.device),
        alpha,
        interpolate=interpolate_quantile,
    )
    return DIVLStats(entropy=entropy, alpha=alpha, quantile_value=quantile)
