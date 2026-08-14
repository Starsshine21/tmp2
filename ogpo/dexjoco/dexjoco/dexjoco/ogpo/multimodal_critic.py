from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .value_critic_protocol import StateFeatures


class MaskedTemporalActionPool(nn.Module):
    """Encode only the executed action prefix into one fixed-width token."""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        max_horizon: int,
        num_attention_heads: int,
    ):
        super().__init__()
        if max_horizon <= 0:
            raise ValueError("max_horizon must be positive")
        if hidden_dim % num_attention_heads:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_horizon = int(max_horizon)
        self.action_projection = nn.Linear(self.action_dim, self.hidden_dim)
        self.temporal_positions = nn.Parameter(torch.zeros(self.max_horizon, self.hidden_dim))
        self.query = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.attention = nn.MultiheadAttention(
            self.hidden_dim,
            int(num_attention_heads),
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))
        nn.init.normal_(self.temporal_positions, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(
        self,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        if action_chunks.ndim != 3:
            raise ValueError("action_chunks must have shape [batch, horizon, action_dim]")
        batch, horizon, action_dim = action_chunks.shape
        if action_dim != self.action_dim:
            raise ValueError(f"expected action_dim={self.action_dim}, got {action_dim}")
        if execution_masks.shape != (batch, horizon):
            raise ValueError("execution_masks must match action chunk batch and horizon")
        if horizon > self.max_horizon:
            raise ValueError(f"horizon={horizon} exceeds max_horizon={self.max_horizon}")
        mask = execution_masks.bool()
        if bool((~mask.any(dim=1)).any()):
            raise ValueError("each sample must contain at least one executed action")

        normalized = (action_chunks - self.action_mean) / self.action_std.clamp_min(1e-6)
        tokens = self.action_projection(normalized)
        tokens = tokens + self.temporal_positions[:horizon].unsqueeze(0)
        query = self.query.expand(batch, -1, -1)
        pooled, _ = self.attention(
            query,
            tokens,
            tokens,
            key_padding_mask=~mask,
            need_weights=False,
        )
        return self.output_norm(pooled[:, 0])


def _prediction_head(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class MultiHeadUdivlCore(nn.Module):
    """Shared action representation with corresponding scalar-Q and Value heads."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        max_horizon: int,
        action_hidden_dim: int,
        head_hidden_dim: int,
        num_attention_heads: int,
        num_value_atoms: int,
        num_pairs: int = 3,
    ):
        super().__init__()
        if num_pairs <= 1:
            raise ValueError("num_pairs must be greater than one")
        self.state_dim = int(state_dim)
        self.num_pairs = int(num_pairs)
        self.num_value_atoms = int(num_value_atoms)
        self.action_pool = MaskedTemporalActionPool(
            action_dim,
            action_hidden_dim,
            max_horizon,
            num_attention_heads,
        )
        q_input_dim = self.state_dim + int(action_hidden_dim)
        self.q_heads = nn.ModuleList(
            [_prediction_head(q_input_dim, head_hidden_dim, 1) for _ in range(self.num_pairs)]
        )
        self.value_heads = nn.ModuleList(
            [
                _prediction_head(self.state_dim, head_hidden_dim, self.num_value_atoms)
                for _ in range(self.num_pairs)
            ]
        )

    @property
    def ensemble_size(self) -> int:
        return self.num_pairs

    def q_from_readout(
        self,
        readout: torch.Tensor,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        if readout.ndim != 2 or readout.shape[-1] != self.state_dim:
            raise ValueError(f"readout must have shape [batch, {self.state_dim}]")
        head_parameter = next(self.q_heads[0].parameters())
        readout = readout.to(device=head_parameter.device, dtype=head_parameter.dtype)
        action_features = self.action_pool(action_chunks, execution_masks)
        action_features = action_features.to(device=head_parameter.device, dtype=head_parameter.dtype)
        if readout.shape[0] != action_features.shape[0]:
            raise ValueError("state and action batch sizes must match")
        fused = torch.cat([readout, action_features], dim=-1)
        return torch.stack([head(fused).squeeze(-1) for head in self.q_heads], dim=0)

    def value_logits_from_readout(self, readout: torch.Tensor) -> torch.Tensor:
        if readout.ndim != 2 or readout.shape[-1] != self.state_dim:
            raise ValueError(f"readout must have shape [batch, {self.state_dim}]")
        head_parameter = next(self.value_heads[0].parameters())
        readout = readout.to(device=head_parameter.device, dtype=head_parameter.dtype)
        return torch.stack([head(readout) for head in self.value_heads], dim=0)


class MultiHeadUdivlCritic(nn.Module):
    """Compose one state encoder with the shared multi-head U-DIVL core."""

    def __init__(self, state_encoder: nn.Module, core: MultiHeadUdivlCore):
        super().__init__()
        self.state_encoder = state_encoder
        self.core = core

    @property
    def ensemble_size(self) -> int:
        return self.core.ensemble_size

    def encode_state(self, batch: Any, *, next_observation: bool = False) -> StateFeatures:
        readout = self.state_encoder(batch, next_observation=next_observation)
        return StateFeatures(readout=readout)

    def q_from_features(
        self,
        features: StateFeatures,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.core.q_from_readout(features.readout, action_chunks, execution_masks)

    def value_logits_from_features(self, features: StateFeatures) -> torch.Tensor:
        return self.core.value_logits_from_readout(features.readout)

    def forward(
        self,
        batch: Any,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
        *,
        next_observation: bool = False,
    ) -> torch.Tensor:
        features = self.encode_state(batch, next_observation=next_observation)
        return self.q_from_features(features, action_chunks, execution_masks)


class MultiHeadScalarQCore(nn.Module):
    """Shared action representation with independent scalar Q heads only."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        max_horizon: int,
        action_hidden_dim: int,
        head_hidden_dim: int,
        num_attention_heads: int,
        num_heads: int,
    ):
        super().__init__()
        if num_heads <= 1:
            raise ValueError("num_heads must be greater than one")
        self.state_dim = int(state_dim)
        self.num_heads = int(num_heads)
        self.action_pool = MaskedTemporalActionPool(
            action_dim,
            action_hidden_dim,
            max_horizon,
            num_attention_heads,
        )
        q_input_dim = self.state_dim + int(action_hidden_dim)
        self.q_heads = nn.ModuleList(
            [_prediction_head(q_input_dim, head_hidden_dim, 1) for _ in range(self.num_heads)]
        )

    @property
    def ensemble_size(self) -> int:
        return self.num_heads

    def q_from_readout(
        self,
        readout: torch.Tensor,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        if readout.ndim != 2 or readout.shape[-1] != self.state_dim:
            raise ValueError(f"readout must have shape [batch, {self.state_dim}]")
        head_parameter = next(self.q_heads[0].parameters())
        readout = readout.to(device=head_parameter.device, dtype=head_parameter.dtype)
        action_features = self.action_pool(action_chunks, execution_masks)
        action_features = action_features.to(device=head_parameter.device, dtype=head_parameter.dtype)
        if readout.shape[0] != action_features.shape[0]:
            raise ValueError("state and action batch sizes must match")
        fused = torch.cat([readout, action_features], dim=-1)
        return torch.stack([head(fused).squeeze(-1) for head in self.q_heads], dim=0)


class MultiHeadScalarQCritic(nn.Module):
    """One multimodal state encoder with an original-OGPO scalar Q ensemble."""

    def __init__(self, state_encoder: nn.Module, core: MultiHeadScalarQCore):
        super().__init__()
        self.state_encoder = state_encoder
        self.core = core

    @property
    def ensemble_size(self) -> int:
        return self.core.ensemble_size

    def encode_state(self, batch: Any, *, next_observation: bool = False) -> StateFeatures:
        cached = (
            batch.next_critic_features
            if next_observation
            else batch.critic_features
        )
        if cached is not None:
            return StateFeatures(readout=cached)
        readout = self.state_encoder(batch, next_observation=next_observation)
        return StateFeatures(readout=readout)

    def q_from_features(
        self,
        features: StateFeatures,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.core.q_from_readout(features.readout, action_chunks, execution_masks)

    def forward(
        self,
        batch: Any,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
        *,
        next_observation: bool = False,
    ) -> torch.Tensor:
        features = self.encode_state(batch, next_observation=next_observation)
        return self.q_from_features(features, action_chunks, execution_masks)
