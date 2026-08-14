from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class StateFeatures:
    """State representation shared by all Q and categorical Value heads."""

    readout: torch.Tensor


class ValueCritic(Protocol):
    @property
    def ensemble_size(self) -> int: ...

    def encode_state(self, batch: Any, *, next_observation: bool = False) -> StateFeatures: ...

    def q_from_features(
        self,
        features: StateFeatures,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor: ...

    def value_logits_from_features(self, features: StateFeatures) -> torch.Tensor: ...

