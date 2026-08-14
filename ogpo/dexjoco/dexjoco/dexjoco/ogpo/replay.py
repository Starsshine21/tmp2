from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import json
import torch
from torch.utils.data import Dataset

from .chunk_transition import compute_chunk_return, compute_transition_discount, make_execution_mask
from .types import ChunkBatch


class OfflineChunkReplay(Dataset):
    """Fixed offline replay dataset for chunk transitions."""

    def __init__(self, batch: ChunkBatch):
        self.batch = batch

    def __len__(self) -> int:
        return self.batch.batch_size

    def __getitem__(self, index: int) -> ChunkBatch:
        return self.batch.index_select(torch.tensor([index], dtype=torch.long))

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> ChunkBatch:
        indices = torch.randint(self.batch.batch_size, (batch_size,), generator=generator)
        sample = self.batch.index_select(indices)
        return sample.to(device) if device is not None else sample


class BalancedCriticReplay:
    """Episode-balanced replay with explicit success and reward-anchor strata."""

    def __init__(
        self,
        batch: ChunkBatch,
        *,
        uniform_fraction: float = 0.5,
        success_fraction: float = 0.25,
        terminal_success_fraction: float = 0.125,
        failure_fraction: float = 0.125,
    ):
        fractions = {
            "uniform": float(uniform_fraction),
            "success": float(success_fraction),
            "terminal_success": float(terminal_success_fraction),
            "failure": float(failure_fraction),
        }
        if any(value < 0.0 for value in fractions.values()):
            raise ValueError("balanced critic replay fractions must be non-negative")
        if abs(sum(fractions.values()) - 1.0) > 1e-6:
            raise ValueError("balanced critic replay fractions must sum to 1")
        self.batch = batch
        self.fractions = fractions
        self._all = self._group_indices(torch.ones(batch.batch_size, dtype=torch.bool))
        success = batch.successes.bool().cpu()
        self._success = self._group_indices(success)
        self._failure = self._group_indices(~success)
        terminal_success = success & batch.dones.bool().cpu()
        self._terminal_success = self._group_indices(terminal_success)

    def _group_indices(self, mask: torch.Tensor) -> list[torch.Tensor]:
        groups = []
        episode_ids = self.batch.episode_ids.cpu()
        for episode_id in torch.unique(episode_ids[mask]):
            indices = torch.nonzero(mask & (episode_ids == episode_id), as_tuple=False).flatten()
            if indices.numel():
                groups.append(indices)
        return groups

    @staticmethod
    def _counts(batch_size: int, fractions: dict[str, float]) -> dict[str, int]:
        raw = {name: batch_size * value for name, value in fractions.items()}
        counts = {name: int(value) for name, value in raw.items()}
        remaining = batch_size - sum(counts.values())
        order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
        for name in order[:remaining]:
            counts[name] += 1
        return counts

    def _sample_groups(
        self,
        groups: list[torch.Tensor],
        count: int,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if count <= 0:
            return torch.empty(0, dtype=torch.long)
        if not groups:
            groups = self._all
        selected_groups = torch.randint(len(groups), (count,), generator=generator)
        selected = []
        for group_index in selected_groups.tolist():
            group = groups[group_index]
            offset = torch.randint(group.numel(), (1,), generator=generator).item()
            selected.append(group[offset])
        return torch.stack(selected)

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> ChunkBatch:
        counts = self._counts(int(batch_size), self.fractions)
        indices = torch.cat(
            [
                self._sample_groups(self._all, counts["uniform"], generator=generator),
                self._sample_groups(self._success, counts["success"], generator=generator),
                self._sample_groups(
                    self._terminal_success,
                    counts["terminal_success"],
                    generator=generator,
                ),
                self._sample_groups(self._failure, counts["failure"], generator=generator),
            ]
        )
        order = torch.randperm(indices.numel(), generator=generator)
        sample = self.batch.index_select(indices[order])
        return sample.to(device) if device is not None else sample


def save_replay(batch: ChunkBatch, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(batch), path)


def save_replay_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    meta_path = Path(path).with_suffix(Path(path).suffix + ".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def load_replay(path: str | Path, *, map_location: str | torch.device = "cpu") -> ChunkBatch:
    payload: dict[str, Any] = torch.load(Path(path), map_location=map_location, weights_only=False)
    return ChunkBatch(**payload)


def split_success_buffers(batch: ChunkBatch) -> dict[str, ChunkBatch]:
    success_mask = batch.successes.bool()
    failure_mask = ~success_mask
    success_indices = torch.nonzero(success_mask, as_tuple=False).flatten()
    failure_indices = torch.nonzero(failure_mask, as_tuple=False).flatten()
    buffers = {"all": batch}
    if success_indices.numel() > 0:
        buffers["success"] = batch.index_select(success_indices)
    if failure_indices.numel() > 0:
        buffers["failure"] = batch.index_select(failure_indices)
    if failure_indices.numel() > 0:
        returns = batch.chunk_returns.index_select(0, failure_indices)
        if bool(returns.max() > returns.min()):
            cutoff = torch.quantile(returns, 0.75)
            near = failure_indices[returns >= cutoff]
            if near.numel() > 0:
                buffers["near_success"] = batch.index_select(near)
    return buffers


def split_replay(
    batch: ChunkBatch,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 0,
) -> dict[str, ChunkBatch]:
    """Create train/validation/held-out splits without episode leakage."""
    if train_ratio <= 0.0 or validation_ratio < 0.0 or train_ratio + validation_ratio >= 1.0:
        raise ValueError("expected train_ratio > 0, validation_ratio >= 0, and train+validation < 1")
    generator = torch.Generator().manual_seed(seed)
    episodes = torch.unique(batch.episode_ids)
    if episodes.numel() >= 3:
        episodes = episodes[torch.randperm(episodes.numel(), generator=generator)]
        train_episode_count = min(max(1, int(episodes.numel() * train_ratio)), episodes.numel() - 2)
        validation_episode_count = min(
            max(1, int(episodes.numel() * validation_ratio)),
            episodes.numel() - train_episode_count - 1,
        )
        train_episodes = episodes[:train_episode_count]
        validation_episodes = episodes[
            train_episode_count : train_episode_count + validation_episode_count
        ]
        heldout_episodes = episodes[train_episode_count + validation_episode_count :]

        def indices_for(selected: torch.Tensor) -> torch.Tensor:
            mask = (batch.episode_ids[:, None] == selected[None, :]).any(dim=1)
            return torch.nonzero(mask, as_tuple=False).flatten()

        return {
            "train": batch.index_select(indices_for(train_episodes)),
            "validation": batch.index_select(indices_for(validation_episodes)),
            "heldout": batch.index_select(indices_for(heldout_episodes)),
        }

    # A one- or two-episode smoke dataset cannot form three disjoint trajectory
    # splits, so retain deterministic transition-level splits for that case.
    perm = torch.randperm(batch.batch_size, generator=generator)
    train_end = max(1, int(batch.batch_size * train_ratio))
    val_end = max(train_end + 1, int(batch.batch_size * (train_ratio + validation_ratio)))
    val_end = min(val_end, batch.batch_size)
    splits = {"train": batch.index_select(perm[:train_end])}
    if val_end > train_end:
        splits["validation"] = batch.index_select(perm[train_end:val_end])
    if batch.batch_size > val_end:
        splits["heldout"] = batch.index_select(perm[val_end:])
    return splits


def make_n_step_replay(batch: ChunkBatch, *, n_step: int) -> ChunkBatch:
    """Fold consecutive chunk transitions into n-step outer-MDP targets."""
    n_step = int(n_step)
    if n_step <= 0:
        raise ValueError("n_step must be positive")
    if n_step == 1:
        return batch

    returns = []
    discounts = []
    dones = []
    successes = []
    next_indices = []
    metadata = []
    for start in range(batch.batch_size):
        total = batch.chunk_returns.new_tensor(0.0)
        cumulative_discount = batch.discounts.new_tensor(1.0)
        last = start
        used = 0
        for offset in range(n_step):
            index = start + offset
            if index >= batch.batch_size:
                break
            if offset > 0:
                previous = index - 1
                contiguous = (
                    batch.episode_ids[index] == batch.episode_ids[previous]
                    and batch.timesteps[index]
                    == batch.timesteps[previous] + batch.executed_lengths[previous]
                )
                if not bool(contiguous.item()):
                    break
            total = total + cumulative_discount * batch.chunk_returns[index]
            cumulative_discount = cumulative_discount * batch.discounts[index]
            last = index
            used += 1
            if bool(batch.dones[index].item()):
                break
        returns.append(total)
        discounts.append(cumulative_discount)
        dones.append(batch.dones[last])
        successes.append(batch.successes[start : last + 1].max())
        next_indices.append(last)
        item_metadata = dict(batch.behavior_metadata[start])
        item_metadata["n_step"] = used
        metadata.append(item_metadata)

    index_tensor = torch.tensor(next_indices, dtype=torch.long, device=batch.next_observations.device)
    nested = {}
    if batch.next_images is not None:
        nested["next_images"] = {
            key: value.index_select(0, index_tensor.to(value.device))
            for key, value in batch.next_images.items()
        }
    return replace(
        batch,
        chunk_returns=torch.stack(returns),
        discounts=torch.stack(discounts),
        next_observations=batch.next_observations.index_select(0, index_tensor),
        next_proprioceptions=batch.next_proprioceptions.index_select(
            0, index_tensor.to(batch.next_proprioceptions.device)
        ),
        dones=torch.stack(dones),
        successes=torch.stack(successes),
        behavior_metadata=metadata,
        **nested,
    )


def add_monte_carlo_returns(batch: ChunkBatch) -> ChunkBatch:
    """Compute discounted return-to-go over contiguous outer-MDP chunks."""
    mc_returns = torch.empty_like(batch.chunk_returns)
    for index in range(batch.batch_size - 1, -1, -1):
        terminal = bool(batch.dones[index].item()) or index == batch.batch_size - 1
        if not terminal:
            next_index = index + 1
            contiguous = (
                batch.episode_ids[next_index] == batch.episode_ids[index]
                and batch.timesteps[next_index]
                == batch.timesteps[index] + batch.executed_lengths[index]
            )
            terminal = not bool(contiguous.item())
        if terminal:
            mc_returns[index] = batch.chunk_returns[index]
        else:
            mc_returns[index] = batch.chunk_returns[index] + batch.discounts[index] * mc_returns[index + 1]
    return replace(batch, mc_returns=mc_returns)


def make_synthetic_replay(
    *,
    num_samples: int = 64,
    obs_dim: int = 12,
    proprio_dim: int = 4,
    generated_horizon: int = 6,
    action_dim: int = 3,
    executed_horizon: int = 3,
    gamma: float = 0.97,
    seed: int = 7,
) -> ChunkBatch:
    """Create deterministic toy replay for unit and smoke tests."""
    if executed_horizon <= 0 or executed_horizon > generated_horizon:
        raise ValueError("executed_horizon must be in [1, generated_horizon]")

    gen = torch.Generator().manual_seed(seed)
    observations = torch.randn(num_samples, obs_dim, generator=gen)
    proprioceptions = torch.randn(num_samples, proprio_dim, generator=gen)
    action_chunks = torch.randn(num_samples, generated_horizon, action_dim, generator=gen)
    execution_masks = make_execution_mask(
        torch.full((num_samples,), executed_horizon), generated_horizon
    )
    executed_lengths = torch.full((num_samples,), executed_horizon, dtype=torch.long)

    prefix = action_chunks[:, :executed_horizon]
    target_direction = torch.tanh(observations[:, :action_dim]).unsqueeze(1)
    per_step_rewards = 1.0 - (prefix - target_direction).pow(2).mean(dim=2)
    chunk_returns = compute_chunk_return(per_step_rewards, gamma, executed_lengths)
    discounts = compute_transition_discount(gamma, executed_lengths)
    next_observations = observations + 0.05 * torch.randn(num_samples, obs_dim, generator=gen)
    next_proprioceptions = proprioceptions + 0.05 * torch.randn(
        num_samples, proprio_dim, generator=gen
    )
    dones = torch.zeros(num_samples)
    successes = (chunk_returns > torch.median(chunk_returns)).float()
    episode_ids = torch.arange(num_samples, dtype=torch.long) // 4
    timesteps = torch.arange(num_samples, dtype=torch.long) % 4
    task_ids = ["synthetic_task"] * num_samples
    languages = ["move toward the synthetic target"] * num_samples
    behavior_metadata = [{"policy": "synthetic_behavior", "seed": seed} for _ in range(num_samples)]

    return ChunkBatch(
        observations=observations,
        proprioceptions=proprioceptions,
        action_chunks=action_chunks,
        execution_masks=execution_masks,
        executed_lengths=executed_lengths,
        chunk_returns=chunk_returns,
        discounts=discounts,
        next_observations=next_observations,
        next_proprioceptions=next_proprioceptions,
        dones=dones,
        successes=successes,
        episode_ids=episode_ids,
        timesteps=timesteps,
        task_ids=task_ids,
        languages=languages,
        behavior_metadata=behavior_metadata,
    )
