from dataclasses import replace

import torch

from dexjoco.ogpo.chunk_transition import (
    assert_suffix_invariant,
    compute_chunk_return,
    compute_transition_discount,
    flatten_masked_action,
    make_execution_mask,
    mask_action_suffix,
)
from dexjoco.ogpo.replay import (
    BalancedCriticReplay,
    add_monte_carlo_returns,
    make_n_step_replay,
    make_synthetic_replay,
    split_replay,
    split_success_buffers,
)


def test_balanced_critic_replay_includes_success_terminal_and_failure_strata():
    batch = make_synthetic_replay(
        num_samples=12,
        generated_horizon=3,
        executed_horizon=1,
        action_dim=2,
    )
    episode_ids = torch.arange(4).repeat_interleave(3)
    successes = (episode_ids < 2).float()
    dones = torch.zeros(12)
    dones[2::3] = 1.0
    batch = replace(
        batch,
        episode_ids=episode_ids,
        timesteps=torch.arange(3).repeat(4),
        successes=successes,
        dones=dones,
    )
    replay = BalancedCriticReplay(batch)

    sample = replay.sample(8, generator=torch.Generator().manual_seed(5))

    assert sample.batch_size == 8
    assert int(sample.successes.sum()) >= 3
    assert int((~sample.successes.bool()).sum()) >= 1
    assert bool((sample.successes.bool() & sample.dones.bool()).any())


def test_execution_mask_and_suffix_invariant():
    action = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    mask = make_execution_mask(2, 5)
    masked = mask_action_suffix(action, mask)
    assert masked[:2].sum() == action[:2].sum()
    assert masked[2:].sum() == 0
    assert_suffix_invariant(action, mask)
    flat_a = flatten_masked_action(action, mask)
    mutated = action.clone()
    mutated[2:] = 999
    flat_b = flatten_masked_action(mutated, mask)
    assert torch.equal(flat_a, flat_b)


def test_chunk_return_and_discount():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    value = compute_chunk_return(rewards, gamma=0.5, executed_length=2)
    assert torch.allclose(value, torch.tensor(2.0))
    discount = compute_transition_discount(0.5, 2)
    assert torch.allclose(discount, torch.tensor(0.25))


def test_success_buffers():
    batch = make_synthetic_replay(num_samples=12)
    buffers = split_success_buffers(batch)
    assert "success" in buffers
    assert "failure" in buffers
    assert buffers["success"].successes.bool().all()
    assert (~buffers["failure"].successes.bool()).all()


def test_near_success_buffer_requires_informative_failure_returns():
    batch = make_synthetic_replay(num_samples=12)
    batch = replace(
        batch,
        successes=torch.zeros(12),
        chunk_returns=torch.zeros(12),
    )

    buffers = split_success_buffers(batch)

    assert "near_success" not in buffers


def test_replay_splits_keep_episodes_disjoint():
    batch = make_synthetic_replay(num_samples=24)

    splits = split_replay(batch, train_ratio=0.6, validation_ratio=0.2, seed=3)
    episode_sets = [set(split.episode_ids.tolist()) for split in splits.values()]

    assert set(splits) == {"train", "validation", "heldout"}
    assert all(left.isdisjoint(right) for i, left in enumerate(episode_sets) for right in episode_sets[i + 1 :])


def test_n_step_replay_accumulates_returns_discount_and_next_state():
    batch = make_synthetic_replay(
        num_samples=4,
        generated_horizon=3,
        executed_horizon=1,
        action_dim=2,
    )
    batch = replace(
        batch,
        episode_ids=torch.zeros(4, dtype=torch.long),
        timesteps=torch.arange(4, dtype=torch.long),
        chunk_returns=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        discounts=torch.full((4,), 0.5),
        dones=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        next_observations=torch.arange(4 * batch.obs_dim, dtype=torch.float32).reshape(4, batch.obs_dim),
    )

    result = make_n_step_replay(batch, n_step=2)

    assert torch.allclose(result.chunk_returns[:3], torch.tensor([2.0, 3.5, 5.0]))
    assert torch.allclose(result.discounts[:3], torch.tensor([0.25, 0.25, 0.25]))
    assert torch.equal(result.next_observations[0], batch.next_observations[1])
    assert result.behavior_metadata[0]["n_step"] == 2


def test_monte_carlo_returns_follow_episode_discounts():
    batch = make_synthetic_replay(num_samples=4, generated_horizon=3, executed_horizon=1, action_dim=2)
    batch = replace(
        batch,
        episode_ids=torch.zeros(4, dtype=torch.long),
        timesteps=torch.arange(4, dtype=torch.long),
        chunk_returns=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        discounts=torch.full((4,), 0.5),
        dones=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )

    result = add_monte_carlo_returns(batch)

    assert result.mc_returns is not None
    assert torch.allclose(result.mc_returns, torch.tensor([3.25, 4.5, 5.0, 4.0]))
