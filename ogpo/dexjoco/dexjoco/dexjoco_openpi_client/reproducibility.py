"""Portable random-number derivation for paired policy evaluation."""

import numpy as np


def policy_noise_seed(seed: int, episode_index: int, request_index: int) -> int:
    """Return a stable common-random-number seed for one policy request."""
    return int(
        (
            int(seed) * 1_000_003
            + int(episode_index) * 10_007
            + int(request_index)
        )
        % (2**31 - 1)
    )


def episode_seed(seed: int, episode_index: int) -> int:
    """Bind each scenario to the evaluation seed and episode index."""
    return int((int(seed) + int(episode_index) * 10_007) % (2**31 - 1))


def policy_noise(seed: int, action_horizon: int, action_dim: int) -> np.ndarray:
    """Generate portable flow noise without depending on a GPU RNG backend."""
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    return generator.standard_normal(
        (int(action_horizon), int(action_dim)),
        dtype=np.float32,
    )
