#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.sim_teleop import BimanualTeleopConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="click_mouse", choices=sorted(CONFIG_MAPPING))
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    config = CONFIG_MAPPING[args.task]()
    env = config.get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=False,
        seed=0,
    )
    action_dim = 46 if isinstance(config.teleop, BimanualTeleopConfig) else 23
    total_return = 0.0
    try:
        observation, _ = env.reset()
        for _ in range(args.steps):
            observation, reward, terminated, truncated, info = env.step(np.zeros(action_dim))
            total_return += float(reward)
            if terminated or truncated:
                break
        image_keys = sorted(
            key for key, value in observation.items() if isinstance(value, np.ndarray) and value.ndim == 3
        )
        print(
            f"task={args.task} steps={args.steps} return={total_return:.6f} "
            f"success={bool(info.get('succeed', False))} cameras={','.join(image_keys)}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
