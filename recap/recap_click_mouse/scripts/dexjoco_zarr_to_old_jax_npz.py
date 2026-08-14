#!/usr/bin/env python
"""Convert the formal replay.zarr episodes to the NPZ schema used by evorl@95f8b6c."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import zarr


EPISODE_RE = re.compile(r"episode_(\d+)_(success|failure)$")


def episode_id(path: Path) -> int:
    match = EPISODE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected episode directory: {path}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--expected-frames", type=int, default=66347)
    parser.add_argument(
        "--task",
        default="Move the mouse to the purple mouse pad and click the left mouse button.",
    )
    args = parser.parse_args()

    episode_dirs = sorted(
        (path for path in args.input_root.iterdir() if EPISODE_RE.fullmatch(path.name)),
        key=episode_id,
    )
    if len(episode_dirs) != args.expected_episodes:
        raise RuntimeError(f"Expected {args.expected_episodes} episodes, found {len(episode_dirs)}")
    if [episode_id(path) for path in episode_dirs] != list(range(args.expected_episodes)):
        raise RuntimeError("Episode IDs are not contiguous from zero")
    if args.output_root.exists():
        raise FileExistsError(f"Output already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)

    total_frames = 0
    successes = 0
    for path in episode_dirs:
        outcome = path.name.rsplit("_", 1)[1]
        group = zarr.open_group(str(path / "replay.zarr"), mode="r")["data"]
        base = np.asarray(group["image_base"][:], dtype=np.uint8)
        wrist = np.asarray(group["image_wrist"][:], dtype=np.uint8)
        state = np.asarray(group["state"][:], dtype=np.float32)
        action = np.asarray(group["action_rotvec"][:], dtype=np.float32)
        success = np.asarray(group["success"][:], dtype=np.bool_)
        lengths = {len(base), len(wrist), len(state), len(action), len(success)}
        if len(lengths) != 1:
            raise RuntimeError(f"Inconsistent array lengths in {path}: {sorted(lengths)}")
        if bool(success.any()) != (outcome == "success"):
            raise RuntimeError(f"Outcome mismatch in {path}")

        idx = episode_id(path)
        np.savez_compressed(
            args.output_root / f"episode_{idx:06d}.npz",
            observation_images_base=base,
            observation_images_wrist=wrist,
            observation_state=state,
            action=action,
            task=np.asarray([args.task]),
            episode_success=np.asarray(outcome == "success"),
        )
        total_frames += len(action)
        successes += int(outcome == "success")
        print(f"episode={idx} outcome={outcome} frames={len(action)}", flush=True)

    if total_frames != args.expected_frames:
        raise RuntimeError(f"Expected {args.expected_frames} frames, got {total_frames}")
    print(
        f"complete episodes={len(episode_dirs)} frames={total_frames} "
        f"successes={successes} failures={len(episode_dirs) - successes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
