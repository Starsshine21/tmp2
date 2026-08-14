#!/usr/bin/env python
"""Convert the exact per-step DexJoCo replay.zarr rollouts to LeRobot v3."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import zarr

from lerobot.datasets.lerobot_dataset import LeRobotDataset


EPISODE_RE = re.compile(r"episode_(\d+)_(success|failure)$")


def _episode_key(path: Path) -> int:
    match = EPISODE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected episode directory name: {path.name}")
    return int(match.group(1))


def _read_episode(path: Path) -> dict[str, np.ndarray]:
    root = zarr.open_group(str(path / "replay.zarr"), mode="r")
    data = root["data"]
    required = ("image_base", "image_wrist", "state", "action_rotvec", "success")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path}: missing replay arrays {missing}")
    arrays = {key: np.asarray(data[key][:]) for key in required}
    lengths = {array.shape[0] for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"{path}: inconsistent replay lengths {sorted(lengths)}")
    return arrays


def convert(args: argparse.Namespace) -> None:
    episode_dirs = sorted(
        [path for path in args.input_root.iterdir() if path.is_dir() and EPISODE_RE.fullmatch(path.name)],
        key=_episode_key,
    )
    if args.episode_limit > 0:
        episode_dirs = episode_dirs[: args.episode_limit]
    if len(episode_dirs) != args.expected_episodes:
        raise RuntimeError(f"Expected {args.expected_episodes} episodes, found {len(episode_dirs)}")
    expected_ids = list(range(args.expected_episodes))
    actual_ids = [_episode_key(path) for path in episode_dirs]
    if actual_ids != expected_ids:
        raise RuntimeError(f"Episode IDs are not contiguous: {actual_ids}")

    outcomes = [path.name.rsplit("_", 1)[1] for path in episode_dirs]
    successes = outcomes.count("success")
    failures = outcomes.count("failure")
    if args.expected_successes >= 0 and args.expected_failures >= 0 and (
        successes,
        failures,
    ) != (args.expected_successes, args.expected_failures):
        raise RuntimeError(
            f"Expected outcomes {args.expected_successes}/{args.expected_failures}, got {successes}/{failures}"
        )
    if successes == 0 or failures == 0:
        raise RuntimeError(f"RECAP requires both outcomes, got successes={successes} failures={failures}")

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_root}")
        shutil.rmtree(args.output_root)

    first = _read_episode(episode_dirs[0])
    base_shape = tuple(int(value) for value in first["image_base"].shape[1:])
    wrist_shape = tuple(int(value) for value in first["image_wrist"].shape[1:])
    state_shape = tuple(int(value) for value in first["state"].shape[1:])
    action_shape = tuple(int(value) for value in first["action_rotvec"].shape[1:])
    features = {
        "observation.images.ego_right": {
            "dtype": "image",
            "shape": base_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": wrist_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": state_shape, "names": None},
        "action": {"dtype": "float32", "shape": action_shape, "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.output_root,
        robot_type="panda_allegro",
        use_videos=False,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    total_frames = 0
    for episode_dir, outcome in zip(episode_dirs, outcomes, strict=True):
        arrays = _read_episode(episode_dir)
        array_success = bool(np.asarray(arrays["success"], dtype=np.bool_).any())
        path_success = outcome == "success"
        if array_success != path_success:
            raise RuntimeError(
                f"Outcome mismatch for {episode_dir.name}: path={path_success}, replay={array_success}"
            )
        length = int(arrays["state"].shape[0])
        for frame in range(length):
            dataset.add_frame(
                {
                    "observation.images.ego_right": np.asarray(
                        arrays["image_base"][frame], dtype=np.uint8
                    ),
                    "observation.images.wrist": np.asarray(
                        arrays["image_wrist"][frame], dtype=np.uint8
                    ),
                    "observation.state": np.asarray(arrays["state"][frame], dtype=np.float32),
                    "action": np.asarray(arrays["action_rotvec"][frame], dtype=np.float32),
                    "task": args.prompt,
                }
            )
        dataset.save_episode(
            extra_episode_metadata={
                "episode_success": outcome,
                "source_episode_id": _episode_key(episode_dir),
            }
        )
        total_frames += length
        print(
            f"converted episode={_episode_key(episode_dir)} outcome={outcome} frames={length}",
            flush=True,
        )

    dataset.finalize()
    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "repo_id": args.repo_id,
        "episodes": len(episode_dirs),
        "successful_episodes": successes,
        "failed_episodes": failures,
        "frames": total_frames,
        "fps": args.fps,
        "prompt": args.prompt,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local_repo")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--prompt",
        default="Move the mouse to the purple mouse pad and click the left mouse button.",
    )
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--expected-successes", type=int, default=-1)
    parser.add_argument("--expected-failures", type=int, default=-1)
    parser.add_argument("--episode-limit", type=int, default=0)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
