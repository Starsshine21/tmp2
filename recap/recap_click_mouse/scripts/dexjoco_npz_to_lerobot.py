#!/usr/bin/env python
"""Convert compact DexJoCo rollout NPZ files into a local LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _scalar_str(data: dict[str, np.ndarray], key: str, fallback: str) -> str:
    if key not in data:
        return fallback
    value = data[key]
    if getattr(value, "shape", ()) == ():
        return str(value.item())
    return str(value)


def _validate_arrays(data: dict[str, np.ndarray]) -> None:
    required = ["base", "wrist", "state", "action", "episode_id", "is_success"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing required NPZ fields: {missing}")

    n = int(data["action"].shape[0])
    for key in required:
        if int(data[key].shape[0]) != n:
            raise ValueError(f"Field '{key}' length {data[key].shape[0]} does not match action length {n}.")
    for key in ("base", "wrist"):
        if data[key].ndim != 4 or data[key].shape[-1] != 3:
            raise ValueError(f"Expected {key} images with shape [N,H,W,3], got {data[key].shape}.")


def _read_match_features(root: Path | None) -> dict[str, dict[str, Any]]:
    if root is None:
        return {}
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Missing features object in {info_path}")
    return features


def _target_image_shape(
    *,
    key: str,
    array: np.ndarray,
    match_features: dict[str, dict[str, Any]],
    image_size: int,
) -> tuple[int, int, int]:
    if key in match_features:
        feature = match_features[key]
        dtype = feature.get("dtype")
        if dtype not in {"image", "video"}:
            raise ValueError(f"Matched feature {key!r} must be image/video, got {dtype!r}.")
        shape = tuple(int(v) for v in feature.get("shape", ()))
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"Matched feature {key!r} has unsupported shape: {shape}.")
        return shape
    if image_size > 0:
        return (int(image_size), int(image_size), 3)
    return tuple(int(v) for v in array.shape[1:])


def _as_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def _resize_image(image: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    image = _as_uint8_image(image)
    if tuple(image.shape) == target_shape:
        return image
    height, width, channels = target_shape
    if channels != 3:
        raise ValueError(f"Only RGB targets are supported, got target shape {target_shape}.")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(Image.fromarray(image).resize((width, height), resampling), dtype=np.uint8)


def _feature_definition(
    *,
    key: str,
    match_features: dict[str, dict[str, Any]],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    if key not in match_features:
        return fallback
    feature = dict(match_features[key])
    if "shape" in feature:
        feature["shape"] = tuple(int(v) for v in feature["shape"])
    return feature


def convert(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root exists: {args.output_root}. Use --overwrite to replace it.")
        shutil.rmtree(args.output_root)

    with np.load(args.input, allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}
    _validate_arrays(data)

    base = data["base"]
    wrist = data["wrist"]
    state = data["state"].astype(np.float32, copy=False)
    action = data["action"].astype(np.float32, copy=False)
    episode_ids = data["episode_id"].astype(np.int64, copy=False)
    is_success = data["is_success"].astype(np.bool_, copy=False)
    task = _scalar_str(data, "base_prompt", _scalar_str(data, "prompt", args.task.replace("_", " "))).strip()
    match_features = _read_match_features(args.match_features_root)

    ego_key = "observation.images.ego_right"
    wrist_key = "observation.images.wrist"
    state_key = "observation.state"
    action_key = "action"
    ego_shape = _target_image_shape(key=ego_key, array=base, match_features=match_features, image_size=args.image_size)
    wrist_shape = _target_image_shape(key=wrist_key, array=wrist, match_features=match_features, image_size=args.image_size)
    use_videos = bool(args.use_videos)
    if match_features:
        use_videos = any(match_features.get(key, {}).get("dtype") == "video" for key in (ego_key, wrist_key))

    features: dict[str, dict[str, Any]] = {
        ego_key: _feature_definition(
            key=ego_key,
            match_features=match_features,
            fallback={"dtype": "image", "shape": ego_shape, "names": ["height", "width", "channel"]},
        ),
        wrist_key: _feature_definition(
            key=wrist_key,
            match_features=match_features,
            fallback={"dtype": "image", "shape": wrist_shape, "names": ["height", "width", "channel"]},
        ),
        state_key: _feature_definition(
            key=state_key,
            match_features=match_features,
            fallback={"dtype": "float32", "shape": tuple(state.shape[1:]), "names": None},
        ),
        action_key: _feature_definition(
            key=action_key,
            match_features=match_features,
            fallback={"dtype": "float32", "shape": tuple(action.shape[1:]), "names": None},
        ),
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.output_root,
        robot_type=args.robot_type or None,
        use_videos=use_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    episode_count = 0
    frame_count = 0
    for episode_id in sorted(np.unique(episode_ids).tolist()):
        indices = np.flatnonzero(episode_ids == int(episode_id))
        if args.max_episodes > 0 and episode_count >= args.max_episodes:
            break
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break
        success = bool(is_success[indices[0]])
        for idx in indices:
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
            dataset.add_frame(
                {
                    ego_key: _resize_image(base[idx], ego_shape),
                    wrist_key: _resize_image(wrist[idx], wrist_shape),
                    "observation.state": state[idx],
                    "action": action[idx],
                    "task": task,
                }
            )
            frame_count += 1
        dataset.save_episode(
            extra_episode_metadata={
                "episode_success": "success" if success else "failure",
                "source_episode_id": int(episode_id),
            }
        )
        episode_count += 1

    dataset.finalize()
    summary = {
        "input": str(args.input),
        "output_root": str(args.output_root),
        "repo_id": args.repo_id,
        "task": task,
        "fps": int(args.fps),
        "robot_type": args.robot_type or None,
        "match_features_root": str(args.match_features_root) if args.match_features_root else None,
        "image_shapes": {
            ego_key: list(ego_shape),
            wrist_key: list(wrist_shape),
        },
        "frames": int(frame_count),
        "episodes": int(episode_count),
        "successful_episodes": int(
            sum(bool(is_success[np.flatnonzero(episode_ids == ep)[0]]) for ep in np.unique(episode_ids)[:episode_count])
        ),
        "use_videos": use_videos,
    }
    summary_path = args.summary_output or args.output_root / "dexjoco_npz_to_lerobot.summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", default="click_mouse")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default=None)
    parser.add_argument("--match-features-root", type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-output", type=Path)
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
