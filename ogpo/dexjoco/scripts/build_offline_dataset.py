#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dexjoco"))

from dexjoco.ogpo.replay import (
    add_monte_carlo_returns,
    make_n_step_replay,
    make_synthetic_replay,
    save_replay,
    save_replay_metadata,
    split_replay,
    split_success_buffers,
)
from dexjoco.ogpo.zarr_replay import ZarrConversionConfig, convert_replay_paths, find_replay_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ogpo/dataset.yaml")
    parser.add_argument("--input-zarr-root", default=None)
    parser.add_argument("--source", choices=["auto", "synthetic", "zarr"], default="auto")
    parser.add_argument(
        "--save-only-splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=None,
        help="Save only selected deterministic splits, omitting the full and success buffers.",
    )
    args = parser.parse_args()
    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    input_root = args.input_zarr_root or data.get("input_zarr_root")
    source = args.source
    if source == "auto":
        source = "zarr" if input_root else str(data.get("source", "synthetic"))

    if source == "zarr":
        if not input_root:
            raise ValueError("zarr source requires --input-zarr-root or data.input_zarr_root")
        paths = find_replay_paths(ROOT / input_root if not Path(input_root).is_absolute() else input_root)
        if not paths:
            raise FileNotFoundError(f"no replay.zarr or .npz files found under {input_root}")
        batch = convert_replay_paths(
            paths,
            ZarrConversionConfig(
                generated_horizon=int(data.get("generated_horizon", 30)),
                executed_horizon=int(data.get("executed_horizon", 4)),
                gamma=float(data.get("gamma", 0.97)),
                action_key=str(data.get("action_key", "action_rotvec")),
                fallback_action_key=str(data.get("fallback_action_key", "action")),
                state_key=str(data.get("state_key", "state")),
                reward_key=str(data.get("reward_key", "reward")),
                done_key=str(data.get("done_key", "done")),
                success_key=str(data.get("success_key", "success")),
                stride=data.get("stride", None),
                terminal_success_reward=float(data.get("terminal_success_reward", 1.0)),
                task_id=str(data.get("task_id", "dexjoco")),
                language=str(data.get("language", "")),
                behavior_policy=str(data.get("behavior_policy", "pi05_or_teleop")),
                success_from_path=bool(data.get("success_from_path", True)),
                image_keys=tuple(data["image_keys"]) if data.get("image_keys") is not None else None,
            ),
        )
        source_paths = [str(p) for p in paths]
    else:
        batch = make_synthetic_replay(
            num_samples=int(data.get("num_samples", 64)),
            obs_dim=int(data.get("obs_dim", 12)),
            proprio_dim=int(data.get("proprio_dim", 4)),
            generated_horizon=int(data.get("generated_horizon", 6)),
            action_dim=int(data.get("action_dim", 3)),
            executed_horizon=int(data.get("executed_horizon", 3)),
            gamma=float(data.get("gamma", 0.97)),
            seed=int(data.get("seed", 7)),
        )
        source_paths = []

    batch = add_monte_carlo_returns(batch)
    batch = make_n_step_replay(batch, n_step=int(data.get("n_step", 1)))
    output = ROOT / data["dataset_path"]
    if args.save_only_splits is None:
        save_replay(batch, output)
    splits = split_replay(
        batch,
        train_ratio=float(data.get("train_ratio", 0.8)),
        validation_ratio=float(data.get("validation_ratio", 0.1)),
        seed=int(data.get("seed", 7)),
    )
    for name, split in splits.items():
        if args.save_only_splits is not None and name not in args.save_only_splits:
            continue
        save_replay(split, output.with_name(f"{output.stem}_{name}{output.suffix}"))
    buffers = split_success_buffers(batch)
    if args.save_only_splits is None:
        for name, split in buffers.items():
            if name == "all":
                continue
            save_replay(split, output.with_name(f"{output.stem}_{name}{output.suffix}"))
    save_replay_metadata(
        output,
        {
            "source": source,
            "source_paths": source_paths,
            "samples": batch.batch_size,
            "generated_horizon": batch.generated_horizon,
            "action_dim": batch.action_dim,
            "n_step": int(data.get("n_step", 1)),
            "success_samples": int(batch.successes.sum().item()),
            "mean_chunk_return": float(batch.chunk_returns.mean().item()),
            "splits": {name: split.batch_size for name, split in splits.items()},
            "buffers": {name: split.batch_size for name, split in buffers.items()},
        },
    )
    print(f"[dataset] wrote {output} samples={batch.batch_size}")


if __name__ == "__main__":
    main()
