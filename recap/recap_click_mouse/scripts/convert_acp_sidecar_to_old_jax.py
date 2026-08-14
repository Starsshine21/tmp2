#!/usr/bin/env python3
"""Convert LeRobot ACP sidecar field names to the old JAX actor schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--smoke-positive-ratio", type=float)
    args = parser.parse_args()

    if args.smoke_positive_ratio is not None:
        if args.input is not None or not 0.0 <= args.smoke_positive_ratio <= 1.0:
            raise ValueError("Smoke mode takes no input and requires a ratio in [0, 1]")
        indices = np.arange(args.expected_frames, dtype=np.int64)
        value = np.zeros(args.expected_frames, dtype=np.float32)
        advantage = np.zeros(args.expected_frames, dtype=np.float32)
        indicator = np.zeros(args.expected_frames, dtype=np.int64)
        indicator[: round(args.expected_frames * args.smoke_positive_ratio)] = 1
    else:
        if args.input is None:
            raise ValueError("--input is required outside smoke mode")
        with np.load(args.input, allow_pickle=False) as data:
            value = np.asarray(data["complementary_info.value"], dtype=np.float32)
            advantage = np.asarray(data["complementary_info.advantage"], dtype=np.float32)
            indicator = np.asarray(data["complementary_info.acp_indicator"], dtype=np.int64)
            indices = np.asarray(data["absolute_indices"], dtype=np.int64)

    expected_indices = np.arange(args.expected_frames, dtype=np.int64)
    if not (
        len(value) == len(advantage) == len(indicator) == args.expected_frames
        and np.array_equal(indices, expected_indices)
    ):
        raise ValueError("ACP sidecar is not aligned to the expected contiguous frame indices")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        value=value,
        advantage=advantage,
        acp_indicator=indicator,
        absolute_indices=indices,
    )
    print(
        f"converted={args.output} frames={len(indicator)} "
        f"positive={int((indicator == 1).sum())} negative={int((indicator == 0).sum())}"
    )


if __name__ == "__main__":
    main()
