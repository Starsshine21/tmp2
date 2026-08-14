#!/usr/bin/env python3
"""Compare exact per-episode traces from two deterministic evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXACT_KEYS = (
    "environment_seed",
    "initial_state_sha256",
    "aligned_state_sha256",
    "aligned_observation_sha256",
    "policy_request_trace_sha256",
    "policy_observation_trace_sha256",
    "policy_action_chunk_first_sha256",
    "policy_action_chunk_trace_sha256",
    "action_trace_sha256",
    "policy_request_count",
    "episode_length",
    "success",
)


def _episodes(path: Path) -> list[dict]:
    metrics = path / "evaluation_metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("record_type") == "episode"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()

    left = _episodes(args.left)
    right = _episodes(args.right)
    if len(left) != len(right):
        raise SystemExit(f"episode count mismatch: {len(left)} != {len(right)}")

    mismatches = []
    for episode_index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        for key in EXACT_KEYS:
            if left_row.get(key) != right_row.get(key):
                mismatches.append(
                    (episode_index, key, left_row.get(key), right_row.get(key))
                )

    if mismatches:
        print(f"reproducible=false mismatches={len(mismatches)}")
        for episode_index, key, left_value, right_value in mismatches[:30]:
            print(
                f"episode={episode_index} key={key} "
                f"left={left_value!r} right={right_value!r}"
            )
        raise SystemExit(1)

    print(f"reproducible=true episodes={len(left)} exact_keys={len(EXACT_KEYS)}")


if __name__ == "__main__":
    main()
