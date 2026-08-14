from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    from .rokoko_mocap import RokokoMocap
except ImportError:
    from rokoko_mocap import RokokoMocap


def build_argparser(default_hand: str = "right") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record canonicalized Rokoko keypoints to a .npy file."
    )
    parser.add_argument(
        "--hand",
        choices=["left", "right"],
        default=default_hand,
        help="Which hand to record from the Rokoko stream",
    )
    parser.add_argument("--listen-ip", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=14044)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where the .npy recording will be written",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Base filename without extension; defaults to rokoko_<hand>_<timestamp>",
    )
    parser.add_argument(
        "--sleep-dt",
        type=float,
        default=0.0,
        help="Optional sleep interval between frames",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(default_hand: str = "right") -> int:
    args = build_argparser(default_hand=default_hand).parse_args()
    is_left = args.hand == "left"

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_name = f"rokoko_{args.hand}_{timestamp}"
    else:
        output_name = args.output_name

    save_path = output_dir / f"{output_name}.npy"

    mocap = RokokoMocap(
        udp_ip=args.listen_ip,
        udp_port=args.listen_port,
        is_left=is_left,
        timeout=args.timeout,
        verbose=args.verbose,
    )

    data = []
    print("===================================")
    print("[INFO] Rokoko recording started")
    print("[INFO] Press Ctrl+C to stop and save")
    print("===================================")

    try:
        while True:
            hand_keypoints = mocap.get()
            data.append(hand_keypoints)
            print(f"\r[INFO] Frames collected: {len(data)}", end="", flush=True)
            if args.sleep_dt > 0:
                time.sleep(args.sleep_dt)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected, stopping recording...")
    finally:
        mocap.close()

    if not data:
        print("[WARNING] No data collected, nothing to save.")
        return 0

    array = np.asarray(data, dtype=np.float32)
    np.save(save_path, array)

    print("===================================")
    print("[INFO] Rokoko mocap data saved successfully")
    print(f"[INFO] Path : {save_path}")
    print(f"[INFO] Shape: {array.shape}")
    print(f"[INFO] Dtype: {array.dtype}")
    print("===================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
