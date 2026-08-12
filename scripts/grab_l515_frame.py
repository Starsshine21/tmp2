#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np


def capture_color_frame() -> np.ndarray:
    try:
        import pyrealsense2.pyrealsense2 as rs
    except ImportError:
        import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    try:
        frame = None
        for _ in range(30):
            frames = pipeline.wait_for_frames()
            frame = frames.get_color_frame()
        if frame is None:
            raise RuntimeError("Failed to read an L515 color frame")
        return np.asanyarray(frame.get_data())
    finally:
        pipeline.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one L515 BGR frame.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), capture_color_frame()):
        raise RuntimeError(f"Failed to write L515 frame to {output_path}")


if __name__ == "__main__":
    main()
