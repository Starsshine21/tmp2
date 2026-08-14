#!/usr/bin/env python3
"""Stream Vive tracker poses to Dexjoco over UDP."""

from __future__ import annotations

import argparse
import socket
import sys
import time
from dataclasses import dataclass

import numpy as np

try:
    import openvr
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "openvr is not installed. Run `pip install openvr` on the SteamVR machine."
    ) from exc


TRACKING_UNIVERSES = {
    "standing": openvr.TrackingUniverseStanding,
    "seated": openvr.TrackingUniverseSeated,
    "raw": openvr.TrackingUniverseRawAndUncalibrated,
}

DEVICE_CLASSES = {
    openvr.TrackedDeviceClass_Controller: "controller",
    openvr.TrackedDeviceClass_GenericTracker: "tracker",
    openvr.TrackedDeviceClass_HMD: "hmd",
    openvr.TrackedDeviceClass_TrackingReference: "tracking_reference",
}


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    serial: str
    model: str
    device_class: str


def _format_pose_line(pose: np.ndarray) -> str:
    flat = pose.reshape(-1)
    values = " ".join(f"{value: .6f}" for value in flat)
    return f"pose(3x4): {values}"


def _format_dual_pose_line(primary_pose: np.ndarray, secondary_pose: np.ndarray) -> str:
    primary_values = " ".join(f"{value: .6f}" for value in primary_pose.reshape(-1))
    secondary_values = " ".join(f"{value: .6f}" for value in secondary_pose.reshape(-1))
    return f"T1: {primary_values} | T2: {secondary_values}"


def _normalize_property(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _get_pose_batch(vr_system, universe: int):
    return vr_system.getDeviceToAbsoluteTrackingPose(
        universe, 0, openvr.k_unMaxTrackedDeviceCount
    )


def _matrix34_to_numpy(matrix34) -> np.ndarray:
    if hasattr(matrix34, "m"):
        return np.asarray(matrix34.m, dtype=np.float64).reshape(3, 4)
    raw = np.array(matrix34)
    if raw.dtype.names and "m" in raw.dtype.names:
        return np.asarray(raw["m"], dtype=np.float64).reshape(3, 4)
    return np.asarray(raw, dtype=np.float64).reshape(3, 4)


def discover_devices(vr_system, universe: int) -> list[DeviceInfo]:
    poses = _get_pose_batch(vr_system, universe)
    devices = []
    for index in range(openvr.k_unMaxTrackedDeviceCount):
        if not poses[index].bDeviceIsConnected:
            continue
        device_class_id = vr_system.getTrackedDeviceClass(index)
        device_class = DEVICE_CLASSES.get(device_class_id, f"class_{device_class_id}")
        serial = _normalize_property(
            vr_system.getStringTrackedDeviceProperty(
                index, openvr.Prop_SerialNumber_String
            )
        )
        model = _normalize_property(
            vr_system.getStringTrackedDeviceProperty(index, openvr.Prop_ModelNumber_String)
        )
        devices.append(
            DeviceInfo(
                index=index,
                serial=serial,
                model=model,
                device_class=device_class,
            )
        )
    return devices


def select_device(
    devices: list[DeviceInfo],
    device_index: int | None,
    serial_contains: str | None,
    excluded_indices: set[int] | None = None,
) -> DeviceInfo:
    excluded_indices = excluded_indices or set()
    if device_index is not None:
        for device in devices:
            if device.index == device_index and device.index not in excluded_indices:
                return device
        raise ValueError(f"tracked device index {device_index} was not found")

    if serial_contains:
        serial_query = serial_contains.lower()
        matches = [
            d
            for d in devices
            if serial_query in d.serial.lower() and d.index not in excluded_indices
        ]
        if not matches:
            raise ValueError(
                f"no tracked device serial contained '{serial_contains}'"
            )
        return matches[0]

    for device in devices:
        if device.device_class == "tracker" and device.index not in excluded_indices:
            return device

    raise ValueError("no Vive tracker device was found; use --list-devices to inspect")


def read_pose_matrix(vr_system, universe: int, device_index: int) -> np.ndarray | None:
    poses = _get_pose_batch(vr_system, universe)
    tracked_pose = poses[device_index]
    if not tracked_pose.bPoseIsValid:
        return None
    return _matrix34_to_numpy(tracked_pose.mDeviceToAbsoluteTracking)


def wait_for_first_pose(
    vr_system, universe: int, device_index: int, timeout_s: float
) -> np.ndarray:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pose = read_pose_matrix(vr_system, universe, device_index)
        if pose is not None:
            return pose
        time.sleep(0.01)
    raise RuntimeError("tracker did not provide a valid pose during warm-up")


def stream_pose(args) -> int:
    interval = 1.0 / args.frequency
    universe = TRACKING_UNIVERSES[args.tracking_universe]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    openvr.init(openvr.VRApplication_Other)
    vr_system = openvr.VRSystem()
    try:
        devices = discover_devices(vr_system, universe)
        if args.list_devices:
            if not devices:
                print("No tracked devices found.")
                return 0
            for device in devices:
                print(
                    f"index={device.index} class={device.device_class} "
                    f"serial={device.serial} model={device.model}"
                )
            return 0

        device = select_device(devices, args.device_index, args.serial_contains)
        primary_pose = wait_for_first_pose(
            vr_system, universe, device.index, args.warmup_seconds
        )
        secondary_device = None
        secondary_pose = None

        if args.two_trackers or args.second_device_index is not None or args.second_serial_contains:
            secondary_device = select_device(
                devices,
                args.second_device_index,
                args.second_serial_contains,
                excluded_indices={device.index},
            )
            secondary_pose = wait_for_first_pose(
                vr_system, universe, secondary_device.index, args.warmup_seconds
            )

        if secondary_device is None:
            print(
                f"Streaming Vive pose from index={device.index} serial={device.serial} "
                f"to {args.host}:{args.port} at {args.frequency:.1f} Hz. Press Ctrl+C to stop."
            )
        else:
            print(
                f"Streaming dual Vive poses from "
                f"index={device.index} serial={device.serial} and "
                f"index={secondary_device.index} serial={secondary_device.serial} "
                f"to {args.host}:{args.port} at {args.frequency:.1f} Hz. Press Ctrl+C to stop."
            )
        while True:
            tick_start = time.time()
            pose = read_pose_matrix(vr_system, universe, device.index)
            if pose is not None:
                primary_pose = pose

            if secondary_device is None:
                sock.sendto(primary_pose.astype(np.float64).tobytes(), (args.host, args.port))
                if not args.quiet:
                    print("\r" + _format_pose_line(primary_pose), end="", flush=True)
            else:
                pose2 = read_pose_matrix(vr_system, universe, secondary_device.index)
                if pose2 is not None:
                    secondary_pose = pose2
                packet = (
                    primary_pose.astype(np.float64).tobytes()
                    + secondary_pose.astype(np.float64).tobytes()
                )
                sock.sendto(packet, (args.host, args.port))
                if not args.quiet:
                    print(
                        "\r" + _format_dual_pose_line(primary_pose, secondary_pose),
                        end="",
                        flush=True,
                    )
            sleep_s = interval - (time.time() - tick_start)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nStopped streaming Vive pose.")
        return 0
    finally:
        sock.close()
        openvr.shutdown()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream Vive tracker poses to Dexjoco over UDP."
    )
    parser.add_argument("--host", default="127.0.0.1", help="UDP target host")
    parser.add_argument(
        "--port", type=int, default=5012, help="UDP target port for tracker pose"
    )
    parser.add_argument(
        "--frequency", type=float, default=90.0, help="Streaming frequency in Hz"
    )
    parser.add_argument(
        "--tracking-universe",
        choices=sorted(TRACKING_UNIVERSES),
        default="standing",
        help="OpenVR tracking universe",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Exact OpenVR tracked-device index to stream",
    )
    parser.add_argument(
        "--serial-contains",
        default=None,
        help="Pick the first tracked device whose serial contains this substring",
    )
    parser.add_argument(
        "--two-trackers",
        action="store_true",
        help="Stream two tracker poses in one UDP packet for bimanual teleoperation",
    )
    parser.add_argument(
        "--second-device-index",
        type=int,
        default=None,
        help="Exact OpenVR tracked-device index for the second tracker",
    )
    parser.add_argument(
        "--second-serial-contains",
        default=None,
        help="Pick the second tracked device whose serial contains this substring",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=0.5,
        help="Seconds to wait for the first valid pose",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available OpenVR devices and exit",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable live pose printing while streaming",
    )
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    try:
        return stream_pose(args)
    except ValueError as exc:
        parser.error(str(exc))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
