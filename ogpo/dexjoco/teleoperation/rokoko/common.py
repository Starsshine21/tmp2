from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict, Optional

import numpy as np

try:
    import lz4.block
    import lz4.frame

    HAS_LZ4 = True
except ImportError:  # pragma: no cover - optional runtime dependency
    HAS_LZ4 = False


MAGIC_LZ4_FRAME = b"\x04\x22\x4D\x18"

LEFT_JOINT_NAMES = [
    "leftHand",
    "leftThumbProximal",
    "leftThumbMedial",
    "leftThumbDistal",
    "leftThumbTip",
    "leftIndexProximal",
    "leftIndexMedial",
    "leftIndexDistal",
    "leftIndexTip",
    "leftMiddleProximal",
    "leftMiddleMedial",
    "leftMiddleDistal",
    "leftMiddleTip",
    "leftRingProximal",
    "leftRingMedial",
    "leftRingDistal",
    "leftRingTip",
    "leftLittleProximal",
    "leftLittleMedial",
    "leftLittleDistal",
    "leftLittleTip",
]

RIGHT_JOINT_NAMES = [name.replace("left", "right") for name in LEFT_JOINT_NAMES]


def try_json(raw: bytes) -> Dict[str, Any]:
    return json.loads(raw.decode("utf-8"))


def try_lz4_decompress(raw: bytes) -> bytes:
    if not HAS_LZ4:
        raise RuntimeError(
            "lz4 is not installed. Install `lz4` if Rokoko Studio sends compressed UDP packets."
        )

    if len(raw) >= 8 and raw[:4] != MAGIC_LZ4_FRAME and raw[4:8] == MAGIC_LZ4_FRAME:
        try:
            return lz4.frame.decompress(raw[4:])
        except Exception:
            pass

    if raw.startswith(MAGIC_LZ4_FRAME):
        return lz4.frame.decompress(raw)

    try:
        return lz4.frame.decompress(raw)
    except Exception:
        pass

    try:
        return lz4.block.decompress(raw)
    except Exception:
        if len(raw) > 4:
            uncompressed_size = struct.unpack("<I", raw[:4])[0]
            return lz4.block.decompress(raw[4:], uncompressed_size=uncompressed_size)
        raise


def parse_rokoko_packet(raw: bytes) -> Dict[str, Any]:
    try:
        return try_json(raw)
    except Exception:
        pass

    decompressed = try_lz4_decompress(raw)
    return json.loads(decompressed.decode("utf-8"))


def extract_body(msg: Dict[str, Any]) -> Dict[str, Any]:
    return msg["scene"]["actors"][0]["body"]


def extract_hand_positions(
    body: Dict[str, Any], joint_names: list[str]
) -> Optional[np.ndarray]:
    points = []
    try:
        for joint_name in joint_names:
            if joint_name not in body:
                return None
            position = body[joint_name]["position"]
            points.append([position["x"], position["y"], position["z"]])
    except Exception:
        return None

    arr = np.asarray(points, dtype=np.float32)
    if arr.shape != (21, 3):
        return None
    return arr


def hand_to_canonical(hand_point: np.ndarray, is_left: bool) -> np.ndarray:
    hand_point = np.asarray(hand_point, dtype=np.float32)
    eps = 1e-6

    z_axis = hand_point[9] - hand_point[0]
    if np.linalg.norm(z_axis) < eps:
        return hand_point - hand_point[0]
    z_axis = z_axis / np.linalg.norm(z_axis)

    if is_left:
        y_axis_aux = hand_point[13] - hand_point[5]
    else:
        y_axis_aux = hand_point[5] - hand_point[13]

    if np.linalg.norm(y_axis_aux) < eps:
        return hand_point - hand_point[0]
    y_axis_aux = y_axis_aux / np.linalg.norm(y_axis_aux)

    x_axis = np.cross(y_axis_aux, z_axis)
    if np.linalg.norm(x_axis) < eps:
        return hand_point - hand_point[0]
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    transform[:3, 3] = hand_point[0]

    hand_homo = np.concatenate(
        [hand_point, np.ones((21, 1), dtype=np.float32)],
        axis=1,
    )
    hand_canonical = hand_homo @ np.linalg.inv(transform).T
    return hand_canonical[:, :3]


class RokokoReceiver:
    def __init__(
        self,
        udp_ip: str = "127.0.0.1",
        udp_port: int = 14044,
        timeout: float = 1.0,
        buffer_size: int = 262144,
        verbose: bool = False,
    ):
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.timeout = timeout
        self.buffer_size = buffer_size
        self.verbose = verbose

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))
        self.sock.settimeout(timeout)

    def recv_message(self) -> Dict[str, Any]:
        while True:
            try:
                data, _ = self.sock.recvfrom(self.buffer_size)
                return parse_rokoko_packet(data)
            except socket.timeout:
                continue
            except Exception as exc:
                if self.verbose:
                    print(f"[RokokoReceiver] packet parse error: {exc}")
                continue

    def recv_hand(self, is_left: bool) -> np.ndarray:
        joint_names = LEFT_JOINT_NAMES if is_left else RIGHT_JOINT_NAMES
        while True:
            msg = self.recv_message()
            try:
                body = extract_body(msg)
            except Exception as exc:
                if self.verbose:
                    print(f"[RokokoReceiver] unexpected message structure: {exc}")
                continue

            raw = extract_hand_positions(body, joint_names)
            if raw is None:
                if self.verbose:
                    side = "left" if is_left else "right"
                    print(f"[RokokoReceiver] missing {side} hand in packet")
                continue
            return hand_to_canonical(raw, is_left=is_left)

    def close(self):
        self.sock.close()
