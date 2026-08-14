from __future__ import annotations

import argparse
import socket
import time

import numpy as np

try:
    from .common import RokokoReceiver
except ImportError:
    from common import RokokoReceiver


DEFAULT_RIGHT_TARGET_PORT = 5013
DEFAULT_LEFT_TARGET_PORT = 5015


class RokokoMocap:
    """External-PC Rokoko receiver that returns canonicalized hand keypoints."""

    def __init__(
        self,
        udp_ip: str = "127.0.0.1",
        udp_port: int = 14044,
        is_left: bool = False,
        timeout: float = 1.0,
        verbose: bool = False,
    ):
        self.is_left = is_left
        self.receiver = RokokoReceiver(
            udp_ip=udp_ip,
            udp_port=udp_port,
            timeout=timeout,
            verbose=verbose,
        )

    def get(self) -> np.ndarray:
        return self.receiver.recv_hand(is_left=self.is_left)

    def close(self):
        self.receiver.close()


def build_argparser(default_hand: str = "right") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive Rokoko Studio packets on another PC and forward one hand over UDP."
    )
    parser.add_argument(
        "--hand",
        choices=["left", "right"],
        default=default_hand,
        help="Which hand to extract from the Rokoko stream",
    )
    parser.add_argument(
        "--listen-ip",
        default="127.0.0.1",
        help="Local UDP bind IP for Rokoko Studio packets",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=14044,
        help="Local UDP bind port for Rokoko Studio packets",
    )
    parser.add_argument(
        "--target-ip",
        default="127.0.0.1",
        help="Destination IP for canonicalized hand keypoints",
    )
    parser.add_argument(
        "--target-port",
        type=int,
        default=None,
        help="Destination UDP port; defaults to 5013 for right hand and 5015 for left hand",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Socket timeout in seconds while waiting for Rokoko packets",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=90.0,
        help="Maximum UDP send rate in Hz",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable live keypoint mean printing while forwarding",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print packet-parse warnings while waiting for Rokoko data",
    )
    return parser


def main(default_hand: str = "right") -> int:
    parser = build_argparser(default_hand=default_hand)
    args = parser.parse_args()

    is_left = args.hand == "left"
    target_port = args.target_port
    if target_port is None:
        target_port = DEFAULT_LEFT_TARGET_PORT if is_left else DEFAULT_RIGHT_TARGET_PORT

    mocap = RokokoMocap(
        udp_ip=args.listen_ip,
        udp_port=args.listen_port,
        is_left=is_left,
        timeout=args.timeout,
        verbose=args.verbose,
    )
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / args.frequency if args.frequency > 0 else 0.0
    side = "left" if is_left else "right"

    print(
        f"Forwarding {side} Rokoko hand from {args.listen_ip}:{args.listen_port} "
        f"to {args.target_ip}:{target_port}. Press Ctrl+C to stop."
    )

    try:
        while True:
            tick_start = time.time()
            hand = np.asarray(mocap.get(), dtype=np.float32)
            send_sock.sendto(hand.tobytes(), (args.target_ip, target_port))
            if not args.quiet:
                mean = hand.mean(axis=0)
                print(
                    f"\r{side} hand mean: {mean[0]: .6f} {mean[1]: .6f} {mean[2]: .6f}",
                    end="",
                    flush=True,
                )
            if interval > 0:
                sleep_s = interval - (time.time() - tick_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nStopped forwarding Rokoko hand.")
        return 0
    finally:
        mocap.close()
        send_sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
