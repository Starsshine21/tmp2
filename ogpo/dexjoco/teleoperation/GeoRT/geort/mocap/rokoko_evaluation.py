import socket
import json
import numpy as np
from geort.env.hand import HandKinematicModel
from geort import load_model, get_config
import argparse
import time

class UDPManusReceiver:
    """
    UDP receiver that accepts raw bytes or JSON and returns a dict compatible with ManusMocap.get():
        {"result": np.ndarray(shape=(21,3), dtype=float32) or None, "status": "recording"/"no data"}
    Non-blocking get(): returns immediately with "no data" if no packet available.
    """

    def __init__(self, bind_ip="10.6.60.137", bind_port=5013, buffer_size=4096):
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.buffer_size = buffer_size

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bind to the provided IP/port. Use '0.0.0.0' if you want to listen on all interfaces.
        self.sock.bind((self.bind_ip, self.bind_port))
        # Use non-blocking recv so the main loop can continue rendering even if no data arrives.
        self.sock.setblocking(False)

        print(f"[UDPManusReceiver] bound to {self.bind_ip}:{self.bind_port}")

    def _parse_binary(self, data):
        """
        Try to parse raw bytes as float32 array of length >= 63 (21*3).
        Returns ndarray (21,3) or raises ValueError.
        """
        arr = np.frombuffer(data, dtype=np.float32)
        if arr.size < 63:
            raise ValueError(f"binary payload too small: {arr.size} floats")
        arr = arr[:63]  # if sender padded or appended, take first 63 floats
        arr = arr.reshape(21, 3)
        return arr

    def _parse_json(self, data):
        """
        Try to parse data as utf-8 JSON text listing points, e.g. [[x,y,z],...]
        Returns ndarray (21,3) or raises ValueError.
        """
        try:
            txt = data.decode("utf-8")
            js = json.loads(txt)
            arr = np.asarray(js, dtype=np.float32)
            if arr.shape != (21, 3):
                raise ValueError(f"json array shape mismatch: {arr.shape}")
            return arr
        except Exception as e:
            raise ValueError(f"json parse failed: {e}")

    def get(self):
        """
        Non-blocking get. If a packet is available, try binary parse first, then JSON.
        Returns: dict like ManusMocap.get()
        """
        try:
            data, addr = self.sock.recvfrom(self.buffer_size)
        except BlockingIOError:
            return {"result": None, "status": "no data"}
        except Exception as e:
            print("[UDPManusReceiver] recv error:", e)
            return {"result": None, "status": "no data"}

        # We received a packet — try to decode
        try:
            arr = self._parse_binary(data)
        except Exception:
            # fallback to json/text parse
            try:
                arr = self._parse_json(data)
            except Exception as e:
                print("[UDPManusReceiver] failed to parse packet from", addr, "err:", e)
                return {"result": None, "status": "no data"}

        # Success
        return {"result": arr.astype(np.float32), "status": "recording"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hand', type=str, default='allegro_left')
    parser.add_argument('-ckpt_tag', type=str, default='wxy_left_1')  # Your CKPT Tag.
    parser.add_argument('--bind_ip', type=str, default='10.6.60.137',
                        help='IP address to bind and listen on (use 0.0.0.0 to bind all interfaces)')
    parser.add_argument('--bind_port', type=int, default=5012, help='UDP port to listen on')
    args = parser.parse_args()

    # GeoRT Model.
    model = load_model(args.ckpt_tag)
    
    # Motion Capture: replace ManusMocap with UDP receiver
    mocap = UDPManusReceiver(bind_ip=args.bind_ip, bind_port=args.bind_port)
    
    # Robot Simulation.
    config = get_config(args.hand)
    hand = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = hand.get_viewer_env()
    
    print("[main] starting loop. waiting for UDP packets...")

    # Run!
    while True:
        # update viewer
        viewer_env.update()

        # try to get a packet (non-blocking)
        result = mocap.get()
        print("[main] mocap.get() result status:", result['result'])

        if result['status'] == 'recording' and result["result"] is not None:
            # result["result"] is (21,3) float32 numpy array — same as ManusMocap behavior
            try:
                qpos = model.forward(result["result"])
                print("[main] computed qpos:", qpos)

                hand.set_qpos_target(qpos)
            except Exception as e:
                print("[main] model.forward or set_qpos_target error:", e)

        # Optional: allow a graceful exit if viewer sets quit flag. 
        # The original ManusMocap returned 'quit' in some designs; adjust if needed.
        # Here we don't produce 'quit' via UDP, so rely on viewer close to raise or exit externally.

        # small sleep to avoid tight busy loop if viewer_env.update doesn't block
        # time.sleep(0.0)


if __name__ == '__main__':
    main()
