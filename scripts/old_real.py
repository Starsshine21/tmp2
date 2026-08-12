#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from openpi.policies import policy_config as policy_config_lib
from openpi.training import checkpoints as checkpoints_lib
from openpi.training import config as train_config_lib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HAND_RANGE = (0.0, 2000.0)


@dataclass(frozen=True)
class DeploymentPreset:
    checkpoint_dir: pathlib.Path
    train_config: str
    assets_dir: pathlib.Path
    control_mode: str


DEPLOYMENT_PRESETS = {
    "eef120k": DeploymentPreset(
        checkpoint_dir=REPO_ROOT
        / "results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_dexhand_full_pytorch/"
        "pi05_pickplace_dexhand_eef_delta_train_full/120000",
        train_config="pi05_pickplace_dexhand_full_pytorch",
        assets_dir=REPO_ROOT / "openpi_official/assets_eef_delta_v2",
        control_mode="delta_eef",
    ),
    "joint52k": DeploymentPreset(
        checkpoint_dir=REPO_ROOT
        / "results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_ur5e_joint_delta_full_pytorch/"
        "pi05_pickplace_ur5e_joint_delta_train_full/52000",
        train_config="pi05_pickplace_ur5e_joint_delta_full_pytorch",
        assets_dir=REPO_ROOT / "openpi_official/assets_ur5e_joint_delta",
        control_mode="joint_position",
    ),
}


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class UR5eRTDE:
    def __init__(self, robot_ip: str, acceleration: float = 0.1, speed: float = 0.1, servo_dt: float = 1 / 500):
        import rtde_control
        import rtde_receive

        self.robot_ip = str(robot_ip)
        self.acceleration = float(acceleration)
        self.speed = float(speed)
        self.servo_dt = float(servo_dt)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)

    def get_pos_j(self) -> np.ndarray:
        return np.asarray(self.rtde_r.getActualQ(), dtype=np.float32)

    def get_pos_eef(self) -> np.ndarray:
        return np.asarray(self.rtde_r.getActualTCPPose(), dtype=np.float32)

    def set_pos_j(self, target_qpos, servo: bool = True):
        target_qpos = np.asarray(target_qpos, dtype=np.float64).tolist()
        if servo:
            self.rtde_c.servoJ(target_qpos, self.speed, self.acceleration, self.servo_dt, 0.1, 300)
        else:
            self.rtde_c.moveJ(target_qpos, self.speed, self.acceleration)

    def set_pos_eef(self, target_tcp_pose, servo: bool = True):
        target_tcp_pose = np.asarray(target_tcp_pose, dtype=np.float64).reshape(6).tolist()
        if servo:
            self.rtde_c.servoL(target_tcp_pose, self.speed, self.acceleration, self.servo_dt, 0.1, 300)
        else:
            self.rtde_c.moveL(target_tcp_pose, self.speed, self.acceleration)

    def inverse_kinematics(self, target_tcp_pose, current_joints):
        target_tcp_pose = np.asarray(target_tcp_pose, dtype=np.float64).reshape(6).tolist()
        current_joints = np.asarray(current_joints, dtype=np.float64).reshape(6).tolist()
        joints = self.rtde_c.getInverseKinematics(target_tcp_pose, current_joints)
        if joints is None:
            return None
        return np.asarray(joints, dtype=np.float32)

    def speed_eef_delta(self, delta_tcp_pose):
        delta_tcp_pose = np.asarray(delta_tcp_pose, dtype=np.float64).reshape(6)
        tcp_speed = delta_tcp_pose / max(self.servo_dt, 1e-6)
        tcp_speed = np.clip(tcp_speed, -self.speed, self.speed).tolist()
        self.rtde_c.speedL(tcp_speed, self.acceleration, self.servo_dt)

    def stop(self):
        try:
            self.rtde_c.speedStop()
        except Exception:
            pass
        try:
            self.rtde_c.servoStop()
        except Exception:
            pass
        try:
            self.rtde_c.stopScript()
        except Exception:
            pass


class InspireHandSerial:
    REGDICT = {"posSet": 1474, "posAct": 1534}

    def __init__(self, port: str, baudrate: int = 115200, hand_id: int = 1):
        import serial

        self.serial_mod = serial
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.hand_id = int(hand_id)
        self.ser = None
        self._last_pos = np.zeros(6, dtype=np.float32)

    def open(self):
        self.ser = self.serial_mod.Serial()
        self.ser.port = self.port
        self.ser.baudrate = self.baudrate
        self.ser.open()

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

    def _write_register(self, add: int, num: int, values):
        packet = [0xEB, 0x90, self.hand_id, num + 3, 0x12, add & 0xFF, (add >> 8) & 0xFF]
        packet.extend(int(v) & 0xFF for v in values)
        checksum = sum(packet[2:]) & 0xFF
        packet.append(checksum)
        self.ser.write(packet)
        time.sleep(0.01)
        self.ser.read_all()

    def _read_register(self, add: int, num: int):
        packet = [0xEB, 0x90, self.hand_id, 0x04, 0x11, add & 0xFF, (add >> 8) & 0xFF, num]
        checksum = sum(packet[2:]) & 0xFF
        packet.append(checksum)
        for _ in range(3):
            self.ser.write(packet)
            time.sleep(0.01)
            recv = self.ser.read_all()
            if len(recv) >= 7:
                data_len = max(0, (recv[3] & 0xFF) - 3)
                if len(recv) >= 7 + data_len:
                    return [recv[7 + i] for i in range(data_len)]
        return []

    def set_hand_pos(self, value):
        payload = []
        for item in value:
            item = int(np.clip(item, 0, 2000))
            payload.append(item & 0xFF)
            payload.append((item >> 8) & 0xFF)
        self._write_register(self.REGDICT["posSet"], 12, payload)

    def get_hand_pos(self) -> np.ndarray:
        raw = self._read_register(self.REGDICT["posAct"], 12)
        if len(raw) < 12:
            return self._last_pos.copy()
        vals = np.asarray([int((raw[2 * i] & 0xFF) + (raw[2 * i + 1] << 8)) for i in range(6)], dtype=np.float32)
        self._last_pos = vals.copy()
        return vals


class L515ColorCamera:
    def __init__(self):
        try:
            import pyrealsense2.pyrealsense2 as rs
        except Exception:
            import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        for _ in range(30):
            try:
                self.pipeline.wait_for_frames()
            except Exception:
                pass

    def get_data(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        if frame is None:
            raise RuntimeError("Failed to read L515 color frame.")
        return np.asanyarray(frame.get_data())

    def close(self):
        self.pipeline.stop()


class OrbbecFemtoBoltColorCamera:
    def __init__(self):
        import pyorbbecsdk as sdk

        self.sdk = sdk
        self.config = sdk.Config()
        self.pipeline = sdk.Pipeline()
        profile_list = self.pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(1280, 720, sdk.OBFormat.BGR, 30)
        self.config.enable_stream(color_profile)
        self.pipeline.start(self.config)

    def get_data(self) -> np.ndarray:
        while True:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            frame = frames.get_color_frame()
            if frame is None:
                continue
            width = frame.get_width()
            height = frame.get_height()
            data = np.asanyarray(frame.get_data())
            return data.reshape((height, width, 3))

    def close(self):
        self.pipeline.stop()


class MockColorCamera:
    def __init__(self, height: int = 480, width: int = 640, value: int = 0):
        self.height = int(height)
        self.width = int(width)
        self.value = int(value)

    def get_data(self) -> np.ndarray:
        return np.full((self.height, self.width, 3), self.value, dtype=np.uint8)

    def close(self):
        pass


class CameraWorker:
    def __init__(self, camera, logger: logging.Logger, name: str, fallback_shape=(480, 640, 3)):
        self.camera = camera
        self.logger = logger
        self.name = name
        self.fallback_shape = tuple(fallback_shape)
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"{self.name}_camera_worker")
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                frame = self.camera.get_data()
                if frame is not None:
                    with self._lock:
                        self._latest = np.asarray(frame).copy()
            except Exception as exc:
                self.logger.warning("%s capture failed: %s", self.name, exc)
                time.sleep(0.05)

    def get_data(self) -> np.ndarray:
        with self._lock:
            if self._latest is not None:
                return self._latest.copy()
        return np.zeros(self.fallback_shape, dtype=np.uint8)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.camera.close()
        except Exception:
            pass


class BridgeCommandColorCamera:
    def __init__(self, command: list[str], image_path: str):
        self.command = [str(x) for x in command]
        self.image_path = pathlib.Path(image_path)
        self._last_image: np.ndarray | None = None

    def get_data(self) -> np.ndarray:
        self.image_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                self.command + ["--output", str(self.image_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        except Exception:
            image = None
        if image is None and self.image_path.exists():
            image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if image is None and self._last_image is not None:
            return self._last_image.copy()
        if image is None:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        self._last_image = image.copy()
        return image

    def close(self):
        pass


class LowPassFilter:
    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.filtered: np.ndarray | None = None

    def reset(self):
        self.filtered = None

    def filter(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if self.filtered is None:
            self.filtered = value.copy()
        else:
            self.filtered = self.alpha * value + (1.0 - self.alpha) * self.filtered
        return self.filtered.copy()


class ArmJointServoWorker:
    def __init__(
        self,
        robot: UR5eRTDE,
        logger: logging.Logger,
        *,
        output_hz: float = 50.0,
        interp_steps: int = 5,
        lpf_alpha: float = 0.1,
    ):
        self.robot = robot
        self.logger = logger
        self.output_hz = float(output_hz)
        self.interp_steps = max(1, int(interp_steps))
        self.lpf = LowPassFilter(alpha=lpf_alpha)
        self._target: np.ndarray | None = None
        self._target_version = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name="arm_joint_servo_worker")
        self._thread.start()

    def set_target(self, target_joints: np.ndarray):
        target_joints = np.asarray(target_joints, dtype=np.float32).reshape(6)
        with self._lock:
            self._target = target_joints.copy()
            self._target_version += 1

    def _snapshot_target(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self._target is None:
                return None, self._target_version
            return self._target.copy(), self._target_version

    def _interpolate(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        alphas = np.linspace(1.0 / self.interp_steps, 1.0, self.interp_steps, dtype=np.float32)
        return np.asarray([(1.0 - a) * start + a * target for a in alphas], dtype=np.float32)

    def _loop(self):
        dt = 1.0 / max(self.output_hz, 1e-6)
        active_version = -1
        hold_warn_time = 0.0
        while not self._stop.is_set():
            target, version = self._snapshot_target()
            if target is None:
                time.sleep(dt)
                continue

            if version != active_version:
                try:
                    current = self.robot.get_pos_j().astype(np.float32)
                    trajectory = self._interpolate(current, target)
                except Exception as exc:
                    now = time.monotonic()
                    if now - hold_warn_time > 1.0:
                        self.logger.warning("Failed to prepare arm joint trajectory: %s", exc)
                        hold_warn_time = now
                    time.sleep(dt)
                    continue
                active_version = version

                for q in trajectory:
                    if self._stop.is_set():
                        break
                    _, latest_version = self._snapshot_target()
                    if latest_version != active_version:
                        break
                    self.robot.set_pos_j(self.lpf.filter(q), servo=True)
                    time.sleep(dt)
            else:
                try:
                    self.robot.set_pos_j(self.lpf.filter(target), servo=True)
                except Exception as exc:
                    now = time.monotonic()
                    if now - hold_warn_time > 1.0:
                        self.logger.warning("Failed to hold arm joint target: %s", exc)
                        hold_warn_time = now
                time.sleep(dt)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def normalize_hand_counts(hand_counts: np.ndarray, max_hand_abs: float = HAND_RANGE[1]) -> np.ndarray:
    hand_counts = np.asarray(hand_counts, dtype=np.float32)
    hand_counts = np.clip(hand_counts, HAND_RANGE[0], float(max_hand_abs))
    return -1.0 + 2.0 * (hand_counts - HAND_RANGE[0]) / (float(max_hand_abs) - HAND_RANGE[0])


def denormalize_hand_action(hand_action: np.ndarray, max_hand_abs: float = HAND_RANGE[1]) -> np.ndarray:
    hand_action = np.asarray(hand_action, dtype=np.float32)
    hand_action = np.clip(hand_action, -1.0, 1.0)
    return (hand_action + 1.0) * 0.5 * (float(max_hand_abs) - HAND_RANGE[0]) + HAND_RANGE[0]


def build_policy_state(
    joints: np.ndarray,
    eef: np.ndarray,
    hand_counts: np.ndarray,
    *,
    max_hand_abs: float = HAND_RANGE[1],
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(joints, dtype=np.float32).reshape(6),
            np.asarray(eef, dtype=np.float32).reshape(6),
            normalize_hand_counts(hand_counts, max_hand_abs=max_hand_abs).reshape(6),
        ],
        axis=0,
    ).astype(np.float32)


def build_joint_policy_state(joints: np.ndarray, hand_counts: np.ndarray) -> np.ndarray:
    """Build the 12-D state used by the joint-target training dataset."""
    return np.concatenate(
        [
            np.asarray(joints, dtype=np.float32).reshape(6),
            np.asarray(hand_counts, dtype=np.float32).reshape(6),
        ],
        axis=0,
    ).astype(np.float32)


def build_policy_observation(
    l515_rgb: np.ndarray,
    orbbec_rgb: np.ndarray,
    state: np.ndarray,
    prompt: str,
) -> dict:
    return {
        "observation/image": l515_rgb,
        "observation/wrist_image": orbbec_rgb,
        "observation/state": state,
        "prompt": prompt,
    }


def split_policy_action(policy_output: dict) -> tuple[np.ndarray, np.ndarray]:
    arm_actions, hand_actions = split_policy_actions(policy_output)
    return arm_actions[0], hand_actions[0]


def split_policy_actions(policy_output: dict) -> tuple[np.ndarray, np.ndarray]:
    if "actions" not in policy_output and "actions_full" not in policy_output:
        raise KeyError("policy_output must contain 'actions' or 'actions_full'")
    action_key = "actions" if "actions" in policy_output else "actions_full"
    action = np.asarray(policy_output[action_key], dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    action = action.reshape(action.shape[0], -1)

    if action_key == "actions_full" and action.shape[1] >= 38:
        return action[:, :6].astype(np.float32), action[:, 32:38].astype(np.float32)
    if action.shape[1] >= 12:
        return action[:, :6].astype(np.float32), action[:, 6:12].astype(np.float32)
    raise ValueError(f"Expected at least 12-D pick-place action, got {action.shape}.")


def parse_joint_limits(value: str) -> tuple[float, float, float, float, float, float]:
    parts = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("joint limits must contain 6 comma-separated values")
    return tuple(float(item) for item in parts)  # type: ignore[return-value]


def validate_control_command_pair(control_mode: str, arm_command_mode: str) -> None:
    if control_mode == "joint_position" and arm_command_mode != "joint":
        raise ValueError("control_mode=joint_position requires --arm-command-mode=joint (or auto)")
    if control_mode != "joint_position" and arm_command_mode == "joint":
        raise ValueError("--arm-command-mode=joint is incompatible with EEF control modes")


def resolve_arm_command_mode(control_mode: str, arm_command_mode: str) -> str:
    resolved = (
        ("joint" if control_mode == "joint_position" else "ik")
        if arm_command_mode == "auto"
        else arm_command_mode
    )
    validate_control_command_pair(control_mode, resolved)
    return resolved


def validate_and_bound_joint_target(
    target: np.ndarray,
    current: np.ndarray,
    lower: np.ndarray | tuple[float, ...],
    upper: np.ndarray | tuple[float, ...],
    max_step: float,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).reshape(6)
    current = np.asarray(current, dtype=np.float32).reshape(6)
    lower = np.asarray(lower, dtype=np.float32).reshape(6)
    upper = np.asarray(upper, dtype=np.float32).reshape(6)
    if not all(np.isfinite(value).all() for value in (target, current, lower, upper)):
        raise ValueError("Joint target, current position, and limits must be finite")
    if np.any(lower >= upper):
        raise ValueError("Each joint lower limit must be below its upper limit")
    if np.any(target < lower) or np.any(target > upper):
        raise ValueError(
            "Predicted absolute joint target exceeds configured limits: "
            f"target={target.tolist()} lower={lower.tolist()} upper={upper.tolist()}"
        )
    if max_step <= 0 or not np.isfinite(max_step):
        raise ValueError("max_joint_step must be finite and positive")
    return current + np.clip(target - current, -float(max_step), float(max_step))


def validate_safetensors_checkpoint(checkpoint_dir: pathlib.Path) -> None:
    weight_path = checkpoint_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"Missing PyTorch checkpoint file: {weight_path}")
    try:
        from safetensors import safe_open

        with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
            next(iter(handle.keys()))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot open {weight_path}. The file is likely incomplete or corrupted; "
            "use a checkpoint directory with a valid model.safetensors."
        ) from exc


def load_policy_norm_stats(train_cfg, checkpoint_dir: pathlib.Path, assets_dir: str | None):
    data_config = train_cfg.data.create_base_config(train_cfg.assets_dirs, train_cfg.model)
    if data_config.asset_id is None:
        return None

    candidates: list[pathlib.Path] = []
    if assets_dir:
        candidates.append(pathlib.Path(assets_dir))
    candidates.append(checkpoint_dir / "assets")
    candidates.append(checkpoint_dir.parent / "assets")

    seen: set[pathlib.Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        norm_file = candidate / data_config.asset_id / "norm_stats.json"
        if norm_file.is_file():
            return checkpoints_lib.load_norm_stats(candidate, data_config.asset_id)
    raise FileNotFoundError(
        f"Could not find norm_stats.json for asset_id={data_config.asset_id!r}. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
    )


@dataclass
class RobotConfig:
    robot_ip: str = "192.168.1.109"
    hand_port: str = "/dev/ttyUSB0"
    control_hz: float = 10.0
    arm_speed: float = 0.1
    arm_acceleration: float = 0.1
    arm_servo_hz: float = 50.0
    arm_interp_steps: int = 5
    arm_lpf_alpha: float = 0.1
    convert_bgr_to_rgb: bool = True
    mock_cameras: bool = False
    bridge_l515: bool = False
    arm_command_mode: str = "auto"
    control_mode: str = "delta_eef"
    eef_delta_scale: float = 1.0
    eef_rot_scale: float = 1.0
    max_eef_delta: float = 0.03
    max_joint_step: float = 0.05
    joint_lower_limits: tuple[float, float, float, float, float, float] = (
        -4.158066,
        -2.743676,
        -2.896363,
        2.991765,
        -2.922274,
        2.274007,
    )
    joint_upper_limits: tuple[float, float, float, float, float, float] = (
        -1.734541,
        -1.007369,
        -0.763186,
        5.433079,
        -0.134594,
        3.806125,
    )
    max_hand_abs: float = 2000.0
    action_ema_alpha: float = 0.25
    arm_ema_alpha: float | None = None
    hand_ema_alpha: float | None = None
    max_hand_step: float = 150.0
    action_chunk_size: int = 3
    policy_noise_seed: int = 0
    save_obs_dir: str | None = "./pi05_obs"
    save_obs_every: int = 1
    record_video: bool = False
    record_dir: str = "real_robot_demos"
    record_fps: float = 10.0
    max_steps: int = 0


class PI05RealRobotRunner:
    def __init__(
        self,
        checkpoint_dir: str,
        train_config_name: str,
        prompt: str,
        robot_cfg: RobotConfig,
        assets_dir: str | None = None,
    ):
        self.logger = _build_logger(self.__class__.__name__)
        self.prompt = prompt
        self.robot_cfg = robot_cfg
        self.obs_step = 0
        self.dt = 1.0 / robot_cfg.control_hz
        self.save_obs_dir = pathlib.Path(robot_cfg.save_obs_dir) if robot_cfg.save_obs_dir else None
        if self.save_obs_dir is not None:
            self.save_obs_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = pathlib.Path(checkpoint_dir).resolve()
        validate_safetensors_checkpoint(checkpoint_path)
        train_cfg = train_config_lib._CONFIGS_DICT[train_config_name]
        if getattr(train_cfg.model, "pytorch_compile_mode", None) is not None:
            train_cfg = dataclasses.replace(
                train_cfg,
                model=dataclasses.replace(train_cfg.model, pytorch_compile_mode=None),
            )
            self.logger.info("Disabled torch.compile for real-robot inference.")
        self.action_horizon = int(train_cfg.model.action_horizon)
        self.action_dim = int(train_cfg.model.action_dim)
        norm_stats = load_policy_norm_stats(train_cfg, checkpoint_path, assets_dir)
        self.control_mode = getattr(train_cfg.model, "control_mode", robot_cfg.control_mode)
        if self.control_mode != robot_cfg.control_mode:
            raise ValueError(
                f"Deployment preset expects control_mode={robot_cfg.control_mode!r}, but train config "
                f"{train_config_name!r} uses {self.control_mode!r}. Refusing to run the wrong action space."
            )
        robot_cfg.arm_command_mode = resolve_arm_command_mode(self.control_mode, robot_cfg.arm_command_mode)
        self.policy = policy_config_lib.create_trained_policy(
            train_cfg,
            checkpoint_path,
            default_prompt=prompt,
            norm_stats=norm_stats,
        )

        self.robot = UR5eRTDE(
            robot_cfg.robot_ip,
            acceleration=robot_cfg.arm_acceleration,
            speed=robot_cfg.arm_speed,
            servo_dt=1.0 / max(robot_cfg.arm_servo_hz, 1e-6),
        )
        self.arm_servo_worker = ArmJointServoWorker(
            self.robot,
            self.logger,
            output_hz=robot_cfg.arm_servo_hz,
            interp_steps=robot_cfg.arm_interp_steps,
            lpf_alpha=robot_cfg.arm_lpf_alpha,
        )
        self.arm_servo_worker.start()
        self.hand = InspireHandSerial(robot_cfg.hand_port)
        self.hand.open()
        if robot_cfg.mock_cameras:
            self.logger.warning("Using mock cameras. Real image devices are disabled.")
            l515_camera = MockColorCamera(value=64)
            orbbec_camera = MockColorCamera(value=128)
        else:
            orbbec_camera = OrbbecFemtoBoltColorCamera()
            if robot_cfg.bridge_l515:
                self.logger.warning("Using bridged L515 capture command.")
                l515_camera = self._make_l515_bridge()
            else:
                try:
                    l515_camera = L515ColorCamera()
                except Exception as exc:
                    self.logger.warning("Direct L515 open failed: %s", exc)
                    self.logger.warning("Falling back to bridged L515 capture command.")
                    l515_camera = self._make_l515_bridge()

        self.l515_camera = CameraWorker(l515_camera, self.logger, "l515")
        self.orbbec_camera = CameraWorker(orbbec_camera, self.logger, "orbbec")
        self.l515_camera.start()
        self.orbbec_camera.start()

        self.logger.info(
            "Camera mapping: L515 -> observation/image -> base_0_rgb; "
            "Orbbec Femto Bolt -> observation/wrist_image -> left_wrist_0_rgb"
        )
        if self.control_mode == "joint_position":
            self.logger.info(
                "Action mapping: arm=action[0:6] absolute joints [rad] after AbsoluteActions; "
                "hand=action[6:12] absolute raw counts"
            )
        else:
            self.logger.info(
                "Action mapping: arm=action[0:6] EEF delta with original signs; "
                "hand=action[6:12] normalized absolute hand counts"
            )
        self.logger.info("Arm command mode: %s", robot_cfg.arm_command_mode)
        self.logger.info(
            "Arm servo worker: hz=%.1f interp_steps=%d lpf_alpha=%.3f",
            robot_cfg.arm_servo_hz,
            robot_cfg.arm_interp_steps,
            robot_cfg.arm_lpf_alpha,
        )
        self.logger.info(
            "Temporal smoothing: action_chunk_size=%d policy_noise_seed=%d action_ema_alpha=%.3f max_hand_step=%.1f",
            robot_cfg.action_chunk_size,
            robot_cfg.policy_noise_seed,
            robot_cfg.action_ema_alpha,
            robot_cfg.max_hand_step,
        )
        self.logger.info("Arm sign convention: unchanged on all 6 dimensions (no last-dimension reversal)")
        self.prev_arm_cmd: np.ndarray | None = None
        self.prev_hand_cmd: np.ndarray | None = None
        self.action_queue: list[tuple[np.ndarray, np.ndarray]] = []
        self.policy_noise = self._make_policy_noise(robot_cfg.policy_noise_seed)
        self.record_dir = pathlib.Path(robot_cfg.record_dir)
        self.video_writers: dict[str, cv2.VideoWriter] = {}
        self.record_frame_idx = 0

    def _make_policy_noise(self, seed: int) -> np.ndarray | None:
        if seed < 0:
            return None
        rng = np.random.default_rng(seed)
        return rng.standard_normal((self.action_horizon, self.action_dim), dtype=np.float32)

    def _make_l515_bridge(self) -> BridgeCommandColorCamera:
        return BridgeCommandColorCamera(
            [
                sys.executable,
                str(pathlib.Path(__file__).resolve().parent / "grab_l515_frame.py"),
            ],
            "/tmp/pi05_l515_bridge.png",
        )

    def _make_video_writer(self, path: pathlib.Path, frame_shape: tuple[int, int, int], fps: float) -> cv2.VideoWriter:
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {path}")
        return writer

    def _ensure_video_writers(self, l515_frame: np.ndarray, orbbec_frame: np.ndarray):
        if self.video_writers or not self.robot_cfg.record_video:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.record_dir / timestamp
        self.video_writers["l515"] = self._make_video_writer(
            run_dir / "l515_observation_image.mp4", l515_frame.shape, self.robot_cfg.record_fps
        )
        self.video_writers["orbbec"] = self._make_video_writer(
            run_dir / "orbbec_observation_wrist_image.mp4", orbbec_frame.shape, self.robot_cfg.record_fps
        )
        combo = np.concatenate([l515_frame, cv2.resize(orbbec_frame, (l515_frame.shape[1], l515_frame.shape[0]))], axis=1)
        self.video_writers["combined"] = self._make_video_writer(run_dir / "combined.mp4", combo.shape, self.robot_cfg.record_fps)
        self.logger.info("Recording demo videos under %s", run_dir)

    def _write_video_frames(self, l515_frame: np.ndarray, orbbec_frame: np.ndarray):
        if not self.robot_cfg.record_video:
            return
        self._ensure_video_writers(l515_frame, orbbec_frame)
        combo = np.concatenate([l515_frame, cv2.resize(orbbec_frame, (l515_frame.shape[1], l515_frame.shape[0]))], axis=1)
        self.video_writers["l515"].write(l515_frame)
        self.video_writers["orbbec"].write(orbbec_frame)
        self.video_writers["combined"].write(combo)
        self.record_frame_idx += 1

    def _resize_rgb(self, image: np.ndarray, shape=(224, 224)) -> np.ndarray:
        image = np.asarray(image)
        if self.robot_cfg.convert_bgr_to_rgb:
            image = image[..., ::-1]
        return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

    def _save_obs_images(self, l515_bgr: np.ndarray, orbbec_bgr: np.ndarray) -> None:
        if self.save_obs_dir is None:
            return
        if self.obs_step % max(1, self.robot_cfg.save_obs_every) != 0:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(str(self.save_obs_dir / f"step_{self.obs_step:06d}_{stamp}_l515_observation_image.png"), l515_bgr)
        cv2.imwrite(str(self.save_obs_dir / f"step_{self.obs_step:06d}_{stamp}_orbbec_observation_wrist_image.png"), orbbec_bgr)

    def _get_obs(self):
        joints = self.robot.get_pos_j().astype(np.float32)
        eef = self.robot.get_pos_eef().astype(np.float32)
        hand_counts = self.hand.get_hand_pos().astype(np.float32)
        l515_frame = self.l515_camera.get_data()
        orbbec_frame = self.orbbec_camera.get_data()
        self._save_obs_images(l515_frame, orbbec_frame)
        self._write_video_frames(l515_frame, orbbec_frame)
        if self.control_mode == "joint_position":
            state = build_joint_policy_state(joints, hand_counts)
        else:
            state = build_policy_state(joints, eef, hand_counts, max_hand_abs=self.robot_cfg.max_hand_abs)
        obs = build_policy_observation(
            self._resize_rgb(l515_frame),
            self._resize_rgb(orbbec_frame),
            state,
            self.prompt,
        )
        return obs, joints, eef, hand_counts

    def _ema(self, current: np.ndarray, previous: np.ndarray | None, alpha: float | None = None) -> np.ndarray:
        if previous is None:
            return current.copy()
        if alpha is None:
            alpha = self.robot_cfg.action_ema_alpha
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return alpha * current + (1.0 - alpha) * previous

    def _apply_action(
        self,
        arm_cmd: np.ndarray,
        hand_cmd: np.ndarray,
        current_joints: np.ndarray,
        current_eef: np.ndarray,
        current_hand: np.ndarray,
    ):
        del current_hand
        arm_cmd = np.asarray(arm_cmd, dtype=np.float32).reshape(6)
        hand_cmd = np.asarray(hand_cmd, dtype=np.float32).reshape(6)

        if self.control_mode == "joint_position":
            arm_target = validate_and_bound_joint_target(
                arm_cmd,
                current_joints,
                self.robot_cfg.joint_lower_limits,
                self.robot_cfg.joint_upper_limits,
                self.robot_cfg.max_joint_step,
            )
            self.prev_arm_cmd = self._ema(arm_target, self.prev_arm_cmd, self.robot_cfg.arm_ema_alpha)
            arm_to_send = self.prev_arm_cmd
            arm_log = f"joint_abs={np.array2string(arm_cmd, precision=4, floatmode='fixed')}"
        elif self.control_mode == "delta_eef":
            arm_delta = np.zeros(6, dtype=np.float32)
            arm_delta[:3] = np.clip(
                arm_cmd[:3] * float(self.robot_cfg.eef_delta_scale),
                -float(self.robot_cfg.max_eef_delta),
                float(self.robot_cfg.max_eef_delta),
            )
            arm_delta[3:6] = np.clip(
                arm_cmd[3:6] * float(self.robot_cfg.eef_delta_scale) * float(self.robot_cfg.eef_rot_scale),
                -float(self.robot_cfg.max_eef_delta),
                float(self.robot_cfg.max_eef_delta),
            )
            arm_target = current_eef + arm_delta
            self.prev_arm_cmd = self._ema(arm_target, self.prev_arm_cmd, self.robot_cfg.arm_ema_alpha)
            arm_to_send = self.prev_arm_cmd
            arm_log = f"eef_delta={np.array2string(arm_delta, precision=4, floatmode='fixed')}"
        else:
            self.prev_arm_cmd = self._ema(arm_cmd, self.prev_arm_cmd, self.robot_cfg.arm_ema_alpha)
            arm_to_send = self.prev_arm_cmd
            arm_log = f"eef_abs={np.array2string(arm_to_send, precision=4, floatmode='fixed')}"

        if self.control_mode == "joint_position":
            if not np.isfinite(hand_cmd).all():
                raise ValueError("Absolute hand action must be finite")
            hand_abs = np.clip(hand_cmd, HAND_RANGE[0], self.robot_cfg.max_hand_abs)
        else:
            hand_abs = denormalize_hand_action(hand_cmd, max_hand_abs=self.robot_cfg.max_hand_abs)
        if self.prev_hand_cmd is not None and self.robot_cfg.max_hand_step > 0:
            hand_abs = self.prev_hand_cmd + np.clip(
                hand_abs - self.prev_hand_cmd,
                -float(self.robot_cfg.max_hand_step),
                float(self.robot_cfg.max_hand_step),
            )
        self.prev_hand_cmd = self._ema(hand_abs, self.prev_hand_cmd, self.robot_cfg.hand_ema_alpha)
        hand_to_send = self.prev_hand_cmd

        arm_command_mode = self.robot_cfg.arm_command_mode
        self.logger.info(
            "%s arm_target=%s arm_mode=%s hand_policy=%s hand_abs=%s",
            arm_log,
            np.array2string(arm_to_send, precision=4, floatmode="fixed"),
            arm_command_mode,
            np.array2string(hand_cmd, precision=3, floatmode="fixed"),
            np.array2string(hand_to_send, precision=1, floatmode="fixed"),
        )
        if arm_command_mode == "joint":
            self.arm_servo_worker.set_target(arm_to_send)
        elif arm_command_mode == "ik":
            # The EEF training target is the component-wise difference of two
            # [xyz, rotation-vector] poses, so all six dimensions are added
            # component-wise here as well. Do not flip or re-compose dimension 6.
            ik_target_eef = arm_to_send
            target_joints = self.robot.inverse_kinematics(ik_target_eef, current_joints)
            if target_joints is None:
                self.logger.warning("IK failed for target_eef=%s", np.array2string(ik_target_eef, precision=4))
            else:
                self.arm_servo_worker.set_target(target_joints)
        elif arm_command_mode == "speed":
            self.robot.speed_eef_delta(arm_to_send - current_eef)
        elif arm_command_mode == "servo":
            self.robot.set_pos_eef(arm_to_send, servo=True)
        else:
            self.robot.set_pos_eef(arm_to_send, servo=False)
        try:
            if self.control_mode == "joint_position":
                actual_joints = self.robot.get_pos_j().astype(np.float32)
                self.logger.info(
                    "actual_joint_delta=%s target_error=%s",
                    np.array2string(actual_joints - current_joints, precision=4, floatmode="fixed"),
                    np.array2string(arm_to_send - actual_joints, precision=4, floatmode="fixed"),
                )
            else:
                actual_eef = self.robot.get_pos_eef().astype(np.float32)
                self.logger.info(
                    "actual_eef_delta=%s target_error=%s",
                    np.array2string(actual_eef - current_eef, precision=4, floatmode="fixed"),
                    np.array2string(arm_to_send - actual_eef, precision=4, floatmode="fixed"),
                )
        except Exception as exc:
            self.logger.warning("Failed to read arm state after command: %s", exc)
        self.hand.set_hand_pos(np.rint(np.clip(hand_to_send, 0, self.robot_cfg.max_hand_abs)).astype(np.int32).tolist())

    def run(self):
        self.logger.info("Press Ctrl+C to stop. Starting real-robot rollout loop.")
        step_idx = 0
        try:
            while True:
                loop_start = time.monotonic()
                if self.robot_cfg.max_steps > 0 and step_idx >= self.robot_cfg.max_steps:
                    self.logger.info("Reached max_steps=%d, stopping rollout.", self.robot_cfg.max_steps)
                    break
                self.obs_step += 1
                obs, joints, eef, hand = self._get_obs()
                if not self.action_queue:
                    policy_output = self.policy.infer(obs, noise=self.policy_noise)
                    self.logger.info("policy_timing=%s", policy_output.get("policy_timing"))
                    arm_actions, hand_actions = split_policy_actions(policy_output)
                    chunk_size = max(1, min(int(self.robot_cfg.action_chunk_size), arm_actions.shape[0]))
                    self.action_queue = [
                        (arm_actions[i].copy(), hand_actions[i].copy())
                        for i in range(chunk_size)
                    ]
                    self.logger.info("queued %d/%d policy actions", chunk_size, arm_actions.shape[0])
                arm_cmd, hand_cmd = self.action_queue.pop(0)
                self._apply_action(arm_cmd, hand_cmd, joints, eef, hand)
                step_idx += 1
                sleep_s = self.dt - (time.monotonic() - loop_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        finally:
            try:
                self.arm_servo_worker.close()
            except Exception:
                pass
            try:
                self.robot.stop()
            except Exception:
                pass
            for dev in (self.hand, self.l515_camera, self.orbbec_camera):
                try:
                    dev.close()
                except Exception:
                    pass
            for writer in self.video_writers.values():
                try:
                    writer.release()
                except Exception:
                    pass
            if self.video_writers:
                self.logger.info("Saved %d recorded frames to %s", self.record_frame_idx, self.record_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Deploy the PyTorch EEF-delta 120k or joint-delta 52k PI05 checkpoint on the real robot.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model",
        choices=sorted(DEPLOYMENT_PRESETS),
        required=True,
        help="eef120k: EEF-delta 120k; joint52k: joint-delta 52k. This also selects matching config/assets.",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--robot-ip", default="192.168.1.109")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--arm-speed", type=float, default=0.1)
    parser.add_argument("--arm-acceleration", type=float, default=0.1)
    parser.add_argument("--arm-servo-hz", type=float, default=50.0)
    parser.add_argument("--arm-interp-steps", type=int, default=5)
    parser.add_argument("--arm-lpf-alpha", type=float, default=0.1)
    parser.add_argument("--mock-cameras", action="store_true")
    parser.add_argument("--bridge-l515", action="store_true")
    parser.add_argument(
        "--arm-command-mode",
        choices=["auto", "joint", "ik", "speed", "servo", "move"],
        default="auto",
        help="auto selects IK for eef120k and direct joint servo for joint52k.",
    )
    parser.add_argument("--eef-delta-scale", type=float, default=1.0)
    parser.add_argument("--eef-rot-scale", type=float, default=1.0)
    parser.add_argument("--max-eef-delta", type=float, default=0.03)
    parser.add_argument("--max-joint-step", type=float, default=0.05)
    parser.add_argument(
        "--joint-lower-limits",
        type=parse_joint_limits,
        default=parse_joint_limits("-4.158066,-2.743676,-2.896363,2.991765,-2.922274,2.274007"),
    )
    parser.add_argument(
        "--joint-upper-limits",
        type=parse_joint_limits,
        default=parse_joint_limits("-1.734541,-1.007369,-0.763186,5.433079,-0.134594,3.806125"),
    )
    parser.add_argument("--max-hand-abs", type=float, default=2000.0)
    parser.add_argument("--action-ema-alpha", type=float, default=0.25)
    parser.add_argument("--arm-ema-alpha", type=float, default=None)
    parser.add_argument("--hand-ema-alpha", type=float, default=None)
    parser.add_argument("--max-hand-step", type=float, default=150.0)
    parser.add_argument("--action-chunk-size", type=int, default=3)
    parser.add_argument("--policy-noise-seed", type=int, default=0)
    parser.add_argument("--convert-bgr-to-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-obs-dir", default="./pi05_obs")
    parser.add_argument("--save-obs-every", type=int, default=1)
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record-dir", default="real_robot_demos")
    parser.add_argument("--record-fps", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    preset = DEPLOYMENT_PRESETS[args.model]

    cfg = RobotConfig(
        robot_ip=args.robot_ip,
        hand_port=args.hand_port,
        control_hz=args.control_hz,
        arm_speed=args.arm_speed,
        arm_acceleration=args.arm_acceleration,
        arm_servo_hz=args.arm_servo_hz,
        arm_interp_steps=args.arm_interp_steps,
        arm_lpf_alpha=args.arm_lpf_alpha,
        mock_cameras=args.mock_cameras,
        bridge_l515=args.bridge_l515,
        arm_command_mode=args.arm_command_mode,
        control_mode=preset.control_mode,
        convert_bgr_to_rgb=args.convert_bgr_to_rgb,
        eef_delta_scale=args.eef_delta_scale,
        eef_rot_scale=args.eef_rot_scale,
        max_eef_delta=args.max_eef_delta,
        max_joint_step=args.max_joint_step,
        joint_lower_limits=args.joint_lower_limits,
        joint_upper_limits=args.joint_upper_limits,
        max_hand_abs=args.max_hand_abs,
        action_ema_alpha=args.action_ema_alpha,
        arm_ema_alpha=args.arm_ema_alpha,
        hand_ema_alpha=args.hand_ema_alpha,
        max_hand_step=args.max_hand_step,
        action_chunk_size=args.action_chunk_size,
        policy_noise_seed=args.policy_noise_seed,
        save_obs_dir=args.save_obs_dir,
        save_obs_every=args.save_obs_every,
        record_video=args.record_video,
        record_dir=args.record_dir,
        record_fps=args.record_fps,
        max_steps=args.max_steps,
    )
    logging.info(
        "Selected model=%s checkpoint=%s train_config=%s assets=%s",
        args.model,
        preset.checkpoint_dir,
        preset.train_config,
        preset.assets_dir,
    )
    runner = PI05RealRobotRunner(
        str(preset.checkpoint_dir),
        preset.train_config,
        args.prompt,
        cfg,
        assets_dir=str(preset.assets_dir),
    )
    runner.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
