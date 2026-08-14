"""Evaluate PI 0.5 policies on DexJoCo simulation environments."""

import multiprocessing as mp
import os
import random
import signal
import time
import json
import hashlib
from collections import deque
from dataclasses import dataclass
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from queue import Empty
from typing import Literal

import imageio
import numpy as np
import yaml
import zarr
from dexjoco.data.episode_store import ZarrEpisodeStore
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation as R

from .dexjoco_openpi_env import DexJoCoOpenPIEnv
from .evaluation_metrics import (
    EpisodeEvaluationMetrics,
    constraint_violation_from_info,
    gripper_tracking_error,
    summarize_evaluation,
)
from .reproducibility import episode_seed, policy_noise, policy_noise_seed


@dataclass
class Observation:
    obs: dict
    timestamp: int


@dataclass
class Action:
    action: np.ndarray
    timestamp: int
    predicted_q: float | None = None
    reference_action_divergence: float | None = None


ActionChunk = Action


def get_latest(q: mp.Queue):
    """Return the newest queued item and discard older buffered items."""
    latest = None
    try:
        while True:
            latest = q.get_nowait()
    except Empty:
        pass
    return latest


def _set_seed(seed: int):
    np.random.seed(seed)
    # torch.manual_seed(seed)
    random.seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def _observation_sha256(observation: dict) -> str:
    """Fingerprint the exact state, images, and language sent to the policy."""
    digest = hashlib.sha256()
    for key in sorted(observation):
        if key in {"_policy_noise", "_policy_noise_seed"}:
            continue
        digest.update(key.encode("utf-8"))
        value = observation[key]
        if isinstance(value, str):
            digest.update(b"str\0")
            digest.update(value.encode("utf-8"))
            continue
        array = np.asarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _fixed_replan_interval(action_horizon: int, replan_ratio: float) -> int:
    """Match the first asynchronous `< threshold` trigger with a fixed period."""
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    if not 0.0 <= replan_ratio < 1.0:
        raise ValueError("replan_ratio must be in [0, 1)")
    return max(
        1,
        int(np.floor((1.0 - float(replan_ratio)) * action_horizon + 1e-9)) + 1,
    )


def _interp_rotvec_geodesic(
    rotvec0: np.ndarray, rotvec1: np.ndarray, t: float
) -> np.ndarray:
    """Interpolate rotation vectors on SO(3) instead of component-wise lerp."""
    if t <= 0.0:
        return rotvec0.copy()
    if t >= 1.0:
        return rotvec1.copy()

    r0 = R.from_rotvec(rotvec0)
    r1 = R.from_rotvec(rotvec1)
    relative_rotvec = (r0.inv() * r1).as_rotvec()
    return (r0 * R.from_rotvec(relative_rotvec * t)).as_rotvec()


def _interp_single_arm_action(
    old_action: np.ndarray, new_action: np.ndarray, t: float
) -> np.ndarray:
    """Interpolate single-arm action [xyz, rotvec, hand]."""
    interp_action = (1.0 - t) * old_action + t * new_action
    rotvec_slice = slice(3, 6)
    interp_action[rotvec_slice] = _interp_rotvec_geodesic(
        old_action[rotvec_slice], new_action[rotvec_slice], t
    ).astype(interp_action.dtype, copy=False)
    return interp_action


def _interp_dual_arm_action(
    old_action: np.ndarray, new_action: np.ndarray, t: float
) -> np.ndarray:
    """Interpolate dual-arm action [r_xyz, r_rotvec, r_hand, l_xyz, l_rotvec, l_hand]."""
    interp_action = (1.0 - t) * old_action + t * new_action
    right_rotvec_slice = slice(3, 6)
    left_rotvec_slice = slice(25, 28)
    interp_action[right_rotvec_slice] = _interp_rotvec_geodesic(
        old_action[right_rotvec_slice], new_action[right_rotvec_slice], t
    ).astype(interp_action.dtype, copy=False)
    interp_action[left_rotvec_slice] = _interp_rotvec_geodesic(
        old_action[left_rotvec_slice], new_action[left_rotvec_slice], t
    ).astype(interp_action.dtype, copy=False)
    return interp_action


def inference_process(
    obs_queue: mp.Queue,
    action_queue: mp.Queue,
    stop_event: MpEvent,
    port: int,
    inferencing_event: MpEvent,
    seed: int,
    host: str,
    verify_policy_repeatability: bool,
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _set_seed(seed)

    # Inference worker: receive observations and query the OpenPI policy server.
    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    while not stop_event.is_set():
        obs: Observation | None = get_latest(obs_queue)
        if obs is None:
            stop_event.wait(0.01)
            continue

        result = client.infer(obs.obs)
        if verify_policy_repeatability:
            repeated = client.infer(obs.obs)
            actions = np.asarray(result["actions"], dtype=np.float32)
            repeated_actions = np.asarray(repeated["actions"], dtype=np.float32)
            max_abs_diff = float(np.max(np.abs(actions - repeated_actions)))
            print(
                "[policy-repeatability] "
                f"timestamp={obs.timestamp} max_abs_diff={max_abs_diff:.9g}",
                flush=True,
            )
        action_chunk = result["actions"]

        action_queue.put(
            ActionChunk(
                action=action_chunk,
                timestamp=obs.timestamp,
                predicted_q=result.get("predicted_q"),
                reference_action_divergence=result.get("policy_reference_action_divergence"),
            )
        )
        inferencing_event.clear()


def receive_actions(
    action_queue: mp.Queue,
    actions_buffer: deque,
    now_timestamp: int,
    dual_arm: bool,
):
    """Receive action chunks and merge them into a timestamped action buffer.

    now_timestamp has not been executed yet.
    """
    interp_action_fn = (
        _interp_dual_arm_action if dual_arm else _interp_single_arm_action
    )

    # Drop expired actions that are older than the current timestamp.
    while actions_buffer and actions_buffer[0].timestamp < now_timestamp:
        actions_buffer.popleft()

    diagnostics = []
    while True:
        try:
            action_chunk: ActionChunk = action_queue.get_nowait()
            action_chunk_array = np.ascontiguousarray(
                np.asarray(action_chunk.action, dtype=np.float32)
            )
            diagnostics.append(
                {
                    "predicted_q": action_chunk.predicted_q,
                    "reference_action_divergence": action_chunk.reference_action_divergence,
                    "action_chunk_sha256": hashlib.sha256(
                        action_chunk_array.tobytes()
                    ).hexdigest(),
                }
            )

            # Chunk timestamp comes from observation, so it should not exceed now_timestamp.
            assert action_chunk.timestamp <= now_timestamp

            # All timestamp ranges below use half-open intervals: [start, end).
            action_chunk_timestamp_range = (
                now_timestamp,
                action_chunk.timestamp + action_chunk.action.shape[0],
            )
            if action_chunk_timestamp_range[1] <= now_timestamp:
                continue

            action = action_chunk.action[
                (action_chunk_timestamp_range[0] - action_chunk.timestamp) : (
                    action_chunk_timestamp_range[1] - action_chunk.timestamp
                )
            ]

            if actions_buffer:
                buffer_timestamp_range = (
                    actions_buffer[0].timestamp,
                    actions_buffer[-1].timestamp + 1,
                )
                assert buffer_timestamp_range[1] - buffer_timestamp_range[0] == len(
                    actions_buffer
                ), "Buffer timestamps must be continuous"
            else:
                buffer_timestamp_range = (now_timestamp, now_timestamp)

            # Blend overlapping actions already in buffer.
            overlap_range = (
                max(action_chunk_timestamp_range[0], buffer_timestamp_range[0]),
                min(action_chunk_timestamp_range[1], buffer_timestamp_range[1]),
            )
            overlap_len = overlap_range[1] - overlap_range[0]
            for ts in range(overlap_range[0], overlap_range[1]):
                buffer_idx = ts - buffer_timestamp_range[0]
                action_idx = ts - action_chunk_timestamp_range[0]

                # Keep interpolation away from 0/1 endpoints for smoother transitions.
                interp_t = (ts - overlap_range[0] + 1) / (overlap_len + 1)

                interp_action = interp_action_fn(
                    actions_buffer[buffer_idx].action,
                    action[action_idx],
                    interp_t,
                )
                actions_buffer[buffer_idx] = Action(action=interp_action, timestamp=ts)

            # Append non-overlapping tail actions.
            non_overlap_timestamp_range = (
                buffer_timestamp_range[1],
                action_chunk_timestamp_range[1],
            )
            for ts in range(
                non_overlap_timestamp_range[0], non_overlap_timestamp_range[1]
            ):
                action_idx = ts - action_chunk_timestamp_range[0]
                actions_buffer.append(Action(action=action[action_idx], timestamp=ts))
        except Empty:
            break
    return diagnostics


def _append_video_frames(video_writers: dict, raw_images: dict):
    for cam_name, writer in video_writers.items():
        writer.append_data(raw_images[cam_name])


def _write_replay_zarr(episode_dir: Path, transitions: list[dict], data_fps: float = 30.0):
    if not transitions:
        return
    actions = np.stack([item["action"] for item in transitions], axis=0).astype(np.float32)
    states = np.stack([item["state"] for item in transitions], axis=0).astype(np.float32)
    rewards = np.asarray([item["reward"] for item in transitions], dtype=np.float32)
    dones = np.asarray([item["done"] for item in transitions], dtype=np.bool_)
    successes = np.asarray([item["success"] for item in transitions], dtype=np.bool_)
    timestamps = np.arange(len(transitions), dtype=np.float32) / float(data_fps)
    episode_data = {
        "action_rotvec": actions,
        "state": states,
        "reward": rewards,
        "done": dones,
        "success": successes,
        "timestamp": timestamps,
    }
    image_keys = sorted(
        set.intersection(*(set(item.get("images", {})) for item in transitions))
    ) if transitions else []
    for key in image_keys:
        episode_data[f"image_{key}"] = np.stack(
            [np.asarray(item["images"][key], dtype=np.uint8) for item in transitions],
            axis=0,
        )
    store = zarr.DirectoryStore(str(episode_dir / "replay.zarr"))
    episode_store = ZarrEpisodeStore.create_empty(storage=store)
    episode_store.append_episode(episode_data, compressors="disk")


def _append_metrics_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        key: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for key, value in payload.items()
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(clean, sort_keys=True) + "\n")


def main(
    config: Path,
    seed: int = 0,
    rand_full: bool = False,
    randomize_dynamics: bool = False,
    port: int = 8000,
    host: str = "0.0.0.0",
    output: Path | None = None,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    replan_ratio: float = 0.8,
    episodes: int = 50,
    pad_state_dim46: bool = False,
    record_pressed_digits: bool | None = None,
    save_replay_zarr: bool = False,
    metrics_output: Path | None = None,
    block_on_inference_when_empty: bool = False,
    deterministic_policy_noise: bool = False,
    block_on_inference_after_replan: bool = False,
    verify_policy_repeatability: bool = False,
    strict_reproducibility: bool = False,
    policy_noise_dim: int = 32,
):
    if render_mode == "rgb_array":
        os.environ.setdefault("MUJOCO_GL", "egl")
    else:
        os.environ.setdefault("MUJOCO_GL", "glfw")
    _set_seed(seed)

    # Load evaluation configuration.
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    env_name = cfg["env_name"]
    camera_mapping = cfg["camera_mapping"]
    robot_type = cfg["robot_type"]
    dual_arm = robot_type == "dual_arm"
    prompt = cfg["prompt"]
    action_horizon = 30  # the policy trained on
    fixed_replan_interval = (
        _fixed_replan_interval(action_horizon, replan_ratio)
        if strict_reproducibility
        else None
    )
    if fixed_replan_interval is not None:
        print(
            "[evaluation] strict fixed replan schedule: "
            f"interval={fixed_replan_interval} timestamps=0,{fixed_replan_interval},...",
            flush=True,
        )

    # Record password input only for iPad tasks unless explicitly configured.
    if record_pressed_digits is None:
        record_pressed_digits = env_name == "bimanual_unlock_ipad"

    # Write episode videos under a temporary name before assigning the result suffix.
    if output is None:
        output_dir = (
            Path("outputs")
            / f"{env_name}{'_rand_full' if rand_full else ''}_seed{seed}"
        )
    else:
        output_dir = output
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_output or output_dir / "evaluation_metrics.jsonl"

    # Create the DexJoCo environment wrapper used by the OpenPI policy.
    env = DexJoCoOpenPIEnv(
        env_name=env_name,
        camera_mapping=camera_mapping,
        seed=seed,
        rand_full=rand_full,
        randomize_dynamics=randomize_dynamics,
        dual_arm=dual_arm,
        prompt=prompt,
        render_mode=render_mode,
        pad_state_dim46=pad_state_dim46,
        password=cfg.get("password", None),  # Pass password from config if available
        deterministic_rendering=strict_reproducibility,
    )
    env.start()

    # Queues connect the control loop with the asynchronous inference worker.
    obs_queue = mp.Queue()
    action_queue = mp.Queue()
    stop_event = mp.Event()
    inferencing_event = mp.Event()

    inference_proc = mp.Process(
        target=inference_process,
        args=(
            obs_queue,
            action_queue,
            stop_event,
            port,
            inferencing_event,
            seed,
            host,
            verify_policy_repeatability,
        ),
    )
    video_writers = None

    try:
        inference_proc.start()
        num_success = 0
        episode_metrics: list[EpisodeEvaluationMetrics] = []

        for ep in range(episodes):
            print(f"Episode {ep + 1}/{episodes}")

            # Setup video writers in a temporary episode directory.
            video_dir = output_dir / f"episode_{ep:02d}_temp"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_writers = {
                cam_name: imageio.get_writer(video_dir / f"{cam_name}.mp4", fps=30)
                for cam_name in camera_mapping.values()
            }

            current_episode_seed = episode_seed(seed, ep)
            _set_seed(current_episode_seed)
            env.reset(seed=current_episode_seed)
            initial_state_sha256 = env.scenario_sha256()

            timestamp = 0
            actions_buffer = deque()
            replay_transitions = []
            current_metrics = EpisodeEvaluationMetrics(
                task_id=env_name,
                seed=seed,
                episode_index=ep,
                initial_state_sha256=initial_state_sha256,
                deterministic_policy_noise=deterministic_policy_noise,
                policy_noise_scheme=(
                    "seed_episode_request_pcg64_v2"
                    if deterministic_policy_noise
                    else "unseeded"
                ),
                environment_seed=current_episode_seed,
            )

            if env_name == "click_mouse":
                # Align with dataset.
                for _ in range(30):
                    # fmt: off
                    env.step(
                        action=np.array([
                            -4.4294e-01, 1.3729e-06, 1.5170e00,
                            -3.14156462e00, -6.91584035e-05, -1.40317984e-03,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0.263, 0, 0, 0
                        ])
                    )
                    # fmt: on

            # Fingerprint the exact simulator state from which policy control
            # starts. ClickMouse performs a fixed 30-step alignment first.
            current_metrics.aligned_state_sha256 = env.scenario_sha256()
            current_metrics.aligned_observation_sha256 = _observation_sha256(env.get_obs())

            # Send the first observation and mark inference as active before enqueueing it.
            policy_request_index = 0

            def request_policy(observation: dict, request_timestamp: int) -> None:
                nonlocal policy_request_index
                observation_sha256 = _observation_sha256(observation)
                if deterministic_policy_noise:
                    policy_noise_seed_value = policy_noise_seed(
                        seed,
                        ep,
                        policy_request_index,
                    )
                    observation["_policy_noise_seed"] = policy_noise_seed_value
                    observation["_policy_noise"] = policy_noise(
                        policy_noise_seed_value,
                        action_horizon,
                        policy_noise_dim,
                    )
                    policy_noise_seed_for_metrics = policy_noise_seed_value
                else:
                    policy_noise_seed_for_metrics = None
                current_metrics.record_policy_request(
                    timestamp=request_timestamp,
                    noise_seed=policy_noise_seed_for_metrics,
                    observation_sha256=observation_sha256,
                )
                policy_request_index += 1
                inferencing_event.set()
                obs_queue.put(Observation(observation, request_timestamp))

            request_policy(env.get_obs(), timestamp)

            # Save the reset frame.
            raw_images = env.get_raw_images()
            _append_video_frames(video_writers, raw_images)

            in_stay_state = (
                False  # Track whether the previous step already used stay().
            )
            password = []
            wait_for_replan_result = False

            # Episode loop.
            while True:
                if not inference_proc.is_alive():
                    raise RuntimeError("policy inference worker exited unexpectedly")
                policy_diagnostics = receive_actions(
                    action_queue,
                    actions_buffer,
                    timestamp,
                    dual_arm,
                )
                for diagnostic in policy_diagnostics:
                    current_metrics.record_policy_diagnostic(
                        predicted_q=diagnostic["predicted_q"],
                        reference_action_divergence=diagnostic["reference_action_divergence"],
                    )
                    current_metrics.record_policy_action_chunk(
                        diagnostic["action_chunk_sha256"]
                    )
                if policy_diagnostics:
                    wait_for_replan_result = False

                # Offline evaluation can run the simulator much faster than a
                # large policy can produce a chunk. Advancing simulation while
                # inference is pending makes the returned chunk expire before
                # it is consumed, so pause wall-clock evaluation until at least
                # one action is available.
                while (
                    (block_on_inference_when_empty and not actions_buffer)
                    or wait_for_replan_result
                ):
                    if not inference_proc.is_alive():
                        raise RuntimeError("policy inference worker exited unexpectedly")
                    policy_diagnostics = receive_actions(
                        action_queue,
                        actions_buffer,
                        timestamp,
                        dual_arm,
                    )
                    for diagnostic in policy_diagnostics:
                        current_metrics.record_policy_diagnostic(
                            predicted_q=diagnostic["predicted_q"],
                            reference_action_divergence=diagnostic["reference_action_divergence"],
                        )
                        current_metrics.record_policy_action_chunk(
                            diagnostic["action_chunk_sha256"]
                        )
                    if policy_diagnostics:
                        wait_for_replan_result = False
                    if not actions_buffer or wait_for_replan_result:
                        time.sleep(0.01)

                # Execute the scheduled action for this timestamp, or hold the pose.
                if actions_buffer:
                    assert actions_buffer[0].timestamp == timestamp, (
                        "Buffer head timestamp must match current timestamp"
                    )
                    action = actions_buffer.popleft().action
                    observation_before = env.get_obs()
                    state_before = observation_before["state"].copy()
                    pressed_digits = env.step(action)
                    in_stay_state = False
                else:
                    observation_before = env.get_obs()
                    state_before = observation_before["state"].copy()
                    pressed_digits = env.stay(continue_stay=in_stay_state)
                    in_stay_state = True

                executed_action = env.last_openpi_action
                if executed_action is not None:
                    current_metrics.record_step(
                        reward=env.last_reward,
                        action=executed_action,
                        gripper_error=gripper_tracking_error(
                            executed_action,
                            env.get_obs()["state"],
                            dual_arm=dual_arm,
                        ),
                        constraint_violation=constraint_violation_from_info(env.last_info),
                    )

                if save_replay_zarr:
                    if executed_action is not None:
                        replay_transitions.append(
                            {
                                "state": np.asarray(state_before, dtype=np.float32),
                                "action": np.asarray(executed_action, dtype=np.float32),
                                "reward": env.last_reward,
                                "done": env.is_done,
                                "success": env.is_success,
                                "images": {
                                    key: np.asarray(value, dtype=np.uint8).copy()
                                    for key, value in observation_before.items()
                                    if key not in {"state", "prompt"}
                                    and isinstance(value, np.ndarray)
                                    and value.ndim == 3
                                },
                            }
                        )

                if record_pressed_digits and pressed_digits:
                    password.append(pressed_digits)

                timestamp += 1

                raw_images = env.get_raw_images()
                _append_video_frames(video_writers, raw_images)

                # Stop after the environment reports terminal state.
                if env.is_done:
                    if env.is_success:
                        num_success += 1
                        print("Success!")
                    else:
                        print("Failed")
                    current_metrics.success = env.is_success
                    break

                if fixed_replan_interval is not None:
                    should_send_obs = timestamp % fixed_replan_interval == 0
                else:
                    # Fast asynchronous mode: request when the buffered horizon
                    # falls below the configured threshold and no work is pending.
                    should_send_obs = (
                        len(actions_buffer) < replan_ratio * action_horizon
                        and obs_queue.empty()
                        and not inferencing_event.is_set()
                        and action_queue.empty()
                    )

                if should_send_obs:
                    request_policy(env.get_obs(), timestamp)
                    wait_for_replan_result = (
                        strict_reproducibility or block_on_inference_after_replan
                    )
                    # inferencing_event is cleared after the action chunk arrives.

            for writer in video_writers.values():
                writer.close()
            video_writers = None

            if deterministic_policy_noise and current_metrics.actions:
                np.save(
                    video_dir / "executed_actions.npy",
                    np.asarray(current_metrics.actions, dtype=np.float32),
                )

            # Rename the temporary episode directory with the final result label.
            if save_replay_zarr:
                _write_replay_zarr(video_dir, replay_transitions)

            # Rename the temporary episode directory with the final result label.
            result_suffix = "success" if env.is_success else "failure"
            if record_pressed_digits:
                if password:
                    password_suffix = "_".join(
                        "".join(str(digit) for digit in digits) for digits in password
                    )
                else:
                    password_suffix = "no_password_input"
                final_video_dir = (
                    output_dir / f"episode_{ep:02d}_{result_suffix}_{password_suffix}"
                )
            else:
                final_video_dir = output_dir / f"episode_{ep:02d}_{result_suffix}"
            video_dir.rename(final_video_dir)
            current_metrics.policy_request_count = policy_request_index
            episode_metrics.append(current_metrics)
            _append_metrics_jsonl(
                metrics_path,
                {"record_type": "episode", **current_metrics.as_dict()},
            )

            # Drain in-flight work before starting the next episode.
            while True:
                try:
                    obs_queue.get_nowait()
                except Empty:
                    break
            while inferencing_event.is_set():
                time.sleep(0.1)
            while not action_queue.empty():
                action_queue.get()

        print(
            f"\nSuccess rate: {num_success}/{episodes} ({100 * num_success / episodes:.1f}%)"
        )
        evaluation_summary = summarize_evaluation(episode_metrics)
        _append_metrics_jsonl(
            metrics_path,
            {"record_type": "aggregate", **evaluation_summary},
        )
        print(f"Evaluation metrics: {metrics_path}")
        (output_dir / f"success_rate_{num_success}_{episodes}.txt").touch()

    finally:
        # Shut down worker and release multiprocessing resources.
        stop_event.set()
        inference_proc.join(timeout=2)
        if inference_proc.is_alive():
            inference_proc.terminate()
            inference_proc.join(timeout=2)
        obs_queue.cancel_join_thread()
        obs_queue.close()
        action_queue.cancel_join_thread()
        action_queue.close()
        env.close()
        if video_writers is not None:
            for writer in video_writers.values():
                writer.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
