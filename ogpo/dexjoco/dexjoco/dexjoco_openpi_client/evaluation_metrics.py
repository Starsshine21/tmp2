from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np


def gripper_tracking_error(action: np.ndarray, state: np.ndarray, *, dual_arm: bool) -> float:
    """Return RMS commanded-versus-observed hand joint error."""
    action = np.asarray(action, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    if dual_arm:
        commanded = np.concatenate([action[6:22], action[28:44]])
        observed = state[14:46]
    else:
        commanded = action[6:22]
        observed = state[7:23]
    if commanded.shape != observed.shape or commanded.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(commanded - observed))))


def constraint_violation_from_info(info: dict[str, Any]) -> float | None:
    """Read an explicit simulator constraint metric without treating normal contacts as failures."""
    for key in (
        "constraint_violation",
        "constraint_violations",
        "collision_violation",
        "collision_violations",
        "illegal_move",
    ):
        if key in info:
            value = np.asarray(info[key], dtype=np.float64)
            return float(value.mean())
    return None


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return float("nan")
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if np.std(left_array) <= 1e-12 or np.std(right_array) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left_array, right_array)[0, 1])


@dataclass
class EpisodeEvaluationMetrics:
    task_id: str
    seed: int
    episode_index: int
    success: bool = False
    rewards: list[float] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    gripper_errors: list[float] = field(default_factory=list)
    constraint_violations: list[float] = field(default_factory=list)
    constraint_observations: int = 0
    predicted_q_values: list[float] = field(default_factory=list)
    reference_action_divergences: list[float] = field(default_factory=list)
    initial_state_sha256: str = ""
    aligned_state_sha256: str = ""
    aligned_observation_sha256: str = ""
    policy_request_count: int = 0
    deterministic_policy_noise: bool = False
    policy_noise_scheme: str = "unseeded"
    policy_request_trace: list[tuple[int, int]] = field(default_factory=list)
    policy_observation_hashes: list[str] = field(default_factory=list)
    policy_action_chunk_hashes: list[str] = field(default_factory=list)
    environment_seed: int = -1

    def record_policy_request(
        self,
        *,
        timestamp: int,
        noise_seed: int | None,
        observation_sha256: str = "",
    ) -> None:
        self.policy_request_trace.append(
            (int(timestamp), -1 if noise_seed is None else int(noise_seed))
        )
        self.policy_observation_hashes.append(str(observation_sha256))

    def record_step(
        self,
        *,
        reward: float,
        action: np.ndarray,
        gripper_error: float,
        constraint_violation: float | None,
    ) -> None:
        self.rewards.append(float(reward))
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        if np.isfinite(gripper_error):
            self.gripper_errors.append(float(gripper_error))
        if constraint_violation is not None and np.isfinite(constraint_violation):
            self.constraint_violations.append(float(constraint_violation))
            self.constraint_observations += 1

    def record_policy_diagnostic(
        self,
        *,
        predicted_q: float | None,
        reference_action_divergence: float | None,
    ) -> None:
        if predicted_q is not None and np.isfinite(predicted_q):
            self.predicted_q_values.append(float(predicted_q))
        if reference_action_divergence is not None and np.isfinite(reference_action_divergence):
            self.reference_action_divergences.append(float(reference_action_divergence))

    def record_policy_action_chunk(self, action_chunk_sha256: str) -> None:
        self.policy_action_chunk_hashes.append(str(action_chunk_sha256))

    @property
    def episode_return(self) -> float:
        return float(sum(self.rewards))

    @property
    def episode_length(self) -> int:
        return len(self.rewards)

    @property
    def action_smoothness(self) -> float:
        if len(self.actions) < 2:
            return float("nan")
        actions = np.stack(self.actions)
        velocity = np.diff(actions, axis=0)
        value = float(np.mean(np.square(velocity)))
        if len(self.actions) >= 3:
            acceleration = np.diff(actions, n=2, axis=0)
            value += 0.1 * float(np.mean(np.square(acceleration)))
        return value

    def as_dict(self) -> dict[str, float | int | str]:
        action_trace_sha256 = ""
        if self.actions:
            action_trace_sha256 = hashlib.sha256(
                np.ascontiguousarray(np.stack(self.actions)).tobytes()
            ).hexdigest()
        policy_request_trace_sha256 = ""
        if self.policy_request_trace:
            policy_request_trace_sha256 = hashlib.sha256(
                np.asarray(self.policy_request_trace, dtype=np.int64).tobytes()
            ).hexdigest()
        policy_noise_seed_first = (
            self.policy_request_trace[0][1] if self.policy_request_trace else -1
        )
        policy_observation_trace_sha256 = ""
        if self.policy_observation_hashes:
            policy_observation_trace_sha256 = hashlib.sha256(
                "\n".join(self.policy_observation_hashes).encode("ascii")
            ).hexdigest()
        policy_action_chunk_trace_sha256 = ""
        if self.policy_action_chunk_hashes:
            policy_action_chunk_trace_sha256 = hashlib.sha256(
                "\n".join(self.policy_action_chunk_hashes).encode("ascii")
            ).hexdigest()
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "episode_index": self.episode_index,
            "environment_seed": self.environment_seed,
            "success": float(self.success),
            "episode_return": self.episode_return,
            "episode_length": self.episode_length,
            "action_smoothness": self.action_smoothness,
            "gripper_error": _mean(self.gripper_errors),
            "constraint_violation_rate": _mean(self.constraint_violations),
            "constraint_metric_available": float(self.constraint_observations > 0),
            "predicted_q_mean": _mean(self.predicted_q_values),
            "policy_reference_action_divergence": _mean(self.reference_action_divergences),
            "initial_state_sha256": self.initial_state_sha256,
            "aligned_state_sha256": self.aligned_state_sha256,
            "aligned_observation_sha256": self.aligned_observation_sha256,
            "action_trace_sha256": action_trace_sha256,
            "policy_request_count": self.policy_request_count,
            "deterministic_policy_noise": float(self.deterministic_policy_noise),
            "policy_noise_scheme": self.policy_noise_scheme,
            "policy_noise_seed_first": policy_noise_seed_first,
            "policy_request_trace_sha256": policy_request_trace_sha256,
            "policy_observation_trace_sha256": policy_observation_trace_sha256,
            "policy_action_chunk_first_sha256": (
                self.policy_action_chunk_hashes[0]
                if self.policy_action_chunk_hashes
                else ""
            ),
            "policy_action_chunk_trace_sha256": policy_action_chunk_trace_sha256,
        }


def summarize_evaluation(episodes: list[EpisodeEvaluationMetrics]) -> dict[str, float | int | str]:
    if not episodes:
        raise ValueError("at least one episode is required")
    task_ids = {episode.task_id for episode in episodes}
    seeds = {episode.seed for episode in episodes}
    deterministic_noise_modes = {
        episode.deterministic_policy_noise for episode in episodes
    }
    policy_noise_schemes = {episode.policy_noise_scheme for episode in episodes}
    if (
        len(task_ids) != 1
        or len(seeds) != 1
        or len(deterministic_noise_modes) != 1
        or len(policy_noise_schemes) != 1
    ):
        raise ValueError("evaluation summary requires one task id and one seed")

    predicted_q = []
    realized_return = []
    for episode in episodes:
        if episode.predicted_q_values:
            predicted_q.append(_mean(episode.predicted_q_values))
            realized_return.append(episode.episode_return)
    all_smoothness = [episode.action_smoothness for episode in episodes if np.isfinite(episode.action_smoothness)]
    all_gripper_errors = [value for episode in episodes for value in episode.gripper_errors]
    all_constraint_values = [value for episode in episodes for value in episode.constraint_violations]
    all_reference_divergences = [
        value for episode in episodes for value in episode.reference_action_divergences
    ]
    success_rate = _mean([float(episode.success) for episode in episodes])
    return {
        "task_id": episodes[0].task_id,
        "seed": episodes[0].seed,
        "deterministic_policy_noise": float(episodes[0].deterministic_policy_noise),
        "policy_noise_scheme": episodes[0].policy_noise_scheme,
        "episode_count": len(episodes),
        "success_rate": success_rate,
        "task_success": success_rate,
        "average_return": _mean([episode.episode_return for episode in episodes]),
        "average_episode_length": _mean([float(episode.episode_length) for episode in episodes]),
        "action_smoothness": _mean(all_smoothness),
        "gripper_error": _mean(all_gripper_errors),
        "constraint_violation_rate": _mean(all_constraint_values),
        "constraint_metric_availability": _mean(
            [float(episode.constraint_observations > 0) for episode in episodes]
        ),
        "predicted_q_mean": _mean(predicted_q),
        "predicted_q_return_gap": _mean(
            [prediction - outcome for prediction, outcome in zip(predicted_q, realized_return, strict=True)]
        ),
        "predicted_q_return_correlation": _correlation(predicted_q, realized_return),
        "policy_reference_action_divergence": _mean(all_reference_divergences),
    }
