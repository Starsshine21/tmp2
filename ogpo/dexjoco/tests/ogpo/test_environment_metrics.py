import hashlib
import math

import numpy as np
import pytest

from dexjoco_openpi_client.evaluation_metrics import (
    EpisodeEvaluationMetrics,
    gripper_tracking_error,
    summarize_evaluation,
)
from dexjoco_openpi_client.reproducibility import (
    episode_seed,
    policy_noise,
    policy_noise_seed,
)


def test_reproducible_episode_seed_and_policy_noise_are_index_bound():
    assert episode_seed(27, 0) == 27
    assert episode_seed(27, 1) == 10_034

    first_seed = policy_noise_seed(27, 3, 5)
    first = policy_noise(first_seed, 30, 32)
    repeated = policy_noise(first_seed, 30, 32)
    different = policy_noise(policy_noise_seed(27, 3, 6), 30, 32)

    assert first.shape == (30, 32)
    assert first.dtype == np.float32
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)


def test_gripper_tracking_error_matches_single_and_dual_arm_layouts():
    single_action = np.zeros(22, dtype=np.float32)
    single_state = np.zeros(23, dtype=np.float32)
    single_state[7:23] = 1.0
    assert gripper_tracking_error(single_action, single_state, dual_arm=False) == 1.0

    dual_action = np.zeros(44, dtype=np.float32)
    dual_state = np.zeros(46, dtype=np.float32)
    dual_state[14:46] = 2.0
    assert gripper_tracking_error(dual_action, dual_state, dual_arm=True) == 2.0


def test_environment_summary_contains_required_metrics_and_metadata():
    first = EpisodeEvaluationMetrics(task_id="click_mouse", seed=7, episode_index=0)
    first.initial_state_sha256 = "scenario-a"
    first.aligned_state_sha256 = "aligned-a"
    first.aligned_observation_sha256 = "observation-a"
    first.record_step(
        reward=0.25,
        action=np.zeros(22),
        gripper_error=0.2,
        constraint_violation=0.0,
    )
    first.record_step(
        reward=0.75,
        action=np.ones(22),
        gripper_error=0.4,
        constraint_violation=1.0,
    )
    first.record_policy_diagnostic(predicted_q=2.0, reference_action_divergence=0.2)
    first.success = False
    first.policy_request_count = 3
    first.deterministic_policy_noise = True
    first.policy_noise_scheme = "seed_episode_request_v1"
    first.environment_seed = 7
    first.record_policy_request(timestamp=0, noise_seed=7, observation_sha256="a" * 64)
    first.record_policy_request(timestamp=7, noise_seed=8, observation_sha256="b" * 64)
    first.record_policy_action_chunk("c" * 64)
    first.record_policy_action_chunk("d" * 64)

    first_dict = first.as_dict()
    assert first_dict["initial_state_sha256"] == "scenario-a"
    assert first_dict["aligned_state_sha256"] == "aligned-a"
    assert first_dict["aligned_observation_sha256"] == "observation-a"
    assert first_dict["policy_request_count"] == 3
    assert first_dict["deterministic_policy_noise"] == 1.0
    assert first_dict["policy_noise_scheme"] == "seed_episode_request_v1"
    assert first_dict["environment_seed"] == 7
    assert first_dict["policy_noise_seed_first"] == 7
    assert first_dict["policy_request_trace_sha256"] == hashlib.sha256(
        np.asarray([[0, 7], [7, 8]], dtype=np.int64).tobytes()
    ).hexdigest()
    assert first_dict["policy_observation_trace_sha256"] == hashlib.sha256(
        (("a" * 64) + "\n" + ("b" * 64)).encode("ascii")
    ).hexdigest()
    assert first_dict["policy_action_chunk_first_sha256"] == "c" * 64
    assert first_dict["policy_action_chunk_trace_sha256"] == hashlib.sha256(
        (("c" * 64) + "\n" + ("d" * 64)).encode("ascii")
    ).hexdigest()
    assert first_dict["action_trace_sha256"] == hashlib.sha256(
        np.ascontiguousarray(np.stack(first.actions)).tobytes()
    ).hexdigest()

    second = EpisodeEvaluationMetrics(task_id="click_mouse", seed=7, episode_index=1)
    second.deterministic_policy_noise = True
    second.policy_noise_scheme = "seed_episode_request_v1"
    second.record_step(
        reward=3.0,
        action=np.zeros(22),
        gripper_error=0.0,
        constraint_violation=None,
    )
    second.record_policy_diagnostic(predicted_q=4.0, reference_action_divergence=0.4)
    second.success = True

    summary = summarize_evaluation([first, second])

    assert summary["task_id"] == "click_mouse"
    assert summary["seed"] == 7
    assert summary["deterministic_policy_noise"] == 1.0
    assert summary["policy_noise_scheme"] == "seed_episode_request_v1"
    assert summary["success_rate"] == 0.5
    assert summary["task_success"] == 0.5
    assert summary["average_return"] == 2.0
    assert summary["average_episode_length"] == 1.5
    assert summary["action_smoothness"] > 0.0
    assert summary["gripper_error"] == pytest.approx(0.2)
    assert summary["constraint_violation_rate"] == 0.5
    assert summary["constraint_metric_availability"] == 0.5
    assert summary["predicted_q_mean"] == 3.0
    assert summary["predicted_q_return_gap"] == 1.0
    assert summary["predicted_q_return_correlation"] == pytest.approx(1.0)
    assert summary["policy_reference_action_divergence"] == pytest.approx(0.3)
    assert not math.isnan(summary["predicted_q_return_correlation"])
