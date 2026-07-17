import numpy as np

from scripts import pi05_real_robot_infer as infer


def test_build_policy_state_normalizes_hand_counts():
    joints = np.arange(6, dtype=np.float32)
    eef = np.arange(10, 16, dtype=np.float32)
    hand_counts = np.array([0, 500, 1000, 1500, 2000, 2500], dtype=np.float32)

    state = infer.build_policy_state(joints, eef, hand_counts, max_hand_abs=2000.0)

    np.testing.assert_allclose(state[:6], joints)
    np.testing.assert_allclose(state[6:12], eef)
    np.testing.assert_allclose(state[12:18], [-1.0, -0.5, 0.0, 0.5, 1.0, 1.0])


def test_split_policy_action_uses_current_12d_pickplace_layout():
    action = np.arange(120, dtype=np.float32).reshape(10, 12)

    arm_cmd, hand_cmd = infer.split_policy_action({"actions": action})

    np.testing.assert_allclose(arm_cmd, np.arange(6, dtype=np.float32))
    np.testing.assert_allclose(hand_cmd, np.arange(6, 12, dtype=np.float32))


def test_denormalize_hand_action_to_counts():
    hand_action = np.array([-1.5, -1.0, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)

    counts = infer.denormalize_hand_action(hand_action, max_hand_abs=2000.0)

    np.testing.assert_allclose(counts, [0.0, 0.0, 1000.0, 1500.0, 2000.0, 2000.0])


def test_build_camera_observation_maps_training_camera_keys():
    l515_rgb = np.full((4, 5, 3), 10, dtype=np.uint8)
    orbbec_rgb = np.full((4, 5, 3), 20, dtype=np.uint8)
    state = np.arange(18, dtype=np.float32)

    obs = infer.build_policy_observation(l515_rgb, orbbec_rgb, state, "place object")

    assert obs["observation/image"] is l515_rgb
    assert obs["observation/wrist_image"] is orbbec_rgb
    assert obs["observation/state"] is state
    assert obs["prompt"] == "place object"


def test_apply_action_sends_eef_pose_and_denormalized_hand_counts():
    class FakeRobot:
        def __init__(self):
            self.eef_target = None

        def set_pos_j(self, *_args, **_kwargs):
            raise AssertionError("EEF control must not call joint command API")

        def set_pos_eef(self, target, servo=True):
            self.eef_target = (np.asarray(target, dtype=np.float32), servo)

    class FakeHand:
        def __init__(self):
            self.target = None

        def set_hand_pos(self, target):
            self.target = target

    runner = infer.PI05RealRobotRunner.__new__(infer.PI05RealRobotRunner)
    runner.robot_cfg = infer.RobotConfig(max_eef_delta=0.03, max_hand_abs=2000.0, action_ema_alpha=1.0)
    runner.control_mode = "delta_eef"
    runner.prev_arm_cmd = None
    runner.prev_hand_cmd = None
    runner.robot = FakeRobot()
    runner.hand = FakeHand()
    runner.logger = infer._build_logger("test_apply_action")

    runner._apply_action(
        np.array([0.01, -0.02, 0.04, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        np.zeros(6, dtype=np.float32),
    )

    eef_target, servo = runner.robot.eef_target
    np.testing.assert_allclose(eef_target, [0.01, -0.02, 0.03, 0.0, 0.0, 0.0])
    assert servo is True
    assert runner.hand.target == [0, 500, 1000, 1500, 2000, 2000]
