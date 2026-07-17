import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies.libero_policy import LiberoInputs


@dataclasses.dataclass(frozen=True)
class PickPlaceInputs(LiberoInputs):
    control_mode: str = "absolute_eef"
    hand_mode: str = "absolute"

    def __call__(self, data: dict) -> dict:
        inputs = super().__call__(data)
        actions = None

        if "label_arm_abs" in data and "label_arm_delta" in data and "label_hand_abs" in data:
            arm_abs = np.asarray(data["label_arm_abs"], dtype=np.float32)
            arm_delta = np.asarray(data["label_arm_delta"], dtype=np.float32)
            hand_abs = np.asarray(data["label_hand_abs"], dtype=np.float32)
            horizon = hand_abs.shape[0]
            actions = np.zeros((horizon, 12), dtype=np.float32)
            actions[:, :6] = arm_delta if self.control_mode == "delta_eef" else arm_abs
            actions[:, 6:12] = hand_abs
        elif "actions" in data:
            raw_actions = np.asarray(data["actions"], dtype=np.float32)
            if raw_actions.shape[-1] != 12:
                raise ValueError(f"Expected 12-dim pick-place actions, got shape {raw_actions.shape}")
            horizon = raw_actions.shape[0]
            actions = np.zeros((horizon, 12), dtype=np.float32)
            actions[:, :6] = raw_actions[:, :6]
            if self.hand_mode in ("delta", "absolute"):
                actions[:, 6:12] = raw_actions[:, 6:12]
            else:
                raise ValueError("12-dim legacy pick-place actions only support hand_mode in {'delta','absolute'}")

        if actions is not None:
            inputs["actions"] = actions
        return inputs


@dataclasses.dataclass(frozen=True)
class PickPlaceOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :12])}
