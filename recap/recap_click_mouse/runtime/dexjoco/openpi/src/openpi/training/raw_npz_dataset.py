from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from lerobot.rl.acp_tags import build_acp_tagged_task


@dataclass(frozen=True)
class RawACPIndex:
    value: np.ndarray
    advantage: np.ndarray
    indicator: np.ndarray


def load_acp_sidecar(path: str | Path) -> RawACPIndex:
    data = np.load(path)
    return RawACPIndex(
        value=np.asarray(data["value"], dtype=np.float32),
        advantage=np.asarray(data["advantage"], dtype=np.float32),
        indicator=np.asarray(data["acp_indicator"], dtype=np.int64),
    )


class ClickMouseRawNpzDataset(Dataset):
    def __init__(self, raw_root: str | Path, acp_sidecar: str | Path | None = None, action_horizon: int = 30):
        self.raw_root = Path(raw_root)
        self.episode_files = sorted(self.raw_root.glob("episode_*.npz"))
        if not self.episode_files:
            raise FileNotFoundError(f"No episode_*.npz files found under {self.raw_root}")

        self.action_horizon = action_horizon
        self.acp = load_acp_sidecar(acp_sidecar) if acp_sidecar else None
        self.lengths: list[int] = []
        self.prefix: list[int] = []
        global_index = 0
        for ep_idx, ep_file in enumerate(self.episode_files):
            with np.load(ep_file, mmap_mode="r") as ep:
                length = int(ep["action"].shape[0])
            self.lengths.append(length)
            self.prefix.append(global_index)
            global_index += length
        self.total_len = global_index
        self._cached_ep_idx: int | None = None
        self._cached_episode: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return self.total_len

    def _resolve_index(self, idx: int) -> tuple[int, int, int]:
        if idx < 0:
            idx += self.total_len
        if idx < 0 or idx >= self.total_len:
            raise IndexError(idx)
        ep_idx = int(np.searchsorted(np.asarray(self.prefix), idx, side="right") - 1)
        frame_idx = idx - self.prefix[ep_idx]
        return ep_idx, frame_idx, idx

    def __getitem__(self, idx: int) -> dict:
        ep_idx, frame_idx, global_idx = self._resolve_index(int(idx))
        if self._cached_ep_idx != ep_idx or self._cached_episode is None:
            ep_file = self.episode_files[ep_idx]
            with np.load(ep_file, mmap_mode="r") as ep:
                self._cached_episode = {
                    "observation_images_base": np.asarray(ep["observation_images_base"]),
                    "observation_images_wrist": np.asarray(ep["observation_images_wrist"]),
                    "observation_state": np.asarray(ep["observation_state"], dtype=np.float32),
                    "action": np.asarray(ep["action"], dtype=np.float32),
                    "task": np.asarray(ep["task"]),
                }
            self._cached_ep_idx = ep_idx

        ep = self._cached_episode
        assert ep is not None

        base = np.asarray(ep["observation_images_base"][frame_idx], dtype=np.uint8)
        wrist = np.asarray(ep["observation_images_wrist"][frame_idx], dtype=np.uint8)
        state = np.asarray(ep["observation_state"][frame_idx], dtype=np.float32)
        all_actions = ep["action"]
        end_idx = min(frame_idx + self.action_horizon, all_actions.shape[0])
        action_chunk = all_actions[frame_idx:end_idx]
        if action_chunk.shape[0] < self.action_horizon:
            pad = np.repeat(action_chunk[-1:], self.action_horizon - action_chunk.shape[0], axis=0)
            action_chunk = np.concatenate([action_chunk, pad], axis=0)
        task_arr = ep["task"]
        if task_arr.shape == ():
            task = str(task_arr.item())
        elif task_arr.shape[0] == 1:
            task = str(task_arr[0])
        else:
            task = str(task_arr[frame_idx])

        prompt = task
        indicator = None
        if self.acp is not None and global_idx < len(self.acp.indicator):
            indicator = int(self.acp.indicator[global_idx])
            prompt = build_acp_tagged_task(task, indicator)

        return {
            "observation.images.ego_right": base,
            "observation.images.wrist": wrist,
            "observation.state": state,
            "base": base,
            "wrist": wrist,
            "state": state,
            "action": action_chunk,
            "actions": action_chunk,
            "prompt": prompt,
            "complementary_info.acp_indicator": np.int64(-1 if indicator is None else indicator),
            "episode_index": np.int64(ep_idx),
            "frame_index": np.int64(frame_idx),
            "index": np.int64(global_idx),
        }
