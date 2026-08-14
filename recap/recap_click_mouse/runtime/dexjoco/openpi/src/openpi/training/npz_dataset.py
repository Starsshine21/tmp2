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


class SingleNpzDataset(Dataset):
    """Dataset backed by a single large NPZ file containing all episodes.

    This avoids the overhead of per-episode file open/close and enables
    fast random access via memory-mapped numpy arrays.

    The NPZ file should contain:
        - base: (N, H, W, 3) uint8 - base camera images
        - wrist: (N, H, W, 3) uint8 - wrist camera images
        - state: (N, state_dim) float32
        - action: (N, action_dim) float32
        - episode_index: (N,) int64
        - frame_index: (N,) int64
        - index: (N,) int64
        - acp_indicator: (N,) int64 (optional, can be in sidecar instead)
        - value: (N,) float32 (optional)
        - advantage: (N,) float32 (optional)
    """

    def __init__(
        self,
        npz_path: str | Path,
        acp_sidecar: str | Path | None = None,
        action_horizon: int = 30,
        load_to_memory: bool = True,
    ):
        self.npz_path = Path(npz_path)
        self.action_horizon = action_horizon

        print(f"Loading NPZ: {self.npz_path}")
        start = time.time()

        if load_to_memory:
            self._data = dict(np.load(self.npz_path))
            self._mmap = None
        else:
            self._mmap = np.load(self.npz_path, mmap_mode="r")
            self._data = None

        elapsed = time.time() - start
        print(f"Loaded in {elapsed:.1f}s")

        # Determine length from action array
        self.total_len = int(self._get_array("action").shape[0])

        # Load ACP sidecar if provided
        if acp_sidecar and Path(acp_sidecar).exists():
            self.acp = load_acp_sidecar(acp_sidecar)
            print(f"Loaded ACP sidecar: {acp_sidecar}")
        else:
            self.acp = None

        # Check if ACP info is in the NPZ itself
        if self.acp is None and "acp_indicator" in self._get_arrays():
            self.acp = RawACPIndex(
                value=self._get_array("value") if "value" in self._get_arrays() else np.zeros(self.total_len, dtype=np.float32),
                advantage=self._get_array("advantage") if "advantage" in self._get_arrays() else np.zeros(self.total_len, dtype=np.float32),
                indicator=self._get_array("acp_indicator"),
            )
            print("Using ACP info from NPZ file")

        # Pre-compute episode boundaries for fast lookup
        if "episode_index" in self._get_arrays():
            ep_indices = self._get_array("episode_index")
            self._episode_starts = []
            self._episode_ends = []
            current_ep = 0
            for i in range(len(ep_indices)):
                if int(ep_indices[i]) != current_ep:
                    self._episode_ends.append(i)
                    self._episode_starts.append(i)
                    current_ep = int(ep_indices[i])
            self._episode_ends.append(len(ep_indices))
            self._num_episodes = len(self._episode_starts)
        else:
            self._episode_starts = [0]
            self._episode_ends = [self.total_len]
            self._num_episodes = 1

        print(f"Dataset: {self.total_len} frames, {self._num_episodes} episodes")

    def _get_arrays(self) -> set[str]:
        if self._data is not None:
            return set(self._data.keys())
        return set(self._mmap.keys())

    def _get_array(self, key: str) -> np.ndarray:
        if self._data is not None:
            return self._data[key]
        return self._mmap[key]

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += self.total_len
        if idx < 0 or idx >= self.total_len:
            raise IndexError(idx)

        # Direct array access - no file I/O, no decompression
        base = np.asarray(self._get_array("base")[idx], dtype=np.uint8)
        wrist = np.asarray(self._get_array("wrist")[idx], dtype=np.uint8)
        state = np.asarray(self._get_array("state")[idx], dtype=np.float32)

        # Action chunk with horizon
        all_actions = self._get_array("action")
        end_idx = min(idx + self.action_horizon, all_actions.shape[0])
        action_chunk = np.asarray(all_actions[idx:end_idx], dtype=np.float32)
        if action_chunk.shape[0] < self.action_horizon:
            pad = np.repeat(action_chunk[-1:], self.action_horizon - action_chunk.shape[0], axis=0)
            action_chunk = np.concatenate([action_chunk, pad], axis=0)

        # Task
        task = "click_mouse"
        if "task" in self._get_arrays():
            task_arr = self._get_array("task")
            if task_arr.ndim == 1 and task_arr.shape[0] == 1:
                task = str(task_arr[0])
            elif task_arr.shape[0] == self.total_len:
                task = str(task_arr[idx])

        # ACP injection
        prompt = task
        indicator = None
        if self.acp is not None and idx < len(self.acp.indicator):
            indicator = int(self.acp.indicator[idx])
            prompt = build_acp_tagged_task(task, indicator)

        # Episode info
        ep_idx = 0
        frame_idx = idx
        if "episode_index" in self._get_arrays():
            ep_idx = int(self._get_array("episode_index")[idx])
        if "frame_index" in self._get_arrays():
            frame_idx = int(self._get_array("frame_index")[idx])

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
            "index": np.int64(idx),
        }


import time


class MultiEpisodeNpzDataset(Dataset):
    """Dataset backed by multiple per-episode NPZ files (like DexJoCo_ReCap).

    Each episode NPZ contains:
        - observation_images_base: (T, H, W, 3) uint8
        - observation_images_wrist: (T, H, W, 3) uint8
        - observation_state: (T, state_dim) float32
        - action: (T, action_dim) float32
        - task: string or (T,) string array
    """

    def __init__(
        self,
        raw_root: str | Path,
        acp_sidecar: str | Path | None = None,
        action_horizon: int = 30,
    ):
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

        print(f"MultiEpisodeNpzDataset: {len(self.episode_files)} episodes, {self.total_len} frames")

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
        ep_file = self.episode_files[ep_idx]

        with np.load(ep_file) as ep:
            base = np.asarray(ep["observation_images_base"][frame_idx], dtype=np.uint8)
            wrist = np.asarray(ep["observation_images_wrist"][frame_idx], dtype=np.uint8)
            state = np.asarray(ep["observation_state"][frame_idx], dtype=np.float32)
            all_actions = np.asarray(ep["action"], dtype=np.float32)
            end_idx = min(frame_idx + self.action_horizon, all_actions.shape[0])
            action_chunk = all_actions[frame_idx:end_idx]
            if action_chunk.shape[0] < self.action_horizon:
                pad = np.repeat(action_chunk[-1:], self.action_horizon - action_chunk.shape[0], axis=0)
                action_chunk = np.concatenate([action_chunk, pad], axis=0)

            task_arr = np.asarray(ep["task"]) if "task" in ep else np.array(["click_mouse"])
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
