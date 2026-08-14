#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import datasets
import numpy as np
from datasets import Dataset

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_FEATURES, get_hf_features_from_features, write_info, write_stats, write_episodes

FPS = 10


def build_features(state_dim: int):
    features = {
        'observation.images.front': {'dtype': 'image', 'shape': (224, 224, 3), 'names': ['height', 'width', 'channel']},
        'observation.images.wrist': {'dtype': 'image', 'shape': (224, 224, 3), 'names': ['height', 'width', 'channel']},
        'observation.state': {'dtype': 'float32', 'shape': (state_dim,), 'names': None},
        'task': {'dtype': 'string', 'shape': (1,), 'names': None},
    }
    features.update(DEFAULT_FEATURES)
    return features


def materialize_for_stats(columns: dict[str, list], features: dict) -> dict:
    stats_data = {}
    for key, feature in features.items():
        if feature['dtype'] == 'image':
            continue
        stats_data[key] = np.asarray(columns[key])
    return stats_data


def main() -> None:
    raw = Path('../../pi05/data/recap_adjust_bottle_sim/raw_rollouts')
    dst = Path('local_data/lerobot_adjust_bottle_value')
    if dst.exists():
        shutil.rmtree(dst)

    ep_dirs = sorted(p for p in raw.glob('episode_*') if p.is_dir())
    if not ep_dirs:
        raise FileNotFoundError(raw)

    sample_npz = np.load(ep_dirs[0] / 'frames.npz')
    state_dim = int(sample_npz['state'].shape[1])
    meta = LeRobotDatasetMetadata.create(
        repo_id='local/adjust_bottle_value',
        root=dst,
        fps=FPS,
        robot_type='aloha',
        features=build_features(state_dim),
        use_videos=False,
    )

    all_episode_stats = []
    episode_rows = []
    global_offset = 0
    for ep_idx, ep_dir in enumerate(ep_dirs):
        frames = np.load(ep_dir / 'frames.npz')
        ep_meta = json.loads((ep_dir / 'meta.json').read_text())
        task = str(ep_meta['prompt'])
        success = 'success' if bool(ep_meta['success']) else 'failure'
        if meta.tasks is None or task not in meta.tasks.index:
            meta.save_episode_tasks([task])
        task_index = meta.get_task_index(task)
        n = int(frames['state'].shape[0])
        columns = {
            'observation.images.front': [img for img in frames['image'][:n]],
            'observation.images.wrist': [img for img in frames['wrist_image'][:n]],
            'observation.state': [row.tolist() for row in frames['state'][:n]],
            'task': [task] * n,
            'episode_index': [ep_idx] * n,
            'frame_index': list(range(n)),
            'index': list(range(global_offset, global_offset + n)),
            'task_index': [task_index] * n,
            'timestamp': [float(x) for x in frames['timestamp'][:n]],
        }
        hf_dataset = Dataset.from_dict(columns, features=get_hf_features_from_features(meta.features))
        episode_rows.append({
            'episode_index': ep_idx,
            'tasks': [task],
            'length': n,
            'episode_success': success,
            'data/chunk_index': 0,
            'data/file_index': ep_idx,
        })
        data_path = dst / f'data/chunk-000/file-{ep_idx:03d}.parquet'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        hf_dataset.to_parquet(str(data_path))
        episode_stats = compute_episode_stats(materialize_for_stats(columns, meta.features), {k: v for k, v in meta.features.items() if v['dtype'] != 'image'})
        all_episode_stats.append(episode_stats)
        global_offset += n

    episodes_ds = datasets.Dataset.from_list(episode_rows)
    write_episodes(episodes_ds, dst)
    write_info({
        'codebase_version': 'v3.0',
        'robot_type': meta.robot_type,
        'total_episodes': len(episode_rows),
        'total_frames': global_offset,
        'total_tasks': meta.total_tasks,
        'total_videos': 0,
        'total_chunks': 1,
        'chunks_size': meta.chunks_size,
        'fps': meta.fps,
        'splits': {'train': f'0:{len(episode_rows)}'},
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'video_path': None,
        'features': meta.features,
    }, dst)
    stats = aggregate_stats(all_episode_stats)
    image_stats = {
        'mean': np.zeros((3, 1, 1), dtype=np.float32),
        'std': np.ones((3, 1, 1), dtype=np.float32),
        'min': np.zeros((3, 1, 1), dtype=np.float32),
        'max': np.ones((3, 1, 1), dtype=np.float32),
        'count': np.array([global_offset], dtype=np.int64),
    }
    stats['observation.images.front'] = image_stats
    stats['observation.images.wrist'] = image_stats
    write_stats(stats, dst)
    print(json.dumps({'output_dir': str(dst), 'episodes': len(episode_rows), 'frames': global_offset}, indent=2))


if __name__ == '__main__':
    main()
