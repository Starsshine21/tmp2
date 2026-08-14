#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.utils import write_episodes, write_info, write_stats, write_tasks

SRC = Path('local_data/lerobot_adjust_bottle_value')
DST = Path('local_data/lerobot_adjust_bottle_filtered')
INDICATOR_FIELD = 'complementary_info.acp_indicator'
MIN_FRAMES_PER_EPISODE = 4


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    (DST / 'data/chunk-000').mkdir(parents=True, exist_ok=True)

    info = json.loads((SRC / 'meta/info.json').read_text())
    tasks_df = pd.read_parquet(SRC / 'meta/tasks.parquet')
    episodes_tbl = pq.read_table(SRC / 'meta/episodes/chunk-000/file-000.parquet')
    episodes = episodes_tbl.to_pydict()
    features = info['features']

    write_tasks(tasks_df, DST)

    all_episode_stats = []
    episode_rows = []
    total_frames = 0
    next_ep_idx = 0

    for src_ep_idx, file_path in enumerate(sorted((SRC / 'data/chunk-000').glob('file-*.parquet'))):
        table = pq.read_table(file_path)
        indicator = np.asarray(table.column(INDICATOR_FIELD).to_pylist(), dtype=np.int64)
        keep_idx = np.where(indicator == 1)[0]
        if keep_idx.size < MIN_FRAMES_PER_EPISODE:
            continue

        filtered = table.take(pa.array(keep_idx))
        data = filtered.to_pydict()
        n = len(data['frame_index'])
        task_name = episodes['tasks'][src_ep_idx][0] if isinstance(episodes['tasks'][src_ep_idx], list) else episodes['tasks'][src_ep_idx]
        ep_success = episodes['episode_success'][src_ep_idx]

        data['episode_index'] = [next_ep_idx] * n
        data['frame_index'] = list(range(n))
        data['index'] = list(range(total_frames, total_frames + n))

        out_table = pa.Table.from_pydict(data, schema=filtered.schema)
        out_path = DST / f'data/chunk-000/file-{next_ep_idx:03d}.parquet'
        pq.write_table(out_table, out_path)

        stats_input = {}
        for key, feature in features.items():
            if feature['dtype'] == 'image':
                continue
            stats_input[key] = np.asarray(data[key])
        episode_stats = compute_episode_stats(stats_input, {k: v for k, v in features.items() if v['dtype'] != 'image'})
        all_episode_stats.append(episode_stats)

        episode_rows.append({
            'episode_index': next_ep_idx,
            'tasks': [task_name],
            'length': n,
            'episode_success': ep_success,
            'data/chunk_index': 0,
            'data/file_index': next_ep_idx,
        })
        total_frames += n
        next_ep_idx += 1

    episodes_ds = datasets.Dataset.from_list(episode_rows)
    write_episodes(episodes_ds, DST)

    new_info = dict(info)
    new_info['codebase_version'] = 'v3.0'
    new_info['total_episodes'] = len(episode_rows)
    new_info['total_frames'] = total_frames
    new_info['total_tasks'] = int(len(tasks_df))
    new_info['data_path'] = 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'
    write_info(new_info, DST)

    stats = aggregate_stats(all_episode_stats) if all_episode_stats else {}
    image_stats = {
        'mean': np.zeros((3, 1, 1), dtype=np.float32),
        'std': np.ones((3, 1, 1), dtype=np.float32),
        'min': np.zeros((3, 1, 1), dtype=np.float32),
        'max': np.ones((3, 1, 1), dtype=np.float32),
        'count': np.array([total_frames], dtype=np.int64),
    }
    for key, feature in features.items():
        if feature['dtype'] == 'image':
            stats[key] = image_stats
    write_stats(stats, DST)

    summary = {
        'src': str(SRC),
        'dst': str(DST),
        'kept_episodes': len(episode_rows),
        'kept_frames': total_frames,
        'min_frames_per_episode': MIN_FRAMES_PER_EPISODE,
        'indicator_field': INDICATOR_FIELD,
    }
    (DST / 'filter_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
