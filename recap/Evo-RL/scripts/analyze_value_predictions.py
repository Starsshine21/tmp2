#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    root = Path('local_data/lerobot_adjust_bottle_value')
    episodes_path = root / 'meta/episodes/chunk-000/file-000.parquet'
    episodes_tbl = pq.read_table(episodes_path)
    episodes = episodes_tbl.to_pydict()
    ep_success = {}
    for i in range(len(episodes['episode_index'])):
        ep_success[int(episodes['episode_index'][i])] = str(episodes['episode_success'][i])

    data_dir = root / 'data/chunk-000'
    rows = []
    for f in sorted(data_dir.glob('file-*.parquet')):
        tbl = pq.read_table(f, columns=['episode_index', 'frame_index', 'complementary_info.value'])
        d = tbl.to_pydict()
        for ep, fr, val in zip(d['episode_index'], d['frame_index'], d['complementary_info.value']):
            rows.append((int(ep), int(fr), float(val)))

    by_ep = {}
    for ep, fr, val in rows:
        by_ep.setdefault(ep, []).append((fr, val))

    ep_summary = []
    for ep, items in sorted(by_ep.items()):
        items = sorted(items)
        vals = np.array([v for _, v in items], dtype=np.float32)
        tail = vals[-5:] if len(vals) >= 5 else vals
        ep_summary.append({
            'episode_index': ep,
            'success': ep_success.get(ep, 'unknown'),
            'num_frames': int(len(vals)),
            'value_mean': float(vals.mean()),
            'value_std': float(vals.std()),
            'value_last': float(vals[-1]),
            'value_tail5_mean': float(tail.mean()),
        })

    def _agg(label: str):
        subset = [r for r in ep_summary if r['success'] == label]
        if not subset:
            return None
        return {
            'episodes': len(subset),
            'value_mean_mean': float(np.mean([r['value_mean'] for r in subset])),
            'value_last_mean': float(np.mean([r['value_last'] for r in subset])),
            'value_tail5_mean': float(np.mean([r['value_tail5_mean'] for r in subset])),
        }

    out = {
        'dataset_root': str(root),
        'num_episodes': len(ep_summary),
        'success_summary': _agg('success'),
        'failure_summary': _agg('failure'),
        'episodes': ep_summary,
    }

    out_dir = Path('outputs/value_infer_adjust_bottle_100000_analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'summary.json'
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
