#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = 'local_data/lerobot_adjust_bottle_filtered'
VALUE_FIELD = 'complementary_info.value'
OUT_DIR = Path('outputs/value_progress_eval')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def resample_1d(values: np.ndarray, n: int = 100) -> np.ndarray:
    if len(values) == 1:
        return np.repeat(values, n)
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, values)


def main() -> None:
    ds = LeRobotDataset(REPO_ID)
    raw = ds.hf_dataset.with_format(None)
    episode_index = np.asarray(raw['episode_index'], dtype=np.int64).reshape(-1)
    values = np.asarray(raw[VALUE_FIELD], dtype=np.float32).reshape(-1)

    unique_eps = sorted(np.unique(episode_index).tolist())
    episode_rows = []
    resampled = []

    for ep in unique_eps:
        mask = episode_index == ep
        ep_values = values[mask]
        if ep_values.size == 0:
            continue
        start_mean = float(np.mean(ep_values[: max(1, ep_values.size // 10)]))
        end_mean = float(np.mean(ep_values[-max(1, ep_values.size // 10) :]))
        delta = end_mean - start_mean
        episode_rows.append(
            {
                'episode_index': int(ep),
                'num_frames': int(ep_values.size),
                'start_value': start_mean,
                'end_value': end_mean,
                'delta': delta,
                'monotonic_fraction': float(np.mean(np.diff(ep_values) >= 0)) if ep_values.size > 1 else 1.0,
            }
        )
        resampled.append(resample_1d(ep_values, 100))

    resampled_arr = np.stack(resampled, axis=0)
    mean_curve = resampled_arr.mean(axis=0)
    std_curve = resampled_arr.std(axis=0)

    with (OUT_DIR / 'episode_stats.json').open('w') as f:
        json.dump(episode_rows, f, indent=2)

    summary = {
        'dataset': REPO_ID,
        'value_field': VALUE_FIELD,
        'num_episodes': len(episode_rows),
        'mean_start_value': float(np.mean([r['start_value'] for r in episode_rows])),
        'mean_end_value': float(np.mean([r['end_value'] for r in episode_rows])),
        'mean_delta': float(np.mean([r['delta'] for r in episode_rows])),
        'positive_delta_fraction': float(np.mean([r['delta'] > 0 for r in episode_rows])),
        'mean_monotonic_fraction': float(np.mean([r['monotonic_fraction'] for r in episode_rows])),
    }
    with (OUT_DIR / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2)

    if plt is not None:
        x = np.linspace(0, 100, 100)
        plt.figure(figsize=(8, 5))
        plt.plot(x, mean_curve, label='mean value')
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.25, label='±1 std')
        plt.xlabel('Normalized episode progress (%)')
        plt.ylabel('Value score')
        plt.title('Value score vs. normalized progress')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'mean_progress_curve.png', dpi=180)
        plt.close()

        plt.figure(figsize=(6, 5))
        deltas = np.asarray([r['delta'] for r in episode_rows], dtype=np.float32)
        plt.hist(deltas, bins=min(20, max(5, len(deltas))))
        plt.xlabel('End value - Start value')
        plt.ylabel('Episode count')
        plt.title('Per-episode value improvement')
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'delta_hist.png', dpi=180)
        plt.close()
    else:
        print('matplotlib not installed; skipped png plots')

    print(json.dumps(summary, indent=2))
    print(f'wrote: {OUT_DIR}')


if __name__ == '__main__':
    main()
