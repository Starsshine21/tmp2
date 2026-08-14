#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dexjoco"))

from dexjoco.ogpo.evaluator import offline_calibration_metrics
from dexjoco.ogpo.replay import OfflineChunkReplay, make_synthetic_replay, save_replay
from dexjoco.ogpo.trainer import (
    build_train_state,
    critic_update,
    flash_actor_update,
    full_actor_update,
    load_checkpoint,
    save_checkpoint,
    sync_old_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dexjoco", action="store_true")
    args = parser.parse_args()
    cfg = {
        "critic": {"ensemble_size": 3, "hidden_dim": 64, "num_layers": 2, "steps_per_actor_step": 1},
        "divl": {"num_atoms": 31, "v_min": -5.0, "v_max": 5.0, "alpha_min": 0.5, "alpha_max": 0.8},
        "actor": {"group_size": 3, "hidden_dim": 64, "ppo_clip_min": 0.05, "ppo_clip_max": 0.2},
        "flow": {"num_steps": 4, "selected_timestep": 2, "stochastic_variance": 0.04},
        "regularization": {"lambda_fm": 0.05, "beta_kl": 0.01},
    }
    batch = make_synthetic_replay(num_samples=24, generated_horizon=5, executed_horizon=2, action_dim=3)
    replay = OfflineChunkReplay(batch)
    state = build_train_state(cfg, batch)
    generator = torch.Generator().manual_seed(23)
    for _ in range(2):
        critic_update(state, replay.sample(12, generator=generator), cfg)
    sample = replay.sample(12, generator=generator)
    full_metrics = full_actor_update(state, sample, cfg)
    sync_old_policy(state)
    flash_metrics = flash_actor_update(state, sample, cfg)
    ckpt = ROOT / "outputs/ogpo/smoke_checkpoint.pt"
    replay_path = ROOT / "outputs/ogpo/smoke_replay.pt"
    save_replay(batch, replay_path)
    save_checkpoint(state, cfg, ckpt)
    load_checkpoint(ckpt, state)
    offline = offline_calibration_metrics(state.critic, batch)
    print("[smoke] critic_update=ok")
    print(f"[smoke] full_actor_loss={full_metrics['actor_loss']:.6f}")
    print(f"[smoke] flash_actor_loss={flash_metrics['actor_loss']:.6f}")
    print(f"[smoke] checkpoint={ckpt}")
    print(f"[smoke] offline_q_rmse={offline['q_rmse']:.6f}")
    if not args.skip_dexjoco:
        dexjoco_python = ROOT / ".conda" / "dexjoco" / "bin" / "python"
        subprocess.run(
            [str(dexjoco_python), str(ROOT / "scripts" / "run_dexjoco_env_smoke.py")],
            cwd=ROOT,
            check=True,
        )
        print("[smoke] dexjoco_eval=ok")


if __name__ == "__main__":
    main()
