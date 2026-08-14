#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dexjoco"))
sys.path.insert(0, str(ROOT / "scripts"))

from dexjoco.ogpo.metrics import add_run_metadata, create_metrics_writer
from dexjoco.ogpo.replay import OfflineChunkReplay, load_replay
from dexjoco.ogpo.trainer import (
    actor_guard_reason,
    actor_start_gate,
    awr_actor_update,
    build_train_state,
    critic_update,
    load_checkpoint,
    save_checkpoint,
    sync_old_policy,
)
from train_udivl_critic import _deep_update, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ogpo/awr.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--overlay", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    for overlay in args.overlay:
        cfg = _deep_update(cfg, load_config(ROOT / overlay))
    if str(cfg.get("actor", {}).get("algorithm")) != "awr":
        parser.error("train_awr.py requires actor.algorithm: awr")

    batch = load_replay(ROOT / cfg["data"]["dataset_path"])
    replay = OfflineChunkReplay(batch)
    validation_path = cfg["data"].get("validation_path")
    validation_batch = load_replay(ROOT / validation_path) if validation_path else batch
    state = build_train_state(cfg, batch, device=cfg["training"].get("device", "cpu"))
    resume = args.resume or cfg["training"].get("resume_checkpoint")
    if resume:
        load_checkpoint(ROOT / resume, state)
        print(f"[awr] resumed checkpoint: {resume}")

    writer = create_metrics_writer(
        ROOT / cfg["training"].get("metrics_path", "outputs/ogpo/awr_metrics.jsonl"),
        ROOT / cfg["training"]["tensorboard_dir"] if cfg["training"].get("tensorboard_dir") else None,
    )
    generator = torch.Generator().manual_seed(int(cfg["training"].get("seed", 0)))
    for step in range(int(cfg["training"].get("actor_steps", 2))):
        sample = replay.sample(int(cfg["training"].get("batch_size", 16)), generator=generator)
        for _ in range(int(cfg["critic"].get("steps_per_actor_step", 4))):
            critic_update(state, sample, cfg)
        gate_reason, gate_metrics = actor_start_gate(state, validation_batch, cfg, outer_step=step)
        if gate_reason:
            metrics: dict[str, float | str] = {
                **gate_metrics,
                "actor_skipped": 1.0,
                "stop_reason": gate_reason,
            }
            add_run_metadata(metrics, config=cfg, step=step)
            writer.write(metrics)
            print(f"[awr] step={step} actor skipped: {gate_reason}")
            continue
        metrics = awr_actor_update(state, sample, cfg)
        metrics.update(gate_metrics)
        sync_period = int(cfg["actor"].get("old_policy_sync_period", 1))
        if sync_period > 0 and (step + 1) % sync_period == 0:
            sync_old_policy(state, ema=float(cfg["actor"].get("old_policy_ema", 0.0)))
        add_run_metadata(metrics, config=cfg, step=step)
        stop_reason = actor_guard_reason(metrics, cfg)
        metrics["stop_reason"] = stop_reason or ""
        writer.write(metrics)
        print(f"[awr] step={step} actor_loss={metrics['actor_loss']:.4f}")
        if stop_reason:
            print(f"[awr] stopping actor extraction: {stop_reason}")
            break
    save_checkpoint(state, cfg, ROOT / cfg["training"].get("checkpoint_path", "outputs/ogpo/awr.pt"))
    print("[awr] checkpoint saved")


if __name__ == "__main__":
    main()
