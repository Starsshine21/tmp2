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
from dexjoco.ogpo.origin_cache import load_or_build_origin_feature_cache
from dexjoco.ogpo.replay import OfflineChunkReplay, load_replay, split_success_buffers
from dexjoco.ogpo.trainer import (
    actor_guard_reason,
    actor_start_gate,
    build_train_state,
    critic_update,
    full_actor_update,
    load_critic_checkpoint,
    load_checkpoint,
    save_checkpoint,
    sync_old_policy,
)
from train_udivl_critic import _deep_update, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ogpo/full_ogpo.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--overlay", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    for overlay in args.overlay:
        cfg = _deep_update(cfg, load_config(ROOT / overlay))
    batch = load_replay(ROOT / cfg["data"]["dataset_path"])
    replay = OfflineChunkReplay(batch)
    validation_path = cfg["data"].get("validation_path")
    validation_batch = load_replay(ROOT / validation_path) if validation_path else batch
    success_batch = split_success_buffers(batch).get("success")
    success_replay = OfflineChunkReplay(success_batch) if success_batch is not None else None
    state = build_train_state(cfg, batch, device=cfg["training"].get("device", "cpu"))
    cache_cfg = cfg.get("data", {}).get("origin_feature_cache", {})
    if bool(cache_cfg.get("enabled", False)):
        batch = load_or_build_origin_feature_cache(
            state,
            batch,
            ROOT / cache_cfg["train_path"],
            inference_batch_size=int(cache_cfg.get("batch_size", 8)),
        )
        validation_batch = load_or_build_origin_feature_cache(
            state,
            validation_batch,
            ROOT / cache_cfg["validation_path"],
            inference_batch_size=int(cache_cfg.get("batch_size", 8)),
        )
        replay = OfflineChunkReplay(batch)
        success_batch = split_success_buffers(batch).get("success")
        success_replay = OfflineChunkReplay(success_batch) if success_batch is not None else None
    resume = args.resume or cfg["training"].get("resume_checkpoint")
    critic_checkpoint = cfg["training"].get("critic_checkpoint")
    critic_steps_per_actor_step = int(cfg["critic"].get("steps_per_actor_step", 4))
    if resume:
        load_checkpoint(ROOT / resume, state)
        print(f"[full] resumed checkpoint: {resume}")
    elif critic_checkpoint:
        load_critic_checkpoint(
            ROOT / critic_checkpoint,
            state,
            load_optimizer=critic_steps_per_actor_step > 0,
        )
        print(f"[full] loaded critic checkpoint: {critic_checkpoint}")
    if critic_steps_per_actor_step == 0:
        state.target_critic.to("cpu")
        if state.target_divl is not None:
            state.target_divl.to("cpu")
        print(
            "[full] actor-only critic mode: critic optimizer state is unused; "
            "target networks moved to CPU"
        )
    writer = create_metrics_writer(
        ROOT / cfg["training"].get("metrics_path", "outputs/ogpo/full_metrics.jsonl"),
        ROOT / cfg["training"]["tensorboard_dir"] if cfg["training"].get("tensorboard_dir") else None,
    )
    generator = torch.Generator().manual_seed(13)
    for step in range(int(cfg["training"].get("actor_steps", 2))):
        sample = replay.sample(int(cfg["training"].get("batch_size", 16)), generator=generator)
        for _ in range(critic_steps_per_actor_step):
            critic_update(state, sample, cfg)
        gate_reason, gate_metrics = actor_start_gate(
            state, validation_batch, cfg, outer_step=step
        )
        if gate_reason:
            metrics: dict[str, float | str] = {
                **gate_metrics,
                "actor_skipped": 1.0,
                "stop_reason": gate_reason,
            }
            add_run_metadata(metrics, config=cfg, step=step)
            writer.write(metrics)
            print(f"[full] step={step} actor skipped: {gate_reason}")
            continue
        fm_sample = replay.sample(int(cfg["training"].get("batch_size", 16)), generator=generator)
        success_sample = None
        if success_replay is not None:
            success_count = max(
                1,
                round(
                    int(cfg["training"].get("batch_size", 16))
                    * float(cfg["data"].get("success_sampling_ratio", 0.5))
                ),
            )
            success_sample = success_replay.sample(
                min(success_count, len(success_replay)),
                generator=generator,
            )
        metrics = full_actor_update(state, sample, cfg, fm_batch=fm_sample, success_batch=success_sample)
        metrics.update(gate_metrics)
        sync_period = int(cfg["actor"].get("old_policy_sync_period", 1))
        if sync_period > 0 and (step + 1) % sync_period == 0:
            sync_old_policy(state, ema=float(cfg["actor"].get("old_policy_ema", 0.0)))
        add_run_metadata(metrics, config=cfg, step=step)
        stop_reason = actor_guard_reason(metrics, cfg)
        metrics["stop_reason"] = stop_reason or ""
        writer.write(metrics)
        print(f"[full] step={step} actor_loss={metrics['actor_loss']:.4f}")
        if stop_reason:
            print(f"[full] stopping actor extraction: {stop_reason}")
            break
    save_checkpoint(state, cfg, ROOT / cfg["training"].get("checkpoint_path", "outputs/ogpo/full.pt"))
    print("[full] checkpoint saved")


if __name__ == "__main__":
    main()
