#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dexjoco"))

from dexjoco.ogpo.metrics import add_run_metadata, create_metrics_writer
from dexjoco.ogpo.evaluator import validation_metrics_for_training
from dexjoco.ogpo.replay import (
    BalancedCriticReplay,
    OfflineChunkReplay,
    load_replay,
    make_n_step_replay,
    make_synthetic_replay,
    save_replay,
)
from dexjoco.ogpo.origin_cache import load_or_build_origin_feature_cache
from dexjoco.ogpo.training_control import TrainableSnapshot, ValidationEarlyStopper
from dexjoco.ogpo.critic import soft_update
from dexjoco.ogpo.trainer import (
    accumulated_critic_update,
    apply_scheduled_critic_stage,
    build_train_state,
    critic_update,
    load_checkpoint,
    maybe_advance_critic_stage,
    save_checkpoint,
)


def _deep_update(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    include = cfg.pop("include", None)
    if include:
        cfg = _deep_update(load_config(ROOT / include), cfg)
    return cfg


def fixed_validation_batch(
    replay: OfflineChunkReplay,
    batch_size: int,
    *,
    seed: int,
    stratified: bool,
):
    generator = torch.Generator().manual_seed(seed)
    if not stratified or replay.batch.mc_returns is None:
        return replay.sample(batch_size, generator=generator)
    targets = replay.batch.mc_returns
    low = torch.nonzero(targets < targets.max(), as_tuple=False).flatten()
    high = torch.nonzero(targets == targets.max(), as_tuple=False).flatten()
    if low.numel() == 0 or high.numel() == 0:
        return replay.sample(batch_size, generator=generator)
    high_count = min(batch_size // 2, int(high.numel()))
    low_count = min(batch_size - high_count, int(low.numel()))
    indices = torch.cat(
        [
            high[torch.randperm(high.numel(), generator=generator)[:high_count]],
            low[torch.randperm(low.numel(), generator=generator)[:low_count]],
        ]
    )
    if indices.numel() < batch_size:
        extra = torch.randint(
            len(replay),
            (batch_size - indices.numel(),),
            generator=generator,
        )
        indices = torch.cat([indices, extra])
    return replay.batch.index_select(indices)


def critic_selection_score(metrics: dict[str, float], config: dict) -> tuple[float, bool]:
    selection = config.get("evaluation", {}).get("checkpoint_selection", {})
    ranking = float(metrics["validation_pairwise_ranking_accuracy"])
    correlation = float(metrics["validation_q_rank_correlation"])
    rmse = float(metrics["validation_q_rmse"])
    gap = abs(float(metrics["validation_q_exploitation_gap"]))
    eligible = (
        ranking >= float(selection.get("min_pairwise_ranking_accuracy", 0.0))
        and correlation >= float(selection.get("min_q_rank_correlation", -1.0))
        and gap <= float(selection.get("max_abs_q_exploitation_gap", float("inf")))
    )
    score = (
        ranking
        + float(selection.get("rank_correlation_weight", 0.25)) * correlation
        - float(selection.get("rmse_weight", 0.5)) * rmse
        - float(selection.get("exploitation_gap_weight", 0.5)) * gap
    )
    return score, eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ogpo/critic_udivl.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--overlay", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    for overlay in args.overlay:
        cfg = _deep_update(cfg, load_config(ROOT / overlay))
    data_path = ROOT / cfg["data"]["dataset_path"]
    if not data_path.exists():
        batch = make_synthetic_replay()
        save_replay(batch, data_path)
        print(f"[critic] created synthetic dataset at {data_path}")
    batch = load_replay(data_path)
    validation_path_value = cfg.get("data", {}).get("validation_path")
    if validation_path_value:
        validation_path = ROOT / validation_path_value
        if not validation_path.exists():
            raise FileNotFoundError(f"configured validation replay does not exist: {validation_path}")
        validation_batch = load_replay(validation_path)
    else:
        validation_batch = batch
    n_step = int(cfg.get("data", {}).get("n_step", 1))
    if bool(cfg.get("data", {}).get("apply_n_step_on_load", False)) and n_step > 1:
        batch = make_n_step_replay(batch, n_step=n_step)
        validation_batch = make_n_step_replay(validation_batch, n_step=n_step)
        print(
            f"[critic] applied n_step={n_step} on load "
            f"train={batch.batch_size} validation={validation_batch.batch_size}",
            flush=True,
        )
    sampling_cfg = cfg.get("training", {}).get("critic_sampling", {})
    if bool(sampling_cfg.get("enabled", False)):
        replay = BalancedCriticReplay(
            batch,
            uniform_fraction=float(sampling_cfg.get("uniform_fraction", 0.5)),
            success_fraction=float(sampling_cfg.get("success_fraction", 0.25)),
            terminal_success_fraction=float(
                sampling_cfg.get("terminal_success_fraction", 0.125)
            ),
            failure_fraction=float(sampling_cfg.get("failure_fraction", 0.125)),
        )
        print(f"[critic] balanced sampling={replay.fractions}", flush=True)
    else:
        replay = OfflineChunkReplay(batch)
    validation_replay = OfflineChunkReplay(validation_batch)
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
        validation_replay = OfflineChunkReplay(validation_batch)
    resume = args.resume or cfg["training"].get("resume_checkpoint")
    if resume:
        load_checkpoint(ROOT / resume, state)
        print(f"[critic] resumed checkpoint: {resume}")
    writer = create_metrics_writer(
        ROOT / cfg["training"].get("metrics_path", "outputs/ogpo/critic_metrics.jsonl"),
        ROOT / cfg["training"]["tensorboard_dir"] if cfg["training"].get("tensorboard_dir") else None,
    )
    generator = torch.Generator().manual_seed(11)
    critic_steps = int(cfg["training"].get("critic_steps", cfg["critic"].get("warmup_steps", 5)))
    evaluation_interval = max(1, int(cfg.get("evaluation", {}).get("interval", 10)))
    if bool(cfg.get("evaluation", {}).get("full_validation", False)):
        validation_batch_size = len(validation_replay)
    else:
        validation_batch_size = min(
            len(validation_replay),
            int(cfg.get("evaluation", {}).get("validation_batch_size", cfg["training"].get("batch_size", 16))),
        )
    fixed_validation_sample = (
        fixed_validation_batch(
            validation_replay,
            validation_batch_size,
            seed=int(cfg["training"].get("seed", 0)) + 2903,
            stratified=bool(
                cfg.get("evaluation", {}).get("stratified_validation_batch", False)
            ),
        )
        if (
            bool(cfg.get("evaluation", {}).get("fixed_validation_batch", False))
            and not bool(cfg.get("evaluation", {}).get("full_validation", False))
        )
        else None
    )
    early_cfg = cfg.get("training", {}).get("early_stopping", {})
    early_stopper = (
        ValidationEarlyStopper(
            mode=str(early_cfg.get("mode", "min")),
            patience=int(early_cfg.get("patience", 20)),
            min_delta=float(early_cfg.get("min_delta", 0.0)),
        )
        if bool(early_cfg.get("enabled", False))
        else None
    )
    early_metric = str(early_cfg.get("metric", "validation_q_huber"))
    early_start_stage = early_cfg.get("start_stage")
    effective_batch_size = int(cfg["training"].get("batch_size", 16))
    microbatch_size = int(cfg["training"].get("microbatch_size", effective_batch_size))
    best_snapshot = None
    checkpoint_interval = int(cfg["training"].get("checkpoint_interval", 0))
    latest_checkpoint = cfg["training"].get("latest_checkpoint_path")
    best_checkpoint = cfg["training"].get("best_checkpoint_path")
    milestone_steps = {int(value) for value in cfg["training"].get("milestone_steps", [])}
    milestone_dir = cfg["training"].get("milestone_checkpoint_dir")
    selection_cfg = cfg.get("evaluation", {}).get("checkpoint_selection", {})
    selection_enabled = bool(selection_cfg.get("enabled", False))
    best_selection_score = float("-inf")
    for step in range(critic_steps):
        stage_advanced = apply_scheduled_critic_stage(state, cfg)
        if stage_advanced and early_stopper is not None:
            early_stopper = ValidationEarlyStopper(
                mode=early_stopper.mode,
                patience=early_stopper.patience,
                min_delta=early_stopper.min_delta,
            )
            best_snapshot = None
        sample = replay.sample(effective_batch_size, generator=generator)
        metrics = (
            accumulated_critic_update(
                state,
                sample,
                cfg,
                microbatch_size=microbatch_size,
            )
            if microbatch_size < effective_batch_size
            else critic_update(state, sample, cfg)
        )
        if stage_advanced:
            metrics["critic_stage_advanced"] = 1.0
        evaluation_due = (
            step == 0
            or state.step % evaluation_interval == 0
            or step == critic_steps - 1
        )
        if evaluation_due:
            validation_sample = (
                validation_batch
                if bool(cfg.get("evaluation", {}).get("full_validation", False))
                else (
                    fixed_validation_sample
                    if fixed_validation_sample is not None
                    else validation_replay.sample(
                        validation_batch_size,
                        generator=generator,
                    )
                )
            )
            metrics.update(validation_metrics_for_training(state, validation_sample, cfg))
            if not cfg.get("critic", {}).get("stage_schedule") and maybe_advance_critic_stage(
                state,
                metrics,
                cfg,
            ):
                metrics["critic_stage_advanced"] = 1.0
                if early_stopper is not None:
                    early_stopper = ValidationEarlyStopper(
                        mode=early_stopper.mode,
                        patience=early_stopper.patience,
                        min_delta=early_stopper.min_delta,
                    )
                    best_snapshot = None
            elif (
                early_stopper is not None
                and (early_start_stage is None or state.critic_stage == str(early_start_stage))
            ):
                if early_metric not in metrics:
                    raise KeyError(f"early-stopping metric is missing: {early_metric}")
                if early_stopper.update(float(metrics[early_metric])):
                    best_snapshot = TrainableSnapshot.capture(state.critic)
                metrics["early_stopping_best"] = early_stopper.best
                metrics["early_stopping_stale_evaluations"] = float(
                    early_stopper.stale_evaluations
                )
            if selection_enabled:
                selection_score, selection_eligible = critic_selection_score(metrics, cfg)
                metrics["validation_checkpoint_selection_score"] = selection_score
                metrics["validation_checkpoint_selection_eligible"] = float(selection_eligible)
                min_delta = float(selection_cfg.get("min_delta", 0.0))
                if (
                    selection_eligible
                    and selection_score > best_selection_score + min_delta
                    and best_checkpoint
                ):
                    best_selection_score = selection_score
                    save_checkpoint(state, cfg, ROOT / best_checkpoint)
                    metrics["validation_best_checkpoint_saved"] = 1.0
                    print(
                        f"[critic] best checkpoint step={state.step} "
                        f"score={selection_score:.6f} path={best_checkpoint}",
                        flush=True,
                    )
        metrics["sample_success_fraction"] = float(sample.successes.float().mean().item())
        metrics["sample_done_fraction"] = float(sample.dones.float().mean().item())
        metrics["sample_unique_episodes"] = float(torch.unique(sample.episode_ids).numel())
        metrics["optimizer_step"] = float(state.step)
        add_run_metadata(metrics, config=cfg, step=step)
        writer.write(metrics)
        print(f"[critic] step={step} loss={metrics['critic_loss']:.4f}")
        if checkpoint_interval > 0 and state.step % checkpoint_interval == 0:
            if latest_checkpoint:
                save_checkpoint(state, cfg, ROOT / latest_checkpoint)
                print(
                    f"[critic] latest checkpoint step={state.step} path={latest_checkpoint}",
                    flush=True,
                )
        if state.step in milestone_steps and milestone_dir:
            milestone_path = ROOT / milestone_dir / f"critic_step_{state.step:05d}.pt"
            save_checkpoint(state, cfg, milestone_path)
            print(f"[critic] milestone checkpoint saved: {milestone_path}", flush=True)
        if early_stopper is not None and early_stopper.should_stop:
            if best_snapshot is not None:
                best_snapshot.restore(state.critic)
                soft_update(state.target_critic, state.critic, 1.0)
            print(
                f"[critic] early stop stage={state.critic_stage} "
                f"metric={early_metric} best={early_stopper.best:.6f}"
            )
            break
    if (
        best_snapshot is not None
        and bool(early_cfg.get("restore_best", True))
        and (early_start_stage is None or state.critic_stage == str(early_start_stage))
    ):
        best_snapshot.restore(state.critic)
        soft_update(state.target_critic, state.critic, 1.0)
        print(
            f"[critic] restored best stage={state.critic_stage} "
            f"metric={early_metric} best={early_stopper.best:.6f}"
        )
    save_checkpoint(state, cfg, ROOT / cfg["training"].get("checkpoint_path", "outputs/ogpo/udivl.pt"))
    print("[critic] checkpoint saved")


if __name__ == "__main__":
    main()
