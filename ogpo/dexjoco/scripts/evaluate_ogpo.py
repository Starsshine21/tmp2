#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dexjoco"))
sys.path.insert(0, str(ROOT / "scripts"))

from dexjoco.ogpo.evaluator import fit_conformal_calibration, offline_calibration_metrics
from dexjoco.ogpo.experiment_matrix import METHOD_SPECS, get_method_spec
from dexjoco.ogpo.replay import load_replay
from dexjoco.ogpo.trainer import build_train_state, load_critic_checkpoint
from train_udivl_critic import _deep_update, load_config


def _replay_compatible(left, right) -> bool:
    return (
        left.obs_dim == right.obs_dim
        and left.generated_horizon == right.generated_horizon
        and left.action_dim == right.action_dim
    )


def _print_dexjoco_eval_command(
    cfg: dict,
    checkpoint: str | None = None,
    *,
    method: str = "udivl_flash",
) -> None:
    evaluation_cfg = cfg.get("evaluation", {})
    tasks = evaluation_cfg.get("dexjoco_tasks", ["click_mouse"])
    seeds = evaluation_cfg.get("seeds", [0])
    episodes = int(evaluation_cfg.get("num_episodes", 1))
    config_set = str(evaluation_cfg.get("config_set", "rand_obj"))
    policy_dir = str(
        evaluation_cfg.get(
            "policy_dir",
            "/nfs_global/S/yangrongzheng/evo-RL/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse",
        )
    )
    flow_cfg = cfg.get("flow", {})
    use_pi05_adapter = method != "pi05_sft" and str(flow_cfg.get("adapter", "")) == "pi05_pytorch"
    for task in tasks:
        for seed in seeds:
            print("cd /nfs_global/S/yangrongzheng/evo-RL/dexjoco")
            print("export PI05_USE_CLEAN_OPENPI_ENV=1")
            print(f"export PI05_TASK={task}")
            print(f"export PI05_CONFIG_SET={config_set}")
            print(f"export PI05_SEED={seed}")
            print(f"export PI05_EPISODES={episodes}")
            print(f"export ROLLOUT_EPISODES={episodes}")
            if use_pi05_adapter:
                ogpo_checkpoint = checkpoint or cfg.get("training", {}).get("checkpoint_path")
                print(f"export PI05_PYTORCH_POLICY_DIR={flow_cfg['checkpoint_dir']}")
                print(f"export OGPO_CHECKPOINT={ogpo_checkpoint}")
                print("bash scripts/run_pi05_ogpo_server.sh")
            else:
                print(f"export PI05_POLICY_DIR={policy_dir}")
                print("bash scripts/run_pi05_server.sh")
            print("# In a second terminal after the server is ready:")
            print("bash scripts/run_pi05_rollout_collect.sh")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--method", choices=sorted(METHOD_SPECS), default="udivl_flash")
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--mode", choices=["offline", "dexjoco-command"], default="offline")
    parser.add_argument("--validation-replay")
    parser.add_argument("--fit-conformal", action="store_true")
    parser.add_argument("--calibration-output")
    parser.add_argument("--calibrated-checkpoint-output")
    parser.add_argument("--overlay", action="append", default=[])
    args = parser.parse_args()
    if args.list_methods:
        for name, spec in METHOD_SPECS.items():
            print(f"{name}: algorithm={spec.algorithm} config={spec.config_path}")
        return
    method_spec = get_method_spec(args.method)
    config_path = args.config or method_spec.config_path
    cfg = load_config(ROOT / config_path)
    for overlay in args.overlay:
        cfg = _deep_update(cfg, load_config(ROOT / overlay))
    if args.mode == "dexjoco-command":
        _print_dexjoco_eval_command(cfg, args.checkpoint, method=args.method)
        return
    if not method_spec.requires_ogpo_checkpoint:
        parser.error("PI0.5 SFT has no OGPO offline checkpoint; use --mode dexjoco-command")
    if args.checkpoint is None:
        parser.error("--checkpoint is required for --mode offline")
    checkpoint_path = ROOT / args.checkpoint
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_cfg = _deep_update(cfg, checkpoint_payload.get("config", {}))
    batch = load_replay(ROOT / state_cfg["data"]["dataset_path"])
    state = build_train_state(state_cfg, batch, device=state_cfg["training"].get("device", "cpu"))
    # Offline critic evaluation must remain compatible with checkpoints made
    # before optimizer parameter-group changes. Actor and optimizer state are
    # irrelevant here; restore only value weights, support, and calibration.
    load_critic_checkpoint(checkpoint_path, state, load_optimizer=False)
    validation_path = Path(args.validation_replay) if args.validation_replay else None
    if validation_path is None:
        dataset_path = ROOT / state_cfg["data"]["dataset_path"]
        base_stem = dataset_path.stem.removesuffix("_train")
        candidate = dataset_path.with_name(f"{base_stem}_validation{dataset_path.suffix}")
        if candidate.exists():
            candidate_batch = load_replay(candidate)
            validation_path = candidate if _replay_compatible(batch, candidate_batch) else dataset_path
        else:
            validation_path = dataset_path
    elif not validation_path.is_absolute():
        validation_path = ROOT / validation_path
    validation_batch = load_replay(validation_path)
    inference_batch_size = int(
        state_cfg.get("evaluation", {}).get(
            "validation_batch_size",
            state_cfg.get("training", {}).get("batch_size", 16),
        )
    )
    if args.fit_conformal:
        fit_conformal_calibration(
            state,
            validation_batch,
            state_cfg,
            inference_batch_size=inference_batch_size,
        )
    metrics = offline_calibration_metrics(
        state.critic,
        validation_batch,
        divl=state.divl,
        conformal_scale=state.conformal_scale,
        inference_batch_size=inference_batch_size,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    if args.calibration_output:
        output = Path(args.calibration_output)
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"calibration_output: {output}")
    if args.calibrated_checkpoint_output:
        output = Path(args.calibrated_checkpoint_output)
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload["conformal_scale"] = state.conformal_scale
        torch.save(checkpoint_payload, output)
        print(f"calibrated_checkpoint: {output}")


if __name__ == "__main__":
    main()
