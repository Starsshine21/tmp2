#!/usr/bin/env python3
"""Evaluate PI0.5 PyTorch checkpoints with the official OpenPI model/data path."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch


DEFAULT_CONFIG_NAME = "pi05_pickplace_dexhand_full_lora_pytorch_32"
DEFAULT_REPO_ID = "local/pi05-pickplace-il"
DEFAULT_DATA_DIR = "/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place_dexhand"
DEFAULT_CHECKPOINT_DIR = (
    "/nfs_global/S/yangrongzheng/pi05/results/openpi_official_pytorch_full_checkpoints/"
    "pi05_pickplace_dexhand_full_lora_pytorch_32/pi05_pickplace_dexhand_eef_delta_train_lora/60000"
)


def _as_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _action_stat(norm_stats: dict, name: str) -> np.ndarray | None:
    stats = norm_stats["actions"]
    if isinstance(stats, dict):
        return _as_array(stats.get(name))
    return _as_array(getattr(stats, name))


def normalize_quantile(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    q01 = np.asarray(q01, dtype=np.float32)
    q99 = np.asarray(q99, dtype=np.float32)
    dim = min(q01.shape[-1], x.shape[-1])
    head = (x[..., :dim] - q01[..., :dim]) / (q99[..., :dim] - q01[..., :dim] + 1e-6) * 2.0 - 1.0
    if dim < x.shape[-1]:
        return np.concatenate([head, x[..., dim:]], axis=-1)
    return head


def unnormalize_quantile(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    q01 = np.asarray(q01, dtype=np.float32)
    q99 = np.asarray(q99, dtype=np.float32)
    dim = min(q01.shape[-1], x.shape[-1])
    head = (x[..., :dim] + 1.0) / 2.0 * (q99[..., :dim] - q01[..., :dim] + 1e-6) + q01[..., :dim]
    if dim < x.shape[-1]:
        return np.concatenate([head, x[..., dim:]], axis=-1)
    return head


def normalize_actions(x: np.ndarray, norm_stats: dict) -> np.ndarray:
    q01 = _action_stat(norm_stats, "q01")
    q99 = _action_stat(norm_stats, "q99")
    if q01 is not None and q99 is not None:
        return normalize_quantile(x, q01, q99)
    mean = _action_stat(norm_stats, "mean")
    std = _action_stat(norm_stats, "std")
    if mean is None or std is None:
        raise ValueError("actions norm stats must contain either q01/q99 or mean/std")
    dim = min(mean.shape[-1], x.shape[-1])
    head = (x[..., :dim] - mean[..., :dim]) / (std[..., :dim] + 1e-6)
    if dim < x.shape[-1]:
        return np.concatenate([head, x[..., dim:]], axis=-1)
    return head


def unnormalize_actions(x: np.ndarray, norm_stats: dict) -> np.ndarray:
    q01 = _action_stat(norm_stats, "q01")
    q99 = _action_stat(norm_stats, "q99")
    if q01 is not None and q99 is not None:
        return unnormalize_quantile(x, q01, q99)
    mean = _action_stat(norm_stats, "mean")
    std = _action_stat(norm_stats, "std")
    if mean is None or std is None:
        raise ValueError("actions norm stats must contain either q01/q99 or mean/std")
    dim = min(mean.shape[-1], x.shape[-1])
    head = x[..., :dim] * (std[..., :dim] + 1e-6) + mean[..., :dim]
    if dim < x.shape[-1]:
        return np.concatenate([head, x[..., dim:]], axis=-1)
    return head


def load_norm_stats(path: str | Path) -> dict[str, dict[str, np.ndarray | None]]:
    with Path(path).open() as f:
        data = json.load(f)
    stats = {}
    for key, val in data["norm_stats"].items():
        stats[key] = {
            "mean": _as_array(val.get("mean")),
            "std": _as_array(val.get("std")),
            "q01": _as_array(val.get("q01")),
            "q99": _as_array(val.get("q99")),
        }
    return stats


def compute_action_mse(
    pred_norm_actions: np.ndarray,
    gt_actions: np.ndarray,
    norm_stats: dict,
    *,
    action_dims: int = 12,
    gt_is_normalized: bool = False,
) -> dict[str, Any]:
    pred_norm = np.asarray(pred_norm_actions, dtype=np.float32)[..., :action_dims]
    gt = np.asarray(gt_actions, dtype=np.float32)[..., :action_dims]

    if gt_is_normalized:
        gt_norm = gt
        gt_phys = unnormalize_actions(gt_norm, norm_stats)[..., :action_dims]
    else:
        gt_phys = gt
        gt_norm = normalize_actions(gt_phys, norm_stats)[..., :action_dims]

    pred_phys = unnormalize_actions(pred_norm, norm_stats)[..., :action_dims]

    sqerr_norm = (pred_norm - gt_norm) ** 2
    sqerr_phys = (pred_phys - gt_phys) ** 2
    flat_norm = sqerr_norm.reshape(-1, sqerr_norm.shape[-1])
    flat_phys = sqerr_phys.reshape(-1, sqerr_phys.shape[-1])
    mse_per_dim_norm = flat_norm.mean(axis=0)
    mse_per_dim_phys = flat_phys.mean(axis=0)

    sample_norm = sqerr_norm.reshape(-1, sqerr_norm.shape[-1])[0]
    return {
        "pred_norm": pred_norm,
        "gt_norm": gt_norm,
        "pred_phys": pred_phys,
        "gt_phys": gt_phys,
        "mse_per_dim_norm": mse_per_dim_norm,
        "mse_per_dim_phys": mse_per_dim_phys,
        "mse_total_norm": float(mse_per_dim_norm.mean()),
        "mse_arm_norm": float(mse_per_dim_norm[:6].mean()) if action_dims >= 6 else float(mse_per_dim_norm.mean()),
        "mse_hand_norm": float(mse_per_dim_norm[6:12].mean()) if action_dims >= 12 else float("nan"),
        "mse_total_phys": float(mse_per_dim_phys.mean()),
        "mse_arm_phys": float(mse_per_dim_phys[:6].mean()) if action_dims >= 6 else float(mse_per_dim_phys.mean()),
        "mse_hand_phys": float(mse_per_dim_phys[6:12].mean()) if action_dims >= 12 else float("nan"),
        "sample_arm_norm": float(sample_norm[:6].mean()) if action_dims >= 6 else float(sample_norm.mean()),
        "sample_hand_norm": float(sample_norm[6:12].mean()) if action_dims >= 12 else float("nan"),
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_openpi_path(repo_root: Path) -> None:
    openpi_src = repo_root / "openpi_official" / "src"
    if str(openpi_src) not in sys.path:
        sys.path.insert(0, str(openpi_src))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _link_dataset(data_dir: Path, repo_root: Path, repo_id: str) -> None:
    data_dir = data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing LeRobot dataset dir: {data_dir}")

    hf_home = Path(os.environ.setdefault("HF_HOME", str(repo_root / ".cache" / "huggingface")))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    roots = [
        Path(os.environ.setdefault("HF_LEROBOT_ROOT", str(repo_root / ".cache" / "lerobot"))),
        hf_home / "lerobot",
    ]

    for root in roots:
        link_path = root / repo_id
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.is_symlink():
            if link_path.resolve() != data_dir:
                link_path.unlink()
                link_path.symlink_to(data_dir, target_is_directory=True)
        elif link_path.exists():
            if link_path.resolve() != data_dir:
                raise FileExistsError(f"{link_path} exists and does not point to {data_dir}")
        else:
            link_path.symlink_to(data_dir, target_is_directory=True)


def _count_frames(data_dir: Path, max_episodes: int | None) -> int:
    parquets = sorted(data_dir.glob("data/chunk-*/*.parquet"))
    if not parquets:
        parquets = sorted(data_dir.glob("*.parquet"))
    if max_episodes is not None:
        parquets = parquets[:max_episodes]
    if not parquets:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    return sum(pq.ParquetFile(p).metadata.num_rows for p in parquets)


def _with_max_episodes(max_episodes: int | None):
    class _Env:
        def __enter__(self):
            self.old = os.environ.get("OPENPI_MAX_EPISODES")
            if max_episodes is not None:
                os.environ["OPENPI_MAX_EPISODES"] = str(max_episodes)

        def __exit__(self, exc_type, exc, tb):
            if self.old is None:
                os.environ.pop("OPENPI_MAX_EPISODES", None)
            else:
                os.environ["OPENPI_MAX_EPISODES"] = self.old

    return _Env()


def _move_to_device(tree, device: str):
    import jax

    return jax.tree.map(lambda x: x.to(device) if torch.is_tensor(x) else x, tree)


def _load_config(config_name: str, assets_base_dir: Path, batch_size: int, num_workers: int):
    from openpi.training import config as _config

    config = _config.get_config(config_name)
    return dataclasses.replace(
        config,
        assets_base_dir=assets_base_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def _make_loader(config, max_episodes: int | None, num_batches: int):
    from openpi.training import data_loader as _data_loader

    with _with_max_episodes(max_episodes):
        return _data_loader.create_data_loader(
            config,
            shuffle=False,
            num_batches=num_batches,
            framework="pytorch",
        )


def _load_model(config, ckpt_dir: Path, device: str):
    weight_path = ckpt_dir / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing checkpoint weights: {weight_path}")
    model = config.model.load_pytorch(config, str(weight_path))
    model.to(device)
    model.eval()
    return model


def _make_noise(shape: tuple[int, ...], device: str, seed: int, offset: int) -> torch.Tensor:
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + offset)
        return torch.randn(shape, generator=generator, dtype=torch.float32, device=device)
    except RuntimeError:
        generator = torch.Generator()
        generator.manual_seed(seed + offset)
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


def _print_metrics(metrics: dict[str, Any], *, action_dims: int, debug_hand: bool, norm_stats: dict) -> None:
    mse_per_dim = metrics["mse_per_dim_phys"]
    print("\n  Results:")
    print(f"    MSE (total, {action_dims}D, phys): {metrics['mse_total_phys']:.10f}")
    print(f"    MSE (arm 6D, phys):       {metrics['mse_arm_phys']:.10f}")
    print(f"    MSE (hand 6D, phys):      {metrics['mse_hand_phys']:.10f}")
    print(f"    MSE (total, {action_dims}D, norm): {metrics['mse_total_norm']:.10f}")
    print(f"    MSE (arm 6D, norm):       {metrics['mse_arm_norm']:.10f}")
    print(f"    MSE (hand 6D, norm):      {metrics['mse_hand_norm']:.10f}")
    print(
        "    Sample sqerr (first item, norm): "
        f"arm={metrics['sample_arm_norm']:.10f}, hand={metrics['sample_hand_norm']:.10f}"
    )
    if action_dims >= 12:
        print("    MSE per dim (phys):")
        print("      arm/eef (0-5):  " + ", ".join(f"{v:.10f}" for v in mse_per_dim[:6]))
        print("      hand (0-5):     " + ", ".join(f"{v:.10f}" for v in mse_per_dim[6:12]))

    pred_phys = metrics["pred_phys"].reshape(-1, metrics["pred_phys"].shape[-1])
    gt_phys = metrics["gt_phys"].reshape(-1, metrics["gt_phys"].shape[-1])
    pred_norm = metrics["pred_norm"].reshape(-1, metrics["pred_norm"].shape[-1])
    gt_norm = metrics["gt_norm"].reshape(-1, metrics["gt_norm"].shape[-1])
    print("\n  Sample (first item, arm/eef delta first 3 dims):")
    print(f"    Pred: {pred_phys[0, :3]}")
    print(f"    GT:   {gt_phys[0, :3]}")

    if debug_hand and action_dims >= 12:
        print("\n  [DEBUG ACTION STATS]")
        for name in ("mean", "std", "q01", "q99"):
            value = _action_stat(norm_stats, name)
            if value is not None:
                print(f"    hand {name}[6:12]: {value[6:12]}")

        print("\n  [DEBUG FIRST ITEM ARM/HAND]")
        print(f"    gt_model_arm_norm:   {gt_norm[0, :6]}")
        print(f"    gt_model_hand_norm:  {gt_norm[0, 6:12]}")
        print(f"    pred_model_arm_norm: {pred_norm[0, :6]}")
        print(f"    pred_model_hand_norm:{pred_norm[0, 6:12]}")
        print(f"    gt_arm_phys:         {gt_phys[0, :6]}")
        print(f"    gt_hand_phys:        {gt_phys[0, 6:12]}")
        print(f"    pred_arm_phys:       {pred_phys[0, :6]}")
        print(f"    pred_hand_phys:      {pred_phys[0, 6:12]}")


def evaluate_checkpoint(config, ckpt_dir: Path, args, num_batches: int, norm_stats: dict) -> dict[str, Any]:
    model = _load_model(config, ckpt_dir, args.device)
    loader = _make_loader(config, args.max_episodes, num_batches)

    pred_chunks = []
    gt_chunks = []
    horizon_slice = slice(None) if args.all_horizon else slice(0, 1)

    print(f"  Running official inference for {num_batches} batches...", flush=True)
    with torch.no_grad():
        for batch_idx, (observation, gt_actions) in enumerate(loader, start=1):
            observation = _move_to_device(observation, args.device)
            gt_actions = gt_actions.to(args.device)
            batch_size = int(gt_actions.shape[0])
            noise = _make_noise(
                (batch_size, config.model.action_horizon, config.model.action_dim),
                args.device,
                args.seed,
                batch_idx,
            )
            pred = model.sample_actions(args.device, observation, noise=noise, num_steps=args.num_steps)
            pred_chunks.append(pred[:, horizon_slice, : args.action_dims].detach().float().cpu().numpy())
            gt_chunks.append(gt_actions[:, horizon_slice, : args.action_dims].detach().float().cpu().numpy())
            if batch_idx == 1 or batch_idx % args.progress_interval == 0 or batch_idx == num_batches:
                print(f"    [{batch_idx}/{num_batches}]", flush=True)

    del model
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    pred_norm = np.concatenate(pred_chunks, axis=0)
    gt_norm = np.concatenate(gt_chunks, axis=0)
    return compute_action_mse(
        pred_norm,
        gt_norm,
        norm_stats,
        action_dims=args.action_dims,
        gt_is_normalized=True,
    )


def main() -> None:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dirs", nargs="+", default=[DEFAULT_CHECKPOINT_DIR])
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--train-config", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--assets-base-dir", default=str(repo_root / "openpi_official" / "assets_eef_delta_v2"))
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--action-dims", type=int, default=12)
    parser.add_argument("--all-horizon", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--debug-hand", action="store_true")
    parser.add_argument("--compare-init", action="store_true", help="Kept for CLI compatibility; not used.")
    args = parser.parse_args()

    _ensure_openpi_path(repo_root)
    data_dir = Path(args.data_dir)
    assets_base_dir = Path(args.assets_base_dir)
    _link_dataset(data_dir, repo_root, args.repo_id)

    frame_count = _count_frames(data_dir, args.max_episodes)
    eval_batch_size = min(max(1, args.eval_batch_size), frame_count)
    num_batches = args.max_batches or max(1, frame_count // eval_batch_size)
    dropped = max(0, frame_count - num_batches * eval_batch_size)
    print(f"Loading data from {data_dir} (max_episodes={args.max_episodes})", flush=True)
    print(f"Frames={frame_count}, eval_batch_size={eval_batch_size}, num_batches={num_batches}, dropped_tail={dropped}")
    print(f"Train config={args.train_config}")
    print(f"Assets base dir={assets_base_dir}")

    config = _load_config(args.train_config, assets_base_dir, eval_batch_size, args.num_workers)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise FileNotFoundError(f"Norm stats not found for asset_id={data_config.asset_id} in {config.assets_dirs}")
    norm_stats = data_config.norm_stats

    results = {}
    for ckpt_dir_str in args.ckpt_dirs:
        ckpt_dir = Path(ckpt_dir_str)
        ckpt_name = ckpt_dir.parent.name + "/" + ckpt_dir.name
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {ckpt_name}", flush=True)
        print(f"{'=' * 60}")

        try:
            metrics = evaluate_checkpoint(config, ckpt_dir, args, num_batches, norm_stats)
        except Exception as exc:
            print(f"  [ERROR] Failed to evaluate {ckpt_dir}: {exc}")
            import traceback

            traceback.print_exc()
            continue

        _print_metrics(metrics, action_dims=args.action_dims, debug_hand=args.debug_hand, norm_stats=norm_stats)
        results[ckpt_name] = metrics

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    for name, res in results.items():
        print(
            f"  {name}: "
            f"phys(total={res['mse_total_phys']:.10f}, arm={res['mse_arm_phys']:.10f}, "
            f"hand={res['mse_hand_phys']:.10f}) | "
            f"norm(total={res['mse_total_norm']:.10f}, arm={res['mse_arm_norm']:.10f}, "
            f"hand={res['mse_hand_norm']:.10f})"
        )


if __name__ == "__main__":
    main()
