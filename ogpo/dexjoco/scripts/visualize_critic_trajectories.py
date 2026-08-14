#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dexjoco"))
sys.path.insert(0, str(ROOT / "scripts"))

from dexjoco.ogpo.conservative_advantage import sign_consensus_advantage
from dexjoco.ogpo.divl import divl_quantile_values
from dexjoco.ogpo.multimodal_critic import MultiHeadUdivlCritic
from dexjoco.ogpo.replay import load_replay
from dexjoco.ogpo.trainer import build_train_state, load_checkpoint
from train_udivl_critic import _deep_update, load_config


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _episode_indices(batch, episode_id: int) -> torch.Tensor:
    indices = torch.nonzero(batch.episode_ids == int(episode_id), as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError(f"episode {episode_id} does not exist in the selected replay")
    order = torch.argsort(batch.timesteps.index_select(0, indices))
    return indices.index_select(0, order)


def _choose_episode(batch, *, success: bool) -> int:
    candidates: list[tuple[int, int]] = []
    for episode_id in torch.unique(batch.episode_ids).tolist():
        indices = _episode_indices(batch, int(episode_id))
        episode_success = bool(batch.successes.index_select(0, indices).max().item())
        if episode_success == success:
            candidates.append((int(indices.numel()), int(episode_id)))
    if not candidates:
        outcome = "successful" if success else "failed"
        raise ValueError(f"replay contains no {outcome} episode")
    # Prefer a long trajectory; use the smallest episode ID to break ties.
    return sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]


@torch.no_grad()
def _score_episode(state, batch, indices: torch.Tensor, config: dict, inference_batch_size: int) -> dict:
    critic = state.critic
    if not isinstance(critic, MultiHeadUdivlCritic):
        raise TypeError("trajectory visualization requires the Gemma+SigLIP multi-head U-DIVL critic")
    critic.eval()
    q_parts = []
    v_parts = []
    alpha_parts = []
    entropy_parts = []
    for start in range(0, indices.numel(), inference_batch_size):
        selected = indices[start : start + inference_batch_size]
        sample = batch.index_select(selected).to(next(critic.parameters()).device)
        features = critic.encode_state(sample)
        q_values = critic.q_from_features(features, sample.action_chunks, sample.execution_masks)
        probabilities = F.softmax(critic.value_logits_from_features(features), dim=-1)
        divl_cfg = config.get("divl", {})
        stats = divl_quantile_values(
            probabilities,
            state.support,
            alpha_min=float(divl_cfg.get("alpha_min", 0.5)),
            alpha_max=float(divl_cfg.get("alpha_max", 0.8)),
            entropy_temperature=float(divl_cfg.get("entropy_temperature", 1.0)),
            alpha_mode=str(divl_cfg.get("alpha_mode", "linear")),
            use_adaptive_quantile=bool(divl_cfg.get("use_adaptive_quantile", True)),
        )
        q_parts.append(q_values.cpu())
        v_parts.append(stats.quantile_value.cpu())
        alpha_parts.append(stats.alpha.cpu())
        entropy_parts.append(stats.entropy.cpu())

    q_values = torch.cat(q_parts, dim=1)
    value_baselines = torch.cat(v_parts, dim=1)
    alpha = torch.cat(alpha_parts, dim=1)
    entropy = torch.cat(entropy_parts, dim=1)
    conservative, _ = sign_consensus_advantage(
        q_values,
        value_baselines,
        positive_margin=0.0,
        negative_margin=0.0,
    )
    selected_batch = batch.index_select(indices)
    count = int(indices.numel())
    return {
        "episode_id": int(selected_batch.episode_ids[0].item()),
        "success": bool(selected_batch.successes.max().item()),
        "timesteps": selected_batch.timesteps.cpu(),
        "progress": torch.linspace(0.0, 1.0, count),
        "chunk_returns": selected_batch.chunk_returns.cpu(),
        "mc_returns": selected_batch.mc_returns.cpu() if selected_batch.mc_returns is not None else None,
        "q": q_values,
        "v": value_baselines,
        "alpha": alpha,
        "entropy": entropy,
        "advantage": q_values - value_baselines,
        "conservative_advantage": conservative,
        "indices": indices.cpu(),
    }


def _write_csv(results: list[tuple[str, dict]], path: Path) -> None:
    fieldnames = [
        "outcome",
        "episode_id",
        "transition_index",
        "timestep",
        "normalized_progress",
        "success",
        "chunk_return",
        "mc_return",
        "q_1",
        "q_2",
        "q_3",
        "q_mean",
        "q_std",
        "v_1",
        "v_2",
        "v_3",
        "v_mean",
        "advantage_1",
        "advantage_2",
        "advantage_3",
        "conservative_advantage",
        "tau_1",
        "tau_2",
        "tau_3",
        "entropy_1",
        "entropy_2",
        "entropy_3",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for outcome, result in results:
            for index in range(result["timesteps"].numel()):
                q = result["q"][:, index]
                v = result["v"][:, index]
                advantage = result["advantage"][:, index]
                row = {
                    "outcome": outcome,
                    "episode_id": result["episode_id"],
                    "transition_index": index,
                    "timestep": int(result["timesteps"][index].item()),
                    "normalized_progress": float(result["progress"][index].item()),
                    "success": int(result["success"]),
                    "chunk_return": float(result["chunk_returns"][index].item()),
                    "mc_return": (
                        float(result["mc_returns"][index].item())
                        if result["mc_returns"] is not None
                        else ""
                    ),
                    "q_mean": float(q.mean().item()),
                    "q_std": float(q.std(unbiased=False).item()),
                    "v_mean": float(v.mean().item()),
                    "conservative_advantage": float(result["conservative_advantage"][index].item()),
                }
                for member in range(3):
                    suffix = member + 1
                    row[f"q_{suffix}"] = float(q[member].item())
                    row[f"v_{suffix}"] = float(v[member].item())
                    row[f"advantage_{suffix}"] = float(advantage[member].item())
                    row[f"tau_{suffix}"] = float(result["alpha"][member, index].item())
                    row[f"entropy_{suffix}"] = float(result["entropy"][member, index].item())
                writer.writerow(row)


def _plot_scores(results: list[tuple[str, dict]], path: Path) -> None:
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex="row")
    for row, (outcome, result) in enumerate(results):
        progress = result["progress"].numpy() * 100.0
        q = result["q"].numpy()
        v = result["v"].numpy()
        advantage = result["advantage"].numpy()

        ax = axes[row, 0]
        ax.plot(progress, q.mean(axis=0), color="#0072B2", linewidth=2.2, label="Q mean")
        ax.fill_between(progress, q.min(axis=0), q.max(axis=0), color="#0072B2", alpha=0.16)
        ax.plot(progress, v.mean(axis=0), color="#D55E00", linewidth=2.2, label="V quantile mean")
        ax.fill_between(progress, v.min(axis=0), v.max(axis=0), color="#D55E00", alpha=0.16)
        if result["mc_returns"] is not None:
            ax.plot(
                progress,
                result["mc_returns"].numpy(),
                color="#222222",
                linewidth=1.5,
                linestyle="--",
                label="Monte Carlo return",
            )
        ax.set_title(f"{outcome.title()} episode {result['episode_id']}: Q and V")
        ax.set_ylabel("Value")
        ax.legend(loc="best", fontsize=8)

        ax = axes[row, 1]
        for member, color in enumerate(colors):
            ax.plot(progress, q[member], color=color, linewidth=1.6, label=f"Q{member + 1}")
            ax.plot(
                progress,
                v[member],
                color=color,
                linewidth=1.2,
                linestyle="--",
                label=f"V{member + 1}",
            )
        ax.set_title("Three Q-V pairs (solid Q, dashed V)")
        ax.legend(ncol=3, loc="best", fontsize=8)

        ax = axes[row, 2]
        for member, color in enumerate(colors):
            ax.plot(
                progress,
                advantage[member],
                color=color,
                linewidth=1.4,
                alpha=0.8,
                label=f"Q{member + 1}-V{member + 1}",
            )
        ax.plot(
            progress,
            result["conservative_advantage"].numpy(),
            color="#111111",
            linewidth=2.3,
            label="Two-sided consensus",
        )
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.set_title("Advantage used before weighting")
        ax.legend(loc="best", fontsize=8)

        for column in range(3):
            axes[row, column].grid(alpha=0.22)
            axes[row, column].set_xlabel("Episode progress (%)")

    fig.suptitle("U-DIVL critic on held-out click_mouse trajectories", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_contact_sheet(batch, results: list[tuple[str, dict]], path: Path) -> None:
    if not batch.images:
        return
    camera_keys = [key for key in ("image_base", "image_wrist") if key in batch.images]
    if not camera_keys:
        camera_keys = sorted(batch.images)[:2]
    samples_per_episode = 5
    fig, axes = plt.subplots(
        len(results) * len(camera_keys),
        samples_per_episode,
        figsize=(15, 3.1 * len(results) * len(camera_keys)),
        squeeze=False,
    )
    row = 0
    for outcome, result in results:
        local_positions = np.linspace(0, result["indices"].numel() - 1, samples_per_episode).round().astype(int)
        for camera_key in camera_keys:
            for column, local_position in enumerate(local_positions):
                replay_index = int(result["indices"][local_position].item())
                frame = batch.images[camera_key][replay_index].cpu().numpy()
                axes[row, column].imshow(frame)
                axes[row, column].axis("off")
                axes[row, column].set_title(
                    f"{outcome} | {camera_key}\n"
                    f"{100.0 * local_position / max(1, result['indices'].numel() - 1):.0f}%"
                )
            row += 1
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_peak_contact_sheet(batch, results: list[tuple[str, dict]], path: Path) -> None:
    """Show consecutive observations around each trajectory's maximum mean Q."""
    if not batch.images:
        return
    camera_keys = [key for key in ("image_base", "image_wrist") if key in batch.images]
    if not camera_keys:
        camera_keys = sorted(batch.images)[:2]
    samples_per_episode = 7
    fig, axes = plt.subplots(
        len(results) * len(camera_keys),
        samples_per_episode,
        figsize=(17, 3.2 * len(results) * len(camera_keys)),
        squeeze=False,
    )
    row = 0
    for outcome, result in results:
        q_mean = result["q"].mean(dim=0)
        peak = int(q_mean.argmax().item())
        offsets = np.arange(-3, 4)
        local_positions = np.clip(peak + offsets, 0, result["indices"].numel() - 1)
        for camera_key in camera_keys:
            for column, local_position in enumerate(local_positions):
                replay_index = int(result["indices"][int(local_position)].item())
                frame = batch.images[camera_key][replay_index].cpu().numpy()
                axes[row, column].imshow(frame)
                axes[row, column].axis("off")
                axes[row, column].set_title(
                    f"{outcome} | {camera_key}\n"
                    f"t={int(result['timesteps'][int(local_position)].item())} "
                    f"Q={float(q_mean[int(local_position)].item()):.3f}"
                )
            row += 1
    fig.suptitle("Observations around maximum trajectory Q", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _summary(results: list[tuple[str, dict]]) -> dict:
    output = {}
    for outcome, result in results:
        q_mean = result["q"].mean(dim=0)
        v_mean = result["v"].mean(dim=0)
        conservative = result["conservative_advantage"]
        episode_summary = {
            "episode_id": result["episode_id"],
            "transitions": int(result["timesteps"].numel()),
            "q_start": float(q_mean[0].item()),
            "q_end": float(q_mean[-1].item()),
            "q_min": float(q_mean.min().item()),
            "q_max": float(q_mean.max().item()),
            "v_start": float(v_mean[0].item()),
            "v_end": float(v_mean[-1].item()),
            "v_min": float(v_mean.min().item()),
            "v_max": float(v_mean.max().item()),
            "positive_consensus_fraction": float((conservative > 0).float().mean().item()),
            "negative_consensus_fraction": float((conservative < 0).float().mean().item()),
            "uncertain_fraction": float((conservative == 0).float().mean().item()),
        }
        if result["mc_returns"] is not None:
            q_error = q_mean - result["mc_returns"]
            episode_summary["q_bias_vs_mc"] = float(q_error.mean().item())
            episode_summary["q_rmse_vs_mc"] = float(q_error.square().mean().sqrt().item())
            episode_summary["mc_return_mean"] = float(result["mc_returns"].mean().item())
        output[outcome] = episode_summary
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/ogpo/click_mouse_gemma_udivl_100ep_calibrated.pt",
    )
    parser.add_argument(
        "--config",
        default="configs/ogpo/pi05_gemma_udivl_critic_100ep.yaml",
    )
    parser.add_argument(
        "--replay",
        default="outputs/ogpo/click_mouse_pi05_replay_100ep_validation.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ogpo/visualizations/click_mouse_critic_15k",
    )
    parser.add_argument("--success-episode", type=int)
    parser.add_argument("--failure-episode", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint_path = _resolve(args.checkpoint)
    config = load_config(_resolve(args.config))
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _deep_update(config, checkpoint_payload.get("config", {}))
    batch = load_replay(_resolve(args.replay))
    state = build_train_state(config, batch, device=args.device)
    load_checkpoint(checkpoint_path, state)

    success_episode = (
        args.success_episode
        if args.success_episode is not None
        else _choose_episode(batch, success=True)
    )
    failure_episode = (
        args.failure_episode
        if args.failure_episode is not None
        else _choose_episode(batch, success=False)
    )
    results = [
        (
            "success",
            _score_episode(
                state,
                batch,
                _episode_indices(batch, success_episode),
                config,
                args.batch_size,
            ),
        ),
        (
            "failure",
            _score_episode(
                state,
                batch,
                _episode_indices(batch, failure_episode),
                config,
                args.batch_size,
            ),
        ),
    ]

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(results, output_dir / "trajectory_q_v_scores.csv")
    _plot_scores(results, output_dir / "trajectory_q_v_scores.png")
    _plot_contact_sheet(batch, results, output_dir / "trajectory_keyframes.png")
    _plot_peak_contact_sheet(batch, results, output_dir / "trajectory_peak_keyframes.png")
    summary = _summary(results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
