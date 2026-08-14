#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dexjoco"))
sys.path.insert(0, str(ROOT / "scripts"))

from dexjoco.ogpo.divl import divl_quantile_values
from dexjoco.ogpo.multimodal_critic import MultiHeadUdivlCritic
from dexjoco.ogpo.pi05_jax_adapter import PI05JaxFlowPolicy
from dexjoco.ogpo.replay import BalancedCriticReplay, OfflineChunkReplay, load_replay
from dexjoco.ogpo.trainer import (
    _critic_execution_mask,
    _policy_condition,
    _sample_frozen_jax_flash_rollout,
    _select_flash_steps,
    build_train_state,
    load_checkpoint,
    policy_observation_to_jax,
)
from dexjoco.ogpo.types import ChunkBatch
from dexjoco.ogpo.value_critic_protocol import StateFeatures
from train_udivl_critic import load_config


def _sign_statistics(q_values: torch.Tensor, baseline: torch.Tensor) -> dict[str, Any]:
    raw = q_values - baseline.unsqueeze(-1)
    positive = raw.min(dim=0).values > 0
    negative = raw.max(dim=0).values < 0
    disagreement = ~(positive | negative)
    return {
        "positive_consensus_ratio": float(positive.float().mean().item()),
        "negative_consensus_ratio": float(negative.float().mean().item()),
        "zero_disagreement_ratio": float(disagreement.float().mean().item()),
        "sign_agreement_ratio": float((positive | negative).float().mean().item()),
        "per_head_positive_ratio": [
            float(value) for value in (raw > 0).float().mean(dim=(1, 2)).tolist()
        ],
        "per_head_advantage_mean": [
            float(value) for value in raw.mean(dim=(1, 2)).tolist()
        ],
        "raw_advantage_mean": float(raw.mean().item()),
        "raw_advantage_std": float(raw.std(unbiased=False).item()),
    }


@torch.inference_mode()
def _score_candidates(
    state,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    group_size: int,
    seed: int,
) -> dict[str, Any]:
    if not isinstance(state.policy, PI05JaxFlowPolicy):
        raise TypeError("this diagnostic requires actor.adapter=pi05_jax")
    if not isinstance(state.critic, MultiHeadUdivlCritic):
        raise TypeError("this diagnostic requires the three-head UDIVL critic")

    device = next(state.critic.parameters()).device
    source = batch.to(device)
    condition = _policy_condition(state.old_policy, source)
    jax_observation = policy_observation_to_jax(condition.observation)
    selected_steps = _select_flash_steps(
        config.get("flow", {}),
        batch_size=source.batch_size,
        num_steps=state.old_policy.num_steps,
        device=device,
        seed=seed + 101,
    )
    actor = nnx.merge(state.old_policy.actor_graphdef, state.old_policy.actor_state)
    endpoints = []
    for candidate in range(group_size):
        _, _, _, endpoint = _sample_frozen_jax_flash_rollout(
            actor,
            jax_observation,
            selected_step=jnp.asarray(selected_steps.cpu().numpy()),
            rng=jax.random.PRNGKey(seed + 17 + candidate * 1009),
            num_steps=state.old_policy.num_steps,
            sde_mode=state.old_policy.sde_mode,
            group_size=1,
        )
        endpoint_torch = torch.as_tensor(
            np.asarray(endpoint).copy(),
            device=device,
            dtype=source.action_chunks.dtype,
        )
        endpoints.append(
            state.old_policy.flat_actions_to_environment(endpoint_torch, condition)
        )
    candidate_flat_actions = torch.stack(endpoints, dim=1)

    batch_size, _, _ = candidate_flat_actions.shape
    chunks = candidate_flat_actions.reshape(
        batch_size,
        group_size,
        source.generated_horizon,
        source.action_dim,
    )
    features = state.critic.encode_state(source)
    grouped_features = StateFeatures(
        readout=features.readout.repeat_interleave(group_size, dim=0)
    )
    critic_mask = _critic_execution_mask(source, config)
    grouped_masks = critic_mask[:, None, :].expand(
        batch_size, group_size, source.generated_horizon
    )
    q_values = state.critic.q_from_features(
        grouped_features,
        chunks.reshape(batch_size * group_size, source.generated_horizon, source.action_dim),
        grouped_masks.reshape(batch_size * group_size, source.generated_horizon),
    ).reshape(state.critic.ensemble_size, batch_size, group_size)
    behavior_q = state.critic.q_from_features(
        features,
        source.action_chunks,
        critic_mask,
    )
    probs = F.softmax(state.critic.value_logits_from_features(features), dim=-1)
    divl_cfg = config.get("divl", {})
    value_stats = divl_quantile_values(
        probs,
        state.support,
        alpha_min=float(divl_cfg.get("alpha_min", 0.5)),
        alpha_max=float(divl_cfg.get("alpha_max", 0.8)),
        entropy_temperature=float(divl_cfg.get("entropy_temperature", 1.0)),
        alpha_mode=str(divl_cfg.get("alpha_mode", "linear")),
        use_adaptive_quantile=bool(divl_cfg.get("use_adaptive_quantile", True)),
        interpolate_quantile=bool(divl_cfg.get("interpolate_quantile", True)),
    )
    value_baseline = value_stats.quantile_value
    group_mean_baseline = q_values.mean(dim=-1)
    candidate_mean = q_values.mean(dim=-1)
    fixed_quantile_baselines = {}
    fixed_quantile_raw = {}
    for alpha in (0.5, 0.6, 0.7, 0.8, 0.9):
        fixed = divl_quantile_values(
            probs,
            state.support,
            alpha_min=alpha,
            alpha_max=alpha,
            use_adaptive_quantile=False,
            interpolate_quantile=bool(divl_cfg.get("interpolate_quantile", True)),
        ).quantile_value
        name = f"fixed_quantile_{alpha:.1f}"
        fixed_quantile_baselines[name] = _sign_statistics(q_values, fixed)
        fixed_quantile_raw[name] = fixed.cpu().tolist()

    return {
        "num_states": batch_size,
        "num_candidates_per_state": group_size,
        "selected_step_histogram": torch.bincount(
            selected_steps.cpu(), minlength=state.old_policy.num_steps
        ).tolist(),
        "q_mean": float(q_values.mean().item()),
        "q_std": float(q_values.std(unbiased=False).item()),
        "v_mean": float(value_baseline.mean().item()),
        "behavior_q_mean": float(behavior_q.mean().item()),
        "candidate_q_minus_v_mean": float((candidate_mean - value_baseline).mean().item()),
        "candidate_q_minus_behavior_q_mean": float((candidate_mean - behavior_q).mean().item()),
        "fraction_candidate_q_above_v_per_head": [
            float(value)
            for value in (q_values > value_baseline.unsqueeze(-1)).float().mean(dim=(1, 2)).tolist()
        ],
        "v_alpha_mean_per_head": [
            float(value) for value in value_stats.alpha.mean(dim=1).tolist()
        ],
        "v_entropy_mean_per_head": [
            float(value) for value in value_stats.entropy.mean(dim=1).tolist()
        ],
        "baselines": {
            "current_divl_v": _sign_statistics(q_values, value_baseline),
            **fixed_quantile_baselines,
            "ogpo_group_mean": _sign_statistics(q_values, group_mean_baseline),
            "replay_behavior_q": _sign_statistics(q_values, behavior_q),
        },
        "raw": {
            "q_values": q_values.cpu().tolist(),
            "value_baseline": value_baseline.cpu().tolist(),
            "behavior_q": behavior_q.cpu().tolist(),
            "fixed_quantile_baselines": fixed_quantile_raw,
            "episode_ids": source.episode_ids.cpu().tolist(),
            "successes": source.successes.cpu().tolist(),
            "dones": source.dones.cpu().tolist(),
            "selected_steps": selected_steps.cpu().tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--states-per-split", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    replay_batch = load_replay(ROOT / config["data"]["dataset_path"])
    state = build_train_state(
        config,
        replay_batch,
        device=config.get("training", {}).get("device", "cuda"),
    )
    load_checkpoint(ROOT / args.checkpoint, state, restore_actor_optimizer=False)
    state.policy.actor_opt_state = None
    state.target_critic.to("cpu")
    state.critic.eval()
    state.old_policy.prepare_inference()

    generators = {
        "uniform": OfflineChunkReplay(replay_batch),
        "balanced": BalancedCriticReplay(replay_batch),
    }
    output: dict[str, Any] = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "splits": {},
    }
    for split_index, (name, replay) in enumerate(generators.items()):
        sampled = replay.sample(
            args.states_per_split,
            generator=torch.Generator().manual_seed(args.seed + split_index * 10000),
        )
        parts = []
        for start in range(0, sampled.batch_size, args.microbatch_size):
            stop = min(start + args.microbatch_size, sampled.batch_size)
            indices = torch.arange(start, stop)
            print(f"[twoside-diagnostic] split={name} states={start}:{stop}", flush=True)
            parts.append(
                _score_candidates(
                    state,
                    sampled.index_select(indices),
                    config,
                    group_size=args.group_size,
                    seed=args.seed + split_index * 10000 + start,
                )
            )
    
        q_values = torch.cat(
            [torch.tensor(part["raw"]["q_values"]) for part in parts], dim=1
        )
        value_baseline = torch.cat(
            [torch.tensor(part["raw"]["value_baseline"]) for part in parts], dim=1
        )
        behavior_q = torch.cat(
            [torch.tensor(part["raw"]["behavior_q"]) for part in parts], dim=1
        )
        fixed_quantile_baselines = {
            name: torch.cat(
                [
                    torch.tensor(part["raw"]["fixed_quantile_baselines"][name])
                    for part in parts
                ],
                dim=1,
            )
            for name in parts[0]["raw"]["fixed_quantile_baselines"]
        }
        combined = {
            "num_states": int(q_values.shape[1]),
            "num_candidates_per_state": int(q_values.shape[2]),
            "q_mean": float(q_values.mean().item()),
            "q_std": float(q_values.std(unbiased=False).item()),
            "v_mean": float(value_baseline.mean().item()),
            "behavior_q_mean": float(behavior_q.mean().item()),
            "candidate_q_minus_v_mean": float(
                (q_values.mean(dim=-1) - value_baseline).mean().item()
            ),
            "candidate_q_minus_behavior_q_mean": float(
                (q_values.mean(dim=-1) - behavior_q).mean().item()
            ),
            "baselines": {
                "current_divl_v": _sign_statistics(q_values, value_baseline),
                **{
                    name: _sign_statistics(q_values, baseline)
                    for name, baseline in fixed_quantile_baselines.items()
                },
                "ogpo_group_mean": _sign_statistics(q_values, q_values.mean(dim=-1)),
                "replay_behavior_q": _sign_statistics(q_values, behavior_q),
            },
            "parts": parts,
        }
        output["splits"][name] = combined
        print(json.dumps({name: combined["baselines"]}, indent=2), flush=True)

    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[twoside-diagnostic] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
