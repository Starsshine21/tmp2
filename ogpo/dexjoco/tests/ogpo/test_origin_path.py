from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml

from dexjoco.ogpo.multimodal_critic import MultiHeadScalarQCore, MultiHeadScalarQCritic
from dexjoco.ogpo.origin_cache import (
    attach_origin_feature_cache,
    build_origin_feature_cache,
)
from dexjoco.ogpo.replay import add_monte_carlo_returns, make_synthetic_replay
from dexjoco.ogpo.trainer import (
    build_train_state,
    conservative_advantages_for_candidates,
    critic_update,
    load_critic_checkpoint,
    save_checkpoint,
)


def _deep_update(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(root: Path, relative_path: str) -> dict:
    payload = yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
    include = payload.pop("include", None)
    if include is None:
        return payload
    return _deep_update(_load_config(root, include), payload)


class ReplayStateEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, batch, *, next_observation: bool = False):
        values = batch.next_observations if next_observation else batch.observations
        return self.projection(values)


def _factory(batch, config):
    return MultiHeadScalarQCritic(
        ReplayStateEncoder(batch.obs_dim, 8),
        MultiHeadScalarQCore(
            state_dim=8,
            action_dim=batch.action_dim,
            max_horizon=batch.generated_horizon,
            action_hidden_dim=8,
            head_hidden_dim=16,
            num_attention_heads=2,
            num_heads=int(config["critic"]["ensemble_size"]),
        ),
    )


def _config():
    return {
        "method": {"name": "ogpo-origin"},
        "critic": {
            "architecture": "gemma_siglip_scalar_q",
            "ensemble_size": 4,
            "learning_rate": 3e-4,
            "bootstrap_target": "ensemble_mean",
            "reference_value_samples": 1,
            "target_tau": 0.05,
        },
        "divl": {
            "enabled": False,
            "num_atoms": 2,
            "v_min": 0.0,
            "v_max": 1.0,
        },
        "uncertainty": {
            "entropy_scale": 0.0,
            "entropy_skip_threshold": 1.1,
            "consensus_skip_threshold": -1.0,
            "use_support_weight": False,
        },
        "actor": {
            "advantage_mode": "group_mean",
            "group_size": 4,
            "hidden_dim": 16,
            "advantage_clip": 100.0,
        },
        "flow": {
            "adapter": "gaussian",
            "num_steps": 3,
            "sde_mode": "ogpo_corrected",
        },
        "regularization": {
            "beta_kl": 0.0,
            "lambda_fm": 0.0,
            "lambda_success": 0.0,
        },
    }


def test_origin_critic_is_q_only_and_uses_standard_td():
    batch = add_monte_carlo_returns(
        make_synthetic_replay(
            num_samples=8,
            generated_horizon=4,
            executed_horizon=2,
            action_dim=2,
        )
    )
    config = _config()
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)

    assert isinstance(state.critic, MultiHeadScalarQCritic)
    assert state.divl is None
    assert state.target_divl is None
    assert not any("value" in name for name, _ in state.critic.named_parameters())

    metrics = critic_update(state, batch, config)
    assert metrics["q_loss_is_mse"] == 1.0
    assert metrics["divl_enabled"] == 0.0
    assert metrics["reference_value_samples"] == 1.0
    assert metrics["bootstrap_active"] == 1.0
    assert torch.isfinite(torch.tensor(metrics["critic_loss"]))


def test_origin_pure_mc_target_does_not_sample_reference_policy():
    batch = add_monte_carlo_returns(
        make_synthetic_replay(
            num_samples=8,
            generated_horizon=4,
            executed_horizon=2,
            action_dim=2,
        )
    )
    config = _config()
    config["critic"].update(
        target_mode="mc_return",
        reference_value_samples=0,
        lambda_mc=1.0,
    )
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("pure MC critic must not sample the reference policy")

    state.reference_policy.rollout = fail_if_called
    metrics = critic_update(state, batch, config)

    assert metrics["bootstrap_active"] == 0.0
    assert metrics["reference_value_samples"] == 0.0
    assert metrics["lambda_mc"] == 1.0
    assert metrics["target_mean"] == torch.mean(batch.mc_returns).item()
    assert "reference_value_mean" not in metrics


def test_origin_advantage_is_mean_centered_without_std_normalization():
    batch = make_synthetic_replay(
        num_samples=3,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config()
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)
    candidates = torch.randn(
        batch.batch_size,
        4,
        batch.generated_horizon * batch.action_dim,
    )

    advantage, metrics = conservative_advantages_for_candidates(
        state,
        batch.observations,
        candidates,
        batch,
        config,
    )

    assert metrics["advantage_mode"] == "group_mean"
    assert torch.allclose(advantage.mean(dim=1), torch.zeros(batch.batch_size), atol=1e-6)
    assert metrics["state_entropy_weight"] == 1.0
    assert metrics["support_weight_mean"] == 1.0


def test_origin_checkpoint_is_separate_and_roundtrips(tmp_path):
    batch = make_synthetic_replay(
        num_samples=4,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config()
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)
    checkpoint = tmp_path / "origin.pt"
    save_checkpoint(state, config, checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["critic_format"] == "gemma_siglip_scalar_q"

    restored = build_train_state(config, batch, multimodal_critic_factory=_factory)
    load_critic_checkpoint(checkpoint, restored)
    for expected, actual in zip(
        state.critic.parameters(),
        restored.critic.parameters(),
        strict=True,
    ):
        assert torch.equal(expected, actual)


def test_origin_frozen_feature_cache_bypasses_encoder():
    batch = make_synthetic_replay(
        num_samples=5,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config()
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)
    state.critic.state_encoder.requires_grad_(False)
    payload = build_origin_feature_cache(
        state,
        batch,
        inference_batch_size=2,
    )
    cached_batch = attach_origin_feature_cache(batch, payload)
    expected = state.critic.encode_state(cached_batch).readout

    def fail_if_called(*args, **kwargs):
        raise AssertionError("frozen encoder should not run for cached replay")

    state.critic.state_encoder.forward = fail_if_called
    actual = state.critic.encode_state(cached_batch).readout
    assert torch.equal(actual, expected)


def test_origin_formal_configs_are_isolated_from_divl_outputs():
    root = Path(__file__).resolve().parents[2]
    critic = _load_config(root, "configs/ogpo/pi05_ogpo_origin_critic_100ep.yaml")
    actor = _load_config(root, "configs/ogpo/pi05_ogpo_origin_actor_100ep.yaml")

    for config in (critic, actor):
        assert config["method"]["name"] == "ogpo-origin"
        assert config["critic"]["architecture"] == "gemma_siglip_scalar_q"
        assert config["critic"]["ensemble_size"] == 10
        assert config["critic"]["target_mode"] == "mc_return"
        assert config["critic"]["reference_value_samples"] == 0
        assert config["critic"]["lambda_mc"] == 1.0
        assert config["critic"]["stage"] == "full_td"
        assert config["critic"]["backbone"]["train_siglip"] is True
        assert config["critic"]["backbone"]["train_vlm_full"] is True
        assert config["data"]["origin_feature_cache"]["enabled"] is False
        assert config["divl"]["enabled"] is False
        assert config["actor"]["advantage_mode"] == "group_mean"
        assert config["actor"]["group_size"] == 32
        assert config["actor"]["full_ratio_mode"] == "ais_joint"
        assert config["regularization"]["beta_kl"] == 0.0
        assert config["regularization"]["lambda_success"] == 0.0
        assert config["training"]["checkpoint_path"].startswith("outputs/ogpo-origin/")
