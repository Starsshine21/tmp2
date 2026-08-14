from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from dexjoco.ogpo.categorical_q import (
    consensus_ranking_loss,
    decode_categorical_q,
    hl_gauss_projection,
    ranking_action_negatives,
)
from dexjoco.ogpo.evaluator import offline_calibration_metrics
from dexjoco.ogpo.multimodal_critic import MultiHeadUdivlCore, MultiHeadUdivlCritic
from dexjoco.ogpo.replay import make_synthetic_replay
from dexjoco.ogpo.trainer import (
    build_train_state,
    critic_update,
    initialize_critic_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


class ReplayEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 8):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, batch, *, next_observation: bool = False):
        values = batch.next_observations if next_observation else batch.observations
        return self.projection(values)


def _factory(batch, config):
    critic_cfg = config["critic"]
    return MultiHeadUdivlCritic(
        ReplayEncoder(batch.obs_dim),
        MultiHeadUdivlCore(
            state_dim=8,
            action_dim=batch.action_dim,
            max_horizon=batch.generated_horizon,
            action_hidden_dim=8,
            head_hidden_dim=16,
            num_attention_heads=2,
            num_value_atoms=int(config["divl"]["num_atoms"]),
            num_pairs=3,
            q_representation=str(critic_cfg.get("q_representation", "scalar")),
            q_num_bins=int(critic_cfg.get("q_num_bins", 201)),
            q_vmin=float(critic_cfg.get("q_vmin", -0.1)),
            q_vmax=float(critic_cfg.get("q_vmax", 1.1)),
        ),
    )


def _config(*, categorical: bool) -> dict:
    return {
        "critic": {
            "architecture": "gemma_siglip_multihead",
            "ensemble_size": 3,
            "learning_rate": 3e-4,
            "q_loss": "mse",
            "q_representation": "categorical" if categorical else "scalar",
            "q_num_bins": 201,
            "q_vmin": -0.1,
            "q_vmax": 1.1,
            "q_hl_gauss_sigma_bins": 0.75,
            "rank_consensus_enabled": categorical,
            "rank_loss_weight": 0.1,
            "rank_margin_bins": 2.0,
            "rank_softmin_tau": 0.02,
            "rank_temperature": 0.02,
            "rank_noise_sigma": 0.15,
            "rank_use_strong_noise": True,
            "rank_use_random_negative": True,
            "rank_only_success": True,
        },
        "divl": {
            "num_atoms": 201,
            "v_min": -0.1,
            "v_max": 1.1,
            "alpha_min": 0.5,
            "alpha_max": 0.6,
        },
        "actor": {"hidden_dim": 16},
        "flow": {"num_steps": 3},
        "training": {"seed": 7},
    }


def test_hl_gauss_projection_is_normalized_and_clips_boundaries():
    support = torch.linspace(-0.1, 1.1, 201)
    targets = torch.tensor([-4.0, -0.1, 0.347, 1.1, 5.0], requires_grad=True)

    projected = hl_gauss_projection(targets, support, sigma_bins=0.75)

    assert projected.shape == (5, 201)
    assert torch.isfinite(projected).all()
    assert (projected >= 0.0).all()
    assert torch.allclose(projected.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert projected.argmax(dim=-1).tolist() == [0, 0, 74, 200, 200]
    assert not projected.requires_grad


def test_categorical_decode_recovers_selected_support_atom():
    support = torch.linspace(-0.1, 1.1, 201)
    logits = torch.full((3, 2, 201), -100.0)
    logits[..., 137] = 100.0

    decoded = decode_categorical_q(logits, support)

    assert torch.allclose(decoded, torch.full((3, 2), support[137]), atol=1e-6)


def test_b2_worst_member_penalizes_one_member_reversal():
    valid = torch.tensor([True])
    positive = torch.tensor([[0.2], [0.15], [0.1]])
    ordered_negative = torch.zeros_like(positive)
    reversed_negative = torch.tensor([[0.0], [0.0], [0.15]])

    ordered_loss, _, ordered_worst = consensus_ranking_loss(
        positive,
        ordered_negative,
        valid,
        margin=0.012,
        softmin_tau=0.02,
        temperature=0.02,
    )
    reversed_loss, _, reversed_worst = consensus_ranking_loss(
        positive,
        reversed_negative,
        valid,
        margin=0.012,
        softmin_tau=0.02,
        temperature=0.02,
    )

    assert reversed_worst.item() < 0.0
    assert ordered_worst.item() > 0.0
    assert reversed_loss.item() > ordered_loss.item() * 5.0


def test_ranking_negatives_only_change_executed_prefix():
    actions = torch.zeros(2, 5, 3)
    mask = torch.tensor([[True, True, False, False, False]] * 2)
    mean = torch.zeros(3)
    std = torch.ones(3)
    action_min = torch.full((3,), -1.0)
    action_max = torch.full((3,), 1.0)

    strong, random = ranking_action_negatives(
        actions,
        mask,
        action_mean=mean,
        action_std=std,
        action_min=action_min,
        action_max=action_max,
        noise_sigma=0.15,
        generator=torch.Generator().manual_seed(5),
    )

    assert torch.equal(strong[:, 2:], actions[:, 2:])
    assert torch.equal(random[:, 2:], actions[:, 2:])
    assert not torch.equal(strong[:, :2], actions[:, :2])
    assert not torch.equal(random[:, :2], actions[:, :2])


def test_b2_gradient_reaches_categorical_logits():
    support = torch.linspace(-0.1, 1.1, 201)
    positive_logits = torch.randn(3, 4, 201, requires_grad=True)
    negative_logits = torch.randn(3, 4, 201, requires_grad=True)
    positive_q = decode_categorical_q(positive_logits, support)
    negative_q = decode_categorical_q(negative_logits, support)

    loss, _, _ = consensus_ranking_loss(
        positive_q,
        negative_q,
        torch.ones(4, dtype=torch.bool),
        margin=0.012,
        softmin_tau=0.02,
        temperature=0.02,
    )
    loss.backward()

    assert positive_logits.grad is not None and torch.isfinite(positive_logits.grad).all()
    assert negative_logits.grad is not None and torch.isfinite(negative_logits.grad).all()
    assert positive_logits.grad.abs().sum().item() > 0.0
    assert negative_logits.grad.abs().sum().item() > 0.0


def test_categorical_critic_forward_loss_gradient_and_validation():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    batch = replace(batch, successes=torch.ones_like(batch.successes, dtype=torch.bool))
    config = _config(categorical=True)
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)
    features = state.critic.encode_state(batch)

    logits = state.critic.q_logits_from_features(
        features, batch.action_chunks, batch.execution_masks
    )
    values = state.critic.q_from_features(
        features, batch.action_chunks, batch.execution_masks
    )
    metrics = critic_update(state, batch, config)
    validation = offline_calibration_metrics(state.critic, batch, config=config)

    assert logits.shape == (3, 8, 201)
    assert values.shape == (3, 8)
    assert state.step == 1
    assert metrics["critic/q_ce_loss"] > 0.0
    assert metrics["critic/rank_pair_count"] == 16.0
    assert metrics["critic_grad_norm"] > 0.0
    for key in (
        "critic/rank_acc_q1",
        "critic/rank_acc_q2",
        "critic/rank_acc_q3",
        "critic/rank_acc_mean",
        "critic/rank_acc_unanimous",
        "critic/rank_margin_satisfied",
    ):
        assert key in validation


def test_scalar_checkpoint_partially_initializes_categorical_q(tmp_path, capsys):
    batch = make_synthetic_replay(
        num_samples=6,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    scalar_config = _config(categorical=False)
    source = build_train_state(scalar_config, batch, multimodal_critic_factory=_factory)
    with torch.no_grad():
        source.critic.state_encoder.projection.weight.fill_(0.125)
        source.critic.core.value_heads[0][-1].bias.fill_(0.25)
    checkpoint = tmp_path / "scalar.pt"
    save_checkpoint(source, scalar_config, checkpoint)

    categorical_config = _config(categorical=True)
    target = build_train_state(categorical_config, batch, multimodal_critic_factory=_factory)
    initialize_critic_from_checkpoint(checkpoint, target)

    assert torch.equal(
        target.critic.state_encoder.projection.weight,
        source.critic.state_encoder.projection.weight,
    )
    assert torch.equal(
        target.critic.core.value_heads[0][-1].bias,
        source.critic.core.value_heads[0][-1].bias,
    )
    assert target.critic.core.q_heads[0][-1].weight.shape[0] == 201
    assert target.critic_optimizer.state_dict()["state"] == {}
    assert all(
        torch.equal(online, target_value)
        for online, target_value in zip(
            target.critic.core.q_heads[0].parameters(),
            target.target_critic.core.q_heads[0].parameters(),
            strict=True,
        )
    )
    output = capsys.readouterr().out
    assert "loaded parameters" in output
    assert "reinitialized categorical Q parameters" in output
    assert "skipped incompatible optimizer states" in output


def test_categorical_checkpoint_roundtrip_preserves_q_support_and_outputs(tmp_path):
    batch = make_synthetic_replay(
        num_samples=6,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    batch = replace(batch, successes=torch.ones_like(batch.successes, dtype=torch.bool))
    config = _config(categorical=True)
    source = build_train_state(config, batch, multimodal_critic_factory=_factory)
    critic_update(source, batch, config)
    checkpoint = tmp_path / "categorical.pt"
    save_checkpoint(source, config, checkpoint)
    with torch.no_grad():
        expected = source.critic(
            batch, batch.action_chunks, batch.execution_masks
        ).clone()

    restored = build_train_state(config, batch, multimodal_critic_factory=_factory)
    load_checkpoint(checkpoint, restored)
    with torch.no_grad():
        actual = restored.critic(batch, batch.action_chunks, batch.execution_masks)
    payload = torch.load(checkpoint, weights_only=False)

    assert torch.equal(expected, actual)
    assert torch.equal(restored.critic.core.q_support, torch.linspace(-0.1, 1.1, 201))
    assert payload["critic_metadata"]["q_representation"] == "categorical"
    assert payload["critic_metadata"]["q_hl_gauss_sigma_bins"] == 0.75
    assert payload["critic_metadata"]["rank_consensus"]["rank_consensus_enabled"]


def test_scalar_mode_keeps_single_output_q_heads():
    batch = make_synthetic_replay(num_samples=4, generated_horizon=3, action_dim=2)
    state = build_train_state(
        _config(categorical=False), batch, multimodal_critic_factory=_factory
    )

    assert state.critic.core.q_representation == "scalar"
    assert state.critic.core.q_heads[0][-1].out_features == 1
    with pytest.raises(RuntimeError, match="categorical Q"):
        state.critic.q_logits_from_features(
            state.critic.encode_state(batch),
            batch.action_chunks,
            batch.execution_masks,
        )
