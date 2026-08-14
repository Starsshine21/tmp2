from __future__ import annotations

import torch

from .ensemble import ensemble_mean_std
from .multimodal_critic import MultiHeadScalarQCritic, MultiHeadUdivlCritic
from .types import ChunkBatch
from .uncertainty import conformal_scale as compute_conformal_scale


def _critic_predictions(
    critic,
    batch: ChunkBatch,
    *,
    divl=None,
    inference_batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    device = next(critic.parameters()).device
    batch_size = batch.batch_size if inference_batch_size is None else max(1, int(inference_batch_size))
    q_chunks = []
    probability_chunks = []
    for start in range(0, batch.batch_size, batch_size):
        indices = torch.arange(start, min(start + batch_size, batch.batch_size))
        sample = batch.index_select(indices).to(device)
        if isinstance(critic, (MultiHeadUdivlCritic, MultiHeadScalarQCritic)):
            features = critic.encode_state(sample)
            q_values = critic.q_from_features(features, sample.action_chunks, sample.execution_masks)
            probabilities = (
                torch.softmax(critic.value_logits_from_features(features), dim=-1)
                if isinstance(critic, MultiHeadUdivlCritic)
                else None
            )
        else:
            q_values = critic(sample.observations, sample.action_chunks, sample.execution_masks)
            probabilities = divl(sample.observations) if divl is not None else None
        q_chunks.append(q_values.detach().cpu())
        if probabilities is not None:
            probability_chunks.append(probabilities.detach().cpu())
    q_values = torch.cat(q_chunks, dim=1)
    if not probability_chunks:
        return q_values, None
    probability_batch_dim = 1 if probability_chunks[0].ndim == 3 else 0
    return q_values, torch.cat(probability_chunks, dim=probability_batch_dim)


def _correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() < 2:
        return x.new_tensor(0.0)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.std(unbiased=False) * y.std(unbiased=False)
    if denominator <= 1e-8:
        return x.new_tensor(0.0)
    return ((x * y).mean() / denominator).clamp(-1.0, 1.0)


def _ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32, device=values.device)
    return ranks


@torch.no_grad()
def offline_calibration_metrics(
    critic,
    batch: ChunkBatch,
    *,
    divl=None,
    conformal_scale: float = 1.0,
    inference_batch_size: int | None = None,
) -> dict[str, float]:
    was_training = critic.training
    critic.eval()
    q_values, probs = _critic_predictions(
        critic,
        batch,
        divl=divl,
        inference_batch_size=inference_batch_size,
    )
    q_mean, q_std = ensemble_mean_std(q_values)
    target_is_mc = batch.mc_returns is not None
    target = (batch.mc_returns if target_is_mc else batch.chunk_returns).cpu()
    error = q_mean - target
    rmse = torch.sqrt(torch.mean(error.pow(2)))
    huber = torch.nn.functional.huber_loss(q_mean, target)
    error_abs = error.abs()
    rank_corr = _correlation(_ranks(q_mean), _ranks(target))
    disagreement_error_corr = _correlation(q_std, error_abs)
    calibrated_std = q_std * float(conformal_scale)
    interval_coverage = (error_abs <= calibrated_std).float().mean()
    member_error = q_values - target.unsqueeze(0)
    all_positive = (member_error > 0).all(dim=0)
    all_negative = (member_error < 0).all(dim=0)
    sign_disagreement = (~(all_positive | all_negative)).float().mean()
    pair_i, pair_j = torch.triu_indices(batch.batch_size, batch.batch_size, offset=1, device=q_mean.device)
    target_delta = target[pair_i] - target[pair_j]
    valid_pairs = target_delta != 0
    if valid_pairs.any():
        q_delta = q_mean[pair_i] - q_mean[pair_j]
        pairwise_accuracy = (torch.sign(q_delta[valid_pairs]) == torch.sign(target_delta[valid_pairs])).float().mean()
    else:
        pairwise_accuracy = q_mean.new_tensor(0.0)
    calibration_terms = []
    order = torch.argsort(calibrated_std)
    for indices in torch.tensor_split(order, min(10, max(1, batch.batch_size))):
        if indices.numel():
            calibration_terms.append((error_abs[indices].mean() - calibrated_std[indices].mean()).abs())
    ece = torch.stack(calibration_terms).mean() if calibration_terms else q_mean.new_tensor(0.0)
    metrics = {
        "q_rmse": float(rmse.item()),
        "q_huber": float(huber.item()),
        "q_rank_correlation": float(rank_corr.item()),
        "pairwise_ranking_accuracy": float(pairwise_accuracy.item()),
        "ensemble_disagreement": float(q_std.mean().item()),
        "ensemble_sign_disagreement": float(sign_disagreement.item()),
        "disagreement_error_correlation": float(disagreement_error_corr.item()),
        "interval_coverage": float(interval_coverage.item()),
        "expected_calibration_error": float(ece.item()),
        "q_exploitation_gap": float(error.mean().item()),
        "conformal_scale": float(conformal_scale),
        "q_mean": float(q_mean.mean().item()),
        "calibration_target_is_mc": float(target_is_mc),
    }
    if probs is not None:
        probs = probs.clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        entropy = entropy / torch.log(torch.tensor(probs.shape[-1], device=probs.device, dtype=probs.dtype))
        metrics["categorical_entropy"] = float(entropy.mean().item())
        metrics["categorical_saturation"] = float(
            ((probs[..., 0] + probs[..., -1]) > 0.5).float().mean().item()
        )
    critic.train(was_training)
    return metrics


@torch.no_grad()
def fit_conformal_calibration(
    state,
    batch: ChunkBatch,
    config: dict,
    *,
    inference_batch_size: int | None = None,
) -> float:
    was_training = state.critic.training
    state.critic.eval()
    q_values, _ = _critic_predictions(
        state.critic,
        batch,
        inference_batch_size=inference_batch_size,
    )
    q_mean, q_std = ensemble_mean_std(q_values)
    uncertainty_cfg = config.get("uncertainty", {})
    scale = compute_conformal_scale(
        q_mean,
        q_std,
        (batch.mc_returns if batch.mc_returns is not None else batch.chunk_returns).cpu(),
        coverage_delta=float(uncertainty_cfg.get("conformal_delta", uncertainty_cfg.get("coverage_delta", 0.1))),
        min_samples=int(uncertainty_cfg.get("min_calibration_samples", 16)),
    )
    state.conformal_scale = scale
    state.critic.train(was_training)
    return scale


def validation_metrics_for_training(state, batch: ChunkBatch, config: dict) -> dict[str, float]:
    """Compute fixed-replay critic diagnostics for periodic training logs."""
    uncertainty_cfg = config.get("uncertainty", {})
    inference_batch_size = config.get("evaluation", {}).get("inference_batch_size")
    if bool(uncertainty_cfg.get("use_conformal", False)):
        fit_conformal_calibration(
            state,
            batch,
            config,
            inference_batch_size=inference_batch_size,
        )
    metrics = offline_calibration_metrics(
        state.critic,
        batch,
        divl=state.divl if bool(config.get("divl", {}).get("enabled", True)) else None,
        conformal_scale=state.conformal_scale,
        inference_batch_size=inference_batch_size,
    )
    return {f"validation_{key}": value for key, value in metrics.items()}
