from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import gc
import math
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import torch
import torch.nn.functional as F

from .conservative_advantage import (
    RunningMAD,
    group_normalized_advantage,
    lcb_advantage,
    scheduled_lambda_abs,
    sign_consensus_advantage,
)
from .categorical_q import (
    categorical_q_entropy,
    consensus_ranking_loss,
    decode_categorical_q,
    hl_gauss_projection,
    ranking_action_negatives,
)
from .critic import ScalarQEnsemble, assert_no_gradients, clone_target, soft_update
from .critic_targets import aggregate_value_heads
from .distributional_value import DistributionalValueEnsemble, make_support, make_support_from_targets
from .divl import divl_projection_targets, divl_quantile_values
from .ensemble import bootstrap_mask, ensemble_mean_std
from .flash_ogpo import flash_ppo_loss, sample_flash_rollout
from .flow_sde import GaussianFlowPolicy
from .full_ogpo import full_chain_ais_ppo_loss, full_chain_ppo_loss
from .gemma_siglip_backbone import (
    build_gemma_siglip_critic,
    build_gemma_siglip_scalar_q_critic,
    configure_critic_stage,
    install_final_gemma_lora,
)
from .losses import (
    action_smoothness_loss,
    flow_matching_anchor_loss,
    success_buffer_loss,
    weighted_flow_matching_loss,
)
from .metrics import grad_norm
from .multimodal_critic import MultiHeadScalarQCritic, MultiHeadUdivlCritic
from .openpi_flow_spec import OpenPIStochasticFlowPolicy
from .pi05_jax_adapter import PI05JaxFlowPolicy, ema_actor_state, load_pi05_jax_flow_policy
from .pi05_jax_flow_core import (
    OpenPIJaxFlowSpec,
    flash_ppo_loss as jax_flash_ppo_loss,
    flow_matching_loss as jax_flow_matching_loss,
    full_chain_ais_ppo_loss as jax_full_chain_ais_ppo_loss,
    full_chain_ppo_loss as jax_full_chain_ppo_loss,
    gaussian_kl_diag as jax_gaussian_kl_diag,
    gaussian_log_prob as jax_gaussian_log_prob,
    rollout as jax_rollout,
    sample_flash_rollout as sample_jax_flash_rollout,
    state_adaptive_kl_penalty as jax_state_adaptive_kl_penalty,
    transition_kl as jax_transition_kl,
    transition_log_prob as jax_transition_log_prob,
    transition_log_std as jax_transition_log_std,
    transition_mean as jax_transition_mean,
)
from .pi05_pytorch_adapter import PI05PytorchFlowPolicy, load_pi05_pytorch_flow_policy
from .temporal_rectification import EmpiricalGradientRectifier, analytic_rectification
from .types import ChunkBatch
from .uncertainty import (
    actor_clip_for_uncertainty,
    kl_uncertainty_scale,
    state_adaptive_kl_penalty,
    state_entropy_weight,
    support_weight,
)
from .value_critic_protocol import StateFeatures


@dataclass
class OGPOTrainState:
    critic: Any
    target_critic: Any
    divl: DistributionalValueEnsemble | None
    target_divl: DistributionalValueEnsemble | None
    policy: OpenPIStochasticFlowPolicy
    old_policy: OpenPIStochasticFlowPolicy
    reference_policy: OpenPIStochasticFlowPolicy
    critic_optimizer: torch.optim.Optimizer
    actor_optimizer: torch.optim.Optimizer
    support: torch.Tensor
    running_mad: RunningMAD
    rectifier: EmpiricalGradientRectifier
    conformal_scale: float = 1.0
    step: int = 0
    critic_stage: str = "head_td"
    critic_stage_step: int = 0
    target_generator: torch.Generator | None = None


def _using_jax_actor(policy: OpenPIStochasticFlowPolicy) -> bool:
    return isinstance(policy, PI05JaxFlowPolicy)


def _single_epoch_on_policy_log_prob(
    new_log_prob: jax.Array,
    old_log_prob: jax.Array,
) -> jax.Array:
    """Use the exact on-policy value while retaining the current-policy gradient."""
    return old_log_prob + new_log_prob - jax.lax.stop_gradient(new_log_prob)


def _anchor_current_to_old_value(current: jax.Array, old: jax.Array) -> jax.Array:
    """Evaluate a one-step surrogate at the old value with the current Jacobian."""
    return old + current - jax.lax.stop_gradient(current)


def _conditionally_anchor_current_to_old_value(
    current: jax.Array,
    old: jax.Array,
    anchor_strength: jax.Array | float,
) -> jax.Array:
    """Anchor the value to old while preserving the current-policy Jacobian."""
    strength = jnp.asarray(anchor_strength, dtype=current.dtype)
    return current + strength * jax.lax.stop_gradient(old - current)


def _numpy_gaussian_kl_diag(
    mean_p: np.ndarray,
    log_std_p: np.ndarray,
    mean_q: np.ndarray,
    log_std_q: np.ndarray,
    *,
    event_dim: int | None = None,
) -> np.ndarray:
    """Host-side diagonal Gaussian KL used for the post-update trust-region gate."""
    mean_p = np.asarray(mean_p, dtype=np.float32)
    log_std_p = np.asarray(log_std_p, dtype=np.float32)
    mean_q = np.asarray(mean_q, dtype=np.float32)
    log_std_q = np.asarray(log_std_q, dtype=np.float32)
    if event_dim is not None:
        if event_dim <= 0 or event_dim > mean_p.shape[-1]:
            raise ValueError(
                f"event_dim must be in [1, {mean_p.shape[-1]}], got {event_dim}"
            )
        mean_p = mean_p[..., :event_dim]
        log_std_p = log_std_p[..., :event_dim]
        mean_q = mean_q[..., :event_dim]
        log_std_q = log_std_q[..., :event_dim]
    var_p = np.exp(2.0 * log_std_p)
    var_q = np.exp(2.0 * log_std_q)
    kl = log_std_q - log_std_p + (var_p + np.square(mean_p - mean_q)) / (2.0 * var_q) - 0.5
    return kl.reshape(kl.shape[0], -1).sum(axis=-1)


@partial(
    nnx.jit,
    static_argnames=("num_steps", "sde_mode", "group_size"),
)
def _sample_frozen_jax_flash_rollout(
    actor,
    observation,
    selected_step,
    rng,
    *,
    num_steps: int,
    sde_mode: str,
    group_size: int,
):
    rollout = sample_jax_flash_rollout(
        actor=actor,
        flow_spec=OpenPIJaxFlowSpec(num_steps),
        observation=observation,
        group_size=group_size,
        selected_step=selected_step,
        rng=rng,
        sde_mode=sde_mode,
    )
    return rollout.x_t, rollout.x_prev, rollout.timestep, rollout.endpoint


def _make_critic_optimizer(
    critic,
    divl,
    critic_cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    named_parameters = list(critic.named_parameters())
    if divl is not None:
        named_parameters.extend((f"divl.{name}", parameter) for name, parameter in divl.named_parameters())
    trainable = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError("critic stage has no trainable parameters")
    base_lr = float(critic_cfg.get("learning_rate", 3e-4))
    backbone_lr = float(critic_cfg.get("backbone", {}).get("learning_rate", base_lr))
    lora_parameters = [parameter for name, parameter in trainable if ".lora_" in name]
    backbone_parameters = [
        parameter
        for name, parameter in trainable
        if ".lora_" not in name
        and (
            name.startswith("state_encoder.gemma_model.")
            or name.startswith("state_encoder.vision_model.")
        )
    ]
    excluded = {id(parameter) for parameter in (*lora_parameters, *backbone_parameters)}
    regular_parameters = [parameter for _, parameter in trainable if id(parameter) not in excluded]
    groups = []
    if regular_parameters:
        groups.append({"params": regular_parameters, "lr": base_lr})
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": backbone_lr})
    if lora_parameters:
        lora_cfg = critic_cfg.get("gemma_lora", {})
        groups.append({"params": lora_parameters, "lr": float(lora_cfg.get("learning_rate", base_lr * 0.1))})
    for group in groups:
        group["initial_lr"] = group["lr"]
    optimizer_name = str(critic_cfg.get("optimizer", "adamw")).lower()
    optimizer_kwargs = {
        "lr": base_lr,
        "weight_decay": float(critic_cfg.get("weight_decay", 1e-4)),
    }
    if optimizer_name == "adam":
        return torch.optim.Adam(groups, **optimizer_kwargs)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(groups, **optimizer_kwargs)
    raise ValueError(f"unsupported critic.optimizer={optimizer_name!r}")


def _apply_critic_lr_schedule(
    state: OGPOTrainState,
    config: dict[str, Any],
) -> float:
    critic_cfg = config.get("critic", {})
    schedule = str(critic_cfg.get("lr_schedule", "constant")).lower()
    if schedule == "constant":
        scale = 1.0
    elif schedule == "cosine":
        total_steps = int(
            critic_cfg.get(
                "lr_schedule_steps",
                config.get("training", {}).get("critic_steps", 1),
            )
        )
        if total_steps <= 0:
            raise ValueError("critic cosine lr_schedule requires positive total steps")
        progress = min(float(state.step) / max(total_steps - 1, 1), 1.0)
        min_ratio = float(critic_cfg.get("min_lr_ratio", 0.0))
        if not 0.0 <= min_ratio <= 1.0:
            raise ValueError("critic.min_lr_ratio must be in [0, 1]")
        scale = min_ratio + 0.5 * (1.0 - min_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )
    else:
        raise ValueError(f"unsupported critic.lr_schedule={schedule!r}")
    for group in state.critic_optimizer.param_groups:
        initial_lr = float(group.get("initial_lr", group["lr"]))
        group["initial_lr"] = initial_lr
        group["lr"] = initial_lr * scale
    return scale


def build_train_state(
    config: dict[str, Any],
    batch: ChunkBatch,
    *,
    device: str | torch.device = "cpu",
    multimodal_critic_factory: Callable[[ChunkBatch, dict[str, Any]], Any] | None = None,
) -> OGPOTrainState:
    method_name = str(config.get("method", {}).get("name", "ogpo-divl"))
    critic_cfg = config.get("critic", {})
    divl_cfg = config.get("divl", {})
    actor_cfg = config.get("actor", {})
    flow_cfg = config.get("flow", {})
    architecture = str(critic_cfg.get("architecture", "mlp"))
    if method_name == "ogpo-origin":
        required = {
            "critic.architecture": architecture == "gemma_siglip_scalar_q",
            "divl.enabled": not bool(divl_cfg.get("enabled", True)),
            "actor.advantage_mode": str(actor_cfg.get("advantage_mode")) == "group_mean",
            "flow.sde_mode": str(flow_cfg.get("sde_mode")) == "ogpo_corrected",
        }
        invalid = [name for name, valid in required.items() if not valid]
        if invalid:
            raise ValueError(
                "ogpo-origin configuration violates original-path invariants: "
                + ", ".join(invalid)
            )
    if architecture not in {"mlp", "gemma_siglip_multihead", "gemma_siglip_scalar_q"}:
        raise ValueError(f"unsupported critic.architecture={architecture!r}")
    if architecture == "mlp":
        feature_source = str(critic_cfg.get("feature_source", "replay_state"))
        if feature_source != "replay_state":
            raise ValueError("critic.feature_source currently supports only 'replay_state'")
        if bool(critic_cfg.get("shared_frozen_encoder", False)):
            raise ValueError(
                "critic.shared_frozen_encoder requires an unavailable PI0.5 feature_source; "
                "use feature_source: replay_state and shared_frozen_encoder: false"
            )
        if int(critic_cfg.get("train_last_n_backbone_layers", 0)) != 0:
            raise ValueError("replay-state critic has no backbone layers to unfreeze")
        if not bool(critic_cfg.get("detach_policy_features", True)):
            raise ValueError("replay-state critic requires critic.detach_policy_features=true")
    flow_adapter = str(flow_cfg.get("adapter", "gaussian_openpi"))
    if flow_adapter not in {"gaussian_openpi", "gaussian", "pi05_pytorch", "pi05_jax"}:
        raise ValueError(
            f"unsupported flow.adapter={flow_adapter!r}; expected gaussian_openpi, pi05_pytorch, or pi05_jax"
        )

    ensemble_size = int(critic_cfg.get("ensemble_size", 3))
    hidden_dim = int(critic_cfg.get("hidden_dim", 128))
    num_layers = int(critic_cfg.get("num_layers", 2))
    if architecture == "gemma_siglip_multihead":
        if ensemble_size != 3:
            raise ValueError("gemma_siglip_multihead requires critic.ensemble_size=3")
        critic_factory = multimodal_critic_factory or build_gemma_siglip_critic
        critic = critic_factory(batch, config)
        if torch.device(device).type == "cuda":
            parameter_bytes = sum(
                parameter.numel() * parameter.element_size()
                for parameter in critic.parameters()
            )
            free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(device))
            print(
                "[ogpo] critic_cuda_load "
                f"parameter_gib={parameter_bytes / 2**30:.3f} "
                f"free_gib={free_bytes / 2**30:.3f} "
                f"total_gib={total_bytes / 2**30:.3f}",
                flush=True,
            )
        critic = critic.to(device)
        if critic.ensemble_size != 3:
            raise ValueError("multimodal critic factory must return three Q-V pairs")
        critic_stage = str(critic_cfg.get("stage", "head_td"))
        lora_cfg = critic_cfg.get("gemma_lora", {})
        if int(lora_cfg.get("final_n_layers", 0)) > 0 and hasattr(critic.state_encoder, "gemma_model"):
            install_final_gemma_lora(
                critic.state_encoder,
                final_n_layers=int(lora_cfg["final_n_layers"]),
                rank=int(lora_cfg.get("rank", 8)),
                alpha=float(lora_cfg.get("alpha", 16.0)),
                target_suffixes=tuple(lora_cfg.get("target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))),
            )
        configure_critic_stage(critic, critic_stage)
        target_critic = clone_target(critic).to(device)
        divl = None
        target_divl = None
    elif architecture == "gemma_siglip_scalar_q":
        if bool(divl_cfg.get("enabled", True)):
            raise ValueError("gemma_siglip_scalar_q requires divl.enabled=false")
        critic_factory = multimodal_critic_factory or build_gemma_siglip_scalar_q_critic
        critic = critic_factory(batch, config)
        if not isinstance(critic, MultiHeadScalarQCritic):
            raise TypeError("gemma_siglip_scalar_q factory must return MultiHeadScalarQCritic")
        if critic.ensemble_size != ensemble_size:
            raise ValueError(
                "gemma_siglip_scalar_q factory ensemble size does not match critic.ensemble_size"
            )
        critic = critic.to(device)
        critic_stage = str(critic_cfg.get("stage", "head_td"))
        target_critic = clone_target(critic).to(device)
        divl = None
        target_divl = None
    else:
        critic_stage = "legacy"
        critic = ScalarQEnsemble(
            ensemble_size,
            batch.obs_dim,
            batch.generated_horizon,
            batch.action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            randomized_prior_scale=float(critic_cfg.get("randomized_prior_scale", 0.0)),
        ).to(device)
        target_critic = clone_target(critic).to(device)
        divl = DistributionalValueEnsemble(
            ensemble_size,
            batch.obs_dim,
            hidden_dim,
            num_layers,
            int(divl_cfg.get("num_atoms", 51)),
        ).to(device)
        target_divl = clone_target(divl).to(device)
    num_atoms = int(divl_cfg.get("num_atoms", 51))
    if bool(divl_cfg.get("auto_support", False)):
        support_targets = batch.mc_returns if batch.mc_returns is not None else batch.chunk_returns
        support = make_support_from_targets(
            support_targets.to(device),
            num_atoms=num_atoms,
            margin_fraction=float(divl_cfg.get("support_margin_fraction", 0.05)),
        )
    else:
        support = make_support(
            float(divl_cfg.get("v_min", -5.0)),
            float(divl_cfg.get("v_max", 5.0)),
            num_atoms,
            device=device,
        )
    if flow_adapter in {"gaussian_openpi", "gaussian"}:
        action_flat_dim = batch.generated_horizon * batch.action_dim
        policy = GaussianFlowPolicy(
            condition_dim=batch.obs_dim,
            action_dim=action_flat_dim,
            hidden_dim=int(actor_cfg.get("hidden_dim", 128)),
            num_steps=int(flow_cfg.get("num_steps", 8)),
            stochastic_variance=float(flow_cfg.get("stochastic_variance", 0.04)),
            sde_mode=str(flow_cfg.get("sde_mode", "gaussian_adapter")),
        ).to(device)
        old_policy = clone_target(policy).to(device)
        reference_policy = clone_target(policy).to(device)
    else:
        if flow_adapter == "pi05_jax":
            _load_flow_policy = load_pi05_jax_flow_policy
        else:
            _load_flow_policy = load_pi05_pytorch_flow_policy
        policy = _load_flow_policy(
            checkpoint_dir=str(flow_cfg["checkpoint_dir"]),
            train_config_name=str(flow_cfg["train_config"]),
            image_mapping=dict(flow_cfg.get("image_mapping", {})),
            environment_action_dim=batch.action_dim,
            num_steps=int(flow_cfg.get("num_steps", 10)),
            stochastic_variance=float(flow_cfg.get("stochastic_variance", 0.04)),
            sde_mode=str(flow_cfg.get("sde_mode", "gaussian_adapter")),
            residual_hidden_dim=int(actor_cfg.get("hidden_dim", 128)),
            device=device,
        )
        if policy.model_horizon != batch.generated_horizon:
            raise ValueError(
                f"PI0.5 action horizon {policy.model_horizon} does not match replay horizon {batch.generated_horizon}"
            )
        old_policy = policy.clone_adapter()
        reference_policy = policy.clone_adapter()
    actor_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if isinstance(policy, PI05JaxFlowPolicy):
        policy.init_actor_optimizer(
            learning_rate=float(actor_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(actor_cfg.get("weight_decay", 0.0)),
            optimizer=str(actor_cfg.get("optimizer", "adafactor")),
            max_grad_norm=float(actor_cfg.get("max_grad_norm", 1.0)),
            preserve_state_for_rollback=bool(actor_cfg.get("reject_update_on_kl", False)),
        )
    return OGPOTrainState(
        critic=critic,
        target_critic=target_critic,
        divl=divl,
        target_divl=target_divl,
        policy=policy,
        old_policy=old_policy,
        reference_policy=reference_policy,
        critic_optimizer=_make_critic_optimizer(critic, divl, critic_cfg),
        actor_optimizer=torch.optim.AdamW(
            actor_parameters,
            lr=float(actor_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(actor_cfg.get("weight_decay", 0.0)),
        ) if actor_parameters else torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-4),
        support=support,
        running_mad=RunningMAD(),
        rectifier=EmpiricalGradientRectifier(
            int(flow_cfg.get("num_steps", 8)),
            momentum=float(flow_cfg.get("rectification_momentum", 0.95)),
            clip_min=float(flow_cfg.get("rectification_clip_min", 0.25)),
            clip_max=float(flow_cfg.get("rectification_clip_max", 4.0)),
        ),
        conformal_scale=float(config.get("uncertainty", {}).get("conformal_scale", 1.0)),
        critic_stage=critic_stage,
        target_generator=torch.Generator(device=torch.device(device).type).manual_seed(
            int(config.get("training", {}).get("seed", 0)) + 1701
        ),
    )


def maybe_advance_critic_stage(
    state: OGPOTrainState,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> bool:
    """Advance staged critic training only after fixed-validation gates pass."""
    if not isinstance(state.critic, MultiHeadUdivlCritic):
        return False
    gates = config.get("critic", {}).get("stage_gates", {})
    required = {
        "validation_pairwise_ranking_accuracy": ("min_pairwise_ranking_accuracy", lambda x, y: x >= y),
        "validation_interval_coverage": ("min_interval_coverage", lambda x, y: x >= y),
        "validation_categorical_saturation": ("max_categorical_saturation", lambda x, y: x <= y),
        "validation_q_exploitation_gap": ("max_abs_exploitation_gap", lambda x, y: abs(x) <= y),
    }
    if state.critic_stage_step < int(gates.get("min_stage_steps", 0)):
        return False
    for metric_name, (gate_name, predicate) in required.items():
        if gate_name not in gates:
            continue
        if metric_name not in metrics or not predicate(float(metrics[metric_name]), float(gates[gate_name])):
            return False
    if not gates or not any(gate_name in gates for gate_name, _ in required.values()):
        return False
    if state.critic_stage == "head_mc":
        next_stage = "head_td"
    elif state.critic_stage == "head_td" and int(
        config.get("critic", {}).get("gemma_lora", {}).get("final_n_layers", 0)
    ) > 0:
        next_stage = "gemma_lora_td"
    else:
        return False
    configure_critic_stage(state.critic, next_stage)
    state.critic_optimizer = _make_critic_optimizer(state.critic, state.divl, config.get("critic", {}))
    state.critic_stage = next_stage
    state.critic_stage_step = 0
    return True


def apply_scheduled_critic_stage(
    state: OGPOTrainState,
    config: dict[str, Any],
) -> bool:
    """Apply a deterministic warmup schedule keyed by optimizer steps."""
    if not isinstance(state.critic, MultiHeadUdivlCritic):
        return False
    schedule = config.get("critic", {}).get("stage_schedule", [])
    if not schedule:
        return False
    elapsed = 0
    desired_stage = str(schedule[-1]["stage"])
    for index, entry in enumerate(schedule):
        desired_stage = str(entry["stage"])
        steps = entry.get("steps")
        if steps is None:
            if index != len(schedule) - 1:
                raise ValueError("only the final critic.stage_schedule entry may omit steps")
            break
        steps = int(steps)
        if steps <= 0:
            raise ValueError("critic.stage_schedule steps must be positive")
        elapsed += steps
        if state.step < elapsed:
            break
    if desired_stage == state.critic_stage:
        return False
    configure_critic_stage(state.critic, desired_stage)
    state.critic_optimizer = _make_critic_optimizer(
        state.critic,
        state.divl,
        config.get("critic", {}),
    )
    state.critic_stage = desired_stage
    state.critic_stage_step = 0
    return True


def _policy_condition(
    policy: OpenPIStochasticFlowPolicy,
    batch: ChunkBatch,
    *,
    next_observation: bool = False,
):
    return policy.condition_from_batch(batch, next_observation=next_observation)


def _jax_actor_step(policy: PI05JaxFlowPolicy, loss_fn: Callable[[Any], jax.Array]) -> float:
    if policy.actor_tx is None or policy.actor_opt_state is None:
        raise RuntimeError("JAX PI0.5 actor optimizer is not initialized")

    def _loss_from_state(actor_state):
        actor = nnx.merge(policy.actor_graphdef, actor_state)
        return loss_fn(actor)

    loss_value, grads = jax.value_and_grad(_loss_from_state)(policy.actor_state)
    policy.apply_actor_gradients(grads)
    return float(loss_value)


def _torch_condition_to_jax(condition):
    if isinstance(condition, PI05FlowCondition):
        return policy_observation_to_jax(condition.observation)
    return condition


def policy_observation_to_jax(observation):
    from .pi05_jax_adapter import _observation_to_jax  # noqa: PLC0415

    return _observation_to_jax(observation)


def _jax_flow_matching_inputs(
    policy: PI05JaxFlowPolicy,
    batch: ChunkBatch,
    *,
    seed: int,
) -> dict[str, Any]:
    device = next(policy.parameters()).device
    source = batch.to(device)
    condition = _policy_condition(policy, source)
    observation = policy_observation_to_jax(condition.observation)
    endpoint = policy.action_chunks_to_flow(source).reshape(source.batch_size, -1)
    actions = jnp.asarray(endpoint.detach().cpu().numpy())
    noise_key, time_key = jax.random.split(jax.random.PRNGKey(int(seed)))
    noise = jax.random.normal(noise_key, actions.shape, dtype=actions.dtype)
    time = jax.random.beta(time_key, 1.5, 1.0, (source.batch_size,)) * 0.999 + 0.001
    return {
        "observation": observation,
        "action_endpoint": actions,
        "noise": noise,
        "timestep": time,
    }


def _prepare_jax_regularization(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    fm_batch: ChunkBatch | None,
    success_batch: ChunkBatch | None,
    enable_success: bool = True,
    seed_step: int | None = None,
) -> dict[str, Any]:
    assert isinstance(state.policy, PI05JaxFlowPolicy)
    regularization_cfg = config.get("regularization", {})
    lambda_smooth = float(regularization_cfg.get("lambda_smooth", 0.0))
    if lambda_smooth != 0.0:
        raise ValueError(
            "JAX PI0.5 full finetuning does not yet support differentiable raw-action smoothness; "
            "set regularization.lambda_smooth=0"
        )
    result: dict[str, Any] = {
        "lambda_fm": float(regularization_cfg.get("lambda_fm", 0.1)),
        "lambda_success": float(regularization_cfg.get("lambda_success", 0.0)),
        "fm": None,
        "success": None,
    }
    stochastic_step = int(state.step if seed_step is None else seed_step)
    if result["lambda_fm"] != 0.0:
        result["fm"] = _jax_flow_matching_inputs(
            state.policy,
            fm_batch if fm_batch is not None else batch,
            seed=stochastic_step + 4101,
        )
    if enable_success and result["lambda_success"] != 0.0:
        success_source = success_batch if success_batch is not None else _success_subset(batch)
        if success_source is not None:
            result["success"] = _jax_flow_matching_inputs(
                state.policy,
                success_source,
                seed=stochastic_step + 5101,
            )
    return result


def _jax_regularization_loss(actor: Any, inputs: dict[str, Any]) -> tuple[jax.Array, jax.Array, jax.Array]:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    fm_loss = zero
    if inputs["fm"] is not None:
        fm_loss = jax_flow_matching_loss(actor=actor, **inputs["fm"])
    success_loss = zero
    if inputs["success"] is not None:
        success_loss = jax_flow_matching_loss(actor=actor, **inputs["success"])
    total = float(inputs["lambda_fm"]) * fm_loss + float(inputs["lambda_success"]) * success_loss
    return total, fm_loss, success_loss


def _accumulate_jax_grads_on_host(existing: Any, grads: Any, *, weight: float) -> Any:
    host_grads = jax.device_get(grads)

    def _scaled_copy(value):
        array = np.asarray(value)
        scale = np.asarray(weight, dtype=array.dtype)
        return np.array(array * scale, copy=True)

    if existing is None:
        return jax.tree_util.tree_map(_scaled_copy, host_grads)

    def _add(accumulator, value):
        array = np.asarray(value, dtype=accumulator.dtype)
        scale = np.asarray(weight, dtype=accumulator.dtype)
        np.add(accumulator, array * scale, out=accumulator)
        return accumulator

    return jax.tree_util.tree_map(_add, existing, host_grads)


def _jax_tree_l2_norm(tree: Any) -> float:
    total = 0.0
    for leaf in jax.tree_util.tree_leaves(jax.device_get(tree)):
        flat = np.asarray(leaf).reshape(-1)
        # Some scanned PI0.5 parameter leaves exceed 4 GiB. Squaring a whole
        # leaf materializes another equally large host array, so accumulate
        # the norm in bounded chunks instead.
        for start in range(0, flat.size, 16 * 1024 * 1024):
            chunk = flat[start : start + 16 * 1024 * 1024]
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)
            total += float(np.dot(chunk, chunk))
    return total**0.5


def _slice_jax_batch(tree: Any, start: int, stop: int) -> Any:
    return jax.tree_util.tree_map(
        lambda value: value[start:stop] if value is not None else None,
        tree,
    )


def _shard_jax_batch(tree: Any, num_devices: int) -> Any:
    """Reshape a leading candidate batch into [device, local_batch, ...]."""
    num_devices = int(num_devices)
    if num_devices <= 0:
        raise ValueError("num_devices must be positive")

    def _shard(value):
        if value is None:
            return None
        if value.shape[0] % num_devices:
            raise ValueError(
                f"candidate batch {value.shape[0]} is not divisible by {num_devices} devices"
            )
        return value.reshape(
            num_devices,
            value.shape[0] // num_devices,
            *value.shape[1:],
        )

    return jax.tree_util.tree_map(_shard, tree)


def _first_pmap_replica(tree: Any) -> Any:
    """Keep one copy of a pmean-replicated pytree on the first local device."""
    return jax.tree_util.tree_map(
        lambda value: _pmap_replica_shard(value, 0).reshape(value.shape[1:]),
        tree,
    )


def _pmap_replica_shard(value: jax.Array, replica: int) -> jax.Array:
    """Return one pmap shard without dispatching a cross-device slice op."""
    replica = int(replica)
    if len(value.addressable_shards) == 1:
        local = value.addressable_shards[0].data
        if local.shape[0] == value.shape[0]:
            # Unit tests and single-device fallback arrays are not sharded.
            # This slice stays on the array's only committed device.
            return local[replica : replica + 1]
    for shard in value.addressable_shards:
        leading_index = shard.index[0]
        start = (
            int(leading_index)
            if isinstance(leading_index, int)
            else int(leading_index.start or 0)
        )
        if start == replica:
            return shard.data
    raise ValueError(
        f"pmap output has no addressable replica {replica}; "
        f"available shards={[shard.index for shard in value.addressable_shards]}"
    )


def _jax_tree_to_device(tree: Any, device: jax.Device) -> Any:
    return jax.tree_util.tree_map(
        lambda value: jax.device_put(
            value,
            device,
            donate=False,
            may_alias=True,
        ),
        tree,
    )


def _jax_tree_copy_to_device(tree: Any, device: jax.Device) -> Any:
    """Copy a pytree so its source buffers can be released or donated."""
    return jax.tree_util.tree_map(
        lambda value: jax.device_put(
            value,
            device,
            donate=False,
            may_alias=False,
        ),
        tree,
    )


def _mean_pmap_gradients_on_host(tree: Any) -> Any:
    """Average data-parallel gradients without a multi-GiB NCCL workspace."""
    def _mean_leaf(value):
        replicas = int(value.shape[0])
        first = np.asarray(jax.device_get(value[0]))
        # Preserve the gradient dtype. PI0.5's largest scanned bf16 leaf is
        # about 255 MiB; promoting every accumulated leaf to fp32 doubles the
        # persistent host gradient tree and can exceed the process VM limit.
        # This matches the established single-GPU host accumulation semantics.
        accumulator = np.array(first, copy=True)
        for replica in range(1, replicas):
            current = np.asarray(jax.device_get(value[replica]), dtype=accumulator.dtype)
            np.add(accumulator, current, out=accumulator)
        accumulator /= float(replicas)
        return accumulator

    return jax.tree_util.tree_map(_mean_leaf, tree)


@partial(jax.jit, donate_argnums=(0,), static_argnames=("other_weight",))
def _jax_tree_add_scaled(tree: Any, other: Any, *, other_weight: float) -> Any:
    return jax.tree_util.tree_map(
        lambda left, right: left + float(other_weight) * right,
        tree,
        other,
    )


@partial(jax.jit, donate_argnums=(0,), static_argnames=("scale",))
def _jax_tree_scale(tree: Any, *, scale: float) -> Any:
    return jax.tree_util.tree_map(lambda value: value * float(scale), tree)


def _mean_pmap_gradients_on_device(tree: Any, device: jax.Device) -> Any:
    """Move local gradient shards to one device and average without NCCL."""
    replicas = int(jax.tree_util.tree_leaves(tree)[0].shape[0])
    accumulator = jax.tree_util.tree_map(
        lambda value: jax.device_put(
            jnp.zeros(value.shape[1:], dtype=value.dtype),
            device,
        ),
        tree,
    )
    for replica in range(replicas):
        current = jax.tree_util.tree_map(
            lambda value: jax.device_put(
                _pmap_replica_shard(value, replica),
                device,
                donate=False,
                may_alias=False,
            ).reshape(value.shape[1:]),
            tree,
        )
        accumulator = _jax_tree_add_scaled(
            accumulator,
            current,
            other_weight=1.0,
        )
        jax.effects_barrier()
        del current
        gc.collect()
    return _jax_tree_scale(accumulator, scale=1.0 / replicas)


@jax.jit
def _jax_tree_l2_norm_device(tree: Any) -> jax.Array:
    squared_norms = [
        jnp.sum(jnp.square(value.astype(jnp.float32)))
        for value in jax.tree_util.tree_leaves(tree)
    ]
    return jnp.sqrt(jnp.sum(jnp.stack(squared_norms)))


def _stack_candidate_rollouts(parts: list[jax.Array]) -> jax.Array:
    """Stack candidate-major [G, B, ...] parts into state-major [B * G, ...]."""
    stacked = jnp.stack(parts, axis=1)
    return stacked.reshape((stacked.shape[0] * stacked.shape[1], *stacked.shape[2:]))


def _critic_execution_mask(batch: ChunkBatch, config: dict[str, Any]) -> torch.Tensor:
    if bool(config.get("data", {}).get("use_execution_mask", True)):
        return batch.execution_masks
    return torch.ones_like(batch.execution_masks, dtype=torch.bool)


@torch.no_grad()
def _reference_value_mean(
    state: OGPOTrainState,
    batch: ChunkBatch,
    *,
    num_samples: int,
    aggregation_mode: str = "ensemble_mean",
) -> torch.Tensor:
    condition = _policy_condition(state.reference_policy, batch, next_observation=True)
    rollout = state.reference_policy.rollout(condition, group_size=num_samples)
    condition_g = state.reference_policy.repeat_condition(condition, num_samples)
    environment_endpoints = state.reference_policy.flat_actions_to_environment(
        rollout.endpoint,
        condition_g,
    )
    endpoints = environment_endpoints.reshape(batch.batch_size, num_samples, -1)
    chunks = endpoints.reshape(batch.batch_size, num_samples, batch.generated_horizon, batch.action_dim)
    if isinstance(state.target_critic, (MultiHeadUdivlCritic, MultiHeadScalarQCritic)):
        features = state.target_critic.encode_state(batch, next_observation=True)
        repeated_features = StateFeatures(
            readout=features.readout.repeat_interleave(num_samples, dim=0)
        )
        flat_chunks = chunks.reshape(batch.batch_size * num_samples, batch.generated_horizon, batch.action_dim)
        flat_masks = batch.execution_masks[:, None, :].expand(
            batch.batch_size, num_samples, batch.generated_horizon
        ).reshape(batch.batch_size * num_samples, batch.generated_horizon)
        q_ref = state.target_critic.q_from_features(repeated_features, flat_chunks, flat_masks)
        q_aggregated, _ = aggregate_value_heads(
            q_ref,
            aggregation_mode,
            generator=state.target_generator,
        )
        return q_aggregated.reshape(batch.batch_size, num_samples).mean(dim=1)
    flat_obs = batch.next_observations[:, None, :].expand(
        batch.batch_size, num_samples, batch.next_observations.shape[-1]
    ).reshape(batch.batch_size * num_samples, -1)
    flat_chunks = chunks.reshape(batch.batch_size * num_samples, batch.generated_horizon, batch.action_dim)
    flat_masks = batch.execution_masks[:, None, :].expand(
        batch.batch_size, num_samples, batch.generated_horizon
    ).reshape(batch.batch_size * num_samples, batch.generated_horizon)
    q_ref = state.target_critic(flat_obs, flat_chunks, flat_masks)
    return q_ref.reshape(state.critic.ensemble_size, batch.batch_size, num_samples).mean(dim=(0, 2))


def _multimodal_scalar_q_update(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    zero_grad: bool = True,
    optimizer_step: bool = True,
    loss_scale: float = 1.0,
) -> dict[str, float]:
    """Scalar-Q Origin update with either policy TD or pure MC targets."""
    critic_cfg = config.get("critic", {})
    batch = batch.to(next(state.critic.parameters()).device)
    if zero_grad:
        state.critic_optimizer.zero_grad(set_to_none=True)
    critic_mask = _critic_execution_mask(batch, config)
    features = state.critic.encode_state(batch)
    q_pred = state.critic.q_from_features(features, batch.action_chunks, critic_mask)
    with torch.no_grad():
        target_mode = str(critic_cfg.get("target_mode", "td")).lower()
        if target_mode == "mc_return":
            if batch.mc_returns is None:
                raise ValueError("critic.target_mode=mc_return requires replay mc_returns")
            target = batch.mc_returns
            next_action_samples = 0
            next_q = None
            bootstrap_active = 0.0
            lambda_mc = 1.0
        elif target_mode == "td":
            next_action_samples = int(critic_cfg.get("reference_value_samples", 1))
            if next_action_samples <= 0:
                raise ValueError("TD Origin requires critic.reference_value_samples >= 1")
            bootstrap_mode = str(critic_cfg.get("bootstrap_target", "ensemble_mean"))
            if bootstrap_mode not in {"ensemble_mean", "mean"}:
                raise ValueError("TD Origin requires critic.bootstrap_target=ensemble_mean")
            next_q = _reference_value_mean(
                state,
                batch,
                num_samples=next_action_samples,
                aggregation_mode="ensemble_mean",
            )
            target = batch.chunk_returns + batch.discounts * (1.0 - batch.dones) * next_q
            bootstrap_active = 1.0
            lambda_mc = 0.0
        else:
            raise ValueError(f"unsupported scalar-Q critic.target_mode={target_mode!r}")

    q_error = (q_pred - target.unsqueeze(0)).square()
    q_loss = q_error.mean()
    (q_loss * float(loss_scale)).backward()
    critic_grad_norm = 0.0
    if optimizer_step:
        critic_grad_norm = float(grad_norm(state.critic.parameters()))
        torch.nn.utils.clip_grad_norm_(
            state.critic.parameters(),
            float(critic_cfg.get("max_grad_norm", 1000.0)),
        )
        state.critic_optimizer.step()
    target_update_period = int(critic_cfg.get("target_update_period", 1))
    if target_update_period <= 0:
        raise ValueError("critic.target_update_period must be positive")
    target_updated = optimizer_step and (state.step + 1) % target_update_period == 0
    if target_updated:
        soft_update(
            state.target_critic,
            state.critic,
            float(critic_cfg.get("target_tau", 0.005)),
        )
    if optimizer_step:
        state.step += 1
        state.critic_stage_step += 1
    member_losses = q_error.mean(dim=1)
    metrics = {
        "critic_loss": float(q_loss.detach().item()),
        "q_loss": float(q_loss.detach().item()),
        "q_loss_is_mse": 1.0,
        "divl_loss": 0.0,
        "divl_enabled": 0.0,
        "target_mean": float(target.mean().item()),
        "target_std": float(target.std(unbiased=False).item()),
        "td_error_abs_mean": float(
            (q_pred.detach() - target.unsqueeze(0)).abs().mean().item()
        ),
        "q_mean": float(q_pred.detach().mean().item()),
        "q_std": float(q_pred.detach().std(unbiased=False).item()),
        "v_divl_mean": 0.0,
        "critic_grad_norm": critic_grad_norm,
        "bootstrap_active": bootstrap_active,
        "bootstrap_target_is_min": 0.0,
        "bootstrap_target_is_subsample_min": 0.0,
        "reference_value_samples": float(next_action_samples),
        "lambda_mc": lambda_mc,
        "target_updated": float(target_updated),
        "critic_stage_head_td": float(state.critic_stage == "head_td"),
        "critic_stage_full_td": float(state.critic_stage == "full_td"),
    }
    if next_q is not None:
        metrics["reference_value_mean"] = float(next_q.mean().item())
    for member, member_loss in enumerate(member_losses):
        metrics[f"q_loss_member_{member}"] = float(member_loss.item())
    return metrics


def _multimodal_q_predictions(
    state: OGPOTrainState,
    batch: ChunkBatch,
    features: StateFeatures,
    critic_mask: torch.Tensor,
    critic_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Score positive and ranking actions with one shared state encoding."""
    rank_enabled = bool(critic_cfg.get("rank_consensus_enabled", False))
    use_strong = rank_enabled and bool(critic_cfg.get("rank_use_strong_noise", True))
    use_random = rank_enabled and bool(critic_cfg.get("rank_use_random_negative", True))
    variants: list[tuple[str, torch.Tensor]] = [("positive", batch.action_chunks)]
    if use_strong or use_random:
        action_pool = state.critic.core.action_pool
        strong, random = ranking_action_negatives(
            batch.action_chunks,
            critic_mask,
            action_mean=action_pool.action_mean,
            action_std=action_pool.action_std,
            action_min=action_pool.action_min,
            action_max=action_pool.action_max,
            noise_sigma=float(critic_cfg.get("rank_noise_sigma", 0.15)),
        )
        if use_strong:
            variants.append(("strong", strong))
        if use_random:
            variants.append(("random", random))

    variant_count = len(variants)
    combined_actions = torch.cat([actions for _, actions in variants], dim=0)
    combined_masks = torch.cat([critic_mask] * variant_count, dim=0)
    combined_features = StateFeatures(readout=features.readout.repeat(variant_count, 1))
    if state.critic.core.q_representation == "categorical":
        combined_logits = state.critic.q_logits_from_features(
            combined_features,
            combined_actions,
            combined_masks,
        )
        q_values = decode_categorical_q(combined_logits, state.critic.core.q_support)
        q_logits = combined_logits.reshape(
            state.critic.ensemble_size,
            variant_count,
            batch.batch_size,
            combined_logits.shape[-1],
        )[:, 0]
    else:
        q_values = state.critic.q_from_features(
            combined_features,
            combined_actions,
            combined_masks,
        )
        q_logits = None
    q_values = q_values.reshape(state.critic.ensemble_size, variant_count, batch.batch_size)
    ranking_values = {
        name: q_values[:, index]
        for index, (name, _) in enumerate(variants)
    }
    return q_values[:, 0], q_logits, ranking_values


def _ranking_loss_and_metrics(
    ranking_values: dict[str, torch.Tensor],
    batch: ChunkBatch,
    critic_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    positive = ranking_values["positive"]
    zero = positive.sum().float() * 0.0
    metrics = {
        "critic/rank_loss": 0.0,
        "critic/rank_loss_strong": 0.0,
        "critic/rank_loss_random": 0.0,
        "critic/rank_pair_count": 0.0,
        "critic/rank_d1_mean": 0.0,
        "critic/rank_d2_mean": 0.0,
        "critic/rank_d3_mean": 0.0,
        "critic/rank_worst_margin_mean": 0.0,
    }
    if not bool(critic_cfg.get("rank_consensus_enabled", False)):
        return zero, metrics
    valid = (
        batch.successes.bool()
        if bool(critic_cfg.get("rank_only_success", True))
        else torch.ones_like(batch.successes, dtype=torch.bool)
    )
    q_num_bins = int(critic_cfg.get("q_num_bins", 201))
    if q_num_bins < 2:
        raise ValueError("critic.q_num_bins must be at least 2")
    bin_width = (
        float(critic_cfg.get("q_vmax", 1.1)) - float(critic_cfg.get("q_vmin", -0.1))
    ) / (q_num_bins - 1)
    margin = float(critic_cfg.get("rank_margin_bins", 2.0)) * bin_width
    losses = []
    margin_parts = []
    worst_parts = []
    for name in ("strong", "random"):
        if name not in ranking_values:
            continue
        rank_loss, member_margins, worst_margin = consensus_ranking_loss(
            positive,
            ranking_values[name],
            valid,
            margin=margin,
            softmin_tau=float(critic_cfg.get("rank_softmin_tau", 0.02)),
            temperature=float(critic_cfg.get("rank_temperature", 0.02)),
        )
        losses.append(rank_loss)
        metrics[f"critic/rank_loss_{name}"] = float(rank_loss.detach().item())
        if bool(valid.any()):
            margin_parts.append(member_margins[:, valid])
            worst_parts.append(worst_margin[valid])
    if not losses:
        return zero, metrics
    loss = torch.stack(losses).mean()
    metrics["critic/rank_loss"] = float(loss.detach().item())
    metrics["critic/rank_pair_count"] = float(int(valid.sum().item()) * len(losses))
    if margin_parts:
        all_margins = torch.cat(margin_parts, dim=1).detach()
        all_worst = torch.cat(worst_parts, dim=0).detach()
        for member in range(min(3, all_margins.shape[0])):
            metrics[f"critic/rank_d{member + 1}_mean"] = float(
                all_margins[member].mean().item()
            )
        metrics["critic/rank_worst_margin_mean"] = float(all_worst.mean().item())
    return loss, metrics


def _multimodal_critic_update(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    zero_grad: bool = True,
    optimizer_step: bool = True,
    loss_scale: float = 1.0,
) -> dict[str, float]:
    critic_cfg = config.get("critic", {})
    divl_cfg = config.get("divl", {})
    batch = batch.to(next(state.critic.parameters()).device)
    if zero_grad:
        state.critic_optimizer.zero_grad(set_to_none=True)
    critic_mask = _critic_execution_mask(batch, config)

    features = state.critic.encode_state(batch)
    q_pred, q_logits, ranking_values = _multimodal_q_predictions(
        state,
        batch,
        features,
        critic_mask,
        critic_cfg,
    )
    with torch.no_grad():
        target_aggregation = str(critic_cfg.get("bootstrap_target", "ensemble_mean"))
        reference_value_samples = int(critic_cfg.get("reference_value_samples", 0))
        lambda_divl_target = float(critic_cfg.get("lambda_divl_target", 1.0))
        reference_value = None
        if state.critic_stage == "head_mc":
            if batch.mc_returns is None:
                raise ValueError("head_mc critic stage requires replay mc_returns")
            y = batch.mc_returns
            v_next = torch.zeros_like(y)
            next_stats = divl_quantile_values(
                F.softmax(state.critic.value_logits_from_features(features), dim=-1),
                state.support,
                alpha_min=float(divl_cfg.get("alpha_min", 0.5)),
                alpha_max=float(divl_cfg.get("alpha_max", 0.8)),
                entropy_temperature=float(divl_cfg.get("entropy_temperature", 1.0)),
                alpha_mode=str(divl_cfg.get("alpha_mode", "linear")),
                use_adaptive_quantile=bool(divl_cfg.get("use_adaptive_quantile", True)),
                interpolate_quantile=bool(divl_cfg.get("interpolate_quantile", True)),
            )
            q_data_target = batch.mc_returns.unsqueeze(0).expand(
                state.critic.ensemble_size,
                batch.batch_size,
            )
            lambda_mc = 1.0
        else:
            next_features = state.target_critic.encode_state(batch, next_observation=True)
            next_logits = state.target_critic.value_logits_from_features(next_features)
            next_stats = divl_quantile_values(
                F.softmax(next_logits, dim=-1),
                state.support,
                alpha_min=float(divl_cfg.get("alpha_min", 0.5)),
                alpha_max=float(divl_cfg.get("alpha_max", 0.8)),
                entropy_temperature=float(divl_cfg.get("entropy_temperature", 1.0)),
                alpha_mode=str(divl_cfg.get("alpha_mode", "linear")),
                use_adaptive_quantile=bool(divl_cfg.get("use_adaptive_quantile", True)),
                interpolate_quantile=bool(divl_cfg.get("interpolate_quantile", True)),
            )
            v_next, _ = aggregate_value_heads(
                next_stats.quantile_value,
                target_aggregation,
                generator=state.target_generator,
            )
            if not 0.0 <= lambda_divl_target <= 1.0:
                raise ValueError("critic.lambda_divl_target must be in [0, 1]")
            if reference_value_samples > 0:
                reference_value = _reference_value_mean(
                    state,
                    batch,
                    num_samples=reference_value_samples,
                    aggregation_mode=target_aggregation,
                )
                v_bootstrap = (
                    lambda_divl_target * v_next
                    + (1.0 - lambda_divl_target) * reference_value
                )
            else:
                v_bootstrap = v_next
            y_td = batch.chunk_returns + batch.discounts * (1.0 - batch.dones) * v_bootstrap
            lambda_mc = float(critic_cfg.get("lambda_mc", 0.0))
            if not 0.0 <= lambda_mc <= 1.0:
                raise ValueError("critic.lambda_mc must be in [0, 1]")
            if lambda_mc > 0.0 and batch.mc_returns is None:
                raise ValueError("critic.lambda_mc requires replay mc_returns; rebuild the offline dataset")
            y = y_td if lambda_mc == 0.0 else (1.0 - lambda_mc) * y_td + lambda_mc * batch.mc_returns
            target_features = state.target_critic.encode_state(batch)
            q_data_target = state.target_critic.q_from_features(
                target_features,
                batch.action_chunks,
                critic_mask,
            )
        z_target = divl_projection_targets(q_data_target, state.support)

    mask = bootstrap_mask(
        state.critic.ensemble_size,
        batch.batch_size,
        float(critic_cfg.get("bootstrap_probability", 0.8)),
        device=q_pred.device,
    )
    q_representation = state.critic.core.q_representation
    q_target = y.unsqueeze(0).expand_as(q_pred)
    if q_representation == "categorical":
        assert q_logits is not None and state.critic.core.q_support is not None
        q_target_distribution = hl_gauss_projection(
            y,
            state.critic.core.q_support,
            sigma_bins=float(critic_cfg.get("q_hl_gauss_sigma_bins", 0.75)),
        )
        q_error = -(
            q_target_distribution.unsqueeze(0)
            * F.log_softmax(q_logits.float(), dim=-1)
        ).sum(dim=-1)
        q_loss_type = "categorical_ce"
        q_entropy = categorical_q_entropy(q_logits.detach())
        support = state.critic.core.q_support
        q_clip_low = (y < support[0]).float().mean()
        q_clip_high = (y > support[-1]).float().mean()
    else:
        q_loss_type = str(critic_cfg.get("q_loss", "huber"))
        if q_loss_type == "mse":
            q_error = (q_pred - q_target).square()
        elif q_loss_type == "huber":
            q_error = F.huber_loss(
                q_pred,
                q_target,
                delta=float(critic_cfg.get("huber_delta", 1.0)),
                reduction="none",
            )
        else:
            raise ValueError(f"unsupported critic.q_loss={q_loss_type!r}")
        q_entropy = q_pred.new_zeros(q_pred.shape)
        q_clip_low = q_pred.new_zeros(())
        q_clip_high = q_pred.new_zeros(())
    q_loss = (q_error * mask.to(q_error.dtype)).sum() / mask.sum().clamp_min(1)
    value_logits = state.critic.value_logits_from_features(features)
    divl_loss = -(z_target * F.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()
    rank_loss, rank_metrics = _ranking_loss_and_metrics(
        ranking_values,
        batch,
        critic_cfg,
    )
    loss = (
        q_loss
        + float(divl_cfg.get("loss_weight", 1.0)) * divl_loss
        + float(critic_cfg.get("rank_loss_weight", 0.1)) * rank_loss
    )
    (loss * float(loss_scale)).backward()
    critic_grad_norm = 0.0
    critic_lr_scale = 1.0
    if optimizer_step:
        critic_grad_norm = float(grad_norm(state.critic.parameters()))
        torch.nn.utils.clip_grad_norm_(
            state.critic.parameters(),
            float(critic_cfg.get("max_grad_norm", 10.0)),
        )
        critic_lr_scale = _apply_critic_lr_schedule(state, config)
        state.critic_optimizer.step()

    target_update_period = int(critic_cfg.get("target_update_period", 1))
    target_updated = optimizer_step and (state.step + 1) % target_update_period == 0
    if target_updated:
        soft_update(state.target_critic, state.critic, float(critic_cfg.get("target_tau", 0.005)))
    if optimizer_step:
        state.step += 1
        state.critic_stage_step += 1
    value_probs = F.softmax(value_logits.detach(), dim=-1)
    value_quantiles = next_stats.quantile_value
    value_quantile_min = value_quantiles.min(dim=0).values
    value_quantile_mean = value_quantiles.mean(dim=0)
    value_quantile_max = value_quantiles.max(dim=0).values
    metrics = {
        "critic_loss": float(loss.detach().item()),
        "q_loss": float(q_loss.detach().item()),
        "q_loss_is_mse": float(q_representation == "scalar" and q_loss_type == "mse"),
        "q_representation_is_categorical": float(q_representation == "categorical"),
        "divl_loss": float(divl_loss.detach().item()),
        "divl_enabled": 1.0,
        "target_mean": float(y.mean().item()),
        "target_std": float(y.std(unbiased=False).item()),
        "td_error_abs_mean": float((q_pred.detach() - y.unsqueeze(0)).abs().mean().item()),
        "q_mean": float(q_pred.detach().mean().item()),
        "q_std": float(q_pred.detach().std(unbiased=False).item()),
        "v_divl_mean": float(v_next.mean().item()),
        "v_head_min_mean": float(value_quantile_min.mean().item()),
        "v_head_mean_mean": float(value_quantile_mean.mean().item()),
        "v_head_max_mean": float(value_quantile_max.mean().item()),
        "v_head_spread_mean": float(
            (value_quantile_max - value_quantile_min).mean().item()
        ),
        "critic_grad_norm": critic_grad_norm,
        "critic_lr_scale": critic_lr_scale,
        "critic_lr": float(state.critic_optimizer.param_groups[0]["lr"]),
        "bootstrap_active": float(mask.float().mean().item()),
        "divl_entropy": float(next_stats.entropy.mean().item()),
        "adaptive_alpha": float(next_stats.alpha.mean().item()),
        "bootstrap_target_is_min": float(target_aggregation in {"ensemble_min", "min"}),
        "bootstrap_target_is_subsample_min": float(target_aggregation == "subsample_min"),
        "lambda_mc": lambda_mc,
        "categorical_saturation": float(
            ((value_probs[..., 0] + value_probs[..., -1]) > 0.5).float().mean().item()
        ),
        "v_quantile": float(next_stats.quantile_value.mean().item()),
        "target_updated": float(target_updated),
        "critic_stage_head_mc": float(state.critic_stage == "head_mc"),
        "critic_stage_head_td": float(state.critic_stage == "head_td"),
        "critic_stage_gemma_lora_td": float(state.critic_stage == "gemma_lora_td"),
        "critic_stage_full_td": float(state.critic_stage == "full_td"),
        "reference_value_samples": float(reference_value_samples),
        "critic/q_ce_loss": float(q_loss.detach().item())
        if q_representation == "categorical"
        else 0.0,
        "critic/q_decoded_mean": float(q_pred.detach().mean().item()),
        "critic/q_decoded_std": float(q_pred.detach().std(unbiased=False).item()),
        "critic/q_target_mean": float(y.mean().item()),
        "critic/q_target_std": float(y.std(unbiased=False).item()),
        "critic/q_target_clip_low_fraction": float(q_clip_low.item()),
        "critic/q_target_clip_high_fraction": float(q_clip_high.item()),
        "critic/q_entropy_mean": float(q_entropy.mean().item()),
    }
    metrics.update(rank_metrics)
    if reference_value is not None:
        metrics["reference_value_mean"] = float(reference_value.mean().item())
    for member, member_loss in enumerate(q_error.detach().mean(dim=1)):
        metrics[f"q_loss_member_{member}"] = float(member_loss.item())
    return metrics


def critic_update(state: OGPOTrainState, batch: ChunkBatch, config: dict[str, Any]) -> dict[str, float]:
    if isinstance(state.critic, MultiHeadUdivlCritic):
        return _multimodal_critic_update(state, batch, config)
    if isinstance(state.critic, MultiHeadScalarQCritic):
        return _multimodal_scalar_q_update(state, batch, config)
    critic_cfg = config.get("critic", {})
    divl_cfg = config.get("divl", {})
    divl_enabled = bool(divl_cfg.get("enabled", True))
    batch = batch.to(next(state.critic.parameters()).device)
    state.critic_optimizer.zero_grad(set_to_none=True)

    critic_mask = _critic_execution_mask(batch, config)
    q_pred = state.critic(batch.observations, batch.action_chunks, critic_mask)
    with torch.no_grad():
        target_aggregation = str(critic_cfg.get("bootstrap_target", "ensemble_mean"))
        reference_value_samples = int(critic_cfg.get("reference_value_samples", 0))
        lambda_divl_target = float(critic_cfg.get("lambda_divl_target", 1.0))
        reference_value = None
        if divl_enabled:
            next_probs = state.target_divl(batch.next_observations)
            next_stats = divl_quantile_values(
                next_probs,
                state.support,
                alpha_min=float(divl_cfg.get("alpha_min", 0.5)),
                alpha_max=float(divl_cfg.get("alpha_max", 0.8)),
                entropy_temperature=float(divl_cfg.get("entropy_temperature", 1.0)),
                alpha_mode=str(divl_cfg.get("alpha_mode", "linear")),
                use_adaptive_quantile=bool(divl_cfg.get("use_adaptive_quantile", True)),
                interpolate_quantile=bool(divl_cfg.get("interpolate_quantile", True)),
            )
            if target_aggregation == "ensemble_mean":
                v_next = next_stats.quantile_value.mean(dim=0)
            elif target_aggregation == "ensemble_min":
                v_next = next_stats.quantile_value.min(dim=0).values
            else:
                raise ValueError(f"unsupported critic.bootstrap_target={target_aggregation!r}")
            if not 0.0 <= lambda_divl_target <= 1.0:
                raise ValueError("critic.lambda_divl_target must be in [0, 1]")
            if reference_value_samples > 0:
                reference_value = _reference_value_mean(
                    state,
                    batch,
                    num_samples=reference_value_samples,
                )
                v_bootstrap = lambda_divl_target * v_next + (1.0 - lambda_divl_target) * reference_value
            else:
                v_bootstrap = v_next
            y_td = batch.chunk_returns + batch.discounts * (1.0 - batch.dones) * v_bootstrap
            lambda_mc = float(critic_cfg.get("lambda_mc", 0.0))
            if not 0.0 <= lambda_mc <= 1.0:
                raise ValueError("critic.lambda_mc must be in [0, 1]")
            if lambda_mc > 0.0 and batch.mc_returns is None:
                raise ValueError("critic.lambda_mc requires replay mc_returns; rebuild the offline dataset")
            y = y_td if lambda_mc == 0.0 else (1.0 - lambda_mc) * y_td + lambda_mc * batch.mc_returns
            q_data_target = state.target_critic(batch.observations, batch.action_chunks, critic_mask)
            z_target = divl_projection_targets(q_data_target, state.support)
        else:
            if batch.mc_returns is None:
                raise ValueError("divl.enabled=false requires replay mc_returns")
            y = batch.mc_returns
            lambda_mc = 1.0
            v_next = torch.zeros_like(y)
            next_stats = None
            z_target = None

    mask = bootstrap_mask(
        state.critic.ensemble_size,
        batch.batch_size,
        float(critic_cfg.get("bootstrap_probability", 0.8)),
        device=q_pred.device,
    )
    q_error = F.huber_loss(
        q_pred,
        y.unsqueeze(0).expand_as(q_pred),
        delta=float(critic_cfg.get("huber_delta", 1.0)),
        reduction="none",
    )
    q_loss = (q_error * mask.to(q_error.dtype)).sum() / mask.sum().clamp_min(1)
    if divl_enabled:
        assert z_target is not None
        divl_logits = state.divl.logits(batch.observations)
        divl_loss = -(z_target * F.log_softmax(divl_logits, dim=-1)).sum(dim=-1).mean()
    else:
        divl_logits = None
        divl_loss = q_loss.new_zeros(())
    loss = q_loss + float(divl_cfg.get("loss_weight", 1.0)) * divl_loss
    loss.backward()
    critic_params = list(state.critic.parameters()) + list(state.divl.parameters())
    critic_grad_norm = float(grad_norm(critic_params))
    torch.nn.utils.clip_grad_norm_(critic_params, float(critic_cfg.get("max_grad_norm", 10.0)))
    state.critic_optimizer.step()
    target_update_period = int(critic_cfg.get("target_update_period", 1))
    if target_update_period <= 0:
        raise ValueError("critic.target_update_period must be positive")
    target_updated = (state.step + 1) % target_update_period == 0
    if target_updated:
        soft_update(state.target_critic, state.critic, float(critic_cfg.get("target_tau", 0.01)))
        soft_update(state.target_divl, state.divl, float(critic_cfg.get("target_tau", 0.01)))
    state.step += 1
    q_member_denominator = mask.sum(dim=1).clamp_min(1).to(q_error.dtype)
    q_member_losses = (q_error * mask.to(q_error.dtype)).sum(dim=1) / q_member_denominator
    divl_probs = F.softmax(divl_logits.detach(), dim=-1) if divl_logits is not None else None
    metrics = {
        "critic_loss": float(loss.detach().item()),
        "q_loss": float(q_loss.detach().item()),
        "divl_loss": float(divl_loss.detach().item()),
        "divl_enabled": float(divl_enabled),
        "target_mean": float(y.mean().item()),
        "target_std": float(y.std(unbiased=False).item()),
        "td_error_abs_mean": float((q_pred.detach() - y.unsqueeze(0)).abs().mean().item()),
        "q_mean": float(q_pred.detach().mean().item()),
        "q_std": float(q_pred.detach().std(unbiased=False).item()),
        "v_divl_mean": float(v_next.mean().item()),
        "critic_grad_norm": critic_grad_norm,
        "bootstrap_active": float(mask.float().mean().item()),
        "divl_entropy": float(next_stats.entropy.mean().item()) if next_stats is not None else 0.0,
        "adaptive_alpha": float(next_stats.alpha.mean().item()) if next_stats is not None else 0.0,
        "bootstrap_target_is_min": float(target_aggregation == "ensemble_min"),
        "lambda_mc": lambda_mc,
        "categorical_saturation": float(
            ((divl_probs[..., 0] + divl_probs[..., -1]) > 0.5).float().mean().item()
        ) if divl_probs is not None else 0.0,
        "v_quantile": float(next_stats.quantile_value.mean().item()) if next_stats is not None else 0.0,
        "target_updated": float(target_updated),
    }
    for member, member_loss in enumerate(q_member_losses):
        metrics[f"q_loss_member_{member}"] = float(member_loss.item())
    if reference_value is not None:
        metrics["reference_value_mean"] = float(reference_value.mean().item())
        metrics["lambda_divl_target"] = lambda_divl_target
    return metrics


def accumulated_critic_update(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    microbatch_size: int,
) -> dict[str, float]:
    """Run one multimodal optimizer step over a larger effective batch."""
    if not isinstance(state.critic, (MultiHeadUdivlCritic, MultiHeadScalarQCritic)):
        if microbatch_size < batch.batch_size:
            raise ValueError("critic gradient accumulation is only implemented for multimodal critics")
        return critic_update(state, batch, config)
    microbatch_size = int(microbatch_size)
    if microbatch_size <= 0:
        raise ValueError("critic microbatch_size must be positive")
    starts = list(range(0, batch.batch_size, microbatch_size))
    total_size = float(batch.batch_size)
    combined: dict[str, float] = {}
    for index, start in enumerate(starts):
        stop = min(start + microbatch_size, batch.batch_size)
        indices = torch.arange(start, stop)
        weight = (stop - start) / total_size
        update_fn = (
            _multimodal_scalar_q_update
            if isinstance(state.critic, MultiHeadScalarQCritic)
            else _multimodal_critic_update
        )
        metrics = update_fn(
            state,
            batch.index_select(indices),
            config,
            zero_grad=index == 0,
            optimizer_step=index == len(starts) - 1,
            loss_scale=weight,
        )
        for key, value in metrics.items():
            if key == "critic/rank_pair_count":
                combined[key] = combined.get(key, 0.0) + float(value)
            else:
                combined[key] = combined.get(key, 0.0) + weight * float(value)
    combined["critic_grad_norm"] = float(metrics["critic_grad_norm"])
    combined["target_updated"] = float(metrics["target_updated"])
    combined["effective_batch_size"] = float(batch.batch_size)
    combined["microbatch_size"] = float(microbatch_size)
    combined["gradient_accumulation_steps"] = float(len(starts))
    return combined


@torch.no_grad()
def conservative_advantages_for_candidates(
    state: OGPOTrainState,
    observations: torch.Tensor,
    candidate_flat_actions: torch.Tensor,
    batch: ChunkBatch,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    uncertainty_cfg = config.get("uncertainty", {})
    divl_cfg = config.get("divl", {})
    divl_enabled = bool(divl_cfg.get("enabled", True))
    batch_size, group_size, action_flat = candidate_flat_actions.shape
    chunks = candidate_flat_actions.reshape(batch_size, group_size, batch.generated_horizon, batch.action_dim)
    flat_obs = observations[:, None, :].expand(batch_size, group_size, observations.shape[-1]).reshape(
        batch_size * group_size, -1
    )
    flat_chunks = chunks.reshape(batch_size * group_size, batch.generated_horizon, batch.action_dim)
    critic_mask = _critic_execution_mask(batch, config)
    flat_masks = critic_mask[:, None, :].expand(batch_size, group_size, batch.generated_horizon).reshape(
        batch_size * group_size, batch.generated_horizon
    )
    if isinstance(state.critic, (MultiHeadUdivlCritic, MultiHeadScalarQCritic)):
        features = state.critic.encode_state(batch)
        grouped_features = StateFeatures(
            readout=features.readout.repeat_interleave(group_size, dim=0)
        )
        q_flat = state.critic.q_from_features(grouped_features, flat_chunks, flat_masks)
        behavior_q_members = state.critic.q_from_features(features, batch.action_chunks, critic_mask)
        probs = (
            F.softmax(state.critic.value_logits_from_features(features), dim=-1)
            if isinstance(state.critic, MultiHeadUdivlCritic)
            else None
        )
    else:
        q_flat = state.critic(flat_obs, flat_chunks, flat_masks)
        behavior_q_members = state.critic(observations, batch.action_chunks, critic_mask)
        probs = state.divl(observations) if divl_enabled else None
    q_values = q_flat.reshape(state.critic.ensemble_size, batch_size, group_size)
    _, q_std = ensemble_mean_std(q_values)
    if divl_enabled:
        assert probs is not None
        divl_stats = divl_quantile_values(
            probs,
            state.support,
            alpha_min=float(divl_cfg.get("alpha_min", 0.5)),
            alpha_max=float(divl_cfg.get("alpha_max", 0.8)),
            entropy_temperature=float(divl_cfg.get("entropy_temperature", 1.0)),
            alpha_mode=str(divl_cfg.get("alpha_mode", "linear")),
            use_adaptive_quantile=bool(divl_cfg.get("use_adaptive_quantile", True)),
            interpolate_quantile=bool(divl_cfg.get("interpolate_quantile", True)),
        )
        value_baselines = divl_stats.quantile_value
        entropy_state = divl_stats.entropy.mean(dim=0)
    else:
        value_baselines = behavior_q_members
        entropy_state = observations.new_zeros(batch_size)
    advantage_mode = str(config.get("actor", {}).get("advantage_mode", "sign_consensus"))
    if advantage_mode == "sign_consensus":
        cons, stats = sign_consensus_advantage(
            q_values,
            value_baselines,
            positive_margin=float(uncertainty_cfg.get("positive_margin", 0.0)),
            negative_margin=float(uncertainty_cfg.get("negative_margin", 0.0)),
        )
        positive_ratio = stats.positive_consensus_ratio
        negative_ratio = stats.negative_consensus_ratio
        zero_ratio = stats.zero_ratio
        sign_agreement_ratio = stats.sign_agreement_ratio
    elif advantage_mode == "lcb":
        calibrated_scale = state.conformal_scale if bool(uncertainty_cfg.get("use_conformal", False)) else 1.0
        cons = lcb_advantage(
            q_values,
            value_baselines,
            kappa=float(uncertainty_cfg.get("lcb_kappa", 1.0)),
            calibrated_scale=calibrated_scale,
        )
        positive_ratio = float((cons > 0).float().mean().item())
        negative_ratio = float((cons < 0).float().mean().item())
        zero_ratio = float((cons == 0).float().mean().item())
        sign_agreement_ratio = 1.0 - zero_ratio
    elif advantage_mode == "group_normalization":
        cons = group_normalized_advantage(q_values)
        positive_ratio = float((cons > 0).float().mean().item())
        negative_ratio = float((cons < 0).float().mean().item())
        zero_ratio = float((cons == 0).float().mean().item())
        sign_agreement_ratio = 1.0 - zero_ratio
    elif advantage_mode == "group_mean":
        q_mean = q_values.mean(dim=0)
        cons = q_mean - q_mean.mean(dim=-1, keepdim=True)
        positive_ratio = float((cons > 0).float().mean().item())
        negative_ratio = float((cons < 0).float().mean().item())
        zero_ratio = float((cons == 0).float().mean().item())
        sign_agreement_ratio = 1.0 - zero_ratio
    elif advantage_mode == "scalar_q":
        behavior_q = behavior_q_members.mean(dim=0)
        cons = q_values.mean(dim=0) - behavior_q.unsqueeze(-1)
        positive_ratio = float((cons > 0).float().mean().item())
        negative_ratio = float((cons < 0).float().mean().item())
        zero_ratio = float((cons == 0).float().mean().item())
        sign_agreement_ratio = 1.0 - zero_ratio
    else:
        raise ValueError(f"unsupported actor.advantage_mode={advantage_mode!r}")
    actor_cfg = config.get("actor", {})
    advantage_clip = float(actor_cfg.get("advantage_clip", 5.0))
    if advantage_mode == "group_mean":
        normalized = cons
        lambda_abs = 1.0
    elif advantage_mode == "group_normalization":
        normalized = cons.clamp(-advantage_clip, advantage_clip)
        lambda_abs = 0.0
    else:
        state.running_mad.update(cons, ignore_zero=True)
        normalized_abs = state.running_mad.normalize(cons, advantage_clip)
        lambda_abs = scheduled_lambda_abs(
            state.step,
            start=float(actor_cfg.get("lambda_abs_start", 1.0)),
            end=float(actor_cfg.get("lambda_abs_end", 1.0)),
            warmup_steps=int(actor_cfg.get("lambda_abs_warmup_steps", 0)),
        )
        if lambda_abs < 1.0:
            group_advantage = group_normalized_advantage(q_values).clamp(-advantage_clip, advantage_clip)
            normalized = lambda_abs * normalized_abs + (1.0 - lambda_abs) * group_advantage
        else:
            normalized = normalized_abs
    state_weights = state_entropy_weight(entropy_state, float(uncertainty_cfg.get("entropy_scale", 0.0))).unsqueeze(-1)
    final_adv = normalized * state_weights
    consensus_per_state = (cons != 0).to(final_adv.dtype).mean(dim=1)
    entropy_skip = entropy_state > float(uncertainty_cfg.get("entropy_skip_threshold", 1.1))
    consensus_skip = consensus_per_state < float(uncertainty_cfg.get("consensus_skip_threshold", -1.0))
    state_skip = entropy_skip | consensus_skip
    final_adv = torch.where(state_skip.unsqueeze(-1), torch.zeros_like(final_adv), final_adv)
    support_distance = torch.zeros_like(final_adv)
    support_weights = torch.ones_like(final_adv)
    if bool(uncertainty_cfg.get("use_support_weight", False)):
        behavior_flat = batch.action_chunks.reshape(batch_size, -1)[:, None, :]
        support_distance = (candidate_flat_actions - behavior_flat).pow(2).mean(dim=-1).sqrt()
        threshold = uncertainty_cfg.get("support_threshold")
        support_weights = support_weight(
            q_std * (state.conformal_scale if bool(uncertainty_cfg.get("use_conformal", False)) else 1.0),
            support_distance,
            lambda_epi=float(uncertainty_cfg.get("lambda_epi", 0.0)),
            lambda_support=float(uncertainty_cfg.get("lambda_support", 0.0)),
            support_threshold=float(threshold) if threshold is not None else None,
        )
        final_adv = final_adv * support_weights
    return final_adv, {
        "advantage_mode": advantage_mode,
        "positive_consensus_ratio": positive_ratio,
        "negative_consensus_ratio": negative_ratio,
        "zero_disagreement_ratio": zero_ratio,
        "sign_agreement_ratio": sign_agreement_ratio,
        "advantage_mad": state.running_mad.value,
        "lambda_abs": lambda_abs,
        "advantage_mean": float(final_adv.mean().item()),
        "advantage_std": float(final_adv.std(unbiased=False).item()),
        "advantage_clip_fraction": float(
            (normalized.abs() >= advantage_clip).float().mean().item()
        ),
        "conservative_advantage_abs_mean": float(cons.abs().mean().item()),
        "state_entropy": float(entropy_state.mean().item()),
        "state_entropy_weight": float(state_weights.mean().item()),
        "state_skip_fraction": float(state_skip.float().mean().item()),
        "entropy_skip_fraction": float(entropy_skip.float().mean().item()),
        "consensus_skip_fraction": float(consensus_skip.float().mean().item()),
        "candidate_ensemble_disagreement": float(q_std.mean().item()),
        "support_distance_mean": float(support_distance.mean().item()),
        "support_weight_mean": float(support_weights.mean().item()),
    }


def _normalized_state_entropy(state: OGPOTrainState, batch: ChunkBatch) -> torch.Tensor:
    if isinstance(state.critic, MultiHeadUdivlCritic):
        features = state.critic.encode_state(batch)
        probs = F.softmax(state.critic.value_logits_from_features(features), dim=-1)
    elif isinstance(state.critic, MultiHeadScalarQCritic):
        return batch.observations.new_zeros(batch.batch_size)
    else:
        assert state.divl is not None
        probs = state.divl(batch.observations)
    entropy = probs.clamp_min(1e-8).mul(probs.clamp_min(1e-8).log()).sum(-1).neg()
    return entropy / torch.log(torch.tensor(probs.shape[-1], dtype=probs.dtype, device=probs.device))


def _success_subset(batch: ChunkBatch) -> ChunkBatch | None:
    indices = torch.nonzero(batch.successes.bool(), as_tuple=False).flatten()
    if indices.numel() == 0:
        return None
    return batch.index_select(indices)


def _actor_regularization_loss(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    fm_batch: ChunkBatch | None = None,
    success_batch: ChunkBatch | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    regularization_cfg = config.get("regularization", {})
    source = fm_batch if fm_batch is not None else batch
    source = source.to(next(state.policy.parameters()).device)
    source_condition = _policy_condition(state.policy, source)
    fm = flow_matching_anchor_loss(
        state.policy,
        source_condition,
        state.policy.action_chunks_to_flow(source).reshape(source.batch_size, -1),
    )
    loss = float(regularization_cfg.get("lambda_fm", 0.1)) * fm.loss
    zero = fm.loss.detach().new_tensor(0.0)
    metrics = {
        "fm_anchor_loss": fm.diagnostics["fm_anchor_loss"],
        "success_buffer_loss": 0.0,
        "action_smoothness": 0.0,
    }

    lambda_success = float(regularization_cfg.get("lambda_success", 0.0))
    success_source = success_batch if success_batch is not None else _success_subset(batch)
    if lambda_success > 0.0 and success_source is not None:
        success_source = success_source.to(next(state.policy.parameters()).device)
        success_condition = _policy_condition(state.policy, success_source)
        success = success_buffer_loss(
            state.policy,
            success_condition,
            state.policy.action_chunks_to_flow(success_source).reshape(success_source.batch_size, -1),
        )
        loss = loss + lambda_success * success.loss
        metrics["success_buffer_loss"] = success.diagnostics["success_buffer_loss"]
    else:
        loss = loss + zero

    lambda_smooth = float(regularization_cfg.get("lambda_smooth", 0.0))
    if lambda_smooth > 0.0:
        smooth_condition = _policy_condition(state.policy, batch)
        smooth_rollout = state.policy.rollout(smooth_condition, group_size=1)
        smooth_endpoint = state.policy.flat_actions_to_environment(smooth_rollout.endpoint, smooth_condition)
        smooth_chunks = smooth_endpoint.reshape(batch.batch_size, batch.generated_horizon, batch.action_dim)
        gripper_mask_value = regularization_cfg.get("gripper_mask")
        gripper_mask = None
        if gripper_mask_value is not None:
            gripper_mask = torch.as_tensor(gripper_mask_value, dtype=torch.bool, device=smooth_chunks.device)
        smooth = action_smoothness_loss(
            smooth_chunks,
            gripper_mask=gripper_mask,
            eta=float(regularization_cfg.get("smooth_acceleration_eta", 0.1)),
        )
        loss = loss + lambda_smooth * smooth
        metrics["action_smoothness"] = float(smooth.detach().item())
    else:
        loss = loss + zero

    return loss, metrics


@torch.no_grad()
def _policy_l2_lag(policy: torch.nn.Module, old_policy: torch.nn.Module) -> float:
    total = 0.0
    count = 0
    for param, old_param in zip(policy.parameters(), old_policy.parameters(), strict=True):
        if not param.requires_grad:
            continue
        diff = param.detach() - old_param.detach()
        total += float(diff.pow(2).sum().item())
        count += diff.numel()
    return (total / max(1, count)) ** 0.5


def _select_flash_steps(
    flow_cfg: dict[str, Any],
    *,
    batch_size: int,
    num_steps: int,
    device: torch.device,
    seed: int | None = None,
) -> torch.Tensor:
    distribution = str(flow_cfg.get("selected_timestep_distribution", "fixed"))
    if distribution == "uniform":
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        return torch.randint(
            num_steps,
            (batch_size,),
            generator=generator,
        ).to(device)
    if distribution == "stratified_uniform":
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        permutations = [
            torch.randperm(num_steps, generator=generator)
            for _ in range(math.ceil(batch_size / num_steps))
        ]
        return torch.cat(permutations)[:batch_size].to(device)
    if distribution == "fixed":
        selected = int(flow_cfg.get("selected_timestep", num_steps // 2))
        if selected < 0 or selected >= num_steps:
            raise ValueError("flow.selected_timestep must be in [0, flow.num_steps)")
        return torch.full((batch_size,), selected, dtype=torch.long, device=device)
    raise ValueError(f"unsupported selected_timestep_distribution={distribution!r}")


def _flash_rectification_weight(
    state: OGPOTrainState,
    flow_cfg: dict[str, Any],
    *,
    timestep: torch.Tensor,
    selected_steps: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    mode = str(flow_cfg.get("temporal_rectification_mode", "analytic"))
    if mode == "none":
        return torch.ones(timestep.shape[0], dtype=timestep.dtype, device=timestep.device)
    if mode == "analytic":
        policy_variance = float(flow_cfg.get("stochastic_variance", 0.04))
        policy_log_std = getattr(state.old_policy, "log_std", None)
        if policy_log_std is not None:
            policy_variance = float(
                policy_log_std.detach().float().exp().square().mean().item()
            )
        return analytic_rectification(
            timestep,
            stochastic_variance=policy_variance,
            sde_mode=str(flow_cfg.get("sde_mode", "gaussian_adapter")),
            clip_min=float(flow_cfg.get("rectification_clip_min", 0.25)),
            clip_max=float(flow_cfg.get("rectification_clip_max", 4.0)),
        )
    if mode == "empirical_ema":
        selected_g = selected_steps.repeat_interleave(group_size)
        weights = [
            state.rectifier.weight(int(step.item()), device=timestep.device)
            for step in selected_g
        ]
        return torch.stack(weights).to(dtype=timestep.dtype, device=timestep.device)
    raise ValueError(f"unsupported temporal_rectification_mode={mode!r}")


def awr_actor_update(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
) -> dict[str, float]:
    """Scalar-Q AWR baseline implemented as weighted replay flow matching."""
    actor_cfg = config.get("actor", {})
    if state.critic.ensemble_size != 1:
        raise ValueError("AWR baseline requires critic.ensemble_size=1")
    batch = batch.to(next(state.policy.parameters()).device)
    state.critic_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        q_data = state.critic(
            batch.observations,
            batch.action_chunks,
            _critic_execution_mask(batch, config),
        ).squeeze(0)
        advantage = q_data - q_data.mean()
        temperature = float(actor_cfg.get("awr_temperature", 1.0))
        max_weight = float(actor_cfg.get("awr_max_weight", 20.0))
        if temperature <= 0.0 or max_weight < 1.0:
            raise ValueError("AWR requires positive temperature and awr_max_weight >= 1")
        log_weight = (advantage / temperature).clamp(min=-20.0, max=math.log(max_weight))
        weights = log_weight.exp()

    state.actor_optimizer.zero_grad(set_to_none=True)
    condition = _policy_condition(state.policy, batch)
    flow_actions = state.policy.action_chunks_to_flow(batch).reshape(batch.batch_size, -1)
    weighted_fm = weighted_flow_matching_loss(state.policy, condition, flow_actions, weights)
    weighted_fm.loss.backward()
    assert_no_gradients(state.critic, "critic")
    assert_no_gradients(state.reference_policy, "reference_policy")
    actor_grad_norm = float(grad_norm(state.policy.parameters()))
    torch.nn.utils.clip_grad_norm_(state.policy.parameters(), float(actor_cfg.get("max_grad_norm", 1.0)))
    state.actor_optimizer.step()
    return {
        "actor_loss": float(weighted_fm.loss.detach().item()),
        "awr_weighted_fm_loss": weighted_fm.diagnostics["weighted_fm_loss"],
        "awr_weight_mean": weighted_fm.diagnostics["awr_weight_mean"],
        "awr_weight_max": weighted_fm.diagnostics["awr_weight_max"],
        "awr_advantage_mean": float(advantage.mean().item()),
        "awr_advantage_std": float(advantage.std(unbiased=False).item()),
        "actor_grad_norm": actor_grad_norm,
        "old_policy_lag": _policy_l2_lag(state.policy, state.old_policy),
    }


def full_actor_update(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    fm_batch: ChunkBatch | None = None,
    success_batch: ChunkBatch | None = None,
) -> dict[str, float]:
    actor_cfg = config.get("actor", {})
    regularization_cfg = config.get("regularization", {})
    uncertainty_cfg = config.get("uncertainty", {})
    batch = batch.to(next(state.policy.parameters()).device)
    group_size = int(actor_cfg.get("group_size", 4))
    actor_epochs = max(1, int(actor_cfg.get("actor_epochs_per_rollout", 1)))
    state.critic_optimizer.zero_grad(set_to_none=True)
    condition = _policy_condition(state.policy, batch)
    condition_g = state.policy.repeat_condition(condition, group_size)

    if _using_jax_actor(state.policy):
        assert isinstance(state.policy, PI05JaxFlowPolicy)
        assert isinstance(state.old_policy, PI05JaxFlowPolicy)
        assert isinstance(state.reference_policy, PI05JaxFlowPolicy)
        flow_spec = OpenPIJaxFlowSpec(state.policy.num_steps)
        jax_observation = policy_observation_to_jax(condition.observation)
        old_actor = nnx.merge(state.old_policy.actor_graphdef, state.old_policy.actor_state)
        reference_actor = nnx.merge(state.reference_policy.actor_graphdef, state.reference_policy.actor_state)

        with torch.no_grad():
            rng = jax.random.PRNGKey(int(state.step) + 31)
            old_rollout_j = jax_rollout(
                actor=old_actor,
                flow_spec=flow_spec,
                observation=jax_observation,
                group_size=group_size,
                rng=rng,
                sde_mode=state.old_policy.sde_mode,
            )
            old_endpoint_torch = torch.as_tensor(np.asarray(old_rollout_j.endpoint).copy(), device=batch.observations.device, dtype=batch.action_chunks.dtype)
            environment_endpoint = state.old_policy.flat_actions_to_environment(old_endpoint_torch, condition_g)
            endpoint = environment_endpoint.reshape(batch.batch_size, group_size, -1)
            advantages, adv_diag = conservative_advantages_for_candidates(
                state, batch.observations, endpoint, batch, config
            )
            entropy_norm = _normalized_state_entropy(state, batch).mean(dim=0)
            eps_clip = actor_clip_for_uncertainty(
                entropy_norm,
                actor_cfg,
                uncertainty_cfg,
            ).repeat_interleave(group_size)
        jax_regularization = _prepare_jax_regularization(
            state,
            batch,
            config,
            fm_batch=fm_batch,
            success_batch=success_batch,
        )
        beta_base = float(regularization_cfg.get("beta_kl", 0.01))
        adapt_kl_beta = bool(uncertainty_cfg.get("adapt_kl_beta", False))
        uncertainty_scale = float(regularization_cfg.get("kl_uncertainty_scale", 1.0))
        normalize_logprob_by_action_dim = bool(
            actor_cfg.get("normalize_logprob_by_action_dim", False)
        )
        logprob_normalizer = float(state.policy.action_dim) if normalize_logprob_by_action_dim else 1.0
        old_states = old_rollout_j.states
        old_next_states = old_rollout_j.next_states
        old_timesteps = old_rollout_j.timesteps
        advantages_j = jnp.asarray(advantages.reshape(-1).detach().cpu().numpy())
        eps_clip_j = jnp.asarray(eps_clip.detach().cpu().numpy())
        entropy_norm_j = jnp.asarray(entropy_norm.detach().cpu().numpy())
        full_ratio_mode = str(actor_cfg.get("full_ratio_mode", "per_transition"))
        normalize_logprob_by_denoising_steps = bool(
            actor_cfg.get("normalize_logprob_by_denoising_steps", False)
        )
        chain_logprob_normalizer = logprob_normalizer
        if full_ratio_mode == "ais_joint" and normalize_logprob_by_denoising_steps:
            chain_logprob_normalizer *= float(state.policy.num_steps)
        gradient_microbatch_size = int(
            actor_cfg.get("gradient_microbatch_size", advantages_j.shape[0])
        )
        if gradient_microbatch_size <= 0:
            raise ValueError("actor.gradient_microbatch_size must be positive")
        loss_total = 0.0
        ppo_total = 0.0
        kl_total = 0.0
        grad_total = 0.0

        if gradient_microbatch_size < advantages_j.shape[0]:
            if beta_base != 0.0 or adapt_kl_beta:
                raise ValueError(
                    "full-chain gradient microbatching currently requires beta_kl=0 "
                    "and uncertainty.adapt_kl_beta=false"
                )
            if (
                float(jax_regularization["lambda_fm"]) != 0.0
                or float(jax_regularization["lambda_success"]) != 0.0
            ):
                raise ValueError(
                    "full-chain gradient microbatching currently requires "
                    "lambda_fm=lambda_success=0"
                )
            jax_observation_g = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, group_size, axis=0) if x is not None else None,
                jax_observation,
            )

            if bool(actor_cfg.get("full_chain_streaming_backward", False)):
                policy_graphdef = state.policy.actor_graphdef
                old_graphdef = state.old_policy.actor_graphdef

                @jax.jit
                def _stream_current_log_prob(
                    actor_state,
                    x_prev,
                    x_t,
                    observation,
                    timestep,
                ):
                    actor = nnx.merge(policy_graphdef, actor_state)
                    return jax_transition_log_prob(
                        actor=actor,
                        flow_spec=flow_spec,
                        x_prev=x_prev,
                        x_t=x_t,
                        observation=observation,
                        timestep=timestep,
                        sde_mode=state.policy.sde_mode,
                    ) / chain_logprob_normalizer

                @jax.jit
                def _stream_old_log_prob(
                    old_actor_state,
                    x_prev,
                    x_t,
                    observation,
                    timestep,
                ):
                    actor = nnx.merge(old_graphdef, old_actor_state)
                    return jax_transition_log_prob(
                        actor=actor,
                        flow_spec=flow_spec,
                        x_prev=x_prev,
                        x_t=x_t,
                        observation=observation,
                        timestep=timestep,
                        sde_mode=state.old_policy.sde_mode,
                    ) / chain_logprob_normalizer

                @jax.jit
                @jax.value_and_grad
                def _stream_weighted_log_prob(
                    actor_state,
                    x_prev,
                    x_t,
                    observation,
                    timestep,
                    coefficient,
                ):
                    log_prob = _stream_current_log_prob(
                        actor_state,
                        x_prev,
                        x_t,
                        observation,
                        timestep,
                    )
                    return jnp.mean(jax.lax.stop_gradient(coefficient) * log_prob)

                torch.cuda.empty_cache()
                ratio = np.ones(int(advantages_j.shape[0]), dtype=np.float32)
                clipped_ratio = ratio.copy()
                for _ in range(actor_epochs):
                    host_grads = None
                    epoch_loss = 0.0
                    ratio_parts = []
                    clipped_parts = []
                    total_candidates = int(advantages_j.shape[0])
                    for start in range(0, total_candidates, gradient_microbatch_size):
                        stop = min(start + gradient_microbatch_size, total_candidates)
                        sample_weight = (stop - start) / total_candidates
                        observation_mb = _slice_jax_batch(jax_observation_g, start, stop)
                        new_step_log_probs = []
                        old_step_log_probs = []
                        for flow_step in range(state.policy.num_steps):
                            args = (
                                old_next_states[start:stop, flow_step],
                                old_states[start:stop, flow_step],
                                observation_mb,
                                old_timesteps[start:stop, flow_step],
                            )
                            new_step_log_probs.append(
                                np.asarray(
                                    _stream_current_log_prob(
                                        state.policy.actor_state,
                                        *args,
                                    )
                                )
                            )
                            old_step_log_probs.append(
                                np.asarray(
                                    _stream_old_log_prob(
                                        state.old_policy.actor_state,
                                        *args,
                                    )
                                )
                            )
                            jax.effects_barrier()

                        log_ratio = np.sum(
                            np.stack(new_step_log_probs, axis=1)
                            - np.stack(old_step_log_probs, axis=1),
                            axis=1,
                        )
                        ratio_mb = np.exp(np.clip(log_ratio, -20.0, 20.0))
                        eps_mb = np.full_like(
                            ratio_mb,
                            float(actor_cfg.get("ppo_clip_chain", 0.01)),
                        )
                        clipped_mb = np.clip(ratio_mb, 1.0 - eps_mb, 1.0 + eps_mb)
                        advantage_mb = np.asarray(advantages_j[start:stop])
                        objective_mb = np.minimum(
                            ratio_mb * advantage_mb,
                            clipped_mb * advantage_mb,
                        )
                        epoch_loss += sample_weight * float(-objective_mb.mean())
                        # Exact derivative of the clipped surrogate with
                        # respect to the normalized joint chain log-ratio.
                        active = np.where(
                            advantage_mb >= 0.0,
                            ratio_mb <= 1.0 + eps_mb,
                            ratio_mb >= 1.0 - eps_mb,
                        )
                        coefficient = np.where(
                            active,
                            -ratio_mb * advantage_mb,
                            0.0,
                        ).astype(np.float32)

                        for flow_step in range(state.policy.num_steps):
                            _, grads_step = _stream_weighted_log_prob(
                                state.policy.actor_state,
                                old_next_states[start:stop, flow_step],
                                old_states[start:stop, flow_step],
                                observation_mb,
                                old_timesteps[start:stop, flow_step],
                                jnp.asarray(coefficient),
                            )
                            host_grads = _accumulate_jax_grads_on_host(
                                host_grads,
                                grads_step,
                                weight=sample_weight,
                            )
                            del grads_step
                            jax.effects_barrier()
                            gc.collect()
                        ratio_parts.append(ratio_mb)
                        clipped_parts.append(clipped_mb)

                    assert host_grads is not None
                    actor_grad_norm = _jax_tree_l2_norm(host_grads)
                    state.policy.apply_actor_gradients(host_grads)
                    loss_total += epoch_loss
                    ppo_total += epoch_loss
                    grad_total += actor_grad_norm
                    ratio = np.concatenate(ratio_parts)
                    clipped_ratio = np.concatenate(clipped_parts)
                return {
                    "actor_loss": loss_total / actor_epochs,
                    "full_ppo_loss": ppo_total / actor_epochs,
                    "reference_kl": 0.0,
                    "reference_kl_beta": 0.0,
                    "ustate_adapt_ppo_clip": 0.0,
                    "ustate_adapt_kl_beta": 0.0,
                    "actor_epochs": float(actor_epochs),
                    "importance_ratio_mean": float(ratio.mean()),
                    "importance_ratio_std": float(ratio.std()),
                    "importance_ratio_min": float(ratio.min()),
                    "importance_ratio_max": float(ratio.max()),
                    "ppo_clip_fraction": float((ratio != clipped_ratio).mean()),
                    "actor_grad_norm": grad_total / actor_epochs,
                    "old_policy_lag": _policy_l2_lag(state.policy, state.old_policy),
                    "fm_anchor_loss": 0.0,
                    "success_buffer_loss": 0.0,
                    "action_smoothness": 0.0,
                    "gradient_microbatch_size": float(gradient_microbatch_size),
                    "full_chain_streaming_backward": 1.0,
                    **adv_diag,
                }

            def _origin_microbatch_loss(
                actor_state,
                states,
                next_states,
                timesteps,
                observation,
                candidate_advantages,
                candidate_eps,
            ):
                actor = nnx.merge(state.policy.actor_graphdef, actor_state)
                current_log_probs = []
                frozen_log_probs = []
                for flow_step in range(state.policy.num_steps):
                    current_log_probs.append(
                        jax_transition_log_prob(
                            actor=actor,
                            flow_spec=flow_spec,
                            x_prev=next_states[:, flow_step],
                            x_t=states[:, flow_step],
                            observation=observation,
                            timestep=timesteps[:, flow_step],
                            sde_mode=state.policy.sde_mode,
                        ) / chain_logprob_normalizer
                    )
                    frozen_log_probs.append(
                        jax.lax.stop_gradient(
                            jax_transition_log_prob(
                                actor=old_actor,
                                flow_spec=flow_spec,
                                x_prev=next_states[:, flow_step],
                                x_t=states[:, flow_step],
                                observation=observation,
                                timestep=timesteps[:, flow_step],
                                sde_mode=state.old_policy.sde_mode,
                            ) / chain_logprob_normalizer
                        )
                    )
                current = jnp.stack(current_log_probs, axis=1)
                frozen = jnp.stack(frozen_log_probs, axis=1)
                if full_ratio_mode != "ais_joint":
                    raise ValueError(
                        "OGPO-origin microbatching requires actor.full_ratio_mode=ais_joint"
                    )
                ppo = jax_full_chain_ais_ppo_loss(
                    current,
                    frozen,
                    candidate_advantages,
                    clip_eps=candidate_eps,
                )
                return ppo.loss, {
                    "ratio": ppo.ratio,
                    "clipped_ratio": ppo.clipped_ratio,
                }

            torch.cuda.empty_cache()
            ratio = np.ones(int(advantages_j.shape[0]), dtype=np.float32)
            clipped_ratio = ratio.copy()
            for _ in range(actor_epochs):
                host_grads = None
                epoch_loss = 0.0
                ratio_parts = []
                clipped_parts = []
                total_candidates = int(advantages_j.shape[0])
                for start in range(0, total_candidates, gradient_microbatch_size):
                    stop = min(start + gradient_microbatch_size, total_candidates)
                    sample_weight = (stop - start) / total_candidates
                    (loss_mb, aux_mb), grads_mb = jax.value_and_grad(
                        _origin_microbatch_loss,
                        has_aux=True,
                    )(
                        state.policy.actor_state,
                        old_states[start:stop],
                        old_next_states[start:stop],
                        old_timesteps[start:stop],
                        _slice_jax_batch(jax_observation_g, start, stop),
                        advantages_j[start:stop],
                        jnp.full(
                            (stop - start,),
                            float(actor_cfg.get("ppo_clip_chain", 0.01)),
                            dtype=advantages_j.dtype,
                        ),
                    )
                    host_grads = _accumulate_jax_grads_on_host(
                        host_grads,
                        grads_mb,
                        weight=sample_weight,
                    )
                    epoch_loss += sample_weight * float(loss_mb)
                    ratio_parts.append(np.asarray(aux_mb["ratio"]))
                    clipped_parts.append(np.asarray(aux_mb["clipped_ratio"]))
                    del grads_mb
                    jax.effects_barrier()
                    gc.collect()
                assert host_grads is not None
                actor_grad_norm = _jax_tree_l2_norm(host_grads)
                state.policy.apply_actor_gradients(host_grads)
                loss_total += epoch_loss
                ppo_total += epoch_loss
                grad_total += actor_grad_norm
                ratio = np.concatenate(ratio_parts)
                clipped_ratio = np.concatenate(clipped_parts)
            return {
                "actor_loss": loss_total / actor_epochs,
                "full_ppo_loss": ppo_total / actor_epochs,
                "reference_kl": 0.0,
                "reference_kl_beta": 0.0,
                "ustate_adapt_ppo_clip": 0.0,
                "ustate_adapt_kl_beta": 0.0,
                "actor_epochs": float(actor_epochs),
                "importance_ratio_mean": float(ratio.mean()),
                "importance_ratio_std": float(ratio.std()),
                "importance_ratio_min": float(ratio.min()),
                "importance_ratio_max": float(ratio.max()),
                "ppo_clip_fraction": float((ratio != clipped_ratio).mean()),
                "actor_grad_norm": grad_total / actor_epochs,
                "old_policy_lag": _policy_l2_lag(state.policy, state.old_policy),
                "fm_anchor_loss": 0.0,
                "success_buffer_loss": 0.0,
                "action_smoothness": 0.0,
                "gradient_microbatch_size": float(gradient_microbatch_size),
                **adv_diag,
            }

        def _loss_fn(actor_state):
            actor = nnx.merge(state.policy.actor_graphdef, actor_state)
            step_log_probs = []
            canonical_old_log_probs = []
            for step in range(state.policy.num_steps):
                step_observation = jax.tree_util.tree_map(
                    lambda x: jnp.repeat(x, group_size, axis=0) if x is not None else None,
                    jax_observation,
                )
                step_log_probs.append(
                    jax_transition_log_prob(
                        actor=actor,
                        flow_spec=flow_spec,
                        x_prev=old_next_states[:, step],
                        x_t=old_states[:, step],
                        observation=step_observation,
                        timestep=old_timesteps[:, step],
                        sde_mode=state.policy.sde_mode,
                    )
                )
                canonical_old_log_probs.append(
                    jax.lax.stop_gradient(
                        jax_transition_log_prob(
                            actor=old_actor,
                            flow_spec=flow_spec,
                            x_prev=old_next_states[:, step],
                            x_t=old_states[:, step],
                            observation=step_observation,
                            timestep=old_timesteps[:, step],
                            sde_mode=state.old_policy.sde_mode,
                        )
                    )
                )
            new_log_probs = jnp.stack(step_log_probs, axis=1) / chain_logprob_normalizer
            old_log_probs = jnp.stack(canonical_old_log_probs, axis=1) / chain_logprob_normalizer
            if full_ratio_mode == "ais_joint":
                ppo = jax_full_chain_ais_ppo_loss(
                    new_log_probs,
                    old_log_probs,
                    advantages_j,
                    clip_eps=jnp.full_like(advantages_j, float(actor_cfg.get("ppo_clip_chain", 0.01))),
                )
            elif full_ratio_mode == "per_transition":
                ppo = jax_full_chain_ppo_loss(
                    new_log_probs,
                    old_log_probs,
                    advantages_j,
                    clip_eps=eps_clip_j,
                )
            else:
                raise ValueError(f"unsupported actor.full_ratio_mode={full_ratio_mode!r}")
            transition_ref_kl = jax_transition_kl(
                actor=actor,
                other_actor=reference_actor,
                flow_spec=flow_spec,
                x_t=old_states[:, 0],
                observation=jax.tree_util.tree_map(lambda x: jnp.repeat(x, group_size, axis=0) if x is not None else None, jax_observation),
                timestep=old_timesteps[:, 0],
                sde_mode=state.policy.sde_mode,
            )
            kl_penalty, ref_kl, beta = jax_state_adaptive_kl_penalty(
                transition_ref_kl,
                entropy_norm_j,
                group_size=group_size,
                beta_base=beta_base,
                adapt_kl_beta=adapt_kl_beta,
                uncertainty_scale=uncertainty_scale,
            )
            regularization_loss, fm_loss, success_loss = _jax_regularization_loss(
                actor,
                jax_regularization,
            )
            total = ppo.loss + kl_penalty + regularization_loss
            return total, {
                "ppo_loss": ppo.loss,
                "ref_kl": ref_kl,
                "beta": beta,
                "fm_loss": fm_loss,
                "success_loss": success_loss,
                "ratio": ppo.ratio,
                "clipped_ratio": ppo.clipped_ratio,
            }

        torch.cuda.empty_cache()
        for _ in range(actor_epochs):
            (total_loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(state.policy.actor_state)
            grad_leaves = jax.tree_util.tree_leaves(grads)
            grad_norm_sq = 0.0
            for leaf in grad_leaves:
                if leaf is not None:
                    grad_norm_sq += float(jnp.sum(jnp.square(leaf)))
            actor_grad_norm = grad_norm_sq ** 0.5
            state.policy.apply_actor_gradients(grads)
            loss_total += float(total_loss)
            ppo_total += float(aux["ppo_loss"])
            kl_total += float(aux["ref_kl"])
            grad_total += actor_grad_norm
        ratio = np.asarray(aux["ratio"])
        clipped_ratio = np.asarray(aux["clipped_ratio"])
        return {
            "actor_loss": loss_total / actor_epochs,
            "full_ppo_loss": ppo_total / actor_epochs,
            "reference_kl": kl_total / actor_epochs,
            "reference_kl_beta": float(aux["beta"]),
            "ustate_adapt_ppo_clip": float(bool(uncertainty_cfg.get("adapt_ppo_clip", False))),
            "ustate_adapt_kl_beta": float(bool(uncertainty_cfg.get("adapt_kl_beta", False))),
            "actor_epochs": float(actor_epochs),
            "importance_ratio_mean": float(ratio.mean()),
            "importance_ratio_std": float(ratio.std()),
            "importance_ratio_min": float(ratio.min()),
            "importance_ratio_max": float(ratio.max()),
            "ppo_clip_fraction": float((ratio != clipped_ratio).mean()),
            "actor_grad_norm": grad_total / actor_epochs,
            "old_policy_lag": _policy_l2_lag(state.policy, state.old_policy),
            "fm_anchor_loss": float(aux["fm_loss"]),
            "success_buffer_loss": float(aux["success_loss"]),
            "action_smoothness": 0.0,
            **adv_diag,
        }

    with torch.no_grad():
        old_rollout = state.old_policy.rollout(condition, group_size=group_size)
        environment_endpoint = state.old_policy.flat_actions_to_environment(old_rollout.endpoint, condition_g)
        endpoint = environment_endpoint.reshape(batch.batch_size, group_size, -1)
        advantages, adv_diag = conservative_advantages_for_candidates(
            state, batch.observations, endpoint, batch, config
        )

    with torch.no_grad():
        entropy_norm = _normalized_state_entropy(state, batch).mean(dim=0)
        eps_clip = actor_clip_for_uncertainty(
            entropy_norm,
            actor_cfg,
            uncertainty_cfg,
        ).repeat_interleave(group_size)
    loss_total = 0.0
    ppo_total = 0.0
    kl_total = 0.0
    grad_total = 0.0
    for _ in range(actor_epochs):
        state.actor_optimizer.zero_grad(set_to_none=True)
        new_log_probs = []
        for step in range(state.policy.num_steps):
            new_log_probs.append(
                state.policy.log_prob(
                    old_rollout.next_states[:, step],
                    old_rollout.states[:, step],
                    condition_g,
                    old_rollout.timesteps[:, step],
                )
            )
        new_log_probs_tensor = torch.stack(new_log_probs, dim=1)
        full_ratio_mode = str(actor_cfg.get("full_ratio_mode", "per_transition"))
        if full_ratio_mode == "ais_joint":
            ppo = full_chain_ais_ppo_loss(
                new_log_probs_tensor,
                old_rollout.log_probs,
                advantages.reshape(-1),
                clip_eps=float(actor_cfg.get("ppo_clip_chain", 0.01)),
            )
        elif full_ratio_mode == "per_transition":
            ppo = full_chain_ppo_loss(
                new_log_probs_tensor,
                old_rollout.log_probs,
                advantages.reshape(-1),
                clip_eps=eps_clip,
            )
        else:
            raise ValueError(f"unsupported actor.full_ratio_mode={full_ratio_mode!r}")
        transition_ref_kl = state.policy.kl_to(
            state.reference_policy,
            old_rollout.states[:, 0],
            condition_g,
            old_rollout.timesteps[:, 0],
        )
        kl_penalty, ref_kl, kl_beta = state_adaptive_kl_penalty(
            transition_ref_kl,
            entropy_norm,
            group_size=group_size,
            beta_base=float(regularization_cfg.get("beta_kl", 0.01)),
            uncertainty_scale=kl_uncertainty_scale(regularization_cfg, uncertainty_cfg),
        )
        reg_loss, reg_diag = _actor_regularization_loss(
            state,
            batch,
            config,
            fm_batch=fm_batch,
            success_batch=success_batch,
        )
        loss = ppo.loss + reg_loss + kl_penalty
        loss.backward()
        assert_no_gradients(state.critic, "critic")
        assert_no_gradients(state.reference_policy, "reference_policy")
        actor_grad_norm = float(grad_norm(state.policy.parameters()))
        torch.nn.utils.clip_grad_norm_(state.policy.parameters(), float(actor_cfg.get("max_grad_norm", 1.0)))
        state.actor_optimizer.step()
        loss_total += float(loss.detach().item())
        ppo_total += float(ppo.loss.detach().item())
        kl_total += float(ref_kl.detach().item())
        grad_total += actor_grad_norm
    return {
        "actor_loss": loss_total / actor_epochs,
        "full_ppo_loss": ppo_total / actor_epochs,
        "reference_kl": kl_total / actor_epochs,
        "reference_kl_beta": float(kl_beta.detach().item()),
        "ustate_adapt_ppo_clip": float(bool(uncertainty_cfg.get("adapt_ppo_clip", False))),
        "ustate_adapt_kl_beta": float(bool(uncertainty_cfg.get("adapt_kl_beta", False))),
        "actor_epochs": float(actor_epochs),
        "importance_ratio_mean": ppo.ratio_mean,
        "importance_ratio_std": ppo.ratio_std,
        "importance_ratio_min": ppo.ratio_min,
        "importance_ratio_max": ppo.ratio_max,
        "ppo_clip_fraction": ppo.clip_fraction,
        "actor_grad_norm": grad_total / actor_epochs,
        "old_policy_lag": _policy_l2_lag(state.policy, state.old_policy),
        **reg_diag,
        **adv_diag,
    }


def flash_actor_update(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    fm_batch: ChunkBatch | None = None,
    success_batch: ChunkBatch | None = None,
    actor_step: int | None = None,
) -> dict[str, float]:
    actor_cfg = config.get("actor", {})
    flow_cfg = config.get("flow", {})
    regularization_cfg = config.get("regularization", {})
    uncertainty_cfg = config.get("uncertainty", {})
    batch = batch.to(next(state.policy.parameters()).device)
    group_size = int(actor_cfg.get("candidate_group_size", actor_cfg.get("group_size", 4)))
    gradient_microbatch_size = int(actor_cfg.get("gradient_microbatch_size", group_size))
    kl_eval_microbatch_size = int(
        actor_cfg.get("kl_eval_microbatch_size", gradient_microbatch_size)
    )
    data_parallel_devices = int(actor_cfg.get("data_parallel_devices", 1))
    distributed_gradient_reduction = str(
        actor_cfg.get("distributed_gradient_reduction", "manual")
    ).lower()
    rollout_state_microbatch_size = int(
        actor_cfg.get("rollout_state_microbatch_size", batch.batch_size)
    )
    success_update_period = int(regularization_cfg.get("success_update_period", 1))
    if (
        group_size <= 0
        or gradient_microbatch_size <= 0
        or kl_eval_microbatch_size <= 0
        or rollout_state_microbatch_size <= 0
        or data_parallel_devices <= 0
        or success_update_period <= 0
    ):
        raise ValueError(
            "candidate_group_size, gradient_microbatch_size, "
            "kl_eval_microbatch_size, rollout_state_microbatch_size, "
            "data_parallel_devices, and success_update_period must be positive"
        )
    if distributed_gradient_reduction not in {"manual", "pmean"}:
        raise ValueError(
            "actor.distributed_gradient_reduction must be 'manual' or 'pmean'; "
            f"got {distributed_gradient_reduction!r}"
        )
    success_update_due = (
        actor_step is None or int(actor_step) % success_update_period == 0
    )
    sampling_step = int(state.step if actor_step is None else actor_step)
    sampling_seed = int(config.get("training", {}).get("seed", 0)) + sampling_step
    actor_epochs = max(1, int(actor_cfg.get("actor_epochs_per_rollout", 1)))
    stabilize_on_policy_statistics = bool(
        actor_cfg.get("stabilize_single_epoch_bf16_statistics", False)
    )
    if stabilize_on_policy_statistics:
        sync_period = int(actor_cfg.get("old_policy_sync_period", 1))
        old_policy_ema = float(actor_cfg.get("old_policy_ema", 0.0))
        if sync_period != 1 or old_policy_ema != 0.0:
            raise ValueError(
                "actor.stabilize_single_epoch_bf16_statistics requires "
                "old_policy_sync_period=1 and old_policy_ema=0"
            )
    regularization_epochs = int(
        regularization_cfg.get("actor_regularization_epochs_per_rollout", 1)
    )
    if regularization_epochs not in {0, 1}:
        raise ValueError(
            "regularization.actor_regularization_epochs_per_rollout currently "
            "supports only 0 or 1"
        )
    selected_steps = _select_flash_steps(
        flow_cfg,
        batch_size=batch.batch_size,
        num_steps=state.policy.num_steps,
        device=batch.observations.device,
        seed=sampling_seed + 101,
    )
    state.critic_optimizer.zero_grad(set_to_none=True)
    condition = _policy_condition(state.policy, batch)
    condition_g = state.policy.repeat_condition(condition, group_size)

    if _using_jax_actor(state.policy):
        assert isinstance(state.policy, PI05JaxFlowPolicy)
        assert isinstance(state.old_policy, PI05JaxFlowPolicy)
        assert isinstance(state.reference_policy, PI05JaxFlowPolicy)
        reference_kl_action_horizon = int(
            actor_cfg.get(
                "reference_kl_action_horizon",
                state.policy.model_horizon,
            )
        )
        if not 1 <= reference_kl_action_horizon <= state.policy.model_horizon:
            raise ValueError(
                "actor.reference_kl_action_horizon must be in "
                f"[1, {state.policy.model_horizon}], got "
                f"{reference_kl_action_horizon}"
            )
        reference_kl_event_dim = (
            reference_kl_action_horizon * state.policy.environment_action_dim
        )
        ppo_action_horizon = int(
            actor_cfg.get("ppo_action_horizon", state.policy.model_horizon)
        )
        if not 1 <= ppo_action_horizon <= state.policy.model_horizon:
            raise ValueError(
                "actor.ppo_action_horizon must be in "
                f"[1, {state.policy.model_horizon}], got {ppo_action_horizon}"
            )
        ppo_event_dim = ppo_action_horizon * state.policy.environment_action_dim
        full_reference_kl_event_dim = (
            state.policy.model_horizon * state.policy.environment_action_dim
        )

        def _ppo_log_prob(x, mean, log_std):
            return jax_gaussian_log_prob(
                x[..., :ppo_event_dim],
                mean[..., :ppo_event_dim],
                log_std[..., :ppo_event_dim],
            )

        def _prefix_reference_kl(
            mean_p,
            log_std_p,
            mean_q,
            log_std_q,
        ):
            return jax_gaussian_kl_diag(
                mean_p[..., :reference_kl_event_dim],
                log_std_p[..., :reference_kl_event_dim],
                mean_q[..., :reference_kl_event_dim],
                log_std_q[..., :reference_kl_event_dim],
            )

        flow_spec = OpenPIJaxFlowSpec(state.policy.num_steps)
        jax_observation = policy_observation_to_jax(condition.observation)
        jax_observation_g = policy_observation_to_jax(condition_g.observation)
        local_devices = jax.local_devices()
        if data_parallel_devices > len(local_devices):
            raise RuntimeError(
                f"actor.data_parallel_devices={data_parallel_devices}, but JAX sees only "
                f"{len(local_devices)} local devices: {local_devices}"
            )
        actor_devices = local_devices[:data_parallel_devices]
        parallel_frozen_statistics = bool(
            actor_cfg.get("parallel_frozen_statistics", False)
        )
        old_statistics_device_index = int(
            actor_cfg.get("old_statistics_device_index", 0)
        )
        reference_statistics_device_index = int(
            actor_cfg.get(
                "reference_statistics_device_index",
                1 if parallel_frozen_statistics and data_parallel_devices > 1 else 0,
            )
        )
        regularization_device_index = int(
            actor_cfg.get(
                "regularization_device_index",
                2 if parallel_frozen_statistics and data_parallel_devices > 2 else 0,
            )
        )
        statistics_device_indices = (
            old_statistics_device_index,
            reference_statistics_device_index,
            regularization_device_index,
        )
        if any(
            index < 0 or index >= data_parallel_devices
            for index in statistics_device_indices
        ):
            raise ValueError(
                "old/reference statistics and regularization device indices must "
                f"be in [0, {data_parallel_devices}); got {statistics_device_indices}"
            )
        if parallel_frozen_statistics and data_parallel_devices < 3:
            raise ValueError(
                "actor.parallel_frozen_statistics requires at least three JAX devices"
            )
        total_candidates = batch.batch_size * group_size
        if total_candidates % data_parallel_devices:
            raise ValueError(
                f"effective candidate batch {total_candidates} must be divisible by "
                f"actor.data_parallel_devices={data_parallel_devices}"
            )
        if data_parallel_devices > 1:
            # Orbax restores arrays with the sharding saved in the checkpoint
            # (e.g. 4-way sharded for a 4-GPU run). On a different topology
            # (8 GPUs) may_alias=True device_put aliases one shard instead of
            # consolidating, so pmap broadcast arguments land on the wrong
            # device. Force a full copy to the target device.
            state.policy.actor_state = _jax_tree_copy_to_device(
                state.policy.actor_state,
                actor_devices[0],
            )
            state.old_policy.actor_state = _jax_tree_copy_to_device(
                state.old_policy.actor_state,
                actor_devices[old_statistics_device_index],
            )
            reference_statistics_storage_index = (
                old_statistics_device_index
                if stabilize_on_policy_statistics
                else reference_statistics_device_index
            )
            state.reference_policy.actor_state = _jax_tree_copy_to_device(
                state.reference_policy.actor_state,
                actor_devices[reference_statistics_storage_index],
            )
            state.policy.actor_opt_state = _jax_tree_copy_to_device(
                state.policy.actor_opt_state,
                actor_devices[0],
            )
            jax.effects_barrier()
            gc.collect()

        with torch.no_grad():
            rng = jax.random.PRNGKey(sampling_seed + 17)
            selected_steps_jax = jnp.asarray(selected_steps.detach().cpu().numpy())
            selected_steps_grouped_jax = jnp.repeat(selected_steps_jax, group_size)
            if data_parallel_devices > 1:
                rollout_cache_key = (
                    data_parallel_devices,
                    state.old_policy.num_steps,
                    state.old_policy.sde_mode,
                )
                rollout_cache = getattr(state.policy, "_flash_dp_rollout_cache", None)
                if rollout_cache is None or rollout_cache[0] != rollout_cache_key:
                    old_graphdef = state.old_policy.actor_graphdef

                    def _distributed_rollout(actor_state, observation, selected_step, device_rng):
                        actor = nnx.merge(old_graphdef, actor_state)
                        rollout = sample_jax_flash_rollout(
                            actor=actor,
                            flow_spec=OpenPIJaxFlowSpec(state.old_policy.num_steps),
                            observation=observation,
                            group_size=1,
                            selected_step=selected_step,
                            rng=device_rng,
                            sde_mode=state.old_policy.sde_mode,
                        )
                        return rollout.x_t, rollout.x_prev, rollout.timestep, rollout.endpoint

                    distributed_rollout = jax.pmap(
                        _distributed_rollout,
                        in_axes=(None, 0, 0, 0),
                        devices=actor_devices,
                    )
                    state.policy._flash_dp_rollout_cache = (
                        rollout_cache_key,
                        distributed_rollout,
                    )
                else:
                    distributed_rollout = rollout_cache[1]
                rollout_result = distributed_rollout(
                    state.old_policy.actor_state,
                    _shard_jax_batch(jax_observation_g, data_parallel_devices),
                    _shard_jax_batch(selected_steps_grouped_jax, data_parallel_devices),
                    jax.random.split(rng, data_parallel_devices),
                )
                old_x_t, old_x_prev, old_timestep, old_endpoint = (
                    value.reshape(total_candidates, *value.shape[2:])
                    for value in rollout_result
                )
            else:
                candidate_rngs = jax.random.split(rng, group_size)
                old_actor = nnx.merge(
                    state.old_policy.actor_graphdef,
                    state.old_policy.actor_state,
                )
                rollout_parts = []
                for candidate in range(group_size):
                    state_parts = [
                        _sample_frozen_jax_flash_rollout(
                            old_actor,
                            _slice_jax_batch(jax_observation, start, stop),
                            selected_step=selected_steps_jax[start:stop],
                            rng=jax.random.fold_in(candidate_rngs[candidate], start),
                            num_steps=state.old_policy.num_steps,
                            sde_mode=state.old_policy.sde_mode,
                            group_size=1,
                        )
                        for start in range(0, batch.batch_size, rollout_state_microbatch_size)
                        for stop in [
                            min(start + rollout_state_microbatch_size, batch.batch_size)
                        ]
                    ]
                    rollout_parts.append(
                        tuple(
                            jnp.concatenate(
                                [part[component] for part in state_parts],
                                axis=0,
                            )
                            for component in range(4)
                        )
                    )
                old_x_t, old_x_prev, old_timestep, old_endpoint = (
                    _stack_candidate_rollouts([part[index] for part in rollout_parts])
                    for index in range(4)
                )
            old_endpoint_torch = torch.as_tensor(
                np.asarray(old_endpoint).copy(),
                device=batch.observations.device,
                dtype=batch.action_chunks.dtype,
            )
            environment_endpoint = state.old_policy.flat_actions_to_environment(old_endpoint_torch, condition_g)
            endpoint = environment_endpoint.reshape(batch.batch_size, group_size, -1)
            advantages, adv_diag = conservative_advantages_for_candidates(
                state, batch.observations, endpoint, batch, config
            )
            entropy_norm = _normalized_state_entropy(state, batch).mean(dim=0)
            eps_clip = (
                actor_clip_for_uncertainty(entropy_norm, actor_cfg, uncertainty_cfg)
                .unsqueeze(-1)
                .expand(batch.batch_size, group_size)
                .reshape(-1)
            )
        selected_counts = torch.bincount(selected_steps.detach().cpu(), minlength=state.policy.num_steps)
        loss_total = 0.0
        flash_total = 0.0
        raw_flash_total = 0.0
        kl_total = 0.0
        rectification_total = 0.0
        raw_loss_by_step = [0.0] * state.policy.num_steps
        raw_grad_by_step = [0.0] * state.policy.num_steps
        rectified_grad_by_step = [0.0] * state.policy.num_steps
        selected_steps_grouped = selected_steps.repeat_interleave(group_size)
        rectification = _flash_rectification_weight(
            state,
            flow_cfg,
            timestep=torch.as_tensor(
                np.asarray(old_timestep).copy(),
                device=batch.observations.device,
                dtype=batch.action_chunks.dtype,
            ),
            selected_steps=selected_steps,
            group_size=group_size,
        )
        jax_regularization = _prepare_jax_regularization(
            state,
            batch,
            config,
            fm_batch=fm_batch,
            success_batch=success_batch,
            enable_success=success_update_due,
            seed_step=sampling_seed,
        )
        beta_base = float(regularization_cfg.get("beta_kl", 0.01))
        adapt_kl_beta = bool(uncertainty_cfg.get("adapt_kl_beta", False))
        uncertainty_scale = float(regularization_cfg.get("kl_uncertainty_scale", 1.0))
        normalize_logprob_by_action_dim = bool(
            actor_cfg.get("normalize_logprob_by_action_dim", False)
        )
        logprob_normalizer = (
            float(ppo_event_dim) if normalize_logprob_by_action_dim else 1.0
        )
        old_timestep = old_timestep.reshape(-1, 1)
        advantages_j = jnp.asarray(advantages.reshape(-1).detach().cpu().numpy())
        eps_clip_j = jnp.asarray(eps_clip.detach().cpu().numpy())
        rectification_j = jnp.asarray(rectification.detach().cpu().numpy())
        entropy_norm_j = jnp.repeat(
            jnp.asarray(entropy_norm.detach().cpu().numpy()),
            group_size,
        )
        num_candidates = int(advantages_j.shape[0])

        def _policy_loss_fn(
            actor_state,
            old_actor_state,
            reference_actor_state,
            x_prev,
            x_t,
            observation,
            timestep,
            candidate_advantages,
            candidate_eps_clip,
            candidate_rectification,
            candidate_entropy,
            fixed_old_log_prob=None,
            fixed_old_mean=None,
            fixed_old_log_std=None,
            fixed_reference_mean=None,
            fixed_reference_log_std=None,
            anchor_on_policy_statistics=0.0,
        ):
            actor = nnx.merge(state.policy.actor_graphdef, actor_state)
            loss_old_actor = nnx.merge(
                state.old_policy.actor_graphdef,
                old_actor_state,
            )
            loss_reference_actor = nnx.merge(
                state.reference_policy.actor_graphdef,
                reference_actor_state,
            )
            # Compute old/current probabilities in the same transformed trace.
            # PI0.5's bf16 fusion otherwise produces a large spurious ratio
            # offset when summing over the 660-dimensional action chunk.
            old_mean = jax_transition_mean(
                actor=loss_old_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                observation=observation,
                timestep=timestep,
                sde_mode=state.old_policy.sde_mode,
            )
            old_log_std = jax_transition_log_std(
                actor=loss_old_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                timestep=timestep,
                sde_mode=state.old_policy.sde_mode,
            )
            old_log_prob = jax.lax.stop_gradient(
                _ppo_log_prob(x_prev, old_mean, old_log_std)
            ) / logprob_normalizer
            current_mean = jax_transition_mean(
                actor=actor,
                flow_spec=flow_spec,
                x_t=x_t,
                observation=observation,
                timestep=timestep,
                sde_mode=state.policy.sde_mode,
            )
            current_log_std = jax_transition_log_std(
                actor=actor,
                flow_spec=flow_spec,
                x_t=x_t,
                timestep=timestep,
                sde_mode=state.policy.sde_mode,
            )
            new_log_prob = _ppo_log_prob(
                x_prev,
                current_mean,
                current_log_std,
            ) / logprob_normalizer
            if stabilize_on_policy_statistics:
                if any(
                    value is None
                    for value in (
                        fixed_old_log_prob,
                        fixed_old_mean,
                        fixed_old_log_std,
                        fixed_reference_mean,
                        fixed_reference_log_std,
                    )
                ):
                    raise ValueError("stabilized on-policy statistics were not supplied")
                old_log_prob = fixed_old_log_prob
                old_mean = fixed_old_mean
                old_log_std = fixed_old_log_std
                ppo_new_log_prob = _conditionally_anchor_current_to_old_value(
                    new_log_prob,
                    old_log_prob,
                    anchor_on_policy_statistics,
                )
                kl_current_mean = _conditionally_anchor_current_to_old_value(
                    current_mean,
                    old_mean,
                    anchor_on_policy_statistics,
                )
                kl_current_log_std = _conditionally_anchor_current_to_old_value(
                    current_log_std,
                    old_log_std,
                    anchor_on_policy_statistics,
                )
            else:
                ppo_new_log_prob = new_log_prob
                kl_current_mean = current_mean
                kl_current_log_std = current_log_std
            raw_flash = jax_flash_ppo_loss(
                ppo_new_log_prob,
                old_log_prob,
                candidate_advantages,
                clip_eps=candidate_eps_clip,
                rectification_weight=jnp.ones_like(candidate_rectification),
            )
            flash = jax_flash_ppo_loss(
                ppo_new_log_prob,
                old_log_prob,
                candidate_advantages,
                clip_eps=candidate_eps_clip,
                rectification_weight=candidate_rectification,
            )
            reference_mean = jax_transition_mean(
                actor=loss_reference_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                observation=observation,
                timestep=timestep,
                sde_mode=state.reference_policy.sde_mode,
            )
            reference_log_std = jax_transition_log_std(
                actor=loss_reference_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                timestep=timestep,
                sde_mode=state.reference_policy.sde_mode,
            )
            if stabilize_on_policy_statistics:
                reference_mean = fixed_reference_mean
                reference_log_std = fixed_reference_log_std
            transition_ref_kl = _prefix_reference_kl(
                kl_current_mean,
                kl_current_log_std,
                reference_mean,
                reference_log_std,
            )
            kl_penalty, ref_kl, beta = jax_state_adaptive_kl_penalty(
                transition_ref_kl,
                candidate_entropy,
                group_size=1,
                beta_base=beta_base,
                adapt_kl_beta=adapt_kl_beta,
                uncertainty_scale=uncertainty_scale,
            )
            total = flash.loss + kl_penalty
            aux = {
                "flash_loss": flash.loss,
                "raw_flash_loss": raw_flash.loss,
                "ref_kl": ref_kl,
                "beta": beta,
                "ratio": flash.ratio,
                "clipped_ratio": flash.clipped_ratio,
                "per_sample_loss": raw_flash.per_sample_loss,
            }
            return total, aux

        def _reference_kl_fn(
            actor_state,
            old_actor_state,
            reference_actor_state,
            x_t,
            observation,
            timestep,
        ):
            actor = nnx.merge(state.policy.actor_graphdef, actor_state)
            old_actor = nnx.merge(state.old_policy.actor_graphdef, old_actor_state)
            reference_actor = nnx.merge(
                state.reference_policy.actor_graphdef,
                reference_actor_state,
            )
            current_mean = jax_transition_mean(
                actor=actor,
                flow_spec=flow_spec,
                x_t=x_t,
                observation=observation,
                timestep=timestep,
                sde_mode=state.policy.sde_mode,
            )
            current_log_std = jax_transition_log_std(
                actor=actor,
                flow_spec=flow_spec,
                x_t=x_t,
                timestep=timestep,
                sde_mode=state.policy.sde_mode,
            )
            reference_mean = jax_transition_mean(
                actor=reference_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                observation=observation,
                timestep=timestep,
                sde_mode=state.reference_policy.sde_mode,
            )
            reference_log_std = jax_transition_log_std(
                actor=reference_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                timestep=timestep,
                sde_mode=state.reference_policy.sde_mode,
            )
            old_mean = jax_transition_mean(
                actor=old_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                observation=observation,
                timestep=timestep,
                sde_mode=state.old_policy.sde_mode,
            )
            old_log_std = jax_transition_log_std(
                actor=old_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                timestep=timestep,
                sde_mode=state.old_policy.sde_mode,
            )
            return (
                jnp.mean(
                    _prefix_reference_kl(
                        current_mean,
                        current_log_std,
                        reference_mean,
                        reference_log_std,
                    )
                ),
                jnp.mean(
                    jax_gaussian_kl_diag(
                        current_mean,
                        current_log_std,
                        reference_mean,
                        reference_log_std,
                    )
                ),
                jnp.mean(
                    jax_gaussian_kl_diag(
                        current_mean[..., :ppo_event_dim],
                        current_log_std[..., :ppo_event_dim],
                        old_mean[..., :ppo_event_dim],
                        old_log_std[..., :ppo_event_dim],
                    )
                ),
            )

        def _flow_matching_component(actor_state, inputs):
            actor = nnx.merge(state.policy.actor_graphdef, actor_state)
            return jax_flow_matching_loss(actor=actor, **inputs)

        if data_parallel_devices > 1:
            dp_cache_key = (
                data_parallel_devices,
                state.policy.num_steps,
                state.policy.sde_mode,
                state.reference_policy.sde_mode,
                beta_base,
                adapt_kl_beta,
                uncertainty_scale,
                normalize_logprob_by_action_dim,
                parallel_frozen_statistics,
                old_statistics_device_index,
                reference_statistics_device_index,
                regularization_device_index,
                distributed_gradient_reduction,
                stabilize_on_policy_statistics,
                reference_kl_event_dim,
                ppo_event_dim,
            )
            dp_cache = getattr(state.policy, "_flash_dp_train_cache", None)
            if dp_cache is None or dp_cache[0] != dp_cache_key:
                policy_graphdef = state.policy.actor_graphdef
                old_graphdef = state.old_policy.actor_graphdef
                reference_graphdef = state.reference_policy.actor_graphdef

                def _frozen_policy_statistics(
                    frozen_actor_state,
                    x_prev,
                    x_t,
                    observation,
                    timestep,
                ):
                    frozen_actor = nnx.merge(old_graphdef, frozen_actor_state)
                    frozen_mean = jax_transition_mean(
                        actor=frozen_actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        observation=observation,
                        timestep=timestep,
                        sde_mode=state.old_policy.sde_mode,
                    )
                    frozen_log_std = jax_transition_log_std(
                        actor=frozen_actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        timestep=timestep,
                        sde_mode=state.old_policy.sde_mode,
                    )
                    frozen_log_prob = (
                        _ppo_log_prob(x_prev, frozen_mean, frozen_log_std)
                        / logprob_normalizer
                    )
                    return frozen_log_prob, frozen_mean, frozen_log_std

                def _reference_policy_statistics(
                    reference_actor_state,
                    x_prev,
                    x_t,
                    observation,
                    timestep,
                ):
                    reference_actor = nnx.merge(reference_graphdef, reference_actor_state)
                    reference_mean = jax_transition_mean(
                        actor=reference_actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        observation=observation,
                        timestep=timestep,
                        sde_mode=state.reference_policy.sde_mode,
                    )
                    reference_log_std = jax_transition_log_std(
                        actor=reference_actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        timestep=timestep,
                        sde_mode=state.reference_policy.sde_mode,
                    )
                    reference_log_prob = (
                        _ppo_log_prob(
                            x_prev,
                            reference_mean,
                            reference_log_std,
                        )
                        / logprob_normalizer
                    )
                    return reference_log_prob, reference_mean, reference_log_std

                def _distributed_policy_loss_fn(
                    actor_state,
                    x_prev,
                    x_t,
                    observation,
                    timestep,
                    candidate_advantages,
                    candidate_eps_clip,
                    candidate_rectification,
                    candidate_entropy,
                    old_log_prob,
                    old_mean,
                    old_log_std,
                    reference_mean,
                    reference_log_std,
                    log_prob_calibration,
                    mean_calibration,
                    log_std_calibration,
                    anchor_on_policy_statistics,
                ):
                    actor = nnx.merge(policy_graphdef, actor_state)
                    current_mean = jax_transition_mean(
                        actor=actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        observation=observation,
                        timestep=timestep,
                        sde_mode=state.policy.sde_mode,
                    )
                    current_log_std = jax_transition_log_std(
                        actor=actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        timestep=timestep,
                        sde_mode=state.policy.sde_mode,
                    )
                    new_log_prob = _ppo_log_prob(
                        x_prev,
                        current_mean,
                        current_log_std,
                    ) / logprob_normalizer
                    if stabilize_on_policy_statistics:
                        calibrated_new_log_prob = new_log_prob + jax.lax.stop_gradient(
                            log_prob_calibration
                        )
                        calibrated_current_mean = current_mean + jax.lax.stop_gradient(
                            mean_calibration
                        )
                        calibrated_current_log_std = (
                            current_log_std
                            + jax.lax.stop_gradient(log_std_calibration)
                        )
                        ppo_new_log_prob = _conditionally_anchor_current_to_old_value(
                            calibrated_new_log_prob,
                            old_log_prob,
                            anchor_on_policy_statistics,
                        )
                        kl_current_mean = _conditionally_anchor_current_to_old_value(
                            calibrated_current_mean,
                            old_mean,
                            anchor_on_policy_statistics,
                        )
                        kl_current_log_std = _conditionally_anchor_current_to_old_value(
                            calibrated_current_log_std,
                            old_log_std,
                            anchor_on_policy_statistics,
                        )
                    else:
                        ppo_new_log_prob = new_log_prob
                        kl_current_mean = current_mean
                        kl_current_log_std = current_log_std
                    raw_flash = jax_flash_ppo_loss(
                        ppo_new_log_prob,
                        old_log_prob,
                        candidate_advantages,
                        clip_eps=candidate_eps_clip,
                        rectification_weight=jnp.ones_like(candidate_rectification),
                    )
                    flash = jax_flash_ppo_loss(
                        ppo_new_log_prob,
                        old_log_prob,
                        candidate_advantages,
                        clip_eps=candidate_eps_clip,
                        rectification_weight=candidate_rectification,
                    )
                    transition_ref_kl = _prefix_reference_kl(
                        kl_current_mean,
                        kl_current_log_std,
                        reference_mean,
                        reference_log_std,
                    )
                    kl_penalty, ref_kl, beta = jax_state_adaptive_kl_penalty(
                        transition_ref_kl,
                        candidate_entropy,
                        group_size=1,
                        beta_base=beta_base,
                        adapt_kl_beta=adapt_kl_beta,
                        uncertainty_scale=uncertainty_scale,
                    )
                    total = flash.loss + kl_penalty
                    return total, {
                        "flash_loss": flash.loss,
                        "raw_flash_loss": raw_flash.loss,
                        "ref_kl": ref_kl,
                        "beta": beta,
                        "ratio": flash.ratio,
                        "clipped_ratio": flash.clipped_ratio,
                        "per_sample_loss": raw_flash.per_sample_loss,
                        "preupdate_mean_abs_diff": jnp.mean(jnp.abs(current_mean - old_mean)),
                        "preupdate_log_prob_abs_diff": jnp.mean(jnp.abs(new_log_prob - old_log_prob)),
                        "surrogate_log_prob_abs_diff": jnp.mean(
                            jnp.abs(ppo_new_log_prob - old_log_prob)
                        ),
                        "next_log_prob_calibration": old_log_prob - new_log_prob,
                        "next_mean_calibration": old_mean - current_mean,
                        "next_log_std_calibration": old_log_std - current_log_std,
                    }

                def _distributed_policy_loss_and_grad(*args):
                    (loss, aux), grads = jax.value_and_grad(
                        _distributed_policy_loss_fn,
                        has_aux=True,
                    )(*args)
                    if distributed_gradient_reduction == "pmean":
                        grads = jax.lax.pmean(grads, axis_name="actor_data")
                    return (loss, aux), grads

                def _distributed_reference_kl(
                    actor_state,
                    x_t,
                    observation,
                    timestep,
                    old_mean,
                    old_log_std,
                    reference_mean,
                    reference_log_std,
                ):
                    actor = nnx.merge(policy_graphdef, actor_state)
                    current_mean = jax_transition_mean(
                        actor=actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        observation=observation,
                        timestep=timestep,
                        sde_mode=state.policy.sde_mode,
                    )
                    current_log_std = jax_transition_log_std(
                        actor=actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        timestep=timestep,
                        sde_mode=state.policy.sde_mode,
                    )
                    prefix_kl = jnp.mean(
                        _prefix_reference_kl(
                            current_mean,
                            current_log_std,
                            reference_mean,
                            reference_log_std,
                        )
                    )
                    full_kl = jnp.mean(
                        jax_gaussian_kl_diag(
                            current_mean,
                            current_log_std,
                            reference_mean,
                            reference_log_std,
                        )
                    )
                    old_policy_kl = jnp.mean(
                        jax_gaussian_kl_diag(
                            current_mean[..., :ppo_event_dim],
                            current_log_std[..., :ppo_event_dim],
                            old_mean[..., :ppo_event_dim],
                            old_log_std[..., :ppo_event_dim],
                        )
                    )
                    return prefix_kl, full_kl, old_policy_kl

                # Frozen-policy forward passes and success BC are independent.
                # Pinning each to a configured device lets JAX enqueue them
                # concurrently while retaining one globally equivalent loss.
                distributed_old_statistics = jax.jit(
                    _frozen_policy_statistics,
                    device=actor_devices[old_statistics_device_index],
                )
                if stabilize_on_policy_statistics:
                    # Equal BF16 policies must use one executable and device.
                    # Cross-trace fourth-decimal errors become O(0.1) after the
                    # 660-dimensional transition KL is summed.
                    distributed_reference_statistics = distributed_old_statistics
                else:
                    distributed_reference_statistics = jax.jit(
                        _reference_policy_statistics,
                        device=actor_devices[reference_statistics_device_index],
                    )
                distributed_regularization_loss_and_grad = jax.jit(
                    jax.value_and_grad(_flow_matching_component),
                    device=actor_devices[regularization_device_index],
                )
                distributed_policy_loss_and_grad = jax.pmap(
                    _distributed_policy_loss_and_grad,
                    axis_name="actor_data",
                    in_axes=(
                        None,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        None,
                    ),
                    devices=actor_devices,
                )
                distributed_reference_kl = jax.pmap(
                    _distributed_reference_kl,
                    axis_name="actor_data",
                    in_axes=(None, 0, 0, 0, 0, 0, 0, 0),
                    devices=actor_devices,
                )
                state.policy._flash_dp_train_cache = (
                    dp_cache_key,
                    distributed_old_statistics,
                    distributed_reference_statistics,
                    distributed_regularization_loss_and_grad,
                    distributed_policy_loss_and_grad,
                    distributed_reference_kl,
                )
            else:
                distributed_old_statistics = dp_cache[1]
                distributed_reference_statistics = dp_cache[2]
                distributed_regularization_loss_and_grad = dp_cache[3]
                distributed_policy_loss_and_grad = dp_cache[4]
                distributed_reference_kl = dp_cache[5]
            fixed_old_log_prob = None
            fixed_old_mean = None
            fixed_old_log_std = None
            fixed_reference_mean = None
            fixed_reference_log_std = None
            reference_kl_eval = None
        else:
            distributed_old_statistics = None
            distributed_reference_statistics = None
            distributed_regularization_loss_and_grad = None
            distributed_policy_loss_and_grad = None
            distributed_reference_kl = None
            fixed_old_log_prob = None
            fixed_old_mean = None
            fixed_old_log_std = None
            fixed_reference_mean = None
            fixed_reference_log_std = None
            reference_kl_eval = jax.jit(_reference_kl_fn)

        # Enqueue success/FM backward before the two frozen-policy forwards.
        # On the accelerated layout all three execute on distinct GPUs. Their
        # gradients are still summed exactly as before.
        precomputed_regularization: dict[str, tuple[Any, Any]] = {}
        if data_parallel_devices > 1:
            assert distributed_regularization_loss_and_grad is not None
            for name in ("fm", "success"):
                inputs = jax_regularization[name]
                weight = float(jax_regularization[f"lambda_{name}"])
                if inputs is None or weight == 0.0:
                    continue
                precomputed_regularization[name] = (
                    distributed_regularization_loss_and_grad(
                        state.policy.actor_state,
                        inputs,
                    )
                )
            assert distributed_old_statistics is not None
            assert distributed_reference_statistics is not None
            fixed_statistics_microbatch_size = int(
                actor_cfg.get(
                    "fixed_statistics_microbatch_size",
                    max(1, num_candidates // data_parallel_devices),
                )
            )
            if fixed_statistics_microbatch_size <= 0:
                raise ValueError("fixed_statistics_microbatch_size must be positive")
            old_statistics_parts: list[tuple[Any, Any, Any]] = []
            reference_statistics_parts: list[tuple[Any, Any, Any]] = []
            for start in range(0, num_candidates, fixed_statistics_microbatch_size):
                stop = min(start + fixed_statistics_microbatch_size, num_candidates)
                observation_part = _slice_jax_batch(jax_observation_g, start, stop)
                old_statistics_parts.append(
                    distributed_old_statistics(
                        state.old_policy.actor_state,
                        old_x_prev[start:stop],
                        old_x_t[start:stop],
                        observation_part,
                        old_timestep[start:stop],
                    )
                )
                reference_statistics_parts.append(
                    distributed_reference_statistics(
                        state.reference_policy.actor_state,
                        old_x_prev[start:stop],
                        old_x_t[start:stop],
                        observation_part,
                        old_timestep[start:stop],
                    )
                )
            # These outputs are tiny compared with PI0.5 parameters and
            # gradients. Concatenate on host to avoid JAX 0.5.3 reusing a
            # same-shape concatenate executable compiled for another GPU.
            # The two frozen forwards and success BC still execute in parallel.
            jax.effects_barrier()
            old_statistics_host = tuple(
                np.concatenate(
                    [np.asarray(jax.device_get(part[index])) for part in old_statistics_parts],
                    axis=0,
                )
                for index in range(3)
            )
            reference_statistics_host = tuple(
                np.concatenate(
                    [
                        np.asarray(jax.device_get(part[index]))
                        for part in reference_statistics_parts
                    ],
                    axis=0,
                )
                for index in range(3)
            )
            old_log_prob_host, old_mean_host, old_log_std_host = old_statistics_host
            _, reference_mean_host, reference_log_std_host = reference_statistics_host
            fixed_old_log_prob, fixed_old_mean, fixed_old_log_std = (
                _shard_jax_batch(jax.device_put(value, actor_devices[0]), data_parallel_devices)
                for value in (old_log_prob_host, old_mean_host, old_log_std_host)
            )
            fixed_reference_mean, fixed_reference_log_std = (
                _shard_jax_batch(jax.device_put(value, actor_devices[0]), data_parallel_devices)
                for value in (reference_mean_host, reference_log_std_host)
            )
            del (
                old_statistics_parts,
                reference_statistics_parts,
                old_statistics_host,
                reference_statistics_host,
            )
            jax.effects_barrier()

            # Do not carry a full auxiliary gradient on its worker GPU into
            # pmap. Move it to the optimizer device and release the source
            # tree before the distributed PPO backward starts.
            for name, (component_loss, component_grads) in tuple(
                precomputed_regularization.items()
            ):
                component_grads_on_optimizer = _jax_tree_copy_to_device(
                    component_grads,
                    actor_devices[0],
                )
                precomputed_regularization[name] = (
                    jax.device_put(component_loss, actor_devices[0]),
                    component_grads_on_optimizer,
                )
                del component_grads
            jax.effects_barrier()
            gc.collect()

        # Release PyTorch's critic cache so JAX can use its full memory cap for
        # the full-finetune value_and_grad backward through PI0.5.
        torch.cuda.empty_cache()
        if data_parallel_devices > 1:
            log_prob_calibration = jnp.zeros_like(fixed_old_log_prob)
            mean_calibration = jnp.zeros_like(fixed_old_mean)
            log_std_calibration = jnp.zeros_like(fixed_old_log_std)
        else:
            log_prob_calibration = None
            mean_calibration = None
            log_std_calibration = None
        post_update_kl_total = 0.0
        post_update_full_kl_total = 0.0
        rejected_updates = 0
        accepted_updates = 0
        attempted_epochs = 0
        success_grad_norm_total = 0.0
        success_grad_scale_total = 0.0
        fm_loss_total = 0.0
        success_loss_total = 0.0
        actor_grad_norms: list[float] = []
        epoch_metrics: dict[str, float] = {}
        final_post_update_kl: float | None = None
        final_post_update_full_kl: float | None = None
        final_post_update_old_policy_kl = 0.0
        ratio = np.ones(num_candidates, dtype=np.float32)
        clipped_ratio = ratio.copy()
        for epoch_index in range(actor_epochs):
            attempted_epochs += 1
            anchor_on_policy_statistics = float(
                stabilize_on_policy_statistics and epoch_index == 0
            )
            host_grads = None
            device_grads = None
            policy_loss_value = 0.0
            flash_loss_value = 0.0
            raw_flash_loss_value = 0.0
            reference_kl_value = 0.0
            beta_value = beta_base
            ratio_parts = []
            clipped_ratio_parts = []
            per_sample_loss_parts = []
            preupdate_mean_diff_parts = []
            preupdate_log_prob_diff_parts = []
            surrogate_log_prob_diff_parts = []
            if data_parallel_devices > 1:
                assert distributed_policy_loss_and_grad is not None
                (policy_loss_devices, aux_devices), policy_grads = (
                    distributed_policy_loss_and_grad(
                        state.policy.actor_state,
                        _shard_jax_batch(old_x_prev, data_parallel_devices),
                        _shard_jax_batch(old_x_t, data_parallel_devices),
                        _shard_jax_batch(jax_observation_g, data_parallel_devices),
                        _shard_jax_batch(old_timestep, data_parallel_devices),
                        _shard_jax_batch(advantages_j, data_parallel_devices),
                        _shard_jax_batch(eps_clip_j, data_parallel_devices),
                        _shard_jax_batch(rectification_j, data_parallel_devices),
                        _shard_jax_batch(entropy_norm_j, data_parallel_devices),
                        fixed_old_log_prob,
                        fixed_old_mean,
                        fixed_old_log_std,
                        fixed_reference_mean,
                        fixed_reference_log_std,
                        log_prob_calibration,
                        mean_calibration,
                        log_std_calibration,
                        anchor_on_policy_statistics,
                    )
                )
                if stabilize_on_policy_statistics and epoch_index == 0:
                    log_prob_calibration = jax.lax.stop_gradient(
                        aux_devices["next_log_prob_calibration"]
                    )
                    mean_calibration = jax.lax.stop_gradient(
                        aux_devices["next_mean_calibration"]
                    )
                    log_std_calibration = jax.lax.stop_gradient(
                        aux_devices["next_log_std_calibration"]
                    )
                if distributed_gradient_reduction == "pmean":
                    # pmean produced the same average on every replica. Copy
                    # replica zero into an independently owned optimizer tree
                    # so the remaining seven output shards can be released.
                    device_grads = _jax_tree_copy_to_device(
                        _first_pmap_replica(policy_grads),
                        actor_devices[0],
                    )
                else:
                    device_grads = _mean_pmap_gradients_on_device(
                        policy_grads,
                        actor_devices[0],
                    )
                del policy_grads
                policy_loss_value = float(np.asarray(policy_loss_devices).mean())
                flash_loss_value = float(np.asarray(aux_devices["flash_loss"]).mean())
                raw_flash_loss_value = float(
                    np.asarray(aux_devices["raw_flash_loss"]).mean()
                )
                reference_kl_value = float(np.asarray(aux_devices["ref_kl"]).mean())
                beta_value = float(np.asarray(aux_devices["beta"]).mean())
                ratio_parts.append(np.asarray(aux_devices["ratio"]).reshape(-1))
                clipped_ratio_parts.append(
                    np.asarray(aux_devices["clipped_ratio"]).reshape(-1)
                )
                per_sample_loss_parts.append(
                    np.asarray(aux_devices["per_sample_loss"]).reshape(-1)
                )
                preupdate_mean_diff_parts.append(
                    np.asarray(aux_devices["preupdate_mean_abs_diff"]).reshape(-1)
                )
                preupdate_log_prob_diff_parts.append(
                    np.asarray(aux_devices["preupdate_log_prob_abs_diff"]).reshape(-1)
                )
                surrogate_log_prob_diff_parts.append(
                    np.asarray(aux_devices["surrogate_log_prob_abs_diff"]).reshape(-1)
                )
                jax.effects_barrier()
                gc.collect()
            else:
                for start in range(0, num_candidates, gradient_microbatch_size):
                    stop = min(start + gradient_microbatch_size, num_candidates)
                    sample_weight = (stop - start) / num_candidates
                    observation_mb = _slice_jax_batch(jax_observation_g, start, stop)
                    (policy_loss_mb, aux_mb), policy_grads = jax.value_and_grad(
                        _policy_loss_fn,
                        has_aux=True,
                    )(
                        state.policy.actor_state,
                        state.old_policy.actor_state,
                        state.reference_policy.actor_state,
                        old_x_prev[start:stop],
                        old_x_t[start:stop],
                        observation_mb,
                        old_timestep[start:stop],
                        advantages_j[start:stop],
                        eps_clip_j[start:stop],
                        rectification_j[start:stop],
                        entropy_norm_j[start:stop],
                        fixed_old_log_prob,
                        fixed_old_mean,
                        fixed_old_log_std,
                        fixed_reference_mean,
                        fixed_reference_log_std,
                        anchor_on_policy_statistics,
                    )
                    host_grads = _accumulate_jax_grads_on_host(
                        host_grads,
                        policy_grads,
                        weight=sample_weight,
                    )
                    del policy_grads
                    policy_loss_value += sample_weight * float(policy_loss_mb)
                    flash_loss_value += sample_weight * float(aux_mb["flash_loss"])
                    raw_flash_loss_value += sample_weight * float(aux_mb["raw_flash_loss"])
                    reference_kl_value += sample_weight * float(aux_mb["ref_kl"])
                    beta_value = float(aux_mb["beta"])
                    ratio_parts.append(np.asarray(aux_mb["ratio"]))
                    clipped_ratio_parts.append(np.asarray(aux_mb["clipped_ratio"]))
                    per_sample_loss_parts.append(np.asarray(aux_mb["per_sample_loss"]))
                    jax.effects_barrier()
                    gc.collect()

            fm_loss = jnp.asarray(0.0, dtype=jnp.float32)
            success_loss = jnp.asarray(0.0, dtype=jnp.float32)
            success_grad_norm = 0.0
            success_grad_scale = 0.0
            for name in ("fm", "success") if epoch_index < regularization_epochs else ():
                inputs = jax_regularization[name]
                weight = float(jax_regularization[f"lambda_{name}"])
                if inputs is None or weight == 0.0:
                    continue
                if data_parallel_devices > 1:
                    # Pop before donating component_grads to the fused tree
                    # addition. Keeping the tuple in the dict prevents XLA
                    # from reusing a multi-GiB gradient buffer.
                    component_loss, component_grads = precomputed_regularization.pop(name)
                else:
                    component_loss, component_grads = jax.value_and_grad(
                        _flow_matching_component,
                    )(state.policy.actor_state, inputs)
                effective_weight = weight
                if name == "success":
                    success_grad_scale = 1.0
                    success_grad_norm = (
                        float(_jax_tree_l2_norm_device(component_grads))
                        if data_parallel_devices > 1
                        else _jax_tree_l2_norm(component_grads)
                    )
                    weighted_norm = abs(weight) * success_grad_norm
                    max_norm = float(
                        regularization_cfg.get(
                            "success_weighted_grad_max_norm",
                            float("inf"),
                        )
                    )
                    if weighted_norm > max_norm:
                        success_grad_scale = max_norm / max(weighted_norm, 1e-12)
                        effective_weight *= success_grad_scale
                if data_parallel_devices > 1:
                    assert device_grads is not None
                    device_grads = _jax_tree_add_scaled(
                        device_grads,
                        component_grads,
                        other_weight=effective_weight,
                    )
                else:
                    host_grads = _accumulate_jax_grads_on_host(
                        host_grads,
                        component_grads,
                        weight=effective_weight,
                    )
                if name == "fm":
                    fm_loss = component_loss
                else:
                    success_loss = component_loss
                del component_grads
                jax.effects_barrier()
                gc.collect()

            fm_loss_total += float(fm_loss)
            success_loss_total += float(success_loss)

            total_loss = (
                policy_loss_value
                + float(jax_regularization["lambda_fm"]) * fm_loss
                + float(jax_regularization["lambda_success"]) * success_loss
            )
            actor_grad_norm = (
                float(_jax_tree_l2_norm_device(device_grads))
                if data_parallel_devices > 1
                else _jax_tree_l2_norm(host_grads)
            )
            actor_grad_norms.append(actor_grad_norm)
            actor_state_before = state.policy.actor_state
            optimizer_state_before = state.policy.actor_opt_state
            if data_parallel_devices > 1:
                grads = device_grads
                device_grads = None
            else:
                grads = jax.device_put(host_grads)
                del host_grads
            state.policy.apply_actor_gradients(grads)
            del grads
            jax.effects_barrier()
            gc.collect()

            if data_parallel_devices > 1 and stabilize_on_policy_statistics:
                assert distributed_old_statistics is not None
                post_statistics_parts = []
                for start in range(0, num_candidates, fixed_statistics_microbatch_size):
                    stop = min(start + fixed_statistics_microbatch_size, num_candidates)
                    post_statistics_parts.append(
                        distributed_old_statistics(
                            state.policy.actor_state,
                            old_x_prev[start:stop],
                            old_x_t[start:stop],
                            _slice_jax_batch(jax_observation_g, start, stop),
                            old_timestep[start:stop],
                        )
                    )
                jax.effects_barrier()
                post_mean_host = np.concatenate(
                    [
                        np.asarray(jax.device_get(part[1]))
                        for part in post_statistics_parts
                    ],
                    axis=0,
                )
                post_log_std_host = np.concatenate(
                    [
                        np.asarray(jax.device_get(part[2]))
                        for part in post_statistics_parts
                    ],
                    axis=0,
                )
                post_update_kl = float(
                    _numpy_gaussian_kl_diag(
                        post_mean_host,
                        post_log_std_host,
                        reference_mean_host,
                        reference_log_std_host,
                        event_dim=reference_kl_event_dim,
                    ).mean()
                )
                post_update_full_kl = float(
                    _numpy_gaussian_kl_diag(
                        post_mean_host,
                        post_log_std_host,
                        reference_mean_host,
                        reference_log_std_host,
                    ).mean()
                )
                post_update_old_policy_kl = float(
                    _numpy_gaussian_kl_diag(
                        post_mean_host,
                        post_log_std_host,
                        old_mean_host,
                        old_log_std_host,
                        event_dim=ppo_event_dim,
                    ).mean()
                )
                del post_statistics_parts, post_mean_host, post_log_std_host
            elif data_parallel_devices > 1:
                assert distributed_reference_kl is not None
                (
                    post_update_kl_devices,
                    post_update_full_kl_devices,
                    post_update_old_policy_kl_devices,
                ) = (
                    distributed_reference_kl(
                        state.policy.actor_state,
                        _shard_jax_batch(old_x_t, data_parallel_devices),
                        _shard_jax_batch(jax_observation_g, data_parallel_devices),
                        _shard_jax_batch(old_timestep, data_parallel_devices),
                        fixed_old_mean,
                        fixed_old_log_std,
                        fixed_reference_mean,
                        fixed_reference_log_std,
                    )
                )
                post_update_kl = float(np.asarray(post_update_kl_devices).mean())
                post_update_full_kl = float(
                    np.asarray(post_update_full_kl_devices).mean()
                )
                post_update_old_policy_kl = float(
                    np.asarray(post_update_old_policy_kl_devices).mean()
                )
            else:
                assert reference_kl_eval is not None
                post_update_kl = 0.0
                post_update_full_kl = 0.0
                post_update_old_policy_kl = 0.0
                for start in range(0, num_candidates, kl_eval_microbatch_size):
                    stop = min(start + kl_eval_microbatch_size, num_candidates)
                    sample_weight = (stop - start) / num_candidates
                    (
                        post_update_kl_part,
                        post_update_full_kl_part,
                        post_update_old_policy_kl_part,
                    ) = (
                        reference_kl_eval(
                            state.policy.actor_state,
                            state.old_policy.actor_state,
                            state.reference_policy.actor_state,
                            old_x_t[start:stop],
                            _slice_jax_batch(jax_observation_g, start, stop),
                            old_timestep[start:stop],
                        )
                    )
                    post_update_kl += sample_weight * float(post_update_kl_part)
                    post_update_full_kl += sample_weight * float(
                        post_update_full_kl_part
                    )
                    post_update_old_policy_kl += sample_weight * float(
                        post_update_old_policy_kl_part
                    )
            reject_update = bool(actor_cfg.get("reject_update_on_kl", False)) and (
                not math.isfinite(post_update_kl)
                or post_update_kl
                > float(actor_cfg.get("max_policy_reference_kl", float("inf")))
            )
            if reject_update:
                # Keep the accepted state live while explicitly dropping the
                # rejected full-model and Adafactor pytrees. Without this,
                # their multi-GiB JAX buffers can survive until a later Python
                # collection and overlap with the next PI0.5 backward.
                rejected_actor_state = state.policy.actor_state
                rejected_optimizer_state = state.policy.actor_opt_state
                state.policy.actor_state = actor_state_before
                state.policy.actor_opt_state = optimizer_state_before
                state.policy._sync_torch_adapter_from_jax()
                rejected_updates += 1
                del rejected_actor_state, rejected_optimizer_state
                jax.effects_barrier()
                if bool(actor_cfg.get("jax_clear_caches_after_kl_rejection", True)):
                    jax.clear_caches()
                gc.collect()
            else:
                # The accepted proposal is now authoritative. Release the
                # rollback-only references before the next actor iteration.
                del actor_state_before, optimizer_state_before
                accepted_updates += 1
                final_post_update_kl = post_update_kl
                final_post_update_full_kl = post_update_full_kl
                final_post_update_old_policy_kl = post_update_old_policy_kl
                gc.collect()

            loss_total += float(total_loss)
            flash_total += flash_loss_value
            raw_flash_total += raw_flash_loss_value
            kl_total += reference_kl_value
            post_update_kl_total += post_update_kl
            post_update_full_kl_total += post_update_full_kl
            success_grad_norm_total += success_grad_norm
            success_grad_scale_total += success_grad_scale
            rectification_total += float(rectification.mean().item())
            ratio = np.concatenate(ratio_parts)
            clipped_ratio = np.concatenate(clipped_ratio_parts)
            per_sample_loss = np.concatenate(per_sample_loss_parts)
            epoch_prefix = f"ppo_epoch_{epoch_index + 1}"
            epoch_metrics.update(
                {
                    f"{epoch_prefix}_importance_ratio_mean": float(ratio.mean()),
                    f"{epoch_prefix}_importance_ratio_std": float(ratio.std()),
                    f"{epoch_prefix}_importance_ratio_min": float(ratio.min()),
                    f"{epoch_prefix}_importance_ratio_max": float(ratio.max()),
                    f"{epoch_prefix}_clip_fraction": float(
                        (ratio != clipped_ratio).mean()
                    ),
                    f"{epoch_prefix}_actor_grad_norm": actor_grad_norm,
                    f"{epoch_prefix}_actor_grad_clip_scale": min(
                        1.0,
                        float(actor_cfg.get("max_grad_norm", 1.0))
                        / max(actor_grad_norm, 1e-12),
                    ),
                    f"{epoch_prefix}_post_reference_kl": post_update_kl,
                    f"{epoch_prefix}_post_full_reference_kl": post_update_full_kl,
                    f"{epoch_prefix}_post_old_policy_kl": post_update_old_policy_kl,
                    f"{epoch_prefix}_accepted": float(not reject_update),
                    f"{epoch_prefix}_on_policy_value_anchor": anchor_on_policy_statistics,
                }
            )
            for step_idx, count in enumerate(selected_counts.tolist()):
                if not count:
                    continue
                step_mask = selected_steps_grouped == step_idx
                step_raw_loss = float(
                    per_sample_loss[step_mask.detach().cpu().numpy()].mean()
                )
                raw_loss_by_step[step_idx] += step_raw_loss
                raw_grad_by_step[step_idx] += actor_grad_norm
                rectified_grad_by_step[step_idx] += actor_grad_norm * float(rectification[step_mask].mean().item())
                if str(flow_cfg.get("temporal_rectification_mode", "analytic")) == "empirical_ema":
                    state.rectifier.update(step_idx, actor_grad_norm, count=count)
            if reject_update:
                # PPO epochs form one transaction over fixed rollout data. If
                # an epoch is rejected, later epochs must not build on a state
                # that the optimizer never accepted.
                break

        # The precomputed success/FM entries retain a full PI0.5 gradient tree.
        # Drop those references before returning so a following distributed
        # backward does not overlap with an auxiliary gradient from this step.
        precomputed_regularization.clear()
        jax.effects_barrier()
        gc.collect()
        if final_post_update_kl is None:
            if data_parallel_devices > 1:
                final_post_update_kl = float(
                    _numpy_gaussian_kl_diag(
                        old_mean_host,
                        old_log_std_host,
                        reference_mean_host,
                        reference_log_std_host,
                        event_dim=reference_kl_event_dim,
                    ).mean()
                )
                final_post_update_full_kl = float(
                    _numpy_gaussian_kl_diag(
                        old_mean_host,
                        old_log_std_host,
                        reference_mean_host,
                        reference_log_std_host,
                    ).mean()
                )
            else:
                assert reference_kl_eval is not None
                final_post_update_kl = 0.0
                final_post_update_full_kl = 0.0
                for start in range(0, num_candidates, kl_eval_microbatch_size):
                    stop = min(start + kl_eval_microbatch_size, num_candidates)
                    sample_weight = (stop - start) / num_candidates
                    final_kl_part, final_full_kl_part, _ = reference_kl_eval(
                        state.policy.actor_state,
                        state.old_policy.actor_state,
                        state.reference_policy.actor_state,
                        old_x_t[start:stop],
                        _slice_jax_batch(jax_observation_g, start, stop),
                        old_timestep[start:stop],
                    )
                    final_post_update_kl += sample_weight * float(final_kl_part)
                    final_post_update_full_kl += sample_weight * float(
                        final_full_kl_part
                    )
        assert final_post_update_full_kl is not None
        epoch_divisor = max(1, attempted_epochs)
        metrics = {
            "actor_loss": loss_total / epoch_divisor,
            "flash_ppo_loss": flash_total / epoch_divisor,
            "flash_raw_ppo_loss": raw_flash_total / epoch_divisor,
            "reference_kl": kl_total / epoch_divisor,
            "selected_step_kl": kl_total / epoch_divisor,
            "post_update_reference_kl": final_post_update_kl,
            "post_update_full_reference_kl": final_post_update_full_kl,
            "post_update_old_policy_kl": final_post_update_old_policy_kl,
            "reference_kl_action_horizon": float(reference_kl_action_horizon),
            "reference_kl_event_dim": float(reference_kl_event_dim),
            "reference_kl_full_event_dim": float(full_reference_kl_event_dim),
            "reference_kl_uses_action_prefix": float(
                reference_kl_action_horizon < state.policy.model_horizon
            ),
            "ppo_action_horizon": float(ppo_action_horizon),
            "ppo_event_dim": float(ppo_event_dim),
            "ppo_uses_action_prefix": float(
                ppo_action_horizon < state.policy.model_horizon
            ),
            "actor_update_rejected": float(accepted_updates == 0),
            "actor_update_accepted": float(accepted_updates > 0),
            "actor_update_partially_rejected": float(
                accepted_updates > 0 and rejected_updates > 0
            ),
            "jax_rejection_cleanup_applied": float(rejected_updates > 0),
            "rejected_update_count": float(rejected_updates),
            "accepted_actor_epochs": float(accepted_updates),
            "attempted_actor_epochs": float(attempted_epochs),
            "actor_sampling_seed": float(sampling_seed),
            "normalize_logprob_by_action_dim": float(normalize_logprob_by_action_dim),
            "logprob_normalizer": logprob_normalizer,
            "reference_kl_beta": beta_value,
            "ustate_adapt_ppo_clip": float(bool(uncertainty_cfg.get("adapt_ppo_clip", False))),
            "ustate_adapt_kl_beta": float(bool(uncertainty_cfg.get("adapt_kl_beta", False))),
            "actor_epochs": float(actor_epochs),
            "actor_regularization_epochs_per_rollout": float(regularization_epochs),
            "selected_step": float(selected_steps.float().mean().item()),
            "selected_step_min": float(selected_steps.min().item()),
            "selected_step_max": float(selected_steps.max().item()),
            "rectification_weight": rectification_total / epoch_divisor,
            "rectification_weight_min": float(rectification.min().item()),
            "rectification_weight_max": float(rectification.max().item()),
            "rectification_weight_std": float(rectification.float().std(unbiased=False).item()),
            "importance_ratio_mean": float(ratio.mean()),
            "importance_ratio_std": float(ratio.std()),
            "importance_ratio_min": float(ratio.min()),
            "importance_ratio_max": float(ratio.max()),
            "preupdate_transition_mean_abs_diff": float(np.mean(np.concatenate(preupdate_mean_diff_parts))) if preupdate_mean_diff_parts else 0.0,
            "preupdate_log_prob_abs_diff": float(np.mean(np.concatenate(preupdate_log_prob_diff_parts))) if preupdate_log_prob_diff_parts else 0.0,
            "surrogate_log_prob_abs_diff": float(
                np.mean(np.concatenate(surrogate_log_prob_diff_parts))
            ) if surrogate_log_prob_diff_parts else 0.0,
            "single_epoch_bf16_statistics_stabilized": float(
                stabilize_on_policy_statistics
            ),
            "on_policy_bf16_statistics_stabilized": float(
                stabilize_on_policy_statistics
            ),
            "ppo_clip_fraction": float((ratio != clipped_ratio).mean()),
            "actor_grad_norm": actor_grad_norm,
            "actor_grad_norm_mean": float(np.mean(actor_grad_norms)),
            "actor_grad_clip_scale": min(
                1.0,
                float(actor_cfg.get("max_grad_norm", 1.0)) / max(actor_grad_norm, 1e-12),
            ),
            "actor_grad_was_clipped": float(
                actor_grad_norm > float(actor_cfg.get("max_grad_norm", 1.0))
            ),
            "success_grad_norm": success_grad_norm_total,
            "success_grad_scale": success_grad_scale_total,
            "success_update_applied": float(success_update_due),
            "success_update_period": float(success_update_period),
            "candidate_group_size": float(group_size),
            "gradient_microbatch_size": float(gradient_microbatch_size),
            "rollout_state_microbatch_size": float(rollout_state_microbatch_size),
            "kl_eval_microbatch_size": float(kl_eval_microbatch_size),
            "actor_data_parallel_devices": float(data_parallel_devices),
            "distributed_gradient_reduction_pmean": float(
                distributed_gradient_reduction == "pmean"
            ),
            "parallel_frozen_statistics": float(parallel_frozen_statistics),
            "old_statistics_device_index": float(old_statistics_device_index),
            "reference_statistics_device_index": float(reference_statistics_device_index),
            "regularization_device_index": float(regularization_device_index),
            "actor_candidates_per_device": float(num_candidates / data_parallel_devices),
            "distributed_backward_calls": float(
                attempted_epochs
                * (
                    1
                    if data_parallel_devices > 1
                    else math.ceil(num_candidates / gradient_microbatch_size)
                )
            ),
            "old_policy_lag": _policy_l2_lag(state.policy, state.old_policy),
            "fm_anchor_loss": fm_loss_total,
            "success_buffer_loss": success_loss_total,
            "action_smoothness": 0.0,
            **epoch_metrics,
            **adv_diag,
        }
        post_update_kl_mean = final_post_update_kl
        kl_target = float(actor_cfg.get("target_policy_reference_kl", float("inf")))
        kl_hard_limit = float(actor_cfg.get("max_policy_reference_kl", float("inf")))
        metrics.update(
            {
                "policy_reference_kl_target": kl_target,
                "policy_reference_kl_hard_limit": kl_hard_limit,
                "policy_reference_kl_target_exceeded": float(post_update_kl_mean > kl_target),
                "policy_reference_kl_hard_limit_exceeded": float(
                    post_update_kl_mean > kl_hard_limit
                ),
                "policy_reference_kl_nonfinite": float(not math.isfinite(post_update_kl_mean)),
                "policy_reference_kl_utilization": (
                    post_update_kl_mean / kl_hard_limit
                    if math.isfinite(kl_hard_limit) and kl_hard_limit > 0.0
                    else 0.0
                ),
            }
        )
        for step_idx, count in enumerate(selected_counts.tolist()):
            metrics[f"selected_step_count_{step_idx}"] = float(count)
            metrics[f"rectifier_count_{step_idx}"] = float(state.rectifier.counts[step_idx].item())
            metrics[f"rectifier_grad_ema_{step_idx}"] = float(state.rectifier.grad_ema[step_idx].item())
            metrics[f"flash_raw_loss_step_{step_idx}"] = raw_loss_by_step[step_idx] / epoch_divisor
            metrics[f"flash_raw_grad_norm_step_{step_idx}"] = raw_grad_by_step[step_idx] / epoch_divisor
            metrics[f"flash_rectified_grad_norm_step_{step_idx}"] = rectified_grad_by_step[step_idx] / epoch_divisor
        return metrics

    with torch.no_grad():
        old_rollout = sample_flash_rollout(
            state.old_policy,
            condition,
            group_size=group_size,
            selected_step=selected_steps,
        )
        environment_endpoint = state.old_policy.flat_actions_to_environment(old_rollout.endpoint, condition_g)
        endpoint = environment_endpoint.reshape(batch.batch_size, group_size, -1)
        advantages, adv_diag = conservative_advantages_for_candidates(
            state, batch.observations, endpoint, batch, config
        )
        entropy_norm = _normalized_state_entropy(state, batch).mean(dim=0)
        eps_clip = (
            actor_clip_for_uncertainty(entropy_norm, actor_cfg, uncertainty_cfg)
            .unsqueeze(-1)
            .expand(batch.batch_size, group_size)
            .reshape(-1)
        )
    selected_counts = torch.bincount(selected_steps.detach().cpu(), minlength=state.policy.num_steps)
    loss_total = 0.0
    flash_total = 0.0
    raw_flash_total = 0.0
    kl_total = 0.0
    grad_total = 0.0
    rectification_total = 0.0
    raw_loss_by_step = [0.0] * state.policy.num_steps
    raw_grad_by_step = [0.0] * state.policy.num_steps
    rectified_grad_by_step = [0.0] * state.policy.num_steps
    selected_steps_grouped = selected_steps.repeat_interleave(group_size)
    trainable_actor_parameters = [parameter for parameter in state.policy.parameters() if parameter.requires_grad]
    for _ in range(actor_epochs):
        state.actor_optimizer.zero_grad(set_to_none=True)
        new_log_prob = state.policy.log_prob(
            old_rollout.x_prev,
            old_rollout.x_t,
            condition_g,
            old_rollout.timestep,
        )
        rectification = _flash_rectification_weight(
            state,
            flow_cfg,
            timestep=old_rollout.timestep,
            selected_steps=selected_steps,
            group_size=group_size,
        )
        raw_flash = flash_ppo_loss(
            new_log_prob,
            old_rollout.old_log_prob,
            advantages.reshape(-1),
            clip_eps=eps_clip,
            rectification_weight=1.0,
        )
        flash = flash_ppo_loss(
            new_log_prob,
            old_rollout.old_log_prob,
            advantages.reshape(-1),
            clip_eps=eps_clip,
            rectification_weight=rectification,
        )
        for step_idx, count in enumerate(selected_counts.tolist()):
            if not count:
                continue
            step_mask = selected_steps_grouped == step_idx
            step_raw_loss = raw_flash.per_sample_loss[step_mask].mean()
            step_gradients = torch.autograd.grad(
                step_raw_loss,
                trainable_actor_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            finite_gradients = [gradient.detach().norm(2) for gradient in step_gradients if gradient is not None]
            raw_grad = (
                float(torch.norm(torch.stack(finite_gradients), 2).item())
                if finite_gradients
                else 0.0
            )
            step_rectification = float(rectification[step_mask].detach().mean().item())
            raw_loss_by_step[step_idx] += float(step_raw_loss.detach().item())
            raw_grad_by_step[step_idx] += raw_grad
            rectified_grad_by_step[step_idx] += raw_grad * step_rectification
            if str(flow_cfg.get("temporal_rectification_mode", "analytic")) == "empirical_ema":
                state.rectifier.update(step_idx, raw_grad, count=count)
        transition_ref_kl = state.policy.kl_to(
            state.reference_policy,
            old_rollout.x_t,
            condition_g,
            old_rollout.timestep,
        )
        kl_penalty, ref_kl, kl_beta = state_adaptive_kl_penalty(
            transition_ref_kl,
            entropy_norm,
            group_size=group_size,
            beta_base=float(regularization_cfg.get("beta_kl", 0.01)),
            uncertainty_scale=kl_uncertainty_scale(regularization_cfg, uncertainty_cfg),
        )
        reg_loss, reg_diag = _actor_regularization_loss(
            state,
            batch,
            config,
            fm_batch=fm_batch,
            success_batch=success_batch,
        )
        loss = flash.loss + reg_loss + kl_penalty
        loss.backward()
        assert_no_gradients(state.critic, "critic")
        assert_no_gradients(state.reference_policy, "reference_policy")
        actor_grad_norm = float(grad_norm(state.policy.parameters()))
        torch.nn.utils.clip_grad_norm_(state.policy.parameters(), float(actor_cfg.get("max_grad_norm", 1.0)))
        state.actor_optimizer.step()
        loss_total += float(loss.detach().item())
        flash_total += float(flash.loss.detach().item())
        raw_flash_total += float(raw_flash.loss.detach().item())
        kl_total += float(ref_kl.detach().item())
        grad_total += actor_grad_norm
        rectification_total += float(rectification.detach().mean().item())
    metrics = {
        "actor_loss": loss_total / actor_epochs,
        "flash_ppo_loss": flash_total / actor_epochs,
        "flash_raw_ppo_loss": raw_flash_total / actor_epochs,
        "reference_kl": kl_total / actor_epochs,
        "selected_step_kl": kl_total / actor_epochs,
        "reference_kl_beta": float(kl_beta.detach().item()),
        "ustate_adapt_ppo_clip": float(bool(uncertainty_cfg.get("adapt_ppo_clip", False))),
        "ustate_adapt_kl_beta": float(bool(uncertainty_cfg.get("adapt_kl_beta", False))),
        "actor_epochs": float(actor_epochs),
        "selected_step": float(selected_steps.float().mean().item()),
        "selected_step_min": float(selected_steps.min().item()),
        "selected_step_max": float(selected_steps.max().item()),
        "rectification_weight": rectification_total / actor_epochs,
        "rectification_weight_min": float(rectification.min().item()),
        "rectification_weight_max": float(rectification.max().item()),
        "rectification_weight_std": float(rectification.float().std(unbiased=False).item()),
        "importance_ratio_mean": flash.ratio_mean,
        "importance_ratio_std": flash.ratio_std,
        "importance_ratio_min": flash.ratio_min,
        "importance_ratio_max": flash.ratio_max,
        "ppo_clip_fraction": flash.clip_fraction,
        "actor_grad_norm": grad_total / actor_epochs,
        "old_policy_lag": _policy_l2_lag(state.policy, state.old_policy),
        **reg_diag,
        **adv_diag,
    }
    for step_idx, count in enumerate(selected_counts.tolist()):
        metrics[f"selected_step_count_{step_idx}"] = float(count)
        metrics[f"rectifier_count_{step_idx}"] = float(state.rectifier.counts[step_idx].item())
        metrics[f"rectifier_grad_ema_{step_idx}"] = float(state.rectifier.grad_ema[step_idx].item())
        metrics[f"flash_raw_loss_step_{step_idx}"] = raw_loss_by_step[step_idx] / actor_epochs
        metrics[f"flash_raw_grad_norm_step_{step_idx}"] = raw_grad_by_step[step_idx] / actor_epochs
        metrics[f"flash_rectified_grad_norm_step_{step_idx}"] = (
            rectified_grad_by_step[step_idx] / actor_epochs
        )
    return metrics


@torch.no_grad()
def sync_old_policy(state: OGPOTrainState, *, ema: float = 0.0) -> None:
    if not 0.0 <= float(ema) < 1.0:
        raise ValueError("old-policy EMA must be in [0, 1)")
    if float(ema) == 0.0:
        if isinstance(state.policy, PI05JaxFlowPolicy):
            assert isinstance(state.old_policy, PI05JaxFlowPolicy)
            state.old_policy.actor_state = jax.tree.map(lambda value: value, state.policy.actor_state)
            state.old_policy._sync_torch_adapter_from_jax()
        elif isinstance(state.policy, PI05PytorchFlowPolicy):
            assert isinstance(state.old_policy, PI05PytorchFlowPolicy)
            state.old_policy.load_adapter_state_dict(state.policy.adapter_state_dict())
        else:
            state.old_policy.load_state_dict(state.policy.state_dict())
        return
    if isinstance(state.policy, PI05JaxFlowPolicy):
        assert isinstance(state.old_policy, PI05JaxFlowPolicy)
        mixed_state = ema_actor_state(
            state.old_policy.actor_state,
            state.policy.actor_state,
            ema=float(ema),
        )
        state.old_policy._replace_actor_state(mixed_state)
        return
    for old_param, param in zip(state.old_policy.parameters(), state.policy.parameters(), strict=True):
        if not param.requires_grad:
            continue
        old_param.mul_(float(ema)).add_(param.detach(), alpha=1.0 - float(ema))


def actor_guard_reason(metrics: dict[str, float], config: dict[str, Any]) -> str | None:
    actor_cfg = config.get("actor", {})
    reference_kl = float(metrics.get("reference_kl", 0.0))
    if (
        not bool(actor_cfg.get("reject_update_on_kl", False))
        and reference_kl > float(actor_cfg.get("max_policy_reference_kl", float("inf")))
    ):
        return "policy_reference_kl_exceeded"
    disagreement = float(metrics.get("candidate_ensemble_disagreement", 0.0))
    if disagreement > float(actor_cfg.get("max_critic_disagreement", float("inf"))):
        return "critic_disagreement_exceeded"
    support_distance = float(metrics.get("support_distance_mean", 0.0))
    if support_distance > float(actor_cfg.get("max_support_distance", float("inf"))):
        return "support_distance_exceeded"
    if float(metrics.get("consecutive_kl_rejections", 0.0)) >= float(
        actor_cfg.get("max_consecutive_kl_rejections", float("inf"))
    ):
        return "repeated_policy_reference_kl_rejections"
    for key in ("actor_loss", "importance_ratio_mean", "importance_ratio_std"):
        value = torch.tensor(float(metrics.get(key, 0.0)))
        if not torch.isfinite(value):
            return f"nonfinite_{key}"
    return None


def actor_delay_active(step: int, config: dict[str, Any]) -> bool:
    return int(step) < int(config.get("actor", {}).get("actor_delay", 0))


def actor_start_gate(
    state: OGPOTrainState,
    validation_batch: ChunkBatch,
    config: dict[str, Any],
    *,
    outer_step: int,
) -> tuple[str | None, dict[str, float]]:
    """Return a Phase-B gate reason and validation diagnostics."""
    if actor_delay_active(outer_step, config):
        return "actor_delay", {"critic_training_step": float(state.step)}
    critic_cfg = config.get("critic", {})
    if bool(critic_cfg.get("force_actor", False)):
        return None, {"critic_training_step": float(state.step), "actor_gate_forced": 1.0}
    if state.step < int(critic_cfg.get("warmup_steps", 0)):
        return "critic_warmup", {"critic_training_step": float(state.step)}

    from .evaluator import offline_calibration_metrics  # noqa: PLC0415

    metrics = offline_calibration_metrics(
        state.critic,
        validation_batch,
        divl=state.divl if bool(config.get("divl", {}).get("enabled", True)) else None,
        conformal_scale=state.conformal_scale,
        inference_batch_size=config.get("evaluation", {}).get(
            "actor_gate_inference_batch_size"
        ),
    )
    metrics["critic_training_step"] = float(state.step)
    metrics["critic_gate_sample_count"] = float(validation_batch.batch_size)
    metrics["critic_gate_forced"] = 0.0
    min_ranking = float(
        critic_cfg.get("min_ranking_accuracy", critic_cfg.get("critic_min_ranking_accuracy", float("-inf")))
    )
    metrics["critic_gate_min_ranking_accuracy"] = min_ranking
    metrics["critic_gate_ranking_margin"] = metrics["pairwise_ranking_accuracy"] - min_ranking
    if metrics["pairwise_ranking_accuracy"] < min_ranking:
        metrics["critic_gate_passed"] = 0.0
        return "critic_ranking_accuracy_below_min", metrics
    min_rank_correlation = float(critic_cfg.get("min_q_rank_correlation", float("-inf")))
    metrics["critic_gate_min_q_rank_correlation"] = min_rank_correlation
    metrics["critic_gate_rank_correlation_margin"] = (
        metrics["q_rank_correlation"] - min_rank_correlation
    )
    if metrics["q_rank_correlation"] < min_rank_correlation:
        metrics["critic_gate_passed"] = 0.0
        return "critic_rank_correlation_below_min", metrics
    max_exploitation_gap = float(
        critic_cfg.get("max_abs_q_exploitation_gap", float("inf"))
    )
    metrics["critic_gate_max_abs_q_exploitation_gap"] = max_exploitation_gap
    metrics["critic_gate_exploitation_gap_margin"] = (
        max_exploitation_gap - abs(metrics["q_exploitation_gap"])
    )
    if abs(metrics["q_exploitation_gap"]) > max_exploitation_gap:
        metrics["critic_gate_passed"] = 0.0
        return "critic_q_exploitation_gap_above_max", metrics
    min_coverage = float(critic_cfg.get("min_coverage", critic_cfg.get("critic_min_coverage", 0.0)))
    metrics["critic_gate_min_coverage"] = min_coverage
    metrics["critic_gate_coverage_margin"] = metrics["interval_coverage"] - min_coverage
    if metrics["interval_coverage"] < min_coverage:
        metrics["critic_gate_passed"] = 0.0
        return "critic_coverage_below_min", metrics
    entropy = metrics.get("categorical_entropy", 0.5)
    if entropy < float(critic_cfg.get("min_divl_entropy", 0.0)):
        metrics["critic_gate_passed"] = 0.0
        return "critic_divl_entropy_too_low", metrics
    if entropy > float(critic_cfg.get("max_divl_entropy", 1.0)):
        metrics["critic_gate_passed"] = 0.0
        return "critic_divl_entropy_too_high", metrics
    max_saturation = float(critic_cfg.get("max_categorical_saturation", float("inf")))
    metrics["critic_gate_max_categorical_saturation"] = max_saturation
    metrics["critic_gate_categorical_saturation_margin"] = (
        max_saturation - metrics.get("categorical_saturation", 0.0)
    )
    if metrics.get("categorical_saturation", 0.0) > max_saturation:
        metrics["critic_gate_passed"] = 0.0
        return "critic_categorical_saturation_above_max", metrics
    metrics["critic_gate_passed"] = 1.0
    return None, metrics


def _policy_checkpoint_state(
    policy: OpenPIStochasticFlowPolicy,
    *,
    jax_sidecar: str | None = None,
    jax_sidecar_has_old_policy: bool = True,
) -> dict[str, Any]:
    if isinstance(policy, PI05JaxFlowPolicy) and jax_sidecar is not None:
        return {
            "format": "pi05_jax_full_finetune",
            "state": policy.adapter_state_dict(),
            "jax_sidecar": jax_sidecar,
            "jax_sidecar_has_old_policy": jax_sidecar_has_old_policy,
        }
    if isinstance(policy, (PI05PytorchFlowPolicy, PI05JaxFlowPolicy)):
        return {"format": "pi05_residual_adapter", "state": policy.adapter_state_dict()}
    return {"format": "full_state_dict", "state": policy.state_dict()}


def _load_policy_checkpoint_state(policy: OpenPIStochasticFlowPolicy, payload: dict[str, Any]) -> None:
    if "format" not in payload:
        policy.load_state_dict(payload)
    elif payload["format"] in {"pi05_residual_adapter", "pi05_jax_full_finetune"}:
        if not isinstance(policy, (PI05PytorchFlowPolicy, PI05JaxFlowPolicy)):
            raise TypeError("PI0.5 residual checkpoint requires a PI0.5 flow policy (pytorch or jax)")
        if payload["format"] == "pi05_jax_full_finetune" and not isinstance(policy, PI05JaxFlowPolicy):
            raise TypeError("full-finetune JAX checkpoint requires PI05JaxFlowPolicy")
        policy.load_adapter_state_dict(payload["state"])
    elif payload["format"] == "full_state_dict":
        policy.load_state_dict(payload["state"])
    else:
        raise ValueError(f"unknown policy checkpoint format: {payload['format']!r}")


def _load_q_ensemble_state(module: ScalarQEnsemble, payload: dict[str, torch.Tensor]) -> None:
    incompatible = module.load_state_dict(payload, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    non_prior_missing = [key for key in incompatible.missing_keys if ".prior." not in key]
    if unexpected or non_prior_missing:
        raise RuntimeError(
            f"incompatible critic checkpoint: missing={non_prior_missing}, unexpected={unexpected}"
        )


def _prepare_restored_critic_stage(state: OGPOTrainState, payload: dict[str, Any]) -> None:
    restored_stage = str(payload.get("critic_stage", state.critic_stage))
    if isinstance(state.critic, MultiHeadUdivlCritic):
        configure_critic_stage(state.critic, restored_stage)
        state.critic_optimizer = _make_critic_optimizer(
            state.critic,
            state.divl,
            payload.get("config", {}).get("critic", {}),
        )
    state.critic_stage = restored_stage
    state.critic_stage_step = int(payload.get("critic_stage_step", 0))


def save_checkpoint(state: OGPOTrainState, config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    jax_sidecar_name = None
    checkpoint_old_policy = True
    if isinstance(state.policy, PI05JaxFlowPolicy):
        if not isinstance(state.old_policy, PI05JaxFlowPolicy):
            raise TypeError("JAX current policy requires a JAX old policy")
        jax_sidecar_path = Path(f"{path}.jax")
        checkpoint_old_policy = bool(
            config.get("actor", {}).get("checkpoint_old_policy", True)
        )
        state.policy.save_training_checkpoint(
            jax_sidecar_path,
            old_policy=state.old_policy if checkpoint_old_policy else None,
        )
        jax_sidecar_name = jax_sidecar_path.name
    payload = {
            "policy": _policy_checkpoint_state(
                state.policy,
                jax_sidecar=jax_sidecar_name,
                jax_sidecar_has_old_policy=checkpoint_old_policy,
            ),
            "old_policy": _policy_checkpoint_state(state.old_policy),
            "reference_metadata": {
                "type": type(state.reference_policy).__name__,
                "flow_convention": "openpi_pi05_euler",
                "checkpoint_dir": getattr(state.reference_policy, "checkpoint_dir", None),
                "train_config": getattr(state.reference_policy, "train_config_name", None),
            },
            "support": state.support.detach().cpu(),
            "critic_optimizer": state.critic_optimizer.state_dict(),
            "actor_optimizer": state.actor_optimizer.state_dict(),
            "schedulers": {},
            "running_mad": state.running_mad.value,
            "rectifier_grad_ema": state.rectifier.grad_ema,
            "rectifier_counts": state.rectifier.counts,
            "conformal_scale": state.conformal_scale,
            "training_step": state.step,
            "critic_stage": state.critic_stage,
            "critic_stage_step": state.critic_stage_step,
            "target_generator_state": (
                None if state.target_generator is None else state.target_generator.get_state()
            ),
            "config": config,
    }
    if isinstance(state.critic, MultiHeadUdivlCritic):
        payload.update(
            {
                "critic_format": "gemma_siglip_multihead",
                "multimodal_critic": state.critic.state_dict(),
                "target_multimodal_critic": state.target_critic.state_dict(),
                "critic_metadata": {
                    **getattr(state.critic, "model_metadata", {}),
                    "action_mean": state.critic.core.action_pool.action_mean.detach().cpu(),
                    "action_std": state.critic.core.action_pool.action_std.detach().cpu(),
                    "q_representation": state.critic.core.q_representation,
                    "q_support": (
                        None
                        if state.critic.core.q_support is None
                        else state.critic.core.q_support.detach().cpu()
                    ),
                    "q_hl_gauss_sigma_bins": config.get("critic", {}).get(
                        "q_hl_gauss_sigma_bins"
                    ),
                    "rank_consensus": {
                        key: config.get("critic", {}).get(key)
                        for key in (
                            "rank_consensus_enabled",
                            "rank_loss_weight",
                            "rank_margin_bins",
                            "rank_softmin_tau",
                            "rank_temperature",
                            "rank_noise_sigma",
                            "rank_use_strong_noise",
                            "rank_use_random_negative",
                            "rank_only_success",
                        )
                    },
                },
            }
        )
    elif isinstance(state.critic, MultiHeadScalarQCritic):
        payload.update(
            {
                "critic_format": "gemma_siglip_scalar_q",
                "multimodal_critic": state.critic.state_dict(),
                "target_multimodal_critic": state.target_critic.state_dict(),
                "critic_metadata": {
                    **getattr(state.critic, "model_metadata", {}),
                    "action_mean": state.critic.core.action_pool.action_mean.detach().cpu(),
                    "action_std": state.critic.core.action_pool.action_std.detach().cpu(),
                },
            }
        )
    else:
        assert state.divl is not None and state.target_divl is not None
        payload.update(
            {
                "critic_format": "mlp",
                "critic_ensemble": state.critic.state_dict(),
                "target_critics": state.target_critic.state_dict(),
                "divl": state.divl.state_dict(),
                "target_divl": state.target_divl.state_dict(),
            }
        )
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    state: OGPOTrainState,
    *,
    restore_actor_optimizer: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location=next(state.policy.parameters()).device, weights_only=False)
    _load_policy_checkpoint_state(state.policy, payload["policy"])
    _load_policy_checkpoint_state(state.old_policy, payload["old_policy"])
    if payload["policy"].get("format") == "pi05_jax_full_finetune":
        if not isinstance(state.policy, PI05JaxFlowPolicy) or not isinstance(
            state.old_policy,
            PI05JaxFlowPolicy,
        ):
            raise TypeError("full-finetune JAX checkpoint requires JAX current and old policies")
        sidecar = path.parent / payload["policy"]["jax_sidecar"]
        sidecar_has_old_policy = bool(
            payload["policy"].get("jax_sidecar_has_old_policy", True)
        )
        state.policy.restore_training_checkpoint(
            sidecar,
            old_policy=state.old_policy if sidecar_has_old_policy else None,
            restore_optimizer=restore_actor_optimizer,
        )
        if not sidecar_has_old_policy:
            state.old_policy.actor_state = jax.tree.map(
                lambda value: value,
                state.policy.actor_state,
            )
            state.old_policy._sync_torch_adapter_from_jax()
    if isinstance(state.critic, MultiHeadUdivlCritic):
        if payload.get("critic_format") != "gemma_siglip_multihead":
            raise ValueError("checkpoint does not contain a multimodal critic")
        state.critic.load_state_dict(payload["multimodal_critic"])
        state.target_critic.load_state_dict(payload["target_multimodal_critic"])
    elif isinstance(state.critic, MultiHeadScalarQCritic):
        if payload.get("critic_format") != "gemma_siglip_scalar_q":
            raise ValueError("checkpoint does not contain an OGPO-origin scalar Q critic")
        state.critic.load_state_dict(payload["multimodal_critic"])
        state.target_critic.load_state_dict(payload["target_multimodal_critic"])
    else:
        _load_q_ensemble_state(state.critic, payload["critic_ensemble"])
        _load_q_ensemble_state(state.target_critic, payload["target_critics"])
        assert state.divl is not None and state.target_divl is not None
        state.divl.load_state_dict(payload["divl"])
        state.target_divl.load_state_dict(payload["target_divl"])
    if "support" in payload:
        state.support = payload["support"].to(next(state.critic.parameters()).device)
    _prepare_restored_critic_stage(state, payload)
    state.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    if restore_actor_optimizer:
        state.actor_optimizer.load_state_dict(payload["actor_optimizer"])
    state.running_mad.value = float(payload["running_mad"])
    state.conformal_scale = float(payload.get("conformal_scale", 1.0))
    if "rectifier_grad_ema" in payload:
        state.rectifier.grad_ema = payload["rectifier_grad_ema"].detach().cpu()
    if "rectifier_counts" in payload:
        state.rectifier.counts = payload["rectifier_counts"].detach().cpu()
    state.step = int(payload["training_step"])
    if state.target_generator is not None and payload.get("target_generator_state") is not None:
        state.target_generator.set_state(payload["target_generator_state"].cpu())
    return payload


def load_critic_checkpoint(
    path: str | Path,
    state: OGPOTrainState,
    *,
    load_optimizer: bool = True,
) -> dict[str, Any]:
    """Restore outer-MDP value state without requiring the same actor type."""
    payload = torch.load(path, map_location=next(state.critic.parameters()).device, weights_only=False)
    if isinstance(state.critic, MultiHeadUdivlCritic):
        if payload.get("critic_format") != "gemma_siglip_multihead":
            raise ValueError("checkpoint does not contain a multimodal critic")
        state.critic.load_state_dict(payload["multimodal_critic"])
        state.target_critic.load_state_dict(payload["target_multimodal_critic"])
    elif isinstance(state.critic, MultiHeadScalarQCritic):
        if payload.get("critic_format") != "gemma_siglip_scalar_q":
            raise ValueError("checkpoint does not contain an OGPO-origin scalar Q critic")
        state.critic.load_state_dict(payload["multimodal_critic"])
        state.target_critic.load_state_dict(payload["target_multimodal_critic"])
    else:
        _load_q_ensemble_state(state.critic, payload["critic_ensemble"])
        _load_q_ensemble_state(state.target_critic, payload["target_critics"])
        assert state.divl is not None and state.target_divl is not None
        state.divl.load_state_dict(payload["divl"])
        state.target_divl.load_state_dict(payload["target_divl"])
    if "support" in payload:
        state.support = payload["support"].to(next(state.critic.parameters()).device)
    _prepare_restored_critic_stage(state, payload)
    if load_optimizer and "critic_optimizer" in payload:
        state.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    state.running_mad.value = float(payload.get("running_mad", state.running_mad.value))
    state.conformal_scale = float(payload.get("conformal_scale", 1.0))
    state.step = int(payload.get("training_step", 0))
    if state.target_generator is not None and payload.get("target_generator_state") is not None:
        state.target_generator.set_state(payload["target_generator_state"].cpu())
    return payload


def initialize_critic_from_checkpoint(
    path: str | Path,
    state: OGPOTrainState,
) -> dict[str, Any]:
    """Initialize a critic while keeping the current run config and optimizer fresh.

    This is intentionally distinct from resume: it supports scalar-Q to
    categorical-Q initialization and does not restore steps, optimizer state,
    training stage, support, or early-stopping state.
    """
    path = Path(path)
    payload = torch.load(
        path,
        map_location=next(state.critic.parameters()).device,
        weights_only=False,
    )
    if not isinstance(state.critic, MultiHeadUdivlCritic):
        raise TypeError("partial critic initialization requires MultiHeadUdivlCritic")
    if payload.get("critic_format") != "gemma_siglip_multihead":
        raise ValueError("initial checkpoint does not contain a multimodal U-DIVL critic")

    def load_compatible(
        module: MultiHeadUdivlCritic,
        source: dict[str, torch.Tensor],
        *,
        skip_q_heads: bool,
    ) -> tuple[list[str], list[str]]:
        destination = module.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for name, value in source.items():
            if skip_q_heads and (name.startswith("core.q_heads.") or name == "core.q_support"):
                skipped.append(name)
                continue
            if name not in destination or destination[name].shape != value.shape:
                skipped.append(name)
                continue
            compatible[name] = value
        module.load_state_dict(compatible, strict=False)
        return sorted(compatible), sorted(skipped)

    source_online = payload["multimodal_critic"]
    source_target = payload["target_multimodal_critic"]
    destination_categorical = state.critic.core.q_representation == "categorical"
    source_categorical = "core.q_support" in source_online
    scalar_to_categorical = destination_categorical and not source_categorical
    online_loaded, online_skipped = load_compatible(
        state.critic,
        source_online,
        skip_q_heads=scalar_to_categorical,
    )
    target_loaded, target_skipped = load_compatible(
        state.target_critic,
        source_target,
        skip_q_heads=scalar_to_categorical,
    )
    if scalar_to_categorical:
        state.target_critic.core.q_heads.load_state_dict(
            state.critic.core.q_heads.state_dict()
        )

    q_reinitialized = sorted(
        name
        for name in state.critic.state_dict()
        if name.startswith("core.q_heads.") or name == "core.q_support"
    ) if scalar_to_categorical else []
    print(
        "[critic-init] loaded parameters: "
        f"online={len(online_loaded)} target={len(target_loaded)} from {path}",
        flush=True,
    )
    print(
        "[critic-init] reinitialized categorical Q parameters: "
        + (", ".join(q_reinitialized) if q_reinitialized else "none"),
        flush=True,
    )
    print(
        "[critic-init] skipped incompatible source parameters: "
        f"online={len(online_skipped)} target={len(target_skipped)}",
        flush=True,
    )
    print("[critic-init] skipped incompatible optimizer states: fresh optimizer", flush=True)
    return payload
