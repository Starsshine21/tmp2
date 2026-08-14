from __future__ import annotations

import abc
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn

from .flow_logprob import gaussian_kl_diag, gaussian_log_prob


@dataclass(frozen=True)
class FlowRollout:
    states: torch.Tensor
    next_states: torch.Tensor
    timesteps: torch.Tensor
    log_probs: torch.Tensor
    endpoint: torch.Tensor


@dataclass(frozen=True)
class OpenPIFlowSpec:
    """OpenPI PI0/PI0.5 flow-matching convention.

    OpenPI uses t=1 for noise and t=0 for the clean action endpoint. The
    learned velocity points from the clean action toward the sampled noise, and
    inference integrates backward with a negative Euler step.
    """

    num_steps: int = 10

    @property
    def dt(self) -> float:
        return -1.0 / max(1, int(self.num_steps))

    def timestep_values(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.linspace(
            1.0,
            1.0 / max(1, int(self.num_steps)),
            int(self.num_steps),
            device=device,
            dtype=dtype,
        )

    def expand_timestep(self, timestep: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=target.dtype, device=target.device)
        while t.ndim < target.ndim:
            t = t.unsqueeze(-1)
        return t

    def sample_training_time(
        self,
        batch_shape: torch.Size | tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        uniform = torch.rand(batch_shape, device=device, dtype=dtype, generator=generator)
        return uniform.pow(1.0 / 1.5) * 0.999 + 0.001

    def training_pair(
        self,
        action_endpoint: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.expand_timestep(timestep, action_endpoint)
        x_t = t * noise + (1.0 - t) * action_endpoint
        target_velocity = noise - action_endpoint
        return x_t, target_velocity

    def euler_step(self, x_t: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        return x_t + self.dt * velocity


class OpenPIStochasticFlowPolicy(nn.Module, abc.ABC):
    """Stochastic transition wrapper around OpenPI's deterministic Euler mean.

    Subclasses provide `predict_velocity(x_t, condition, timestep)`. The
    wrapper turns each deterministic Euler update into a diagonal Gaussian
    transition so PPO-style objectives can evaluate transition log-probability
    ratios without learning an inner flow-state value function.
    """

    def __init__(
        self,
        action_dim: int,
        *,
        num_steps: int = 10,
        stochastic_variance: float = 0.04,
        sde_mode: str = "gaussian_adapter",
    ):
        super().__init__()
        if sde_mode not in {"gaussian_adapter", "ogpo_corrected"}:
            raise ValueError(f"unsupported SDE mode: {sde_mode}")
        self.action_dim = int(action_dim)
        self.num_steps = int(num_steps)
        self.sde_mode = sde_mode
        self.flow_spec = OpenPIFlowSpec(num_steps=self.num_steps)
        init_std = math.sqrt(float(stochastic_variance))
        self.log_std = nn.Parameter(torch.full((self.action_dim,), math.log(init_std)))

    @abc.abstractmethod
    def predict_velocity(self, x_t: torch.Tensor, condition: Any, timestep: torch.Tensor) -> torch.Tensor:
        """Predict the PI0.5 flow velocity for one batched latent state."""

    def condition_batch_size(self, condition: Any) -> int:
        return int(condition.shape[0])

    def condition_device_dtype(self, condition: Any) -> tuple[torch.device, torch.dtype]:
        return condition.device, condition.dtype

    def repeat_condition(self, condition: Any, repeats: int) -> Any:
        return condition.repeat_interleave(repeats, dim=0)

    def action_chunks_to_flow(self, batch: Any) -> torch.Tensor:
        """Map replay action chunks into the flow model's action space."""
        return batch.action_chunks

    def flat_actions_to_environment(
        self,
        flat_actions: torch.Tensor,
        condition: Any | None = None,
    ) -> torch.Tensor:
        """Map flat flow endpoints into the critic/environment action space."""
        del condition
        return flat_actions

    def transition_mean(self, x_t: torch.Tensor, condition: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        velocity = self.predict_velocity(x_t, condition, timestep)
        if self.sde_mode == "ogpo_corrected":
            # OGPO is written in cleanward time tau=1-t. Mapping its tapered
            # CondOT correction back to PI's reverse-time derivative gives
            # v_pi + sigma_base^2 / 2 * ((1-t) v_pi + x_t).
            t = self.flow_spec.expand_timestep(timestep, x_t)
            sigma_squared = self.log_std.to(x_t.device, x_t.dtype).exp().square().expand_as(x_t)
            velocity = velocity + 0.5 * sigma_squared * ((1.0 - t) * velocity + x_t)
        return self.flow_spec.euler_step(x_t, velocity)

    def transition_std(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        base_std = self.log_std.to(x_t.device, x_t.dtype).exp().expand_as(x_t)
        if self.sde_mode == "gaussian_adapter":
            return base_std
        t = self.flow_spec.expand_timestep(timestep, x_t).clamp(min=0.0, max=1.0)
        return base_std * t.sqrt()

    def transition_log_std(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.transition_std(x_t, timestep).clamp_min(torch.finfo(x_t.dtype).tiny).log()

    def sample_transition(
        self,
        x_t: torch.Tensor,
        condition: Any,
        timestep: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool | torch.Tensor = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.transition_mean(x_t, condition, timestep)
        log_std = self.transition_log_std(x_t, timestep)
        if isinstance(deterministic, bool) and deterministic:
            x_prev = mean
        elif isinstance(deterministic, torch.Tensor):
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
            stochastic = mean + noise * log_std.exp()
            mask = deterministic.to(dtype=torch.bool, device=mean.device)
            while mask.ndim < mean.ndim:
                mask = mask.unsqueeze(-1)
            x_prev = torch.where(mask, mean, stochastic)
        else:
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
            x_prev = mean + noise * log_std.exp()
        return x_prev, gaussian_log_prob(x_prev, mean, log_std)

    def log_prob(
        self,
        x_prev: torch.Tensor,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        mean = self.transition_mean(x_t, condition, timestep)
        return gaussian_log_prob(x_prev, mean, self.transition_log_std(x_t, timestep))

    def kl_to(
        self,
        other: "OpenPIStochasticFlowPolicy",
        x_t: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return gaussian_kl_diag(
            self.transition_mean(x_t, condition, timestep),
            self.transition_log_std(x_t, timestep),
            other.transition_mean(x_t, condition, timestep),
            other.transition_log_std(x_t, timestep),
        )

    def rollout(
        self,
        condition: torch.Tensor,
        *,
        group_size: int = 1,
        selected_timestep: int | torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        deterministic_except_selected: bool = False,
    ) -> FlowRollout:
        batch = self.condition_batch_size(condition)
        device, dtype = self.condition_device_dtype(condition)
        condition_g = self.repeat_condition(condition, group_size)
        selected_g: int | torch.Tensor | None
        if isinstance(selected_timestep, torch.Tensor):
            selected = selected_timestep.to(device=device, dtype=torch.long)
            if selected.shape == (batch,):
                selected_g = selected.repeat_interleave(group_size)
            elif selected.shape == (batch * group_size,):
                selected_g = selected
            else:
                raise ValueError("selected_timestep tensor must have shape [batch] or [batch * group_size]")
        else:
            selected_g = selected_timestep
        x_t = torch.randn(
            batch * group_size,
            self.action_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        states = []
        next_states = []
        timesteps = []
        log_probs = []
        timestep_values = self.flow_spec.timestep_values(device=device, dtype=dtype)
        for step_index, t_scalar in enumerate(timestep_values):
            t_value = t_scalar.expand(batch * group_size, 1)
            deterministic = (
                deterministic_except_selected
                and selected_g is not None
                and (
                    (step_index != selected_g)
                    if isinstance(selected_g, torch.Tensor)
                    else step_index != selected_g
                )
            )
            x_prev, log_prob = self.sample_transition(
                x_t,
                condition_g,
                t_value,
                generator=generator,
                deterministic=deterministic,
            )
            states.append(x_t)
            next_states.append(x_prev)
            timesteps.append(t_value)
            log_probs.append(log_prob)
            x_t = x_prev
        return FlowRollout(
            states=torch.stack(states, dim=1),
            next_states=torch.stack(next_states, dim=1),
            timesteps=torch.stack(timesteps, dim=1),
            log_probs=torch.stack(log_probs, dim=1),
            endpoint=x_t,
        )
