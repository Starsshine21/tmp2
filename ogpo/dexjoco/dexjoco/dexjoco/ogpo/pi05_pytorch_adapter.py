from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .openpi_flow_spec import OpenPIStochasticFlowPolicy


def _tree_map_tensor(value: Any, fn) -> Any:
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {key: _tree_map_tensor(item, fn) for key, item in value.items()}
    if dataclasses.is_dataclass(value):
        updates = {
            field.name: _tree_map_tensor(getattr(value, field.name), fn)
            for field in dataclasses.fields(value)
        }
        return dataclasses.replace(value, **updates)
    if value is None:
        return None
    return value


@dataclass(frozen=True)
class PI05FlowCondition:
    """Batched, model-ready PI0.5 observation used by flow transitions."""

    observation: Any

    @property
    def state(self) -> torch.Tensor:
        return self.observation.state

    @property
    def batch_size(self) -> int:
        return int(self.state.shape[0])

    def repeat_interleave(self, repeats: int) -> "PI05FlowCondition":
        return PI05FlowCondition(
            _tree_map_tensor(self.observation, lambda tensor: tensor.repeat_interleave(repeats, dim=0))
        )


def _stack_tree(values: list[Any], *, device: torch.device | str) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: _stack_tree([value[key] for value in values], device=device) for key in first}
    arrays = [np.asarray(value) for value in values]
    return torch.as_tensor(np.stack(arrays, axis=0), device=device)


@dataclass(frozen=True)
class PI05ReplayConditionBuilder:
    """Convert raw replay images/state/language into model-ready observations."""

    input_transform: Any
    observation_type: Any
    image_mapping: dict[str, str]
    output_transform: Any | None = None
    model_action_dim: int | None = None
    environment_action_dim: int | None = None

    def _raw_sample(self, batch, index: int, *, next_observation: bool = False) -> dict[str, Any]:
        images = batch.next_images if next_observation else batch.images
        if images is None:
            raise ValueError(
                "PI0.5 flow adapter requires replay RGB observations; rebuild the dataset from replay.zarr "
                "after collecting synchronized image arrays"
            )
        missing = set(self.image_mapping.values()) - set(images)
        if missing:
            raise KeyError(f"replay is missing PI0.5 camera arrays: {sorted(missing)}")
        states = batch.next_proprioceptions if next_observation else batch.proprioceptions
        raw = {
            role: images[key][index].detach().cpu().numpy()
            for role, key in self.image_mapping.items()
        }
        raw["state"] = states[index].detach().cpu().numpy()
        raw["prompt"] = np.asarray(batch.languages[index])
        return raw

    def __call__(
        self,
        batch,
        *,
        next_observation: bool = False,
        device: torch.device | str = "cpu",
    ) -> PI05FlowCondition:
        transformed = []
        for index in range(batch.batch_size):
            transformed.append(self.input_transform(self._raw_sample(batch, index, next_observation=next_observation)))
        model_inputs = _stack_tree(transformed, device=device)
        return PI05FlowCondition(self.observation_type.from_dict(model_inputs))

    def action_chunks_to_flow(self, batch) -> torch.Tensor:
        if self.environment_action_dim is None:
            raise RuntimeError("PI0.5 environment action dimension is not configured")
        transformed_actions = []
        for index in range(batch.batch_size):
            raw = self._raw_sample(batch, index)
            raw["actions"] = batch.action_chunks[index].detach().cpu().numpy()
            transformed = self.input_transform(raw)
            if "actions" not in transformed:
                raise KeyError("PI0.5 input transform did not return normalized actions")
            transformed_actions.append(
                np.asarray(transformed["actions"], dtype=np.float32)[..., : self.environment_action_dim]
            )
        return torch.as_tensor(
            np.stack(transformed_actions),
            device=batch.action_chunks.device,
            dtype=batch.action_chunks.dtype,
        )

    def flat_actions_to_environment(
        self,
        flat_actions: torch.Tensor,
        *,
        model_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.output_transform is None or self.model_action_dim is None or self.environment_action_dim is None:
            raise RuntimeError("PI0.5 output action transform is not configured")
        batch_size = flat_actions.shape[0]
        if flat_actions.shape[1] % self.environment_action_dim:
            raise ValueError("flat PI0.5 actions are not divisible by the environment action dimension")
        horizon = flat_actions.shape[1] // self.environment_action_dim
        flow_actions = flat_actions.detach().reshape(batch_size, horizon, self.environment_action_dim).cpu().numpy()
        states = model_states.detach().cpu().numpy() if model_states is not None else None
        environment_actions = []
        for index in range(batch_size):
            padded = np.zeros((horizon, self.model_action_dim), dtype=flow_actions.dtype)
            padded[..., : self.environment_action_dim] = flow_actions[index]
            output = {"actions": padded}
            if states is not None:
                output["state"] = states[index]
            transformed = self.output_transform(output)
            environment_actions.append(
                np.asarray(transformed["actions"], dtype=np.float32)[..., : self.environment_action_dim]
            )
        return torch.as_tensor(
            np.stack(environment_actions),
            device=flat_actions.device,
            dtype=flat_actions.dtype,
        ).reshape(batch_size, -1)


class PI05PytorchFlowPolicy(OpenPIStochasticFlowPolicy):
    """Stochastic OGPO adapter backed by the real PyTorch PI0.5 action expert.

    The large PI0.5 backend is frozen and can be shared by policy, old policy,
    and reference policy. A zero-initialized residual velocity adapter and the
    transition variance are the only trainable policy parameters.
    """

    def __init__(
        self,
        backend: nn.Module,
        *,
        environment_action_dim: int,
        num_steps: int = 10,
        stochastic_variance: float = 0.04,
        sde_mode: str = "gaussian_adapter",
        residual_hidden_dim: int = 128,
        condition_builder: PI05ReplayConditionBuilder | None = None,
        checkpoint_dir: str | None = None,
        train_config_name: str | None = None,
    ):
        model_horizon = int(backend.config.action_horizon)
        model_action_dim = int(backend.config.action_dim)
        environment_action_dim = int(environment_action_dim)
        if environment_action_dim > model_action_dim:
            raise ValueError("environment action dimension cannot exceed PI0.5 model action dimension")
        super().__init__(
            action_dim=model_horizon * environment_action_dim,
            num_steps=num_steps,
            stochastic_variance=stochastic_variance,
            sde_mode=sde_mode,
        )
        self.backend = backend
        self.model_horizon = model_horizon
        self.model_action_dim = model_action_dim
        self.environment_action_dim = environment_action_dim
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.condition_builder = condition_builder
        self.checkpoint_dir = checkpoint_dir
        self.train_config_name = train_config_name
        self.residual = nn.Sequential(
            nn.Linear(2 * environment_action_dim + 1, self.residual_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.residual_hidden_dim, environment_action_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.backend.requires_grad_(False)
        self.backend.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backend.eval()
        return self

    def condition_batch_size(self, condition: PI05FlowCondition) -> int:
        return condition.batch_size

    def condition_device_dtype(self, condition: PI05FlowCondition) -> tuple[torch.device, torch.dtype]:
        return condition.state.device, torch.float32

    def repeat_condition(self, condition: PI05FlowCondition, repeats: int) -> PI05FlowCondition:
        return condition.repeat_interleave(repeats)

    def condition_from_batch(self, batch, *, next_observation: bool = False) -> PI05FlowCondition:
        if self.condition_builder is None:
            raise RuntimeError("PI0.5 replay condition builder is not configured")
        return self.condition_builder(
            batch,
            next_observation=next_observation,
            device=self.log_std.device,
        )

    def action_chunks_to_flow(self, batch) -> torch.Tensor:
        if self.condition_builder is None or not hasattr(self.condition_builder, "action_chunks_to_flow"):
            return super().action_chunks_to_flow(batch)
        return self.condition_builder.action_chunks_to_flow(batch)

    def flat_actions_to_environment(
        self,
        flat_actions: torch.Tensor,
        condition: PI05FlowCondition | None = None,
    ) -> torch.Tensor:
        if self.condition_builder is None or not hasattr(self.condition_builder, "flat_actions_to_environment"):
            return super().flat_actions_to_environment(flat_actions, condition)
        model_states = None if condition is None else condition.state
        converted = self.condition_builder.flat_actions_to_environment(
            flat_actions,
            model_states=model_states,
        )
        if flat_actions.requires_grad:
            converted = converted + flat_actions - flat_actions.detach()
        return converted

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        condition: PI05FlowCondition,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        batch = x_t.shape[0]
        x_env = x_t.reshape(batch, self.model_horizon, self.environment_action_dim)
        x_model = x_env.new_zeros(batch, self.model_horizon, self.model_action_dim)
        x_model[..., : self.environment_action_dim] = x_env
        time = timestep.reshape(batch, -1)[:, 0].to(dtype=torch.float32)
        with torch.no_grad():
            base_model = self.backend.predict_velocity(
                condition.observation,
                x_model,
                time,
                train=False,
            )
        base = base_model[..., : self.environment_action_dim].to(dtype=x_env.dtype)
        time_features = time.to(dtype=x_env.dtype)[:, None, None].expand(batch, self.model_horizon, 1)
        residual = self.residual(torch.cat([x_env, base, time_features], dim=-1))
        return (base + residual).reshape(batch, -1)

    def clone_adapter(self, *, trainable: bool = False) -> "PI05PytorchFlowPolicy":
        clone = PI05PytorchFlowPolicy(
            self.backend,
            environment_action_dim=self.environment_action_dim,
            num_steps=self.num_steps,
            stochastic_variance=float(self.log_std.detach().exp().square().mean().item()),
            sde_mode=self.sde_mode,
            residual_hidden_dim=self.residual_hidden_dim,
            condition_builder=self.condition_builder,
            checkpoint_dir=self.checkpoint_dir,
            train_config_name=self.train_config_name,
        ).to(self.log_std.device)
        clone.log_std.data.copy_(self.log_std.data)
        clone.residual.load_state_dict(copy.deepcopy(self.residual.state_dict()))
        if not trainable:
            clone.requires_grad_(False)
        return clone

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        state = {"log_std": self.log_std.detach().cpu()}
        state.update({f"residual.{key}": value.detach().cpu() for key, value in self.residual.state_dict().items()})
        return state

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.log_std.data.copy_(state["log_std"].to(self.log_std.device))
        residual_state = {
            key.removeprefix("residual."): value.to(self.log_std.device)
            for key, value in state.items()
            if key.startswith("residual.")
        }
        self.residual.load_state_dict(residual_state)


def load_pi05_pytorch_flow_policy(
    *,
    checkpoint_dir: str | Path,
    train_config_name: str,
    image_mapping: dict[str, str],
    environment_action_dim: int,
    num_steps: int,
    stochastic_variance: float,
    sde_mode: str,
    residual_hidden_dim: int,
    device: torch.device | str,
) -> PI05PytorchFlowPolicy:
    """Load a converted PI0.5 checkpoint and construct the OGPO adapter."""
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    weights = checkpoint_dir / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(
            f"{weights} is missing. Convert the JAX checkpoint with "
            "openpi/examples/convert_jax_model_to_pytorch.py before OGPO actor training."
        )

    from openpi.models import model as openpi_model  # noqa: PLC0415
    from openpi.policies import policy_config  # noqa: PLC0415
    from openpi.training import config as training_config  # noqa: PLC0415

    trained_policy = policy_config.create_trained_policy(
        training_config.get_config(train_config_name),
        checkpoint_dir,
        pytorch_device=str(device),
    )
    if not getattr(trained_policy, "_is_pytorch_model", False):
        raise TypeError("OGPO requires a PyTorch PI0.5 checkpoint")
    builder = PI05ReplayConditionBuilder(
        input_transform=trained_policy._input_transform,
        output_transform=trained_policy._output_transform,
        observation_type=openpi_model.Observation,
        image_mapping=dict(image_mapping),
        model_action_dim=int(trained_policy._model.config.action_dim),
        environment_action_dim=int(environment_action_dim),
    )
    return PI05PytorchFlowPolicy(
        trained_policy._model,
        environment_action_dim=environment_action_dim,
        num_steps=num_steps,
        stochastic_variance=stochastic_variance,
        sde_mode=sde_mode,
        residual_hidden_dim=residual_hidden_dim,
        condition_builder=builder,
        checkpoint_dir=str(checkpoint_dir),
        train_config_name=train_config_name,
    ).to(device)
