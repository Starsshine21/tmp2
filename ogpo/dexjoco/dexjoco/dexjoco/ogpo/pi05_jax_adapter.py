from __future__ import annotations

import copy
import functools
import math
from pathlib import Path
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import optax
import numpy as np
import orbax.checkpoint as ocp
import torch
import torch.nn as nn

from .openpi_flow_spec import OpenPIStochasticFlowPolicy
from .pi05_pytorch_adapter import PI05FlowCondition, PI05ReplayConditionBuilder


def _ensure_orbax_jax_compatibility() -> None:
    if not hasattr(jax.monitoring, "record_scalar"):
        setattr(jax.monitoring, "record_scalar", lambda *args, **kwargs: None)


@functools.partial(jax.jit, static_argnames=("ema",))
def ema_actor_state(old_state, new_state, *, ema: float):
    """Fuse full-actor EMA so XLA does not materialize multiply temporaries."""
    return jax.tree.map(
        lambda old, new: old * ema + new * (1.0 - ema),
        old_state,
        new_state,
    )


class PI05JaxResidualHead(nnx.Module):
    """Trainable residual head on top of the jointly trainable PI0.5 actor."""

    def __init__(self, environment_action_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        in_dim = 2 * int(environment_action_dim) + 1
        self.fc1 = nnx.Linear(in_dim, int(hidden_dim), rngs=rngs)
        self.fc2 = nnx.Linear(int(hidden_dim), int(environment_action_dim), rngs=rngs)
        self.fc2.kernel.value = jnp.zeros_like(self.fc2.kernel.value)
        self.fc2.bias.value = jnp.zeros_like(self.fc2.bias.value)
        self.environment_action_dim = int(environment_action_dim)

    def __call__(self, x_env: jax.Array, base: jax.Array, time: jax.Array) -> jax.Array:
        time_features = jnp.broadcast_to(time[:, None, None], x_env.shape[:-1] + (1,))
        hidden = self.fc1(jnp.concatenate([x_env, base, time_features], axis=-1))
        hidden = nnx.silu(hidden)
        return self.fc2(hidden)


class PI05JaxActorModule(nnx.Module):
    """JAX-trainable OGPO actor containing the full PI0.5 backend."""

    def __init__(
        self,
        backend: Any,
        *,
        model_horizon: int,
        model_action_dim: int,
        environment_action_dim: int,
        residual_hidden_dim: int,
        init_log_std: float,
        num_steps: int,
        sde_mode: str,
        rngs: nnx.Rngs,
    ):
        self.backend = backend
        self.model_horizon = int(model_horizon)
        self.model_action_dim = int(model_action_dim)
        self.environment_action_dim = int(environment_action_dim)
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.num_steps = int(num_steps)
        self.sde_mode = str(sde_mode)
        self.residual = PI05JaxResidualHead(
            environment_action_dim=self.environment_action_dim,
            hidden_dim=self.residual_hidden_dim,
            rngs=rngs,
        )
        self.log_std = nnx.Param(jnp.full((self.model_horizon * self.environment_action_dim,), init_log_std))

    def predict_velocity(self, observation: Any, x_env: jax.Array, time: jax.Array) -> jax.Array:
        batch = x_env.shape[0]
        x_model = jnp.zeros(
            (batch, self.model_horizon, self.model_action_dim),
            dtype=x_env.dtype,
        )
        x_model = x_model.at[..., : self.environment_action_dim].set(x_env)
        # OGPO needs a deterministic velocity (no image augmentation) so PPO
        # transition log-prob ratios are well-defined; train=False skips the
        # augmax pipeline that would otherwise require an RNG.
        #
        # Rematerialize the (multi-billion-param) PI0.5 forward: full finetune
        # must backprop through the base model, and storing every Gemma layer
        # activation for the backward of all PPO transition forwards OOMs a
        # 40GB GPU. `jax.checkpoint` (nothing_saveable) drops the saved
        # activations and recomputes the forward during the transpose, bounding
        # peak memory to ~one forward at a time. Closed-over backend params are
        # still differentiated (standard remat pattern).
        # Wrap in a closure so `train=False` is a Python-level constant baked
        # into the checkpointed function (passing it as a checkpoint arg makes
        # JAX trace the bool, which trips linen's `deterministic=not train`).
        def _backend_forward(obs: Any, x_model_arg: jax.Array, time_arg: jax.Array) -> jax.Array:
            return self.backend.predict_velocity(obs, x_model_arg, time_arg, train=False)

        v_t = jax.checkpoint(
            _backend_forward,
            policy=jax.checkpoint_policies.nothing_saveable(),
        )(observation, x_model, time)
        base = v_t[..., : self.environment_action_dim]
        residual = self.residual(x_env, base, time)
        return (base + residual).reshape(batch, -1)

    def _corrected_velocity(
        self,
        x_env: jax.Array,
        velocity: jax.Array,
        time: jax.Array,
    ) -> jax.Array:
        if self.sde_mode != "ogpo_corrected":
            return velocity
        sigma_squared = jnp.exp(2.0 * self.log_std.value).reshape(
            1,
            self.model_horizon,
            self.environment_action_dim,
        )
        return velocity + 0.5 * sigma_squared * (
            (1.0 - time) * velocity + x_env
        )

    def _sample_actions_generic(self, observation: Any, noise: jax.Array) -> jax.Array:
        """Fallback for test/minimal backends without PI0's KV-cache API."""
        dt = jnp.asarray(-1.0 / self.num_steps, dtype=noise.dtype)
        batch_size = noise.shape[0]

        def step(step_index: int, x_env: jax.Array) -> jax.Array:
            time = jnp.asarray(1.0, dtype=noise.dtype) + dt * step_index
            time_batch = jnp.broadcast_to(time, (batch_size,))
            velocity = self.predict_velocity(observation, x_env, time_batch).reshape(
                x_env.shape
            )
            velocity = self._corrected_velocity(x_env, velocity, time)
            return x_env + dt * velocity

        return jax.lax.fori_loop(0, self.num_steps, step, noise)

    def _sample_actions_cached(self, observation: Any, noise: jax.Array) -> jax.Array:
        """PI0.5 sampling with one prefix forward and one compiled flow loop."""
        from openpi.models import model as openpi_model  # noqa: PLC0415
        from openpi.models.pi0 import make_attn_mask  # noqa: PLC0415

        observation = openpi_model.preprocess_observation(
            None,
            observation,
            train=False,
        )
        backend = self.backend
        batch_size = observation.state.shape[0]
        dt = jnp.asarray(-1.0 / self.num_steps, dtype=noise.dtype)

        prefix_tokens, prefix_mask, prefix_ar_mask = backend.embed_prefix(observation)
        prefix_attention_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = backend.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attention_mask,
            positions=prefix_positions,
        )

        def step(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            x_env, time = carry
            x_model = jnp.zeros(
                (batch_size, self.model_horizon, self.model_action_dim),
                dtype=x_env.dtype,
            )
            x_model = x_model.at[..., : self.environment_action_dim].set(x_env)
            time_batch = jnp.broadcast_to(time, (batch_size,))
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = backend.embed_suffix(
                observation,
                x_model,
                time_batch,
            )
            suffix_attention_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_to_suffix_mask = jnp.broadcast_to(
                prefix_mask[:, None, :],
                (
                    batch_size,
                    suffix_tokens.shape[1],
                    prefix_tokens.shape[1],
                ),
            )
            full_attention_mask = jnp.concatenate(
                [prefix_to_suffix_mask, suffix_attention_mask],
                axis=-1,
            )
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (prefix_out, suffix_out), _ = backend.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attention_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            if prefix_out is not None:
                raise AssertionError("cached PI0.5 suffix forward returned prefix output")
            base = backend.action_out_proj(
                suffix_out[:, -self.model_horizon :]
            )[..., : self.environment_action_dim]
            velocity = base + self.residual(x_env, base, time_batch)
            velocity = self._corrected_velocity(x_env, velocity, time)
            return x_env + dt * velocity, time + dt

        def cond(carry: tuple[jax.Array, jax.Array]) -> jax.Array:
            _, time = carry
            return time >= -dt / 2

        actions, _ = jax.lax.while_loop(
            cond,
            step,
            (noise, jnp.asarray(1.0, dtype=noise.dtype)),
        )
        return actions

    def sample_actions(self, observation: Any, noise: jax.Array) -> jax.Array:
        backend_supports_cache = all(
            hasattr(self.backend, name)
            for name in ("embed_prefix", "embed_suffix", "PaliGemma", "action_out_proj")
        )
        if backend_supports_cache:
            return self._sample_actions_cached(observation, noise)
        return self._sample_actions_generic(observation, noise)


def _torch_to_jax(value: Any) -> Any:
    """Copy a torch tensor (or pytree of tensors) into JAX arrays via host memory.

    The base PI0.5 model is a frozen, no-grad velocity oracle, so values never
    need to carry autograd across the framework boundary; a CPU round-trip is
    correct and avoids DLPack lifetime pitfalls between JAX and PyTorch.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return jnp.asarray(value.detach().to("cpu").numpy())
    if isinstance(value, dict):
        return {key: _torch_to_jax(item) for key, item in value.items()}
    return jnp.asarray(np.asarray(value))


def _jax_to_torch(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Copy a JAX array back into a detached torch tensor on ``device``.

    PI0.5 runs in ``bfloat16`` (``ml_dtypes.bfloat16``), which ``torch`` cannot
    ingest from a NumPy buffer directly; promote any non-torch-supported dtype
    to ``float32`` first (the caller already requests a torch ``dtype``).
    """
    arr = np.asarray(value)
    if str(arr.dtype) not in {
        "float64", "float32", "float16", "complex64", "complex128",
        "int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8", "bool",
    }:
        arr = arr.astype(np.float32)
    else:
        arr = np.array(arr, copy=True)
    return torch.as_tensor(arr, device=device, dtype=dtype)


def _observation_to_jax(observation: Any) -> Any:
    """Convert a torch-backed openpi ``Observation`` into a JAX-backed one."""
    from openpi.models import model as openpi_model  # noqa: PLC0415

    def _image_to_jax(value: Any) -> Any:
        # openpi ``Observation.from_dict`` permutes torch uint8 images to NCHW
        # (for the PyTorch SigLIP); the JAX SigLIP expects NHWC, so undo it.
        arr = value.detach().to("cpu").numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (0, 2, 3, 1))
        return jnp.asarray(arr)

    return openpi_model.Observation(
        images={key: _image_to_jax(value) for key, value in observation.images.items()},
        image_masks={key: _torch_to_jax(value) for key, value in observation.image_masks.items()},
        state=_torch_to_jax(observation.state),
        tokenized_prompt=(
            _torch_to_jax(observation.tokenized_prompt)
            if observation.tokenized_prompt is not None
            else None
        ),
        tokenized_prompt_mask=(
            _torch_to_jax(observation.tokenized_prompt_mask)
            if observation.tokenized_prompt_mask is not None
            else None
        ),
    )


class PI05JaxFlowPolicy(OpenPIStochasticFlowPolicy):
    """Stochastic OGPO adapter backed by the JAX PI0.5 action expert.

    Mirrors :class:`PI05PytorchFlowPolicy` but keeps the base PI0.5 model in
    JAX (no JAX->PyTorch checkpoint conversion). Unlike the PyTorch adapter,
    actor optimization lives entirely in JAX so gradients can flow through the
    native PI0.5 graph during OGPO PPO / flow-matching updates.
    """

    def __init__(
        self,
        backend: Any,
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
        model_horizon = int(backend.action_horizon)
        model_action_dim = int(backend.action_dim)
        environment_action_dim = int(environment_action_dim)
        if environment_action_dim > model_action_dim:
            raise ValueError("environment action dimension cannot exceed PI0.5 model action dimension")
        super().__init__(
            action_dim=model_horizon * environment_action_dim,
            num_steps=num_steps,
            stochastic_variance=stochastic_variance,
            sde_mode=sde_mode,
        )
        # ``backend`` is a Flax NNX module, not a torch.nn.Module, so it is NOT
        # registered as a child module and ``.to()`` / ``parameters()`` ignore it.
        self.backend = backend
        self.model_horizon = model_horizon
        self.model_action_dim = model_action_dim
        self.environment_action_dim = environment_action_dim
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.condition_builder = condition_builder
        self.checkpoint_dir = checkpoint_dir
        self.train_config_name = train_config_name
        self.rng = jax.random.PRNGKey(0)
        self.actor = PI05JaxActorModule(
            backend,
            model_horizon=self.model_horizon,
            model_action_dim=self.model_action_dim,
            environment_action_dim=self.environment_action_dim,
            residual_hidden_dim=self.residual_hidden_dim,
            init_log_std=math.log(math.sqrt(float(stochastic_variance))),
            num_steps=self.num_steps,
            sde_mode=self.sde_mode,
            rngs=nnx.Rngs(0),
        )
        self.actor_graphdef, self.actor_state = nnx.split(self.actor)
        self._inference_predict_velocity = None
        self._inference_sample_actions = None
        self.actor_tx: optax.GradientTransformation | None = None
        self.actor_opt_state = None
        self._actor_optimizer_step = None
        self.residual = nn.Sequential(
            nn.Linear(2 * environment_action_dim + 1, self.residual_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.residual_hidden_dim, environment_action_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        if hasattr(backend, "eval"):
            backend.eval()

    def init_actor_optimizer(
        self,
        *,
        learning_rate: float,
        weight_decay: float = 0.0,
        optimizer: str = "adafactor",
        max_grad_norm: float = 1.0,
        preserve_state_for_rollback: bool = False,
    ) -> None:
        # Full finetuning of the ~2.3B PI0.5 on a single 40GB GPU is memory
        # bound: AdamW keeps two per-parameter moment buffers (mu, nu) that,
        # together with the params and the value_and_grad activation graph, OOM
        # the JAX allocation. Adafactor stores only row/column second-moment
        # statistics (~1/18 of AdamW's state) and is the standard low-memory
        # choice for large-model finetuning; AdamW remains selectable for
        # bigger GPUs. Gradient clipping is folded into the optax chain (the
        # JAX actor update path does not clip in PyTorch).
        if optimizer == "adamw":
            inner = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adafactor":
            inner = optax.adafactor(learning_rate=learning_rate, weight_decay_rate=weight_decay)
        else:
            raise ValueError(f"unsupported actor.optimizer={optimizer!r}")
        self.actor_tx = (
            optax.chain(optax.clip_by_global_norm(max_grad_norm), inner)
            if max_grad_norm > 0
            else inner
        )
        self.actor_opt_state = self.actor_tx.init(self.actor_state)
        actor_tx = self.actor_tx

        def _optimizer_step(grads, opt_state, actor_state):
            updates, new_opt_state = actor_tx.update(grads, opt_state, actor_state)
            return optax.apply_updates(actor_state, updates), new_opt_state

        # The full PI0.5 gradient tree is several GiB. Fusing Adafactor and
        # apply_updates lets XLA reuse donated gradient/optimizer buffers for
        # squared-gradient temporaries and the new parameters instead of
        # materializing grads, updates, and new params at the same time.
        # actor_state itself is intentionally not donated: current, reference,
        # and old policies may still share the original immutable JAX buffers.
        donate_argnums = (0,) if preserve_state_for_rollback else (0, 1)
        self._actor_optimizer_step = jax.jit(_optimizer_step, donate_argnums=donate_argnums)

    def apply_actor_gradients(self, grads) -> None:
        if self._actor_optimizer_step is None or self.actor_opt_state is None:
            raise RuntimeError("JAX actor optimizer is not initialized")
        new_actor_state, new_opt_state = self._actor_optimizer_step(
            grads,
            self.actor_opt_state,
            self.actor_state,
        )
        self.actor_opt_state = new_opt_state
        self._replace_actor_state(new_actor_state)

    def _sync_torch_adapter_from_jax(self) -> None:
        actor = nnx.merge(self.actor_graphdef, self.actor_state)
        log_std = np.asarray(actor.log_std.value).copy()
        self.log_std.data.copy_(torch.as_tensor(log_std, device=self.log_std.device, dtype=self.log_std.dtype))
        fc1_w = torch.as_tensor(np.asarray(actor.residual.fc1.kernel.value).T.copy(), device=self.log_std.device, dtype=self.log_std.dtype)
        fc1_b = torch.as_tensor(np.asarray(actor.residual.fc1.bias.value).copy(), device=self.log_std.device, dtype=self.log_std.dtype)
        fc2_w = torch.as_tensor(np.asarray(actor.residual.fc2.kernel.value).T.copy(), device=self.log_std.device, dtype=self.log_std.dtype)
        fc2_b = torch.as_tensor(np.asarray(actor.residual.fc2.bias.value).copy(), device=self.log_std.device, dtype=self.log_std.dtype)
        self.residual[0].weight.data.copy_(fc1_w)
        self.residual[0].bias.data.copy_(fc1_b)
        self.residual[2].weight.data.copy_(fc2_w)
        self.residual[2].bias.data.copy_(fc2_b)

    def _replace_actor_state(self, actor_state) -> None:
        self.actor_state = actor_state
        self._inference_predict_velocity = None
        self._inference_sample_actions = None
        self._sync_torch_adapter_from_jax()

    def prepare_inference(self) -> None:
        """Freeze the current actor state into compiled velocity and rollout paths."""
        from openpi.shared import nnx_utils  # noqa: PLC0415

        actor = nnx.merge(self.actor_graphdef, self.actor_state)
        self._inference_predict_velocity = nnx_utils.module_jit(actor.predict_velocity)
        self._inference_sample_actions = nnx_utils.module_jit(actor.sample_actions)

    def sample_actions_jax(
        self,
        observation: Any,
        *,
        noise: Any | None = None,
        noise_seed: int | None = None,
    ) -> jax.Array:
        """Sample one action chunk without entering the Torch compatibility path."""
        if noise is None:
            if noise_seed is None:
                self.rng, key = jax.random.split(self.rng)
            else:
                key = jax.random.key(int(noise_seed))
            noise_array = jax.random.normal(
                key,
                (
                    observation.state.shape[0],
                    self.model_horizon,
                    self.environment_action_dim,
                ),
                dtype=jnp.float32,
            )
        else:
            noise_array = jnp.asarray(noise, dtype=jnp.float32)
            if noise_array.ndim == 2:
                noise_array = noise_array[None, ...]
            if noise_array.shape[-1] == self.model_action_dim:
                noise_array = noise_array[..., : self.environment_action_dim]
            expected = (
                observation.state.shape[0],
                self.model_horizon,
                self.environment_action_dim,
            )
            if noise_array.shape != expected:
                raise ValueError(
                    f"expected inference noise shape {expected}, got {noise_array.shape}"
                )

        if self._inference_sample_actions is None:
            actor = nnx.merge(self.actor_graphdef, self.actor_state)
            return actor.sample_actions(observation, noise_array)
        return self._inference_sample_actions(observation, noise_array)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self.backend, "eval"):
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
        time = timestep.reshape(batch, -1)[:, 0].to(dtype=torch.float32)

        jax_obs = _observation_to_jax(condition.observation)
        jax_x = _torch_to_jax(x_env)
        jax_time = _torch_to_jax(time)
        if self._inference_predict_velocity is None:
            actor = nnx.merge(self.actor_graphdef, self.actor_state)
            velocity = actor.predict_velocity(jax_obs, jax_x, jax_time)
        else:
            velocity = self._inference_predict_velocity(jax_obs, jax_x, jax_time)
        return _jax_to_torch(velocity, device=x_env.device, dtype=x_env.dtype)

    def clone_adapter(self, *, trainable: bool = False) -> "PI05JaxFlowPolicy":
        clone = PI05JaxFlowPolicy(
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
        # JAX arrays are immutable and optax updates are functional, so old /
        # reference policies can share the actor_state arrays (no deep copy);
        # they only diverge once sync_old_policy EMA-mixes them. Avoiding the
        # copy keeps full-finetune memory to one param copy + optimizer state.
        clone.actor_state = jax.tree.map(lambda x: x, self.actor_state)
        clone._sync_torch_adapter_from_jax()
        if not trainable:
            clone.requires_grad_(False)
        return clone

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        self._sync_torch_adapter_from_jax()
        state = {"log_std": self.log_std.detach().cpu().clone()}
        state.update(
            {
                f"residual.{key}": value.detach().cpu().clone()
                for key, value in self.residual.state_dict().items()
            }
        )
        return state

    def save_training_checkpoint(
        self,
        directory: str | Path,
        *,
        old_policy: "PI05JaxFlowPolicy | None" = None,
    ) -> None:
        """Persist full-finetune actor state and optimizer state as an Orbax sidecar."""
        item: dict[str, Any] = {
            "policy_actor_state": self.actor_state.to_pure_dict(),
        }
        if old_policy is not None:
            item["old_policy_actor_state"] = old_policy.actor_state.to_pure_dict()
        if self.actor_opt_state is not None:
            item["actor_opt_state"] = self.actor_opt_state
        directory = Path(directory).expanduser().resolve()
        directory.parent.mkdir(parents=True, exist_ok=True)
        _ensure_orbax_jax_compatibility()
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(
                directory,
                args=ocp.args.PyTreeSave(item=item),
                force=True,
            )

    def restore_training_checkpoint(
        self,
        directory: str | Path,
        *,
        old_policy: "PI05JaxFlowPolicy | None" = None,
        restore_optimizer: bool = True,
    ) -> None:
        """Restore a full-finetune Orbax sidecar into existing NNX state templates."""
        target: dict[str, Any] = {
            "policy_actor_state": self.actor_state.to_pure_dict(),
        }
        if old_policy is not None:
            target["old_policy_actor_state"] = old_policy.actor_state.to_pure_dict()
        if restore_optimizer:
            if self.actor_opt_state is None:
                raise RuntimeError("initialize the JAX actor optimizer before restoring its state")
            target["actor_opt_state"] = self.actor_opt_state
        directory = Path(directory).expanduser().resolve()
        _ensure_orbax_jax_compatibility()
        with ocp.PyTreeCheckpointer() as checkpointer:
            restored = checkpointer.restore(
                directory,
                args=ocp.args.PyTreeRestore(item=target, partial_restore=True),
            )
        self.actor_state.replace_by_pure_dict(restored["policy_actor_state"])
        self._inference_predict_velocity = None
        self._inference_sample_actions = None
        self._sync_torch_adapter_from_jax()
        if old_policy is not None:
            old_policy.actor_state.replace_by_pure_dict(restored["old_policy_actor_state"])
            old_policy._inference_predict_velocity = None
            old_policy._inference_sample_actions = None
            old_policy._sync_torch_adapter_from_jax()
        if restore_optimizer:
            self.actor_opt_state = restored["actor_opt_state"]

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.log_std.data.copy_(state["log_std"].to(self.log_std.device))
        residual_state = {
            key.removeprefix("residual."): value.to(self.log_std.device)
            for key, value in state.items()
            if key.startswith("residual.")
        }
        self.residual.load_state_dict(residual_state)
        actor = nnx.merge(self.actor_graphdef, self.actor_state)
        actor.log_std.value = jnp.asarray(state["log_std"].detach().cpu().numpy())
        actor.residual.fc1.kernel.value = jnp.asarray(
            residual_state["0.weight"].detach().cpu().numpy().T
        )
        actor.residual.fc1.bias.value = jnp.asarray(residual_state["0.bias"].detach().cpu().numpy())
        actor.residual.fc2.kernel.value = jnp.asarray(
            residual_state["2.weight"].detach().cpu().numpy().T
        )
        actor.residual.fc2.bias.value = jnp.asarray(residual_state["2.bias"].detach().cpu().numpy())
        self.actor_graphdef, self.actor_state = nnx.split(actor)
        self._inference_predict_velocity = None
        self._inference_sample_actions = None


def load_pi05_jax_flow_policy(
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
) -> PI05JaxFlowPolicy:
    """Load a JAX PI0.5 checkpoint and construct the OGPO adapter.

    Unlike the PyTorch adapter this does NOT require ``model.safetensors``; it
    loads the native Orbax ``params/`` checkpoint directly into the JAX model.
    """
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    params_dir = checkpoint_dir / "params"
    if not params_dir.exists():
        raise FileNotFoundError(
            f"{params_dir} is missing. Point flow.checkpoint_dir at the native JAX checkpoint "
            "(the directory containing params/), not the converted PyTorch checkpoint."
        )

    from openpi.models import model as openpi_model  # noqa: PLC0415
    from openpi.policies import policy_config  # noqa: PLC0415
    from openpi.training import config as training_config  # noqa: PLC0415
    import jax  # noqa: PLC0415

    _ensure_orbax_jax_compatibility()

    trained_policy = policy_config.create_trained_policy(
        training_config.get_config(train_config_name),
        checkpoint_dir,
    )
    if getattr(trained_policy, "_is_pytorch_model", False):
        raise TypeError(
            "pi05_jax adapter requires a native JAX checkpoint, but model.safetensors was found. "
            "Use flow.adapter: pi05_pytorch instead, or remove model.safetensors."
        )
    builder = PI05ReplayConditionBuilder(
        input_transform=trained_policy._input_transform,
        output_transform=trained_policy._output_transform,
        observation_type=openpi_model.Observation,
        image_mapping=dict(image_mapping),
        model_action_dim=int(trained_policy._model.action_dim),
        environment_action_dim=int(environment_action_dim),
    )
    return PI05JaxFlowPolicy(
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
