"""PyTorch LoRA utilities for PI0/PI05 fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: float = 1.0
    rslora: bool = False
    dropout: float = 0.0
    init_std: float = 0.01

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


@dataclass(frozen=True)
class LoRATrainingConfig:
    enabled: bool = False
    attn_rank: int = 16
    ffn_rank: int = 16
    attn_alpha: float = 16.0
    ffn_alpha: float = 16.0
    use_rslora: bool = False
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    apply_to: Literal["all", "paligemma_only", "expert_only", "paligemma_attn", "expert_attn"] = "all"
    train_vision_encoder: bool = False
    train_non_lora_layers: bool = True
    trainable_modules: list[str] = field(
        default_factory=lambda: [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
    )

    def get_lora_configs(self) -> dict[str, LoRAConfig]:
        return {
            "attn": LoRAConfig(
                rank=self.attn_rank,
                alpha=self.attn_alpha,
                rslora=self.use_rslora,
                dropout=self.dropout,
            ),
            "ffn": LoRAConfig(
                rank=self.ffn_rank,
                alpha=self.ffn_alpha,
                rslora=self.use_rslora,
                dropout=self.dropout,
            ),
        }


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_config: LoRAConfig,
        *,
        bias: bool,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_config = lora_config

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.lora_a = nn.Parameter(torch.empty(lora_config.rank, in_features, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(out_features, lora_config.rank, device=device, dtype=dtype))
        nn.init.normal_(self.lora_a, std=lora_config.init_std)
        self.lora_dropout = nn.Dropout(lora_config.dropout) if lora_config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_a), self.lora_b)
        return out + lora_out * self.lora_config.scaling_value

    def merge_lora_weights(self) -> None:
        with torch.no_grad():
            self.weight.data += (self.lora_b @ self.lora_a) * self.lora_config.scaling_value


def _replace_linear_with_lora(linear: nn.Linear, lora_config: LoRAConfig) -> LoRALinear:
    lora_linear = LoRALinear(
        linear.in_features,
        linear.out_features,
        lora_config,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    lora_linear.weight.data.copy_(linear.weight.data)
    if linear.bias is not None:
        lora_linear.bias.data.copy_(linear.bias.data)
    return lora_linear


def apply_lora_to_pi0_pytorch(model: nn.Module, lora_config: LoRATrainingConfig) -> tuple[int, int]:
    """Inject LoRA layers into PI0Pytorch and freeze parameters for LoRA training."""
    if not lora_config.enabled:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return 0, trainable

    lora_configs = lora_config.get_lora_configs()
    target_modules = lora_config.target_modules
    if lora_config.apply_to in {"paligemma_attn", "expert_attn"}:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    apply_to_paligemma = lora_config.apply_to in {"all", "paligemma_only", "paligemma_attn"}
    apply_to_expert = lora_config.apply_to in {"all", "expert_only", "expert_attn"}
    attn_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}
    ffn_modules = {"gate_proj", "up_proj", "down_proj"}

    applied = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        module_name = name.rsplit(".", 1)[-1]
        if module_name not in target_modules:
            continue
        if "vision_tower" in name or "vision_model" in name:
            continue

        is_paligemma = "paligemma" in name and "language_model" in name
        is_expert = "gemma_expert" in name
        if not ((is_paligemma and apply_to_paligemma) or (is_expert and apply_to_expert)):
            continue

        if module_name in attn_modules:
            cfg = lora_configs["attn"]
        elif module_name in ffn_modules:
            cfg = lora_configs["ffn"]
        else:
            continue

        parent_name, attr_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, attr_name, _replace_linear_with_lora(module, cfg))
        applied += 1

    logging.info("Applied LoRA to %d PyTorch linear layers", applied)
    return freeze_for_lora_training(model, lora_config)


def freeze_for_lora_training(model: nn.Module, lora_config: LoRATrainingConfig) -> tuple[int, int]:
    frozen_count = 0
    trainable_count = 0
    trainable_modules = set(lora_config.trainable_modules) if lora_config.train_non_lora_layers else set()

    for name, param in model.named_parameters():
        is_lora_param = "lora_" in name
        is_vision_param = "vision_tower" in name or "vision_model" in name
        is_trainable_module = any(module_name in name for module_name in trainable_modules)

        should_train = is_lora_param or (lora_config.train_vision_encoder and is_vision_param) or is_trainable_module
        param.requires_grad = should_train
        if should_train:
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    total = frozen_count + trainable_count
    ratio = 100.0 * trainable_count / total if total else 0.0
    logging.info("LoRA training: %d trainable params, %d frozen params (%.4f%% trainable)", trainable_count, frozen_count, ratio)
    return frozen_count, trainable_count
