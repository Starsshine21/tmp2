#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dexjoco"))
sys.path.insert(0, str(ROOT / "scripts"))

from dexjoco.ogpo.gemma_siglip_backbone import LoRALinear
from dexjoco.ogpo.replay import load_replay
from dexjoco.ogpo.trainer import build_train_state, critic_update, load_checkpoint, save_checkpoint
from train_udivl_critic import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ogpo/pi05_gemma_udivl_critic_smoke.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    device = config["training"].get("device", "cuda")
    replay = load_replay(ROOT / config["data"]["dataset_path"])
    count = int(config["training"].get("batch_size", 1))
    batch = replay.index_select(torch.arange(count)).to(device)
    state = build_train_state(config, replay, device=device)

    vision_parameters = list(state.critic.state_encoder.vision_model.parameters())
    if not vision_parameters or any(parameter.requires_grad for parameter in vision_parameters):
        raise AssertionError("SigLIP must remain fully frozen")
    lora_modules = [
        module
        for module in state.critic.state_encoder.gemma_model.modules()
        if isinstance(module, LoRALinear)
    ]
    if not lora_modules:
        raise AssertionError("no Gemma LoRA modules were installed")
    if any(module.base.weight.requires_grad for module in lora_modules):
        raise AssertionError("LoRA base weights must remain frozen")

    with torch.no_grad():
        features = state.critic.encode_state(batch)
        expected_q = state.critic.q_from_features(
            features,
            batch.action_chunks,
            batch.execution_masks,
        )
        corrupted_actions = batch.action_chunks.clone()
        corrupted_actions[~batch.execution_masks] += 1000.0
        suffix_q = state.critic.q_from_features(
            features,
            corrupted_actions,
            batch.execution_masks,
        )
    if not torch.allclose(expected_q, suffix_q, atol=1e-5, rtol=1e-5):
        raise AssertionError("critic Q changed after modifying only the unexecuted suffix")

    metrics = critic_update(state, batch, config)
    if not any(module.lora_b.grad is not None for module in lora_modules):
        raise AssertionError("Gemma LoRA did not receive gradients")

    checkpoint = ROOT / config["training"]["checkpoint_path"]
    save_checkpoint(state, config, checkpoint)
    with torch.no_grad():
        restored_expected = state.critic(
            batch,
            batch.action_chunks,
            batch.execution_masks,
        ).clone()
        next(parameter for parameter in state.critic.parameters() if parameter.requires_grad).add_(1.0)
    load_checkpoint(checkpoint, state)
    with torch.no_grad():
        restored_actual = state.critic(
            batch,
            batch.action_chunks,
            batch.execution_masks,
        )
    if not torch.equal(restored_expected, restored_actual):
        raise AssertionError("multimodal critic checkpoint reload changed Q outputs")

    print(
        json.dumps(
            {
                "status": "ok",
                "critic_loss": metrics["critic_loss"],
                "lora_modules": len(lora_modules),
                "suffix_invariant": True,
                "checkpoint": str(checkpoint),
                "max_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
