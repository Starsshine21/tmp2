from __future__ import annotations

import dataclasses

import openpi.models.pi0_config as pi0_config
import openpi.policies.pickplace_policy as pickplace_policy
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


def register(config_module):
    @dataclasses.dataclass(frozen=True)
    class PickPlaceLeRobotDataConfig(config_module.SimpleDataConfig):
        repo_id: str = "local/pi05-pickplace-il"
        base_config: config_module.DataConfig = dataclasses.field(
            default_factory=lambda: config_module.DataConfig(prompt_from_task=True)
        )
        data_transforms: object = dataclasses.field(default=None)
        model_transforms: object = dataclasses.field(default=None)
        repack_transforms: _transforms.Group = dataclasses.field(
            default_factory=lambda: _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
        )

        def create(self, assets_dirs, model_config):
            data_transforms = _transforms.Group(
                inputs=[pickplace_policy.PickPlaceInputs(model_type=model_config.model_type)],
                outputs=[pickplace_policy.PickPlaceOutputs()],
            )
            model_transforms = config_module.ModelTransformFactory()(model_config)
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=self.repack_transforms,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=("actions",),
            )

    return config_module.TrainConfig(
        name="pi05_pickplace_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=12,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=PickPlaceLeRobotDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=12,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20000,
        batch_size=32,
        log_interval=50,
        save_interval=1000,
        num_workers=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=20000,
            decay_lr=1e-5,
        ),
    )
