from pathlib import Path

import yaml

from dexjoco.ogpo.experiment_matrix import METHOD_SPECS


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


def test_required_evaluation_methods_have_real_configs():
    expected = {
        "pi05_sft",
        "scalar_single_q_awr",
        "scalar_q_ensemble_full",
        "divl_full",
        "divl_flash",
        "udivl_flash",
    }
    root = Path(__file__).resolve().parents[2]

    assert set(METHOD_SPECS) == expected
    assert all((root / spec.config_path).is_file() for spec in METHOD_SPECS.values())


def test_trainable_method_configs_use_pi05_and_match_method_semantics():
    root = Path(__file__).resolve().parents[2]
    expected = {
        "scalar_single_q_awr": ("awr", False, 1),
        "scalar_q_ensemble_full": ("full_ogpo", False, 3),
        "divl_full": ("full_ogpo", True, 3),
        "divl_flash": ("flash_ogpo", True, 3),
        "udivl_flash": ("flash_ogpo", True, 3),
    }

    for method, (algorithm, divl_enabled, ensemble_size) in expected.items():
        cfg = _load_config(root, METHOD_SPECS[method].config_path)
        assert cfg["flow"]["adapter"] == "pi05_pytorch", method
        assert cfg["actor"]["algorithm"] == algorithm, method
        assert cfg["divl"].get("enabled", True) is divl_enabled, method
        assert cfg["critic"]["ensemble_size"] == ensemble_size, method

    udivl = _load_config(root, METHOD_SPECS["udivl_flash"].config_path)
    assert udivl["uncertainty"]["entropy_scale"] > 0.0
    assert udivl["uncertainty"]["use_support_weight"] is True


def test_sft_method_points_to_original_jax_checkpoint():
    root = Path(__file__).resolve().parents[2]
    cfg = _load_config(root, METHOD_SPECS["pi05_sft"].config_path)

    assert cfg["evaluation"]["policy_dir"].endswith(
        "click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse"
    )


def test_trainable_methods_have_non_overlapping_outputs():
    root = Path(__file__).resolve().parents[2]
    configs = [
        _load_config(root, spec.config_path)
        for spec in METHOD_SPECS.values()
        if spec.requires_ogpo_checkpoint
    ]

    for key in ("checkpoint_path", "metrics_path", "tensorboard_dir"):
        paths = [cfg["training"][key] for cfg in configs]
        assert len(paths) == len(set(paths)), key


def test_pi05_actor_configs_match_production_critic_architecture():
    root = Path(__file__).resolve().parents[2]
    critic_cfg = _load_config(root, "configs/ogpo/pi05_gemma_udivl_critic.yaml")

    for actor_path in (
        "configs/ogpo/pi05_full_ogpo.yaml",
        "configs/ogpo/pi05_flash_ogpo.yaml",
    ):
        actor_cfg = _load_config(root, actor_path)
        for key in ("architecture", "ensemble_size", "action_hidden_dim", "head_hidden_dim"):
            assert actor_cfg["critic"][key] == critic_cfg["critic"][key], (actor_path, key)
        for key in ("num_atoms", "auto_support", "support_margin_fraction"):
            assert actor_cfg["divl"].get(key) == critic_cfg["divl"].get(key), (actor_path, key)
        assert actor_cfg["actor"]["actor_delay"] == 0, actor_path


def test_gemma_udivl_production_config_locks_approved_ogpo_choices():
    root = Path(__file__).resolve().parents[2]
    cfg = _load_config(root, "configs/ogpo/pi05_gemma_udivl_critic.yaml")

    assert cfg["critic"]["architecture"] == "gemma_siglip_multihead"
    assert cfg["critic"]["ensemble_size"] == 3
    assert cfg["critic"]["backbone"]["train_siglip"] is False
    assert cfg["critic"]["gemma_lora"]["final_n_layers"] == 4
    assert cfg["critic"]["target_tau"] == 0.005
    assert cfg["critic"]["bootstrap_target"] == "subsample_min"
    assert cfg["critic"]["reference_value_samples"] > 1
    assert cfg["actor"]["advantage_mode"] == "sign_consensus"
    assert cfg["actor"]["lambda_abs_start"] == cfg["actor"]["lambda_abs_end"] == 1.0
    assert cfg["flow"]["sde_mode"] == "ogpo_corrected"
    assert cfg["evaluation"]["best_of_n"] == 1
