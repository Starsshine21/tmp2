#!/usr/bin/env bash
set -euo pipefail

patch_openpi_for_evorl_lerobot_acp() {
  conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python - "$DEXJOCO_DIR/openpi" <<'PY'
from pathlib import Path
import sys

openpi_dir = Path(sys.argv[1])
data_loader = openpi_dir / "src/openpi/training/data_loader.py"
dexjoco_configs = openpi_dir / "src/openpi/training/dexjoco_configs.py"
pi0_config = openpi_dir / "src/openpi/models/pi0_config.py"

text = data_loader.read_text()
if "\nimport random\n" not in text:
    text = text.replace("import os\n", "import os\nimport random\n", 1)
if "class ACPLeRobotPromptDataset" not in text:
    marker = "\n\nclass FakeDataset(Dataset):"
    insert = r'''

class ACPLeRobotPromptDataset(Dataset):
    """Apply Evo-RL ACP prompt tags to LeRobot samples before OpenPI repacking."""

    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._indicator_field = os.environ.get(
            "OPENPI_LEROBOT_ACP_INDICATOR_FIELD",
            "complementary_info.acp_indicator",
        )
        self._dropout = float(os.environ.get("OPENPI_LEROBOT_ACP_INDICATOR_DROPOUT_PROB", "0.0"))
        if not 0.0 <= self._dropout <= 1.0:
            raise ValueError("OPENPI_LEROBOT_ACP_INDICATOR_DROPOUT_PROB must be within [0, 1].")
        self._tag_key = os.environ.get("OPENPI_LEROBOT_ACP_TAG_KEY", "Advantage").strip() or "Advantage"
        raw_tag_values = os.environ.get("OPENPI_LEROBOT_ACP_TAG_VALUES", "negative,positive")
        self._tag_values = [value.strip() for value in raw_tag_values.split(",") if value.strip()]
        if not self._tag_values:
            raise ValueError("OPENPI_LEROBOT_ACP_TAG_VALUES must contain at least one comma-separated value.")
        self._rng = random.Random(int(os.environ.get("OPENPI_LEROBOT_ACP_SEED", "0")))

    @staticmethod
    def _scalar(value):
        arr = np.asarray(value)
        if arr.shape == ():
            return arr.item()
        return arr.reshape(-1)[0].item()

    def _tagged_task(self, task, tag_index: int) -> str:
        base = str(ACPLeRobotPromptDataset._scalar(task)).rstrip()
        tag = f"{self._tag_key}: {self._tag_values[tag_index]}"
        return f"{base}\n{tag}" if base else tag

    def __getitem__(self, index: SupportsIndex) -> dict:
        sample = dict(self._dataset[index])
        if self._indicator_field not in sample:
            raise KeyError(f"ACP indicator field '{self._indicator_field}' is missing from LeRobot sample.")
        raw_indicator = int(self._scalar(sample[self._indicator_field]))
        if raw_indicator < 0 or raw_indicator >= len(self._tag_values):
            raise ValueError(
                f"ACP indicator must be in [0, {len(self._tag_values) - 1}], got {raw_indicator}."
            )
        if self._dropout > 0.0 and self._rng.random() < self._dropout:
            return sample
        task_source = sample.get("task", sample.get("prompt", ""))
        tagged = self._tagged_task(task_source, tag_index=raw_indicator)
        sample["task"] = tagged
        sample["prompt"] = tagged
        return sample

    def __len__(self) -> int:
        return len(self._dataset)
'''
    text = text.replace(marker, insert + marker)

old = '''    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset
'''
new = '''    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    if os.environ.get("OPENPI_LEROBOT_ACP_ENABLE", "0") == "1":
        dataset = ACPLeRobotPromptDataset(dataset)

    return dataset
'''
if "OPENPI_LEROBOT_ACP_ENABLE" not in text:
    if old not in text:
        raise RuntimeError("Could not find OpenPI LeRobot dataset prompt insertion point.")
    text = text.replace(old, new, 1)
data_loader.write_text(text)

cfg_text = dexjoco_configs.read_text()
if "import os" not in cfg_text.splitlines()[:8]:
    cfg_text = cfg_text.replace("from datetime import datetime\n", "from datetime import datetime\nimport os\n")
cfg_text = cfg_text.replace(
    "warmup_steps=10_000,",
    'warmup_steps=int(os.environ.get("DEXJOCO_RECAP_WARMUP_STEPS", "10000")),',
)
cfg_text = cfg_text.replace(
    "num_workers=4,",
    'num_workers=int(os.environ.get("DEXJOCO_RECAP_NUM_WORKERS", "4")),',
)
cfg_text = cfg_text.replace(
    "save_interval=10000,",
    'save_interval=int(os.environ.get("DEXJOCO_RECAP_SAVE_INTERVAL", "10000")),',
)
dexjoco_configs.write_text(cfg_text)

pi0_text = pi0_config.read_text()
if "import os" not in pi0_text.splitlines()[:8]:
    pi0_text = pi0_text.replace("import dataclasses\n", "import dataclasses\nimport os\n")
lora_only_patch = '''    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if os.environ.get("OPENPI_RECAP_LORA_ONLY", "0") == "1":
            return nnx.Not(nnx_utils.PathRegex(".*lora.*"))
'''
if "OPENPI_RECAP_LORA_ONLY" not in pi0_text:
    pi0_text = pi0_text.replace(
        '''    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
''',
        lora_only_patch,
    )
pi0_config.write_text(pi0_text)
print("[dexjoco] patched OpenPI for Evo-RL LeRobot ACP prompt tagging")
PY
}

patch_openpi_config_yaml_for_lerobot_pool() {
  local dataset_parent="$1"
  local pretrained_model_path="$2"
  conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python - "$DEXJOCO_DIR/openpi/config.yaml" "$DEXJOCO_TASK" "$DEXJOCO_RECAP_BATCH_SIZE" "$DEXJOCO_RECAP_TRAIN_STEPS" "$pretrained_model_path" "$dataset_parent" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
task = sys.argv[2]
batch_size = int(sys.argv[3])
steps = int(sys.argv[4])
pretrained_model_path = sys.argv[5]
dataset_parent = sys.argv[6]

cfg = yaml.safe_load(path.read_text())
cfg["pretrained_model_path"] = pretrained_model_path
cfg["ckpts_root"] = "../checkpoints/evorl_lerobot_ckpts"
cfg["wandb_enabled"] = False
cfg["batch_size"] = batch_size
cfg["single_arm_steps"] = steps
cfg["dataset_root"] = dataset_parent
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"[dexjoco] patched openpi/config.yaml for task={task} dataset_parent={dataset_parent}")
print(path.read_text())
PY
}
