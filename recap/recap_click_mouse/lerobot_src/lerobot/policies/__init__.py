# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RECAP value training imports ``lerobot.policies.pretrained`` directly and does
# not need to eagerly register every policy family.  Keeping this package init
# intentionally light isolates the value environment from unrelated optional
# policy dependencies (GR00T, Diffusers/PEFT, robot SDKs, ...).
__all__: list[str] = []
