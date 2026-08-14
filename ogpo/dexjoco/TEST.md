# TEST Log

## 2026-06-14

### Step 1
- Created `TEST.md` to record every experiment and code/config change during DexJoCo + pi05 debugging.
- Next: audit why `../pi05/.conda-pi05-openpi-final` is importing packages from `RLinf-main/.venv`.

### Step 2
- Audited the active `../pi05/.conda-pi05-openpi-final` environment.
- Found the contamination source is not `VIRTUAL_ENV`, but an explicit `PYTHONPATH` injection.
- Observed:
  - `PYTHONPATH=/nfs_global/S/yangrongzheng/RLinf-main/RLinf_deps/libero:/nfs_global/S/yangrongzheng/RLinf-main/.venv/lib/python3.11/site-packages:/nfs_global/S/yangrongzheng/RLinf-main:/nfs_global/S/yangrongzheng/pi05`
  - `sys.executable=/nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final/bin/python`
  - but `sys.path` still prioritizes `RLinf-main/.venv/lib/python3.11/site-packages` before the conda site-packages.
- Conclusion: the previously reused `pi05` conda env is being polluted by inherited `PYTHONPATH` from the shell/session.
- Next: retry `serve_policy.py` with a sanitized `PYTHONPATH`, keeping only DexJoCo OpenPI source and required local paths.

### Step 3
- Retried `serve_policy.py` after sanitizing `PYTHONPATH` down to:
  - `/nfs_global/S/yangrongzheng/evo-RL/dexjoco/openpi/src`
  - `/nfs_global/S/yangrongzheng/evo-RL/dexjoco/openpi/packages/openpi-client/src`
- Result: the previous `RLinf-main/.venv` contamination path disappeared, but the underlying conda env proved incomplete.
- New failure:
  - `ModuleNotFoundError: No module named 'tyro'`
- Conclusion: `../pi05/.conda-pi05-openpi-final` is not a self-sufficient environment for DexJoCo OpenPI serving once the inherited `PYTHONPATH` crutch is removed.
- Decision: stop here on the reused-env route to avoid an endless cycle of patching individual missing packages. The next clean step would be to provision a dedicated DexJoCo OpenPI environment.

### Step 4
- Added a clean-runtime helper path for DexJoCo OpenPI serving without inheriting the shell's polluted `PYTHONPATH`.
- Updated `scripts/common_env.sh` to define:
  - `CLEAN_OPENPI_ENV_PREFIX`
  - `PI05_USE_CLEAN_OPENPI_ENV`
  - `PI05_EFFECTIVE_OPENPI_ENV_PREFIX`
  - `DEXJOCO_CLEAN_PYTHONPATH_DEFAULT`
  - `DEXJOCO_STRIP_PYTHONPATH`
- Updated `scripts/run_pi05_server.sh` to:
  - activate `PI05_EFFECTIVE_OPENPI_ENV_PREFIX`
  - optionally replace `PYTHONPATH` with DexJoCo OpenPI source paths only
  - print the effective env and python path before serving
- Added `scripts/check_clean_openpi_runtime.sh` to audit import provenance for `openpi`, `lerobot`, and `tyro` under the sanitized runtime.
- Goal of this step: keep the previously reused method available, but make the contamination boundary explicit and observable before serving.

### Step 5
- Ran `bash scripts/check_clean_openpi_runtime.sh` with sanitized `PYTHONPATH`.
- Verified that the contamination from `RLinf-main/.venv` is gone:
  - `openpi` now resolves to `dexjoco/openpi/src/openpi/__init__.py`
- Verified the reused `../pi05/.conda-pi05-openpi-final` environment is still incomplete for clean DexJoCo serving:
  - `lerobot IMPORT FAILED = ModuleNotFoundError("No module named 'lerobot'")`
  - `tyro IMPORT FAILED = ModuleNotFoundError("No module named 'tyro'")`
- Also observed extra RoboTwin paths still present in `sys.path` from the reused environment:
  - `/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/envs/curobo/src`
  - `/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05/src`
  - `/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05/packages/openpi-client/src`
- Conclusion: the new helper scripts successfully expose a clean import boundary, and the remaining blocker is no longer hidden contamination but missing runtime dependencies inside the reused `pi05` conda env itself.
- Decision boundary: do not keep patching this reused env blindly. The next meaningful step is to provision a dedicated clean OpenPI environment, then point it at the downloaded DexJoCo checkpoint.

### Step 6
- Attempted to provision a dedicated clean conda prefix at `dexjoco/.conda/openpi` as the next meaningful step after proving the reused `../pi05` env was incomplete.
- Minimal probe command used:
  - `conda create -y -p /nfs_global/S/yangrongzheng/evo-RL/dexjoco/.conda/openpi python=3.11 pip git git-lfs`
- Result inside the Codex sandbox:
  - failed before solve/install due to blocked outbound network/proxy access
  - representative error: `PermissionError: [Errno 1] Operation not permitted`
- This confirms the remaining blocker is no longer repository state but external install access needed to create a fresh environment.
- Decision boundary: do not keep retrying conda creation inside the restricted sandbox. Either run the creation in the user's shell, or grant the escalated install request when prompted.

### Step 7
- Summarized the root cause chain explicitly for handoff:
  1. The reused `../pi05/.conda-pi05-openpi-final` runtime was polluted by inherited `PYTHONPATH`, causing imports to resolve to `RLinf-main/.venv` packages instead of DexJoCo's `openpi`.
  2. After sanitizing `PYTHONPATH`, the reused conda env itself still proved incomplete, with missing runtime dependencies such as `tyro` and `lerobot`.
- This confirms the problem is not the downloaded `click_mouse` checkpoint and not the DexJoCo simulator layer.
- The unresolved blocker is now singular and explicit: a clean dedicated OpenPI environment has not yet been provisioned at `dexjoco/.conda/openpi`.

### Step 8
- Successfully created the clean dedicated conda prefix at `dexjoco/.conda/openpi`.
- Command that succeeded:
  - `conda create -y -p /nfs_global/S/yangrongzheng/evo-RL/dexjoco/.conda/openpi python=3.11 pip git git-lfs`
- Result: the repository now has a real clean OpenPI base prefix independent from `../pi05/.conda-pi05-openpi-final`.
- Next: install the remaining documented OpenPI runtime pieces into this clean prefix (`ffmpeg`, `lerobot --no-deps`, editable `openpi`, editable `openpi-client`) and then verify imports.

### Step 9
- Tried to follow `openpi/environment-openpi.yaml` literally by installing `ffmpeg=7` into `dexjoco/.conda/openpi`.
- Result under the currently available conda channels:
  - `PackagesNotFoundError: ffmpeg=7`
- Effective channel set during install was only:
  - `defaults`
  - `https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main`
  - `https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free`
- Conclusion: the documented `ffmpeg=7` requirement cannot be satisfied from the channels currently visible in this environment. This is a channel availability problem, not a repository packaging bug.
- Next: continue with the Python-side OpenPI installs first, since they are required to validate import/runtime completeness; revisit ffmpeg only if serving/runtime actually needs it.

### Step 10
- Installed `lerobot` into the clean `dexjoco/.conda/openpi` prefix successfully.
- Verified the clean prefix with a targeted import probe using sanitized `PYTHONPATH`:
  - `lerobot OK`
  - `openpi OK` from `dexjoco/openpi/src/openpi/__init__.py`
  - `tyro FAIL = ModuleNotFoundError("No module named 'tyro'")`
- Conclusion: the clean environment is now real and correctly isolated from the old `../pi05` runtime. The next blocker is ordinary missing Python dependencies for OpenPI, starting with `tyro`.

### Step 11
- Installed `tyro` into the clean `dexjoco/.conda/openpi` prefix.
- This removes the first import blocker previously seen when trying to use the clean environment for `serve_policy.py`.
- Next: validate the serve path again to expose the next actual missing dependency, if any.

### Step 12
- Re-ran the clean serve-path validation with:
  - `PYTHONPATH=<dexjoco openpi srcs> python scripts/serve_policy.py --help`
- Result:
  - `ModuleNotFoundError: No module named 'flax'`
- This is progress: the clean environment now gets past the previous `tyro` blocker and reaches the core OpenPI policy import chain.
- The next concrete missing runtime dependency for the documented serve path is `flax`.

### Step 13
- Installed local editable `openpi-client` into the clean `dexjoco/.conda/openpi` prefix.
- Verified with `pip show openpi-client` that it now resolves to:
  - `dexjoco/openpi/packages/openpi-client`
- This completes the Python-side package layout intended by the repository docs, aside from the remaining core runtime dependencies still missing from the clean prefix.

### Step 14
- Re-ran `serve_policy.py --help` inside the clean `dexjoco/.conda/openpi` environment with sanitized `PYTHONPATH`.
- Result: the script now runs successfully and prints the expected CLI help.
- This verifies that the clean OpenPI environment is usable enough to import the documented serve path, and that the earlier blockers from the reused `../pi05` runtime have been removed.
- Observed warning:
  - installed `jax_cuda12_plugin` version does not match `jaxlib` version, so the CUDA plugin is ignored during this help-only validation.
- This warning does not block the documented environment creation itself, but it should be revisited before GPU policy serving.

### Step 15
- Ran a real `click_mouse` server smoke test with the clean `dexjoco/.conda/openpi` environment and the downloaded checkpoint:
  - `PI05_USE_CLEAN_OPENPI_ENV=1`
  - `PI05_TASK=click_mouse`
  - `PI05_POLICY_DIR=/nfs_global/S/yangrongzheng/evo-RL/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse`
  - `bash scripts/run_pi05_server.sh`
- Observed behavior:
  - the clean environment and config wiring are correct (`config=click_mouse`, checkpoint path correct)
  - the model loading path starts successfully (`INFO:root:Loading model...`)
  - Orbax checkpoint restore starts reading the checkpoint tree
- Observed warnings so far:
  - JAX CUDA plugin / jaxlib version mismatch warning
  - missing `_CHECKPOINT_METADATA` file inside `params/`, but restore did not immediately abort on that warning
- At the last observed point, the server process was still alive, so the next step is to test the client side (`eval`) or collect further serve logs.

### Step 16
- Ran a real client-side eval smoke test against the running `click_mouse` policy server:
  - `conda activate dexjoco/.conda/dexjoco`
  - `PI05_TASK=click_mouse`
  - `PI05_CONFIG_SET=rand_obj`
  - `PI05_EPISODES=1`
  - `bash scripts/run_pi05_eval.sh`
- Result inside the Codex environment:
  - the eval client starts and resolves the correct config file
  - but the websocket connection fails immediately with:
    - `PermissionError: [Errno 1] Operation not permitted`
- Root cause: the Codex sandbox forbids creating the localhost socket needed by the websocket client.
- Conclusion: the remaining blocker for rollout/eval/collection is no longer the DexJoCo/OpenPI environment setup; it is the sandbox's local networking restriction.
- This means the next real step (formal rollout or collection) must be run in the user's own terminal, not inside this Codex sandbox.

### Step 17
- Added a formal policy-rollout collection wrapper: `scripts/run_pi05_rollout_collect.sh`.
- Purpose:
  - reuse the existing DexJoCo OpenPI eval client as the rollout engine
  - standardize rollout output directory, episode count, host/port, and randomization flags for model-driven collection
- This fills the gap that the repository's existing `scripts/run_pi05_collect.sh` only covered teleoperation demonstration collection, not policy rollout collection.

### Step 18
- Fixed the formal rollout collection wrapper to use the actual installed module path:
  - from `dexjoco.dexjoco_openpi_client.eval_dexjoco_openpi`
  - to `dexjoco_openpi_client.eval_dexjoco_openpi`
- This aligns the wrapper with how the local DexJoCo package is installed in the environment.

### Step 19
- Investigated the official `click_mouse` checkpoint structure and confirmed it looks like an Orbax/JAX checkpoint, not a PyTorch safetensors checkpoint.
- Evidence:
  - contains `_CHECKPOINT_METADATA`, `params/_METADATA`, `params/manifest.ocdbt`, `train_state/manifest.ocdbt`
  - does not contain `model.safetensors` or `model.safetensors.index.json`
- Traced the failure to `openpi/src/openpi/models/model.py:restore_params`, which assumed `ckptr.metadata(...)` was always a dict and indexed it as `metadata["params"]`.
- Added a compatibility patch so `restore_params` also handles newer Orbax metadata objects returned directly for a pytree checkpoint path.

### Step 20
- Re-tested `click_mouse` server startup after patching `restore_params`.
- Result:
  - the previous fatal error `TypeError: 'StepMetadata' object is not subscriptable` no longer appears
  - server proceeds past checkpoint metadata parsing and remains alive in the loading phase
- Conclusion: the official DexJoCo checkpoint is indeed compatible with the JAX/Orbax serving path, and the immediate blocker was an Orbax metadata API compatibility bug in `restore_params`, not an incorrect checkpoint format.

### Step 21
- Continued the Orbax/JAX compatibility fix for the official `click_mouse` checkpoint.
- New failure showed a mismatch between the user-provided restore item (`StepMetadata`) and on-disk metadata tree (`dict`).
- Patched `openpi/src/openpi/models/model.py` again to call `ocp.args.PyTreeRestore(..., partial_restore=True)`.
- Rationale: this allows Orbax to restore only the matching subtree instead of requiring exact structural equality between the metadata wrapper object and the on-disk parameter tree.

### Step 22
- Observed a new client-side local connection blocker unrelated to model loading:
  - `InvalidProxy('127.0.0.1:10808', "scheme  isn't supported")`
- Root cause: inherited proxy environment variables were interfering with websocket connections to the local policy server.
- Patched `scripts/run_pi05_rollout_collect.sh` to automatically clear proxy variables for `127.0.0.1` / `localhost` rollout connections and force `NO_PROXY` / `no_proxy` to include local addresses.

### Step 23
- Investigated the remaining Orbax `StepMetadata` vs `dict` restore mismatch more directly.
- Confirmed from `params/_METADATA` that the on-disk checkpoint tree is a normal dict-shaped pytree with tuple-string keypaths.
- Updated `openpi/src/openpi/models/model.py` again so that, for newer Orbax metadata wrapper objects, it reconstructs a dict-shaped restore template from `params/_METADATA` instead of passing the wrapper object itself to `PyTreeRestore`.
- This is intended to eliminate the root-level `StepMetadata` vs `dict` type mismatch during partial restore.

### Step 24
- Downgraded the clean OpenPI environment from JAX/JAXLIB 0.10.1 to `0.5.3`, matching the CUDA12 plugin and the user's request.
- Result:
  - `jax==0.5.3`
  - `jaxlib==0.5.3`
  - `jax-cuda12-plugin==0.5.3`
- Pip reported two dependency drifts that may matter later:
  - `ml-dtypes` is `0.5.4` but `openpi` pins `0.4.1`
  - `tensorstore` is `0.1.84` but `openpi` pins `0.1.74`
- Next: re-run the `click_mouse` server smoke test under the downgraded JAX stack.

### Step 25
- Switched focus to the trained value model.
- Determined that the currently available local dataset already contains inferred value signals:
  - `complementary_info.value`
  - `complementary_info.advantage`
  - `complementary_info.acp_indicator`
- Added a minimal analysis script at `Evo-RL/scripts/eval_value_progress.py` that:
  - groups frames by episode
  - computes per-episode start/end/delta value statistics
  - resamples each successful trajectory to 100 points
  - exports a mean normalized progress curve and a delta histogram
