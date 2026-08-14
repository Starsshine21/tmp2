# ReCap / OGPO migration bundle

This repository contains the code and experiment configuration needed to move the current ReCap and OGPO work to another server.

## Layout

- `recap/Evo-RL/`: current ReCap-side Evo-RL working tree, including local source and configuration changes.
- `recap/recap_click_mouse/`: click-mouse ReCap experiment scripts, configs, and the small source/runtime overlays used by those jobs.
- `ogpo/dexjoco/`: current OGPO/DexJoCo working tree, including OGPO source, OpenPI integration, tests, configs, SLURM launchers, and documentation.

## Intentionally omitted

Checkpoints, model weights, datasets, local environments, caches, generated outputs, logs, videos, images, PDFs, and simulator mesh/assets are not included. Absolute cluster paths in SLURM and shell scripts may need to be updated on the destination server.

The pre-existing PI05 deployment files in this repository are retained unchanged.
