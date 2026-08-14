#!/usr/bin/env python3
"""Run the isolated ReCap OpenPI server with the local Orbax/JAX shim."""

from __future__ import annotations

import runpy
from pathlib import Path

import jax


if not hasattr(jax.monitoring, "record_scalar"):
    setattr(jax.monitoring, "record_scalar", lambda *args, **kwargs: None)

runpy.run_path(str(Path(__file__).with_name("serve_policy.py")), run_name="__main__")
