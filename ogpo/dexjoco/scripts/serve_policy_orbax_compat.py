#!/usr/bin/env python3
"""Run OpenPI's policy server with the Orbax/JAX compatibility shim installed."""

from __future__ import annotations

import runpy
from pathlib import Path

import jax


if not hasattr(jax.monitoring, "record_scalar"):
    setattr(jax.monitoring, "record_scalar", lambda *args, **kwargs: None)

root = Path(__file__).resolve().parents[1]
runpy.run_path(str(root / "openpi" / "scripts" / "serve_policy.py"), run_name="__main__")
