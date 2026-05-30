"""Test helpers.

These tests deliberately load the engine's *pure-Python* modules directly by
file path, bypassing ``engine/__init__.py`` (which imports numpy and other
heavy ML dependencies that are not installed in CI). Only modules that depend
solely on the standard library are exercised here:

  - engine/pulse.py        (thermal pulse controller)
  - engine/monitor.py      (hardware snapshot dataclasses)
  - engine/personality.py  (SQLite-backed routing learner)

The heavy-dependency modules (dispatcher, ops, executor, the NPU/GPU/ONNX
schedulers, etc.) require numpy/torch/onnx/kp and real hardware, so they are
not imported or executed by this test suite.
"""

import importlib.util
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent / "engine"


def load_module(name: str):
    """Load a single engine module by file path, without importing the package."""
    path = ENGINE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"engine_{name}_isolated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
