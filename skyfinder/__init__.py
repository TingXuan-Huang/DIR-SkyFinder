"""SkyFinder: Deep Imbalanced Regression on outdoor webcam temperature data.

Top-level re-exports for ergonomics:
    from skyfinder import Config, run_baseline

For the CLI, install with `pip install -e .` and use:
    skyfinder train --family main
    skyfinder analyze all
    skyfinder figures
"""
from __future__ import annotations

__version__ = "0.2.0"

# Re-export the most-used symbols. Imports are inside __all__ resolution to keep
# `import skyfinder` fast — only loaded on attribute access.
__all__ = ["Config", "run_baseline"]


def __getattr__(name: str):
    if name == "Config":
        from skyfinder.training.config import Config
        return Config
    if name == "run_baseline":
        from skyfinder.training.trainer import run_baseline
        return run_baseline
    raise AttributeError(f"module 'skyfinder' has no attribute {name!r}")
