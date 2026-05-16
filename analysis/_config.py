"""Loader for the analysis-pipeline config (single YAML at project root).

Every function in `analysis/` (and `inference.py`) takes `config: dict` as its
first argument. The config is loaded once at CLI entry and passed through; no
module reads `dir_skyfinder.baseline` paths/constants directly.

Why a flat dict and not a dataclass: the YAML evolves often (new ablation paths,
new reference baselines). A flat dict + `Path(config[key])` at the use site keeps
edits localized.

Usage:
    from analysis._config import load_config
    cfg = load_config()                              # reads analysis_config.yaml
    cfg = load_config("path/to/other_config.yaml")   # explicit override
"""
from __future__ import annotations

from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path("analysis_config.yaml")


def load_config(path: Path | str | None = None) -> dict:
    """Read the analysis config YAML. Returns a flat dict.

    Path-valued entries stay as strings here -- consumers cast to `Path(...)` at
    the point of use (keeps the loader trivial and the dict round-trippable).
    """
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"analysis config not found: {p}\n"
            f"create {DEFAULT_CONFIG_PATH} (see analysis_config.yaml in the repo for the schema)"
        )
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{p} did not parse to a dict (got {type(data).__name__})")
    return data
