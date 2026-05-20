"""Analysis side of SkyFinder DIR.

Files:
    config_loader.py        — load_config() reads configs/analysis.yaml
    style.py                — Nature-style matplotlib helpers
    aggregate.py            — JSON results → flat DataFrame
    baselines_constant.py   — C1: constant predictors (no GPU)
    baselines_metadata.py   — C2: HistGradientBoostingRegressor on metadata
    skymask_inference.py    — D1: sky-mask inference (no retraining)
    linear_probe.py         — D4: linear-probe delta summary
    extract_embeddings.py   — penultimate features for trained checkpoints
    extract_trajectory.py   — per-epoch feature extraction
    corrupt_labels.py       — F-family train-label corruption
    dist.py                 — distribution-shift analysis
    figures/                — all publication figures (subpackage)
"""
from __future__ import annotations

# Public re-exports for the most common analysis entry points.
from skyfinder.analysis.config_loader import load_config

__all__ = ["load_config"]
