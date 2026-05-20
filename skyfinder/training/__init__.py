"""Training side of SkyFinder DIR.

Files (ML-pipeline conventions):
    config.py       — Config dataclass + paths
    dataloader.py   — SkyFinderDataset, build_loaders, transforms
    model.py        — build_model, FDSModel
    engine.py       — train_one_epoch, predict_split, per_bin_mae, get_device
    checkpoint.py   — save/load model weights + training state
    trainer.py      — run_baseline orchestration
    families.py     — FAMILIES registry
    lds.py          — Label Distribution Smoothing (DIR-specific; kept by name)
    fds.py          — Feature Distribution Smoothing (DIR-specific; kept by name)
    migrate.py      — one-shot results-layout migration
    diagnostics.py  — saved-run convergence diagnostics
"""
from __future__ import annotations

from skyfinder.training.config import Config, DATA, IMG_DIR, LABELS, RESULTS_DIR, SPLITS
from skyfinder.training.trainer import run_baseline

__all__ = ["Config", "run_baseline", "DATA", "IMG_DIR", "LABELS", "RESULTS_DIR", "SPLITS"]
