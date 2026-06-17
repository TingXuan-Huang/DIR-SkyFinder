"""Training config dataclass + module-level paths.

`Config` holds every training option (baseline, LDS, FDS, corruption, snapshots).
All LDS/FDS fields default off so existing baseline calls are unchanged.

Paths default to the module-level constants (DATA, LABELS, SPLITS, IMG_DIR), but are
now also Config fields (`labels_path`, `splits_path`, `img_dir`) so you can inject them
per-run instead of reassigning module globals:
    Config(labels_path=Path("/content/drive/.../labels_with_images.csv"), ...)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Project root = parent of skyfinder/ package.
# __file__ = .../skyfinder/training/config.py -> parents[2] = repo root.
PROJ = Path(__file__).resolve().parents[2]
DATA = PROJ / "data"
LABELS = DATA / "labels_with_images.csv"
SPLITS = DATA / "splits" / "loco_5fold.json"
IMG_DIR = DATA / "images"
RESULTS_DIR = PROJ / "results"


@dataclass
class Config:
    """Training config. Pass to `run_baseline(cfg=...)` or via kwargs."""

    model: str = "resnet50"          # "resnet50" or "vit_b_16"
    fold: int = 0                    # 0 through 4 for the 5-fold split
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    num_workers: int = 2
    train_subset: int | None = None  # cap train rows for smoke tests
    val_subset: int | None = None
    seed: int = 0
    run_name: str | None = None      # auto-generated from model/fold/epochs/time if None

    # Data paths (injectable; default to the module-level constants).
    labels_path: Path = LABELS
    splits_path: Path = SPLITS
    img_dir: Path = IMG_DIR

    # LDS (loss-side reweighting; no architecture change)
    use_lds: bool = False
    lds_kernel: str = "gaussian"     # "gaussian" | "triang" | "laplace"
    lds_ks: int = 5                  # odd
    lds_sigma: float = 2.0
    lds_reweight: str = "sqrt_inv"   # "none" | "sqrt_inv" | "inverse"

    # FDS (architecture-side feature calibration)
    use_fds: bool = False
    fds_kernel: str = "gaussian"
    fds_ks: int = 5
    fds_sigma: float = 2.0
    fds_momentum: float = 0.9
    fds_start_smooth: int = 1        # epoch to begin applying calibration

    # Bucket scheme — applies to both LDS and FDS when active
    bin_width: float = 1.0           # °C per bucket; matches DIR paper convention

    # Ablation hooks
    freeze_backbone: bool = False    # D4: linear probe — train only the regression head
    corruption: dict | None = None   # F-family: train-label corruption

    # Embedding-trajectory hook (see skyfinder.analysis.extract_trajectory).
    # When >0, dumps state_dict to <run>_ep{N}.pt at epoch 0 and every Nth epoch.
    snapshot_every: int = 0
