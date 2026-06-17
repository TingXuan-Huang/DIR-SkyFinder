"""Smoke test for the refactored build_loaders(cfg).

Run from repo root:  python refactor/smoke_dataloader.py

Injects real repo-root data paths into Config (during the transition the module-level
defaults resolve to refactor/data/, which does not exist — injecting paths is exactly
what Option 2 made possible). Once this package is promoted to skyfinder/, the defaults
resolve correctly and the explicit paths become optional.
"""
from pathlib import Path

from skyfinder.training.config import Config
from skyfinder.training.dataloader import build_loaders

REPO = Path(__file__).resolve().parents[1]  # repo root

cfg = Config(
    fold=0, train_subset=8, val_subset=8, num_workers=0,
    labels_path=REPO / "data" / "labels_with_images.csv",
    splits_path=REPO / "data" / "splits" / "loco_5fold.json",
    img_dir=REPO / "data" / "images",
)

train_loader, val_loader, train_df, val_df = build_loaders(cfg)
x, y, w = next(iter(train_loader))

print("x", tuple(x.shape), x.dtype, "| y", tuple(y.shape), y.dtype, "| w", tuple(w.shape))
print("train_df", len(train_df), "val_df", len(val_df))
print("y sample (°C):", [round(v, 1) for v in y.tolist()])
assert tuple(x.shape) == (8, 3, 224, 224), x.shape
assert tuple(y.shape) == (8,) and tuple(w.shape) == (8,)
print("OK")
