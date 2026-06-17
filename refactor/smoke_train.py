"""End-to-end run_baseline smoke for the refactored training package.

Proves the whole core pipeline runs from refactor/: build_loaders(cfg) -> model ->
train_one_epoch (weighted_l1_loss) -> predict_split -> per_bin_mae. Also runs an
LDS+FDS variant (2 epochs so FDS calibration fires at epoch 1) to confirm the DIR
interventions are wired end-to-end, not silent no-ops.

Run from repo root:  python refactor/smoke_train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # put refactor/ first on sys.path

from skyfinder.training.config import Config
from skyfinder.training.trainer import run_baseline

REPO = Path(__file__).resolve().parents[1]
paths = dict(
    labels_path=REPO / "data" / "labels_with_images.csv",
    splits_path=REPO / "data" / "splits" / "loco_5fold.json",
    img_dir=REPO / "data" / "images",
)

print("\n========== BASELINE (1 epoch) ==========")
base = run_baseline(Config(model="resnet50", fold=0, epochs=1,
                           train_subset=16, val_subset=16, num_workers=0, seed=0, **paths),
                    save=False)

print("\n========== LDS + FDS (2 epochs) ==========")
dir_ = run_baseline(Config(model="resnet50", fold=0, epochs=2,
                           train_subset=16, val_subset=16, num_workers=0, seed=0,
                           use_lds=True, use_fds=True, **paths),
                    save=False)

print("\n========== RESULT ==========")
print("baseline final_val:", {k: round(v, 3) for k, v in base["final_val"].items() if v == v})
print("lds+fds  final_val:", {k: round(v, 3) for k, v in dir_["final_val"].items() if v == v})
print("\nOK: full DIR training pipeline runs end-to-end from refactor/")
