"""Create 5-fold leave-one-camera-out splits.

Partitions the trainable cameras into 5 disjoint groups, then for each fold:
  - test  = rows from cameras in that fold's group  (~1/5 of cameras)
  - val   = random 10% of the remaining rows
  - train = the other 90% of remaining rows

Test cameras never appear in train/val (leave-one-camera-out for evaluation).
Val rows come from train cameras — val is for early-stopping / hyperparam picks,
test is what we report.

Writes data/splits/loco_5fold.json: a list of {fold, test_cams, train, val, test}
where train/val/test are row indices into data/labels_with_images.csv.

Run from project root:  python data/splits.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
LABELS_PATH = DATA_DIR / "labels_with_images.csv"
OUT_PATH = DATA_DIR / "splits" / "loco_5fold.json"

N_FOLDS = 5
VAL_FRAC = 0.10
SEED = 0


def main() -> None:
    df = pd.read_csv(LABELS_PATH)
    n = len(df)
    rng = np.random.RandomState(SEED)

    cams = sorted(df["CamId"].unique().tolist())
    rng.shuffle(cams)
    cam_groups = [g.tolist() for g in np.array_split(cams, N_FOLDS)]

    folds = []
    for f, test_cams in enumerate(cam_groups):
        test_mask = df["CamId"].isin(test_cams)
        test_idx = df.index[test_mask].tolist()
        rem_idx = df.index[~test_mask].to_numpy()
        rng.shuffle(rem_idx)
        n_val = int(len(rem_idx) * VAL_FRAC)
        val_idx = rem_idx[:n_val].tolist()
        train_idx = rem_idx[n_val:].tolist()

        # Sanity: disjoint and complete coverage
        assert len(set(train_idx) | set(val_idx) | set(test_idx)) == n
        assert len(train_idx) + len(val_idx) + len(test_idx) == n

        folds.append({
            "fold": f,
            "test_cams": sorted(test_cams),
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        })
        print(f"[fold {f}]  test_cams={len(test_cams):2d}  "
              f"train={len(train_idx):>6,}  val={len(val_idx):>5,}  test={len(test_idx):>6,}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(folds))
    print(f"[saved] {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
