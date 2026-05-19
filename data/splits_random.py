"""Generate random row-level 5-fold splits, *ignoring* camera identity.

This is the control for Analysis #8 in REPORT.md §11: if val ≈ test under a
random row split but val << test under LOCO, the val→test gap is caused by
camera-shift, not just sample variance.

Output: data/splits/random_5fold.json — same schema as loco_5fold.json so the
existing trainer can consume it by swapping `splits_path` in config.yaml.

For each fold:
  - test = a random 1/5 of all rows (every camera contributes)
  - val  = random 10% of the remaining rows
  - train = the rest
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
LABELS_PATH = DATA_DIR / "labels_with_images.csv"
OUT_PATH = DATA_DIR / "splits" / "random_5fold.json"

N_FOLDS = 5
VAL_FRAC = 0.10
SEED = 0


def main() -> None:
    df = pd.read_csv(LABELS_PATH)
    n = len(df)
    rng = np.random.RandomState(SEED)

    perm = rng.permutation(n)
    fold_chunks = np.array_split(perm, N_FOLDS)

    folds = []
    for f in range(N_FOLDS):
        test_idx = fold_chunks[f].tolist()
        rem = np.concatenate([fold_chunks[j] for j in range(N_FOLDS) if j != f])
        rng.shuffle(rem)
        n_val = int(len(rem) * VAL_FRAC)
        val_idx = rem[:n_val].tolist()
        train_idx = rem[n_val:].tolist()

        assert len(set(train_idx) | set(val_idx) | set(test_idx)) == n
        assert len(train_idx) + len(val_idx) + len(test_idx) == n

        # Diagnostic: how many cameras are in each split?
        train_cams = df.iloc[train_idx]["CamId"].nunique()
        val_cams = df.iloc[val_idx]["CamId"].nunique()
        test_cams = df.iloc[test_idx]["CamId"].nunique()

        folds.append({
            "fold": f,
            # No leave-one-camera concept here; keep field for schema parity.
            "test_cams": sorted(df.iloc[test_idx]["CamId"].unique().tolist()),
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        })
        print(f"[fold {f}]  train={len(train_idx):>6,} ({train_cams} cams)  "
              f"val={len(val_idx):>5,} ({val_cams} cams)  "
              f"test={len(test_idx):>6,} ({test_cams} cams)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(folds))
    print(f"[saved] {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes)")
    print()
    print("To use this split in an experiment, set in config.yaml:")
    print(f"  splits_path: {OUT_PATH}")
    print("then run the standard pipeline. Expected result if camera-shift "
          "is the cause:  val MAE ≈ test MAE ≈ 2.8 °C.")


if __name__ == "__main__":
    main()
