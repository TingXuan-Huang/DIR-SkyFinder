"""C2: metadata-only baseline.

Trains a `HistGradientBoostingRegressor` per fold on (CamId, Hour, Month, Latitude,
Longitude) -- no image features at all. Reports val + test per-bin MAE.

Output: `<baselines_metadata_path>` (from config). Same per_fold schema as the constant baselines, so
`analysis/aggregate.py` ingests them with one code path.

Implication if C2 beats the CNN: vision isn't adding much; the CNN is mostly
learning per-camera priors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyfinder.training.engine import per_bin_mae


def run_baselines_metadata(config: dict, out_path: Path | str | None = None, seed: int = 0) -> dict:
    """Fit HistGradientBoostingRegressor per fold on metadata; save per-fold MAE.

    Reports MAE on:
      val  -- held-out images from train cameras (in-distribution; matches CNN's val)
      test -- LOCO held-out cameras (out-of-distribution; the honest comparison)

    Historical note: this was the "C2" ablation in `docs/ablation_catalog.md`.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor   # heavy: local import

    labels_path = Path(config["labels_path"])
    splits_path = Path(config["splits_path"])
    out_path = Path(out_path) if out_path is not None else Path(config["baselines_metadata_path"])

    df = pd.read_csv(labels_path)
    splits = json.loads(splits_path.read_text())

    # HistGBR's native categorical support: single column of int codes.
    # Use ALL CamId values seen in the full df so codes are consistent across folds.
    cam_dtype = pd.CategoricalDtype(categories=sorted(df["CamId"].unique()))
    df = df.copy()
    df["CamId_cat"] = df["CamId"].astype(cam_dtype).cat.codes
    feature_cols = ["CamId_cat", "Hour", "Month", "Latitude", "Longitude"]

    results: dict = {"per_fold": []}
    for f in splits:
        fold = f["fold"]
        train_df = df.iloc[f["train"]]
        val_df = df.iloc[f["val"]]
        test_df = df.iloc[f["test"]]

        X_train = train_df[feature_cols].to_numpy()
        y_train = train_df["TempM"].to_numpy()

        # CamId_cat is at index 0; mark it as categorical for the splitter.
        model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=6,
            categorical_features=[0],
            random_state=seed,
        )
        model.fit(X_train, y_train)

        val_pred = model.predict(val_df[feature_cols].to_numpy())
        test_pred = model.predict(test_df[feature_cols].to_numpy())

        val_mae = per_bin_mae(val_df["TempM"].to_numpy(), val_pred, y_train)
        test_mae = per_bin_mae(test_df["TempM"].to_numpy(), test_pred, y_train)

        results["per_fold"].append({
            "fold": fold,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "test_cams": f["test_cams"],
            "val": val_mae,
            "test": test_mae,
        })
        print(f"[fold {fold}] val={val_mae['overall']:.3f}  test={test_mae['overall']:.3f}  "
              f"(val few={val_mae['few']:.3f}  test few={test_mae['few']:.3f})")

    # Summary across folds.
    val_overall = np.array([r["val"]["overall"] for r in results["per_fold"]])
    test_overall = np.array([r["test"]["overall"] for r in results["per_fold"]])
    results["summary"] = {
        "val_mae_mean":  float(val_overall.mean()),  "val_mae_std":  float(val_overall.std()),
        "test_mae_mean": float(test_overall.mean()), "test_mae_std": float(test_overall.std()),
        "seed": seed,
    }
    print(f"[summary] val MAE:  {results['summary']['val_mae_mean']:.3f} "
          f"± {results['summary']['val_mae_std']:.3f}")
    print(f"[summary] test MAE: {results['summary']['test_mae_mean']:.3f} "
          f"± {results['summary']['test_mae_std']:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[saved] {out_path}")
    return results
