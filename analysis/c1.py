"""C1: constant predictors. No GPU, no training -- just pandas groupby.

Three predictors, fit on each fold's train slice, evaluated on val + test:
  - global_mean        : single scalar  = mean(train TempM)
  - per_cam_mean       : groupby CamId  -> mean(TempM); fall back to global
  - per_cam_month_mean : groupby (CamId, Month) -> mean(TempM); falls back per-cam, then global

Output: `ablations/results/c1_constants.json` with `per_fold` rows shaped like
C2's, so `analysis/aggregate.py` can ingest them with the same code path.

Run:
    python analysis.py c1
        [--labels data/labels_with_images.csv]
        [--splits data/splits/loco_5fold.json]
        [--out ablations/results/c1_constants.json]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the same per-bin convention as baseline.per_bin_mae and ablation._per_bin_mae.
from ablation import _per_bin_mae


PREDICTORS = ("global_mean", "per_cam_mean", "per_cam_month_mean")


def _fit_predict(train_df: pd.DataFrame, kind: str) -> dict:
    """Returns a dict with predict(df) -> ndarray, plus a stash of `global_mean` for backstop."""
    gmean = float(train_df["TempM"].mean())
    if kind == "global_mean":
        def predict(d):
            return np.full(len(d), gmean, dtype=np.float32)
    elif kind == "per_cam_mean":
        cam_tbl = train_df.groupby("CamId")["TempM"].mean()
        def predict(d):
            return d["CamId"].map(cam_tbl).fillna(gmean).to_numpy(dtype=np.float32)
    elif kind == "per_cam_month_mean":
        cam_month = train_df.groupby(["CamId", "Month"])["TempM"].mean().to_dict()
        cam = train_df.groupby("CamId")["TempM"].mean().to_dict()
        def predict(d):
            keys = list(zip(d["CamId"].to_numpy(), d["Month"].to_numpy()))
            out = np.empty(len(d), dtype=np.float32)
            for i, k in enumerate(keys):
                v = cam_month.get(k)
                if v is None or np.isnan(v):
                    v = cam.get(k[0], gmean)
                out[i] = v
            return out
    else:
        raise ValueError(f"unknown predictor: {kind!r}")
    return {"predict": predict}


def run_c1(labels_path: Path, splits_path: Path, out_path: Path) -> dict:
    df = pd.read_csv(labels_path)
    splits = json.loads(Path(splits_path).read_text())

    results: dict = {"per_fold": [], "predictors": list(PREDICTORS)}
    for f in splits:
        fold = f["fold"]
        train_df = df.iloc[f["train"]]
        val_df   = df.iloc[f["val"]]
        test_df  = df.iloc[f["test"]]
        y_train = train_df["TempM"].to_numpy()
        for kind in PREDICTORS:
            mdl = _fit_predict(train_df, kind)
            val_pred  = mdl["predict"](val_df)
            test_pred = mdl["predict"](test_df)
            val_mae  = _per_bin_mae(val_df["TempM"].to_numpy(), val_pred, y_train)
            test_mae = _per_bin_mae(test_df["TempM"].to_numpy(), test_pred, y_train)
            results["per_fold"].append({
                "fold": fold,
                "predictor": kind,
                "n_train": int(len(train_df)),
                "n_val":   int(len(val_df)),
                "n_test":  int(len(test_df)),
                "val":  val_mae,
                "test": test_mae,
            })
            print(f"[fold {fold}] {kind:>18s}  "
                  f"val={val_mae['overall']:.3f}  test={test_mae['overall']:.3f}  "
                  f"(val few={val_mae['few']:.3f}  test few={test_mae['few']:.3f})")

    # Summary: mean +/- std over folds, per predictor.
    summary = {}
    for kind in PREDICTORS:
        rows = [r for r in results["per_fold"] if r["predictor"] == kind]
        v = np.array([r["val"]["overall"]  for r in rows])
        t = np.array([r["test"]["overall"] for r in rows])
        summary[kind] = {
            "val_mae_mean":  float(v.mean()), "val_mae_std":  float(v.std()),
            "test_mae_mean": float(t.mean()), "test_mae_std": float(t.std()),
        }
        print(f"[summary] {kind:>18s}  "
              f"val={summary[kind]['val_mae_mean']:.3f} +/- {summary[kind]['val_mae_std']:.3f}  "
              f"test={summary[kind]['test_mae_mean']:.3f} +/- {summary[kind]['test_mae_std']:.3f}")
    results["summary"] = summary

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[saved] {out_path}")
    return results
