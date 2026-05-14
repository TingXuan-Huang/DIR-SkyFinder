"""Ablation utilities for DIR-SkyFinder.

Two roles:

  1. **Training-time corruption** for F-family experiments. `baseline.py` imports
     `corrupt_train_labels` and calls it on the per-fold train df before building
     loaders. Val/test are never touched.

  2. **C2 metadata-only baseline** (no GPU). HistGradientBoostingRegressor on
     (CamId, Hour, Month, Latitude, Longitude). Reports per-fold val + test MAE.

Run from project root:
    python ablation.py c2
        [--labels data/labels_with_images.csv]
        [--splits data/splits/loco_5fold.json]
        [--out ablations/results/c2_metadata_only.json]

See ablation.md for the full plan and ablations/config_ab.yaml for the
GPU-driven experiments (D4 / E1 / F1 / F2 / F3 / F5).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Imputers (used by F1 / F2 / F3 to fill corrupted labels)
# ============================================================

def fit_imputer(df: pd.DataFrame, kind: str = "per_cam_month_mean"):
    """Fit a metadata -> TempM imputer on `df`. Returns predict(df_new) -> ndarray.

    Available imputers (simplest -> most expressive):
      "global_mean"        : one scalar
      "per_cam_mean"       : groupby CamId
      "per_cam_month_mean" : groupby (CamId, Month), fall back to per-cam, then global

    Note: Lat/Lon are redundant with CamId inside F (the imputer is fit on kept
    rows of one fold and applied to corrupted rows of the SAME fold -- same
    camera set), so we only use (CamId, Month) for the default imputer.
    """
    if kind == "global_mean":
        mean = float(df["TempM"].mean())
        return lambda d: np.full(len(d), mean, dtype=np.float32)

    if kind == "per_cam_mean":
        table = df.groupby("CamId")["TempM"].mean()
        global_mean = float(df["TempM"].mean())

        def predict(d: pd.DataFrame) -> np.ndarray:
            return d["CamId"].map(table).fillna(global_mean).to_numpy(dtype=np.float32)
        return predict

    if kind == "per_cam_month_mean":
        cam_month = df.groupby(["CamId", "Month"])["TempM"].mean().to_dict()
        cam = df.groupby("CamId")["TempM"].mean().to_dict()
        global_mean = float(df["TempM"].mean())

        def predict(d: pd.DataFrame) -> np.ndarray:
            keys = list(zip(d["CamId"].to_numpy(), d["Month"].to_numpy()))
            out = np.empty(len(d), dtype=np.float32)
            for i, k in enumerate(keys):
                v = cam_month.get(k)
                if v is None or np.isnan(v):
                    v = cam.get(k[0], global_mean)
                out[i] = v
            return out
        return predict

    raise ValueError(f"unknown imputer kind: {kind!r}")


# ============================================================
# F-family label corruption
# ============================================================

def corrupt_train_labels(train_df: pd.DataFrame, cfg: dict, seed: int = 42) -> pd.DataFrame:
    """Apply F-family corruption to a per-fold training df. Returns a NEW df.

    Always operates on a copy; never mutates `train_df`. Val/test are not seen
    here -- the caller (baseline.run_baseline) only passes the train slice.

    cfg schema (one mode per config block):

      mode: "random"          # F1: drop X% uniformly at random
        rate: 0.25            # fraction in [0, 1]
        drop_mode: "impute"   # "impute" -> fill with imputer; "drop" -> remove rows
        imputer: "per_cam_month_mean"  # any kind from fit_imputer (only used if drop_mode=impute)

      mode: "range"           # F2: drop labels whose true TempM is in [lo, hi)
        range: [25.0, 30.0]
        drop_mode: "impute"   # or "drop"
        imputer: "per_cam_month_mean"

      mode: "rare_bin"        # F3: drop fraction `drop_prob` from rare bins only
        drop_prob: 0.8
        bin_width: 1.0
        few_max: 20           # bins with count < few_max are "few" (matches per_bin_mae)
        drop_mode: "impute"
        imputer: "per_cam_month_mean"

      mode: "noise"           # F5: add Gaussian noise to ALL labels (no mask)
        noise_std: 3.0
    """
    rng = np.random.RandomState(seed)
    out = train_df.reset_index(drop=True).copy()
    mode = cfg["mode"]

    # F5: pure additive noise on every row, no mask, no imputer
    if mode == "noise":
        std = float(cfg["noise_std"])
        noise = rng.normal(0.0, std, size=len(out)).astype(np.float32)
        out["TempM"] = (out["TempM"].astype(np.float32).to_numpy() + noise).astype(np.float32)
        return out

    # F1 / F2 / F3: build a per-row boolean mask of rows to corrupt
    if mode == "random":
        mask = rng.rand(len(out)) < float(cfg["rate"])

    elif mode == "range":
        lo, hi = cfg["range"]
        mask = ((out["TempM"] >= float(lo)) & (out["TempM"] < float(hi))).to_numpy()

    elif mode == "rare_bin":
        bin_width = float(cfg.get("bin_width", 1.0))
        few_max = int(cfg.get("few_max", 20))
        # Same bucketization convention as lds.py: bin = floor((temp - MIN_TEMP) / bin_width)
        MIN_TEMP = -30.0
        bins = ((out["TempM"].to_numpy() - MIN_TEMP) / bin_width).astype(int)
        bins = np.clip(bins, 0, None)
        counts = np.bincount(bins)
        is_few_bin = counts < few_max
        in_few = is_few_bin[bins]
        drop_prob = float(cfg["drop_prob"])
        mask = in_few & (rng.rand(len(out)) < drop_prob)

    else:
        raise ValueError(f"unknown corruption mode: {mode!r}")

    n_corrupt = int(mask.sum())
    n_total = len(out)
    print(f"[corrupt] mode={mode}  n_corrupt={n_corrupt:,}/{n_total:,}  "
          f"({100*n_corrupt/max(n_total,1):.1f}%)")

    drop_mode = cfg.get("drop_mode", "impute")

    if drop_mode == "drop":
        kept = out.loc[~mask].reset_index(drop=True)
        print(f"[corrupt] drop_mode=drop -> {len(kept):,} rows remain")
        return kept

    # drop_mode == "impute"
    kept = out.loc[~mask]
    if len(kept) == 0:
        raise ValueError("corruption mask covers ALL rows; no rows left to fit imputer")
    if n_corrupt == 0:
        return out  # nothing to impute
    imputer_kind = cfg.get("imputer", "per_cam_month_mean")
    predict = fit_imputer(kept, imputer_kind)
    imputed = predict(out.loc[mask])
    out.loc[mask, "TempM"] = imputed.astype(out["TempM"].dtype)
    print(f"[corrupt] drop_mode=impute imputer={imputer_kind}  "
          f"imputed range [{imputed.min():.2f}, {imputed.max():.2f}] "
          f"mean={imputed.mean():.2f}")
    return out


# ============================================================
# C2: metadata-only baseline (no GPU, CPU only)
# ============================================================

def _per_bin_mae(y_true: np.ndarray, y_pred: np.ndarray, train_y: np.ndarray,
                 bin_w: float = 2.0) -> dict:
    """Same many/medium/few convention as baseline.per_bin_mae."""
    lo = min(train_y.min(), y_true.min())
    hi = max(train_y.max(), y_true.max())
    edges = np.arange(np.floor(lo / bin_w) * bin_w,
                      np.ceil(hi / bin_w) * bin_w + bin_w, bin_w)
    train_hist, _ = np.histogram(train_y, bins=edges)
    idx = np.clip(np.digitize(y_true, edges) - 1, 0, len(edges) - 2)
    err = np.abs(y_true - y_pred)
    out = {"overall": float(err.mean())}
    for name, n_lo, n_hi in [("many", 100, np.inf), ("medium", 20, 100), ("few", 0, 20)]:
        sel = np.isin(idx, np.where((train_hist >= n_lo) & (train_hist < n_hi))[0])
        out[name] = float(err[sel].mean()) if sel.any() else float("nan")
    return out


def run_c2(labels_path: Path, splits_path: Path, out_path: Path, seed: int = 0) -> dict:
    """Fit HistGradientBoostingRegressor per fold on metadata; save per-fold MAE.

    Reports MAE on:
      val  -- held-out images from train cameras (in-distribution; matches CNN's val)
      test -- LOCO held-out cameras (out-of-distribution; the honest comparison)
    """
    from sklearn.ensemble import HistGradientBoostingRegressor  # local import: sklearn is heavy

    df = pd.read_csv(labels_path)
    splits = json.loads(Path(splits_path).read_text())

    # HistGBR's native categorical support: we need a single column of int category codes.
    # Use ALL CamId values seen in the full df so codes are consistent across folds.
    cam_dtype = pd.CategoricalDtype(categories=sorted(df["CamId"].unique()))
    df = df.copy()
    df["CamId_cat"] = df["CamId"].astype(cam_dtype).cat.codes
    feature_cols = ["CamId_cat", "Hour", "Month", "Latitude", "Longitude"]

    results = {"per_fold": []}
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

        val_mae = _per_bin_mae(val_df["TempM"].to_numpy(), val_pred, y_train)
        test_mae = _per_bin_mae(test_df["TempM"].to_numpy(), test_pred, y_train)

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

    # Summary across folds
    val_overall = np.array([r["val"]["overall"] for r in results["per_fold"]])
    test_overall = np.array([r["test"]["overall"] for r in results["per_fold"]])
    results["summary"] = {
        "val_mae_mean": float(val_overall.mean()),
        "val_mae_std": float(val_overall.std()),
        "test_mae_mean": float(test_overall.mean()),
        "test_mae_std": float(test_overall.std()),
        "seed": seed,
    }
    print(f"[summary] val MAE:  {results['summary']['val_mae_mean']:.3f} "
          f"± {results['summary']['val_mae_std']:.3f}")
    print(f"[summary] test MAE: {results['summary']['test_mae_mean']:.3f} "
          f"± {results['summary']['test_mae_std']:.3f}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[saved] {out_path}")
    return results


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="task", required=True)

    p_c2 = sub.add_parser("c2", help="metadata-only baseline (no GPU)")
    p_c2.add_argument("--labels", type=Path, default=Path("data/labels_with_images.csv"))
    p_c2.add_argument("--splits", type=Path, default=Path("data/splits/loco_5fold.json"))
    p_c2.add_argument("--out", type=Path, default=Path("ablations/results/c2_metadata_only.json"))
    p_c2.add_argument("--seed", type=int, default=0)

    a = ap.parse_args()
    if a.task == "c2":
        run_c2(a.labels, a.splits, a.out, seed=a.seed)


if __name__ == "__main__":
    main()
