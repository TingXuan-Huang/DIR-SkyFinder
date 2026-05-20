"""F-family training-label corruption.

Used during training (via `baseline.build_loaders`) to mutate the per-fold train df
before LDS bins are computed. Val/test are never touched.

Two helpers:
  - `fit_imputer(df, kind)` — pandas-only imputers (global / per-cam / per-cam-month means)
  - `corrupt_train_labels(train_df, cfg, seed)` — applies one of the F1/F2/F3/F5 modes

The schema for the F-family corruption block in YAML lives in `ablation.md`.
"""
from __future__ import annotations

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
    here -- the caller (baseline.build_loaders) only passes the train slice.

    cfg schema (one mode per config block):

      mode: "random"          # F1: drop X% uniformly at random
        rate: 0.25            # fraction in [0, 1]
        drop_mode: "impute"   # "impute" -> fill with imputer; "drop" -> remove rows
        imputer: "per_cam_month_mean"  # any kind from fit_imputer

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
          f"({100 * n_corrupt / max(n_total, 1):.1f}%)")

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
