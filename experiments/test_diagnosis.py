"""Diagnostic figures for the val (in-camera) vs test (LOCO) MAE gap.

Produces eight PNG/PDF pairs in `figures_diag/`:

  diag_per_camera_test_mae.{png,pdf}        -- Analysis #1
  diag_per_fold_temp_hist.{png,pdf}         -- Analysis #2
  diag_geo_scatter.{png,pdf}                -- Analysis #3
  diag_ood_distance_vs_error.{png,pdf}      -- Analysis #4
  diag_per_camera_pred_vs_true.{png,pdf}    -- Analysis #5
  diag_per_bin_test_with_baselines.{png,pdf}-- Analysis #6 (CNN vs C1 vs C2 per bin)
  diag_month_hour_coverage.{png,pdf}        -- Analysis #9 (train vs test, month x hour)
  diag_worst_camera_case_study.{png,pdf}    -- Analysis #10 (worst test cams + ImageNet kNN)

Run:
    python -m analysis.test_diagnosis

Inputs (relative to repo root):
  - data/labels_with_images.csv
  - data/splits/loco_5fold.json
  - final_results/results/test_inference.json     (ResNet-50, fold 0)
  - server_results/results/*.json                 (val_preds across folds)
  - server_results/ablations/results/c1_constants.json
  - server_results/ablations/results/c2_metadata_only.json
  - data/images/<CamId>/<Filename>.jpg            (only needed for #10)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skyfinder.analysis.style import PALETTE, apply_nature_style, figsize

apply_nature_style()

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "labels_with_images.csv"
SPLITS = ROOT / "data" / "splits" / "loco_5fold.json"
TEST_INF = ROOT / "final_results" / "results" / "test_inference.json"
VAL_DIR = ROOT / "server_results" / "results"
C1_JSON = ROOT / "server_results" / "ablations" / "results" / "c1_constants.json"
C2_JSON = ROOT / "server_results" / "ablations" / "results" / "c2_metadata_only.json"
IMG_DIR = ROOT / "data" / "images"
OUT = ROOT / "figures_diag"
OUT.mkdir(exist_ok=True)


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / name}.png")


def _load_test_inference() -> tuple[dict, pd.DataFrame, list]:
    df = pd.read_csv(LABELS)
    splits = json.loads(SPLITS.read_text())
    runs = json.loads(TEST_INF.read_text())["runs"]
    return runs, df, splits


def _align_preds_to_split(test_ys: np.ndarray, sub: pd.DataFrame) -> np.ndarray:
    """Return per-pred row indices into `sub` (or -1 for dropped extras).

    The labels CSV may have lost a handful of rows since inference ran (we
    observed 3 extras for fold 0 baseline). We walk both sequences and, when
    a pred y doesn't match the next split y, treat that pred as a dropped row
    and tentatively assign it to the *current* camera block.
    """
    out = np.full(len(test_ys), -1, dtype=np.int64)
    j = 0  # index into sub
    sub_ys = sub["TempM"].to_numpy()
    n_sub = len(sub_ys)
    for i, y in enumerate(test_ys):
        if j < n_sub and abs(float(y) - float(sub_ys[j])) < 1e-3:
            out[i] = j
            j += 1
        else:
            # Extra/dropped row. Inherit current row's index so it still
            # gets assigned to the surrounding camera in aggregations.
            out[i] = min(j, n_sub - 1)
    return out


def _per_camera_records(runs: dict, df: pd.DataFrame, splits: list) -> pd.DataFrame:
    """Long-format: one row per (run, image) with cam_id and abs_err."""
    rows = []
    for run_name, r in runs.items():
        fold = r["fold"]
        kind = r["config_kind"]
        preds = np.asarray(r["test_preds"], dtype=np.float32)
        ys = np.asarray(r["test_ys"], dtype=np.float32)
        idx = splits[fold]["test"]
        sub = df.iloc[idx].reset_index(drop=True)
        if len(sub) != len(preds):
            print(f"  [align] {run_name}: split rows {len(sub)} != preds {len(preds)} — using LCS walk")
        align_idx = _align_preds_to_split(ys, sub)
        # Drop unalignable preds (shouldn't happen with our fallback but be safe)
        keep = align_idx >= 0
        meta = sub.iloc[align_idx[keep]].reset_index(drop=True)
        rows.append(pd.DataFrame({
            "run": run_name,
            "config_kind": kind,
            "fold": fold,
            "cam_id": meta["CamId"].to_numpy(),
            "lat": meta["Latitude"].to_numpy(),
            "lon": meta["Longitude"].to_numpy(),
            "month": meta["Month"].to_numpy(),
            "hour": meta["Hour"].to_numpy(),
            "y_true": ys[keep],
            "y_pred": preds[keep],
            "abs_err": np.abs(preds[keep] - ys[keep]),
        }))
    return pd.concat(rows, ignore_index=True)


# ----- Analysis #1: per-camera test MAE -----------------------------------

def fig_per_camera_test_mae(long_df: pd.DataFrame) -> None:
    """Horizontal bar: per-camera test MAE for each variant."""
    # Aggregate to (cam_id, config_kind) -> mean abs_err.
    grp = long_df.groupby(["cam_id", "config_kind"], as_index=False)["abs_err"].mean()
    # Cam-level count and mean-true-temp (for sorting + annotation).
    cam_info = long_df.groupby("cam_id").agg(
        n=("abs_err", "size"),
        mean_true=("y_true", "mean"),
    )
    # Sort cameras by baseline MAE so the worst are on top.
    base = grp[grp["config_kind"] == "baseline"].set_index("cam_id")["abs_err"]
    cam_order = base.sort_values(ascending=True).index.tolist()

    kinds = ["baseline", "lds", "fds", "lds_fds"]
    n_cams = len(cam_order)
    width = 0.2

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.28 * n_cams)))
    y = np.arange(n_cams)
    for i, k in enumerate(kinds):
        vals = [
            grp[(grp["cam_id"] == c) & (grp["config_kind"] == k)]["abs_err"].iloc[0]
            if not grp[(grp["cam_id"] == c) & (grp["config_kind"] == k)].empty else np.nan
            for c in cam_order
        ]
        ax.barh(y + (i - 1.5) * width, vals, height=width, label=k, color=PALETTE[k])

    labels = [
        f"{c} (n={cam_info.loc[c, 'n'] // len(kinds)}, "
        f"$\\bar{{T}}$={cam_info.loc[c, 'mean_true']:.1f}°C)"
        for c in cam_order
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("test MAE (°C)")
    ax.axvline(grp["abs_err"].mean(), color="gray", ls=":", lw=0.8,
               label=f"mean = {grp['abs_err'].mean():.2f} °C")
    ax.set_title("Per-camera test MAE (fold 0 LOCO, ResNet-50)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    _save(fig, "diag_per_camera_test_mae")


# ----- Analysis #2: per-fold train vs test temp distribution --------------

def fig_per_fold_temp_hist(df: pd.DataFrame, splits: list) -> None:
    fig, axes = plt.subplots(nrows=len(splits), figsize=(7.5, 1.6 * len(splits)),
                              sharex=True)
    if len(splits) == 1:
        axes = [axes]
    bins = np.arange(df["TempM"].min(), df["TempM"].max() + 1, 1.0)

    for ax, fold in zip(axes, splits):
        train_t = df.iloc[fold["train"]]["TempM"].to_numpy()
        test_t = df.iloc[fold["test"]]["TempM"].to_numpy()
        ax.hist(train_t, bins=bins, color="lightgray", label=f"train (n={len(train_t)})", density=True)
        ax.hist(test_t, bins=bins, histtype="step", color=PALETTE["lds_fds"],
                label=f"test (n={len(test_t)})", density=True, lw=1.3)
        # vertical line at each test-camera's mean temperature
        cam_means = df.iloc[fold["test"]].groupby("CamId")["TempM"].mean()
        for cam, mu in cam_means.items():
            ax.axvline(mu, color="darkred", lw=0.6, alpha=0.6)
        ax.axvline(train_t.mean(), color="black", lw=1.2, ls="--",
                   label=f"train mean {train_t.mean():.1f}")
        ax.axvline(test_t.mean(), color=PALETTE["lds_fds"], lw=1.2, ls="--",
                   label=f"test mean {test_t.mean():.1f}")
        ax.set_ylabel(f"fold {fold['fold']}\ndensity")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

    axes[-1].set_xlabel("TempM (°C)")
    fig.suptitle("Train vs test temperature distribution by fold\n"
                 "(dark red verticals: per-test-camera mean temperature)",
                 fontsize=10)
    _save(fig, "diag_per_fold_temp_hist")


# ----- Analysis #3: geographic scatter ------------------------------------

def fig_geo_scatter(df: pd.DataFrame, splits: list) -> None:
    cam = df.groupby("CamId").agg(
        lat=("Latitude", "mean"),
        lon=("Longitude", "mean"),
        mean_temp=("TempM", "mean"),
        n=("TempM", "size"),
    ).reset_index()

    fig, axes = plt.subplots(1, len(splits), figsize=(3.2 * len(splits), 3.3),
                              sharex=True, sharey=True)
    if len(splits) == 1:
        axes = [axes]
    vmin, vmax = cam["mean_temp"].min(), cam["mean_temp"].max()
    for ax, fold in zip(axes, splits):
        test_cams = set(fold["test_cams"])
        train_mask = ~cam["CamId"].isin(test_cams)
        test_mask = cam["CamId"].isin(test_cams)
        ax.scatter(cam.loc[train_mask, "lon"], cam.loc[train_mask, "lat"],
                   c=cam.loc[train_mask, "mean_temp"], s=40, vmin=vmin, vmax=vmax,
                   cmap="coolwarm", edgecolors="gray", linewidths=0.4, label="train")
        sc = ax.scatter(cam.loc[test_mask, "lon"], cam.loc[test_mask, "lat"],
                        c=cam.loc[test_mask, "mean_temp"], s=110, vmin=vmin, vmax=vmax,
                        cmap="coolwarm", edgecolors="black", linewidths=1.6,
                        marker="*", label="test (LOCO)")
        # Annotate test cams with their CamId
        for _, r in cam.loc[test_mask].iterrows():
            ax.text(r["lon"], r["lat"] + 1.5, int(r["CamId"]), fontsize=6,
                    ha="center", color="black")
        ax.set_title(f"fold {fold['fold']}")
        ax.set_xlabel("longitude")
        if fold["fold"] == 0:
            ax.set_ylabel("latitude")
        ax.grid(lw=0.3, alpha=0.5)

    cb = fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.02)
    cb.set_label("camera mean TempM (°C)")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Camera geography by fold (* = held-out LOCO test camera)",
                 fontsize=10)
    _save(fig, "diag_geo_scatter")


# ----- Analysis #4: "OOD" distance vs error -------------------------------

def fig_ood_distance_vs_error(long_df: pd.DataFrame, df: pd.DataFrame,
                               splits: list) -> None:
    """For each test camera in fold 0, compute two distances and plot vs MAE.

    d_T:  |cam_mean_T - train_mean_T| (climate distance)
    d_geo:min geodesic distance (lat/long L2 deg) to any TRAIN camera mean.
    """
    fold = splits[0]
    test_cams = set(fold["test_cams"])
    train_df = df.iloc[fold["train"]]
    train_mean_T = train_df["TempM"].mean()

    train_cam = train_df.groupby("CamId").agg(
        lat=("Latitude", "mean"),
        lon=("Longitude", "mean"),
    )
    test_cam_meta = df.iloc[fold["test"]].groupby("CamId").agg(
        lat=("Latitude", "mean"),
        lon=("Longitude", "mean"),
        mean_T=("TempM", "mean"),
        n=("TempM", "size"),
    )

    def nearest_train_geo(lat, lon):
        d = np.hypot(train_cam["lat"] - lat, train_cam["lon"] - lon)
        return float(d.min())

    test_cam_meta["d_T"] = (test_cam_meta["mean_T"] - train_mean_T).abs()
    test_cam_meta["d_geo"] = test_cam_meta.apply(
        lambda r: nearest_train_geo(r["lat"], r["lon"]), axis=1
    )

    # Per-cam MAE per kind, fold 0 only
    f0 = long_df[long_df["fold"] == 0]
    cam_mae = f0.groupby(["cam_id", "config_kind"])["abs_err"].mean().unstack("config_kind")
    cam_mae = cam_mae.join(test_cam_meta, how="inner")
    assert set(cam_mae.index).issubset(test_cams), "joined non-test cameras"

    kinds = ["baseline", "lds", "fds", "lds_fds"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))

    for ax, xcol, xlabel in zip(
        axes,
        ["d_T", "d_geo"],
        ["|cam mean T − train mean T| (°C)", "lat/long L2 to nearest train cam (deg)"],
    ):
        for k in kinds:
            ax.scatter(cam_mae[xcol], cam_mae[k], color=PALETTE[k], s=55,
                       label=k, edgecolors="black", linewidths=0.4, alpha=0.85)
        # Linear fit on baseline only for a guide
        x = cam_mae[xcol].to_numpy()
        y = cam_mae["baseline"].to_numpy()
        if len(x) > 2 and np.std(x) > 0:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            r = np.corrcoef(x, y)[0, 1]
            ax.plot(xs, m * xs + b, "k--", lw=0.8,
                    label=f"baseline fit r={r:.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("per-camera test MAE (°C)")
        ax.legend(fontsize=7, framealpha=0.9)
        ax.grid(lw=0.4, alpha=0.5)

    fig.suptitle("Test-camera OOD distance vs per-camera test MAE (fold 0)",
                 fontsize=10)
    _save(fig, "diag_ood_distance_vs_error")


# ----- Analysis #5: per-camera predicted mean vs true mean ----------------

def fig_per_camera_pred_vs_true(long_df: pd.DataFrame, df: pd.DataFrame,
                                 splits: list) -> None:
    """Two panels: (a) test-cam per-cam mean pred vs true; (b) val analogue.

    The val analogue uses server_results val_preds for the matching runs to
    show the contrast.
    """
    fold = splits[0]
    f0 = long_df[long_df["fold"] == 0]

    cam_pt = f0.groupby(["cam_id", "config_kind"]).agg(
        mean_true=("y_true", "mean"),
        mean_pred=("y_pred", "mean"),
        n=("y_true", "size"),
    ).reset_index()

    # Now build the val analogue from server_results.
    val_pt_rows = []
    for run_name in ["baseline_resnet50_fold0", "lds_resnet50_fold0",
                     "fds_resnet50_fold0", "lds_fds_resnet50_fold0"]:
        path = VAL_DIR / f"{run_name}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        preds = np.asarray(d.get("val_preds", []))
        ys = np.asarray(d.get("val_ys", []))
        if len(preds) == 0:
            continue
        val_idx = splits[0]["val"]
        sub = df.iloc[val_idx].reset_index(drop=True)
        if len(sub) != len(preds):
            print(f"  [align-val] {run_name}: split rows {len(sub)} != preds {len(preds)} — using LCS walk")
        align_idx = _align_preds_to_split(ys, sub)
        keep = align_idx >= 0
        meta = sub.iloc[align_idx[keep]].reset_index(drop=True)
        kind = (
            "lds_fds" if "lds_fds" in run_name else
            "lds" if "lds_" in run_name else
            "fds" if "fds_" in run_name else
            "baseline"
        )
        per_cam = pd.DataFrame({
            "cam_id": meta["CamId"].to_numpy(),
            "y_true": ys[keep], "y_pred": preds[keep], "config_kind": kind,
        })
        agg = per_cam.groupby(["cam_id", "config_kind"]).agg(
            mean_true=("y_true", "mean"),
            mean_pred=("y_pred", "mean"),
            n=("y_true", "size"),
        ).reset_index()
        val_pt_rows.append(agg)
    val_pt = pd.concat(val_pt_rows, ignore_index=True) if val_pt_rows else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=True, sharey=True)
    kinds = ["baseline", "lds", "fds", "lds_fds"]

    for ax, pt, title in [
        (axes[0], val_pt, "val (in-camera holdout, fold 0)"),
        (axes[1], cam_pt, "test (LOCO, fold 0)"),
    ]:
        # Identity line
        lo, hi = -10, 35
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.6, alpha=0.6, label="y = x")
        # Global train mean (the easy "guess the mean" fallback)
        train_mean = df.iloc[fold["train"]]["TempM"].mean()
        ax.axhline(train_mean, color="gray", ls=":", lw=0.7,
                   label=f"train mean ({train_mean:.1f} °C)")
        if not pt.empty:
            for k in kinds:
                sub = pt[pt["config_kind"] == k]
                ax.scatter(sub["mean_true"], sub["mean_pred"],
                           color=PALETTE[k], s=45, label=k,
                           edgecolors="black", linewidths=0.3, alpha=0.85)
        ax.set_xlabel("per-camera mean true TempM (°C)")
        ax.set_title(title)
        ax.grid(lw=0.4, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    axes[0].set_ylabel("per-camera mean predicted TempM (°C)")
    axes[0].legend(fontsize=7, framealpha=0.9, loc="upper left")
    fig.suptitle("Per-camera mean prediction vs truth — val (memorisable) vs test (LOCO)",
                 fontsize=10)
    _save(fig, "diag_per_camera_pred_vs_true")


# ----- Analysis #6: per-bin test panel with C1/C2 overlaid ----------------

def fig_per_bin_test_with_baselines(long_df: pd.DataFrame) -> None:
    """Bar chart: per-bin test MAE for the 4 DIR variants + C1 + C2.

    DIR variants come from `final_results/results/test_inference.json`
    (fold 0 only). C1 / C2 are mean ± std across all 5 folds.
    """
    # CNN bins per (kind) — re-derive from preds via per_bin_mae using fold-0 train.
    from skyfinder.training.engine import per_bin_mae  # local import
    df_lbl = pd.read_csv(LABELS)
    splits = json.loads(SPLITS.read_text())
    train_y = df_lbl.iloc[splits[0]["train"]]["TempM"].to_numpy()

    bins = ["overall", "many", "medium", "few"]
    cnn_bin = {}
    for kind in ["baseline", "lds", "fds", "lds_fds"]:
        sub = long_df[(long_df["fold"] == 0) & (long_df["config_kind"] == kind)]
        cnn_bin[kind] = per_bin_mae(sub["y_true"].to_numpy(),
                                     sub["y_pred"].to_numpy(), train_y)

    # C1: use the per_cam_month_mean variant (the strongest C1 row).
    c1 = json.loads(C1_JSON.read_text())
    c1_rows = [r for r in c1["per_fold"] if r["predictor"] == "per_cam_month_mean"]
    c2 = json.loads(C2_JSON.read_text())
    c2_rows = c2["per_fold"]

    def agg(rows):
        out = {}
        for b in bins:
            xs = [r["test"][b] for r in rows
                  if r["test"].get(b) is not None and not (isinstance(r["test"][b], float) and np.isnan(r["test"][b]))]
            out[b] = (float(np.mean(xs)), float(np.std(xs))) if xs else (np.nan, 0.0)
        return out

    c1_agg = agg(c1_rows)
    c2_agg = agg(c2_rows)

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    kinds = ["baseline", "lds", "fds", "lds_fds"]
    n_grp = len(bins)
    width = 0.12
    x = np.arange(n_grp)

    for i, k in enumerate(kinds):
        vals = [cnn_bin[k][b] for b in bins]
        ax.bar(x + (i - 2.5) * width, vals, width=width, label=k, color=PALETTE[k])
    # C1
    vals = [c1_agg[b][0] for b in bins]
    errs = [c1_agg[b][1] for b in bins]
    ax.bar(x + 1.5 * width, vals, width=width, yerr=errs, color="#999999",
           label="C1 (per-cam-month, falls back to global)", edgecolor="black", linewidth=0.4)
    # C2
    vals = [c2_agg[b][0] for b in bins]
    errs = [c2_agg[b][1] for b in bins]
    ax.bar(x + 2.5 * width, vals, width=width, yerr=errs, color="#666666",
           label="C2 (metadata GBM)", edgecolor="black", linewidth=0.4)

    # Annotate where C2 beats the CNN
    for j, b in enumerate(bins):
        cnn_vals = [cnn_bin[k][b] for k in kinds if not np.isnan(cnn_bin[k][b])]
        if not cnn_vals:
            continue
        cnn_min = min(cnn_vals)
        c2_val = c2_agg[b][0]
        if not np.isnan(c2_val) and c2_val < cnn_min:
            ax.text(j + 2.5 * width, c2_val + 0.4, "*", color="darkgreen",
                    ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(bins)
    ax.set_ylabel("test MAE (°C)")
    ax.set_xlabel("DIR bin")
    ax.set_title("Per-bin test MAE: CNN variants (fold 0) vs C1, C2 (5-fold mean ± std)\n"
                 "★ = metadata baseline beats every CNN variant in this bin")
    ax.legend(fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    _save(fig, "diag_per_bin_test_with_baselines")


# ----- Analysis #9: per-(month, hour) coverage heatmaps -------------------

def fig_month_hour_coverage(df: pd.DataFrame, splits: list) -> None:
    """For each fold: train vs test image counts in (month x hour) cells.

    A test cell that train doesn't cover = the model has never seen that
    combination of season + time-of-day.
    """
    fig, axes = plt.subplots(nrows=len(splits), ncols=3,
                              figsize=(13.0, 3.3 * len(splits)),
                              sharex=True, sharey=True,
                              gridspec_kw={"width_ratios": [1, 1, 1]})
    if len(splits) == 1:
        axes = np.array([axes])

    months = np.arange(1, 13)
    hours = np.arange(0, 24)

    for row_i, fold in enumerate(splits):
        train_df = df.iloc[fold["train"]]
        test_df = df.iloc[fold["test"]]
        train_cov = pd.crosstab(train_df["Month"], train_df["Hour"]).reindex(
            index=months, columns=hours, fill_value=0)
        test_cov = pd.crosstab(test_df["Month"], test_df["Hour"]).reindex(
            index=months, columns=hours, fill_value=0)
        # Per-cell train density divided by global train density: if a test
        # cell has high density but train coverage is low, that's bad.
        eps = 1e-9
        train_norm = train_cov / (train_cov.sum().sum() + eps)
        test_norm = test_cov / (test_cov.sum().sum() + eps)
        gap = test_norm - train_norm  # positive = test over-represents

        vmax = max(train_norm.values.max(), test_norm.values.max())
        for col_i, (mat, ttl, cmap, vlim) in enumerate([
            (train_norm, "train", "Blues", (0, vmax)),
            (test_norm, "test", "Oranges", (0, vmax)),
            (gap, "test − train", "RdBu_r",
             (-abs(gap.values).max(), abs(gap.values).max())),
        ]):
            ax = axes[row_i, col_i]
            im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                           vmin=vlim[0], vmax=vlim[1],
                           extent=(hours[0] - 0.5, hours[-1] + 0.5,
                                   months[-1] + 0.5, months[0] - 0.5))
            if row_i == 0:
                ax.set_title(ttl)
            if col_i == 0:
                ax.set_ylabel(f"fold {fold['fold']}\nmonth")
            if row_i == len(splits) - 1:
                ax.set_xlabel("hour")
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    fig.suptitle("Train vs test temporal coverage (density per month × hour bucket)",
                 fontsize=10)
    _save(fig, "diag_month_hour_coverage")


# ----- Analysis #10: worst-camera case study with ImageNet kNN ------------

def _imagenet_features(paths: list[Path], device: str = "cpu") -> np.ndarray:
    """Run ImageNet ResNet-50 once on a list of image paths, return Nx2048 features.

    Cached to disk so re-runs are fast.
    """
    cache = OUT / "_imagenet_features_cache.npz"
    str_paths = [str(p) for p in paths]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["paths"]) == str_paths:
            return z["feats"]

    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T
    from PIL import Image

    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    feats = np.zeros((len(paths), 2048), dtype=np.float32)
    with torch.no_grad():
        for i, p in enumerate(paths):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            x = tf(img).unsqueeze(0).to(device)
            f = model(x).squeeze(0).cpu().numpy()
            feats[i] = f
            if (i + 1) % 50 == 0:
                print(f"    encoded {i + 1}/{len(paths)}")
    np.savez(cache, feats=feats, paths=np.array(str_paths))
    return feats


def fig_worst_camera_case_study(long_df: pd.DataFrame, df: pd.DataFrame,
                                 splits: list, n_worst: int = 3,
                                 n_samples: int = 3, seed: int = 0) -> None:
    """For each of the N worst test cameras, show:
       (1) 3 query images from that camera
       (2) nearest training image by ImageNet ResNet-50 features
       (3) true / predicted temperature on each query

    Uses only baseline predictions for clarity.
    """
    rng = np.random.default_rng(seed)
    fold = splits[0]
    f0 = long_df[(long_df["fold"] == 0) & (long_df["config_kind"] == "baseline")]
    cam_mae = f0.groupby("cam_id")["abs_err"].mean().sort_values(ascending=False)
    worst_cams = cam_mae.head(n_worst).index.tolist()
    print(f"  worst {n_worst} test cams (by baseline MAE): {worst_cams}")

    # Build sample sets
    train_pool = df.iloc[fold["train"]]
    # Subsample train pool aggressively to keep ImageNet encoding cheap
    train_sample = train_pool.sample(min(600, len(train_pool)), random_state=seed).reset_index(drop=True)
    test_pool = df.iloc[fold["test"]]

    query_records = []  # (cam, true, pred, path)
    test_paths = []
    for cam in worst_cams:
        cam_rows = test_pool[test_pool["CamId"] == cam]
        # Stratify the 3 query samples by absolute error percentile so we see
        # a typical, a hard, and an extreme failure
        cam_long = f0[f0["cam_id"] == cam].reset_index(drop=True)
        if len(cam_long) == 0 or len(cam_rows) == 0:
            continue
        # Pick rows at err percentile 50, 75, 99
        order = np.argsort(cam_long["abs_err"].to_numpy())
        idxs = [order[int(p * (len(order) - 1) / 100)] for p in (50, 75, 99)]
        for i in idxs[:n_samples]:
            cam_long_row = cam_long.iloc[i]
            true_T = cam_long_row["y_true"]
            pred_T = cam_long_row["y_pred"]
            # Match this back to the test_pool row by TempM (closest match within camera)
            d = (cam_rows["TempM"] - true_T).abs()
            best = cam_rows.iloc[d.values.argmin()]
            path = IMG_DIR / str(int(best["CamId"])) / best["Filename"]
            if not path.exists():
                continue
            query_records.append({
                "cam_id": cam, "path": path, "true": float(true_T),
                "pred": float(pred_T), "err": float(abs(pred_T - true_T)),
                "month": int(best["Month"]), "hour": int(best["Hour"]),
            })
            test_paths.append(path)

    # Train paths
    train_paths = []
    train_meta = []
    for _, r in train_sample.iterrows():
        p = IMG_DIR / str(int(r["CamId"])) / r["Filename"]
        if p.exists():
            train_paths.append(p)
            train_meta.append({"CamId": int(r["CamId"]), "TempM": float(r["TempM"])})

    if not query_records or not train_paths:
        print("  skipping case study — missing images")
        return

    print(f"  encoding {len(query_records) + len(train_paths)} images with ImageNet ResNet-50…")
    all_paths = test_paths + train_paths
    feats = _imagenet_features(all_paths, device="cpu")
    test_f = feats[: len(test_paths)]
    train_f = feats[len(test_paths):]
    # Cosine similarity
    test_n = test_f / (np.linalg.norm(test_f, axis=1, keepdims=True) + 1e-9)
    train_n = train_f / (np.linalg.norm(train_f, axis=1, keepdims=True) + 1e-9)
    sims = test_n @ train_n.T   # (Nq, Ntrain)
    nn_idx = sims.argmax(axis=1)
    nn_meta = [train_meta[i] for i in nn_idx]
    nn_paths = [train_paths[i] for i in nn_idx]

    # Render as grouped rows: 3 cams × 3 percentiles × (query, nn) pair
    from PIL import Image
    n_per_cam = n_samples
    n_cams_rendered = len(query_records) // n_per_cam
    fig, axes = plt.subplots(n_cams_rendered, n_per_cam * 2,
                              figsize=(2.0 * n_per_cam * 2, 2.6 * n_cams_rendered))
    if n_cams_rendered == 1:
        axes = np.array([axes])

    for cam_i in range(n_cams_rendered):
        for samp_i in range(n_per_cam):
            qi = cam_i * n_per_cam + samp_i
            q = query_records[qi]; nn_p = nn_paths[qi]; nn_m = nn_meta[qi]
            col_q = samp_i * 2
            col_n = samp_i * 2 + 1
            for col, (img_path, ttl) in enumerate([
                (q["path"], f"cam {q['cam_id']} (m{q['month']},h{q['hour']})\n"
                            f"true={q['true']:.1f} pred={q['pred']:.1f} err={q['err']:.1f}°"),
                (nn_p, f"NN: train cam {nn_m['CamId']}\nT={nn_m['TempM']:.1f}°"),
            ]):
                ax = axes[cam_i, col_q + col]
                try:
                    ax.imshow(np.asarray(Image.open(img_path).convert("RGB")))
                except Exception:
                    ax.text(0.5, 0.5, "image not found", ha="center", va="center")
                ax.set_title(ttl, fontsize=7)
                ax.set_axis_off()
        # Row-leading annotation: which worst cam this row is
        cam_label = f"Worst #{cam_i + 1}\ncam {worst_cams[cam_i]}"
        axes[cam_i, 0].annotate(cam_label, xy=(-0.15, 0.5),
                                 xycoords="axes fraction",
                                 fontsize=9, fontweight="bold",
                                 ha="right", va="center")
    fig.suptitle("Worst-test-camera case study — left of each pair: TEST query (err 50/75/99 pctile); "
                 "right: nearest TRAIN image by ImageNet ResNet-50 cosine",
                 fontsize=9)
    _save(fig, "diag_worst_camera_case_study")


# ----- Main ---------------------------------------------------------------

def main() -> None:
    print(f"writing diagnostics to {OUT}/")
    runs, df, splits = _load_test_inference()
    long_df = _per_camera_records(runs, df, splits)

    print("[1/8] per-camera test MAE")
    fig_per_camera_test_mae(long_df)
    print("[2/8] per-fold temperature histograms")
    fig_per_fold_temp_hist(df, splits)
    print("[3/8] geographic scatter")
    fig_geo_scatter(df, splits)
    print("[4/8] OOD distance vs error")
    fig_ood_distance_vs_error(long_df, df, splits)
    print("[5/8] per-camera pred vs true (val vs test)")
    fig_per_camera_pred_vs_true(long_df, df, splits)
    print("[6/8] per-bin test panel with C1/C2 baselines")
    fig_per_bin_test_with_baselines(long_df)
    print("[7/8] month×hour coverage train vs test")
    fig_month_hour_coverage(df, splits)
    print("[8/8] worst-test-camera case study (loads images, runs ImageNet)")
    fig_worst_camera_case_study(long_df, df, splits)

    # also dump per-camera CSV for downstream analysis
    csv_out = OUT / "per_camera_test_metrics.csv"
    summary = long_df.groupby(["cam_id", "config_kind"]).agg(
        n=("abs_err", "size"),
        mae=("abs_err", "mean"),
        mean_true=("y_true", "mean"),
        mean_pred=("y_pred", "mean"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
    ).reset_index()
    summary.to_csv(csv_out, index=False)
    print(f"  wrote {csv_out}")


if __name__ == "__main__":
    main()
