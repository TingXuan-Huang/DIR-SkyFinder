"""Trajectory figures (per-epoch snapshots): PCA grid, per-bin cosine-sim grid,
k-NN MAE over epochs, CKA between epochs.

All public functions accept arbitrary `run_name` strings."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skyfinder.analysis.figures._helpers import (_linear_cka, _list_traj_epochs,
                                                  _load_traj_npz,
                                                  per_bin_feature_cosine_sim)
from skyfinder.analysis.style import figsize, save_fig


def fig_traj_pca(config: dict, run_name: str, fold: int = 0, split: str = "val",
                 color_by: str = "temp") -> None:
    """PCA grid across snapshot epochs. `color_by` ∈ {'temp', 'cam'}."""
    from sklearn.decomposition import PCA
    fig_dir = Path(config["figures_dir"])
    eps = _list_traj_epochs(config, run_name, fold, split)
    if not eps:
        print(f"[skip] fig_traj_pca: no snapshots for {run_name}/{split}")
        return
    n = len(eps)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize("double", 0.30 * max(nrows, 1)),
                            squeeze=False)
    axs_flat = axs.ravel()
    sc = None
    for ax, ep in zip(axs_flat, eps):
        data = _load_traj_npz(config, run_name, fold, ep, split)
        if data is None:
            ax.set_axis_off(); continue
        xy = PCA(n_components=2, random_state=0).fit_transform(data["features"])
        if color_by == "temp":
            cvals, cmap = data["ys"], "viridis"
        else:
            cvals, cmap = pd.Categorical(data["cam_ids"]).codes, "tab20"
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=cvals, cmap=cmap,
                        s=1, alpha=0.5, edgecolor="none")
        ax.set_title(f"ep {ep}")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axs_flat[n:]:
        ax.set_axis_off()
    if sc is not None:
        cbar_label = "true TempM (°C)" if color_by == "temp" else "camera id"
        fig.colorbar(sc, ax=axs_flat.tolist(), shrink=0.6, label=cbar_label)
    fig.suptitle(f"{run_name} fold{fold} {split} — PCA trajectory ({color_by})", fontsize=8, y=0.99)
    save_fig(fig, f"fig_traj_pca_{run_name}_fold{fold}_{split}_{color_by}", fig_dir)


def fig_traj_per_bin(config: dict, run_name: str, fold: int = 0, split: str = "val",
                     bin_w: float = 2.0) -> None:
    fig_dir = Path(config["figures_dir"])
    eps = _list_traj_epochs(config, run_name, fold, split)
    if not eps:
        print(f"[skip] fig_traj_per_bin: no snapshots for {run_name}/{split}")
        return
    snaps = []
    for ep in eps:
        d = _load_traj_npz(config, run_name, fold, ep, split)
        if d is not None:
            snaps.append((ep, d))
    if not snaps:
        print(f"[skip] fig_traj_per_bin: no readable npz files"); return
    all_ys = np.concatenate([d["ys"] for _, d in snaps])
    lo, hi = float(all_ys.min()), float(all_ys.max())
    edges = np.arange(np.floor(lo / bin_w) * bin_w, np.ceil(hi / bin_w) * bin_w + bin_w, bin_w)

    n = len(snaps)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize("double", 0.30 * max(nrows, 1)),
                            sharex=True, sharey=True, squeeze=False)
    axs_flat = axs.ravel()
    im = None
    for ax, (ep, d) in zip(axs_flat, snaps):
        sim = per_bin_feature_cosine_sim(d["features"], d["ys"], edges)
        im = ax.imshow(sim, origin="lower", cmap="viridis", vmin=-1, vmax=1,
                       extent=[edges[0], edges[-1], edges[0], edges[-1]])
        ax.set_title(f"ep {ep}")
    for ax in axs_flat[n:]:
        ax.set_axis_off()
    if im is not None:
        fig.colorbar(im, ax=axs_flat.tolist(), shrink=0.6, label="cosine sim")
    fig.suptitle(f"{run_name} fold{fold} {split} — per-bin cosine trajectory", fontsize=8, y=0.99)
    save_fig(fig, f"fig_traj_per_bin_{run_name}_fold{fold}_{split}", fig_dir)


def fig_traj_knn_mae(config: dict, run_name: str, fold: int = 0,
                     splits: tuple[str, ...] = ("train", "val", "test")) -> None:
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsRegressor
    fig_dir = Path(config["figures_dir"])
    eps_per_split = {s: _list_traj_epochs(config, run_name, fold, s) for s in splits}
    if not any(eps_per_split.values()):
        print(f"[skip] fig_traj_knn_mae: no snapshots for {run_name}")
        return
    fig, ax = plt.subplots(figsize=figsize("single"))
    markers = {"train": "o", "val": "s", "test": "^"}
    for split, eps in eps_per_split.items():
        if not eps:
            continue
        maes = []
        for ep in eps:
            data = _load_traj_npz(config, run_name, fold, ep, split)
            if data is None:
                maes.append(np.nan); continue
            X = data["features"]; y = data["ys"]
            fold_maes = []
            for tr, te in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
                knn = KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
                knn.fit(X[tr], y[tr])
                fold_maes.append(float(np.mean(np.abs(knn.predict(X[te]) - y[te]))))
            maes.append(float(np.mean(fold_maes)))
        ax.plot(eps, maes, marker=markers.get(split, "o"), label=split, linewidth=1.0)
    ax.set_xlabel("epoch"); ax.set_ylabel("5-fold k-NN MAE on features (°C)")
    ax.set_title(f"{run_name} fold{fold}")
    ax.legend(frameon=False)
    save_fig(fig, f"fig_traj_knn_{run_name}_fold{fold}", fig_dir)


def fig_traj_cka(config: dict, run_name: str, fold: int = 0, split: str = "val") -> None:
    fig_dir = Path(config["figures_dir"])
    eps = _list_traj_epochs(config, run_name, fold, split)
    if len(eps) < 2:
        print(f"[skip] fig_traj_cka: need >=2 snapshots for {run_name}/{split}")
        return
    feats = []
    for ep in eps:
        data = _load_traj_npz(config, run_name, fold, ep, split)
        feats.append(data["features"] if data is not None else None)
    n = len(eps)
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if feats[i] is None or feats[j] is None:
                continue
            M[i, j] = _linear_cka(feats[i], feats[j])
    fig, ax = plt.subplots(figsize=figsize("single"))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(eps, rotation=0)
    ax.set_yticks(range(n)); ax.set_yticklabels(eps)
    ax.set_xlabel("epoch"); ax.set_ylabel("epoch")
    ax.set_title(f"{run_name} fold{fold} {split} — CKA across epochs")
    fig.colorbar(im, ax=ax, shrink=0.8, label="linear CKA")
    save_fig(fig, f"fig_traj_cka_{run_name}_fold{fold}_{split}", fig_dir)


def render_trajectory(config: dict, runs: tuple[str, ...] | None = None, fold: int = 0) -> None:
    """Render all 5 trajectory figure types for each run that has snapshots."""
    from skyfinder.analysis.extract_trajectory import DEFAULT_RUNS
    runs = runs if runs is not None else DEFAULT_RUNS
    for r in runs:
        for s in ("val", "test"):
            fig_traj_pca(config, r, fold=fold, split=s, color_by="temp")
            fig_traj_pca(config, r, fold=fold, split=s, color_by="cam")
            fig_traj_per_bin(config, r, fold=fold, split=s)
            fig_traj_cka(config, r, fold=fold, split=s)
        fig_traj_knn_mae(config, r, fold=fold)


# Backwards-compatible alias.
make_trajectory = render_trajectory
