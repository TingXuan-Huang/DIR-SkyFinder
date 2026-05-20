"""Embedding-space figures (single-snapshot per run): PCA/UMAP scatter, k-NN MAE,
per-bin feature similarity, CKA between configs.

Every public function accepts `runs` so you can swap in any subset (ResNet, ViT,
mixed). Default is the 4-ResNet headline list (`DEFAULT_EMBED_RUNS`)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skyfinder.analysis.figures._helpers import (DEFAULT_EMBED_RUNS, KIND_LABEL,
                                                  _linear_cka, _load_embedding,
                                                  _project_2d,
                                                  per_bin_feature_cosine_sim)
from skyfinder.analysis.style import PALETTE, figsize, save_fig


def _fig_embed_scatter(config: dict, color_by: str, name: str, split: str,
                        runs: list[tuple[str, str]]) -> None:
    fig_dir = Path(config["figures_dir"])
    loaded = [(label, _load_embedding(config, run, split=split)) for run, label in runs]
    if not any(d is not None for _, d in loaded):
        print(f"[skip] {name}_{split}: no embedding npz files for split={split}")
        return
    n = len(runs)
    fig, axs = plt.subplots(2, n, figsize=figsize("double", 0.55), squeeze=False)
    sc = None
    for col, (label, data) in enumerate(loaded):
        for row, method in enumerate(["pca", "umap"]):
            ax = axs[row, col]
            if data is None:
                ax.set_axis_off(); ax.set_title(f"{label} (missing)"); continue
            xy = _project_2d(data["features"], method)
            if xy is None:
                ax.set_axis_off(); continue
            color_vals = data["ys"] if color_by == "temp" else pd.Categorical(data["cam_ids"]).codes
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=color_vals,
                            cmap="viridis" if color_by == "temp" else "tab20",
                            s=1, alpha=0.5, edgecolor="none")
            if row == 0:
                ax.set_title(KIND_LABEL.get(label, label))
            if col == 0:
                ax.set_ylabel(method.upper())
            ax.set_xticks([]); ax.set_yticks([])
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.6, pad=0.02)
        cbar.set_label("true TempM (°C)" if color_by == "temp" else "camera id")
    fig.suptitle(f"split={split}", fontsize=7, y=0.99)
    save_fig(fig, f"{name}_{split}", fig_dir)


def fig_embed_temp(config: dict, runs: list[tuple[str, str]] | None = None,
                   split: str = "val"):
    _fig_embed_scatter(config, "temp", "fig_embed_temp", split, runs or DEFAULT_EMBED_RUNS)


def fig_embed_cam(config: dict, runs: list[tuple[str, str]] | None = None,
                  split: str = "val"):
    _fig_embed_scatter(config, "cam", "fig_embed_cam", split, runs or DEFAULT_EMBED_RUNS)


def fig_embed_knn(config: dict, runs: list[tuple[str, str]] | None = None,
                  split: str = "val") -> None:
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsRegressor
    runs = runs or DEFAULT_EMBED_RUNS
    fig_dir = Path(config["figures_dir"])
    rows = []
    for run, label in runs:
        data = _load_embedding(config, run, split=split)
        if data is None:
            continue
        X = data["features"]; y = data["ys"]
        maes = []
        for tr, te in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
            knn = KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
            knn.fit(X[tr], y[tr])
            maes.append(np.mean(np.abs(knn.predict(X[te]) - y[te])))
        rows.append((label, float(np.mean(maes)), float(np.std(maes))))
    if not rows:
        print(f"[skip] fig_embed_knn_{split}: no embeddings"); return
    fig, ax = plt.subplots(figsize=figsize("single"))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], yerr=[r[2] for r in rows], capsize=3,
           color=[PALETTE.get(r[0], "#888888") for r in rows], edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels([KIND_LABEL.get(r[0], r[0]) for r in rows])
    ax.set_ylabel(f"5-fold k-NN MAE on features (°C, split={split})")
    save_fig(fig, f"fig_embed_knn_{split}", fig_dir)


def fig_embed_per_bin(config: dict, runs: list[tuple[str, str]] | None = None,
                       split: str = "val") -> None:
    """Per-bin mean-feature cosine-similarity heatmap, one panel per run."""
    runs = runs or DEFAULT_EMBED_RUNS
    fig_dir = Path(config["figures_dir"])
    bin_w = 2.0
    panels = [(label, _load_embedding(config, run, split=split)) for run, label in runs]
    panels = [(l, d) for l, d in panels if d is not None]
    if not panels:
        print(f"[skip] fig_embed_per_bin_{split}: no embeddings"); return
    all_ys = np.concatenate([d["ys"] for _, d in panels])
    lo, hi = float(all_ys.min()), float(all_ys.max())
    edges = np.arange(np.floor(lo / bin_w) * bin_w, np.ceil(hi / bin_w) * bin_w + bin_w, bin_w)

    fig, axs = plt.subplots(1, len(panels), figsize=figsize("double", 0.30), sharey=True,
                            squeeze=False)
    im = None
    for ax, (label, data) in zip(axs[0], panels):
        sim = per_bin_feature_cosine_sim(data["features"], data["ys"], edges)
        im = ax.imshow(sim, origin="lower", cmap="viridis", vmin=-1, vmax=1,
                       extent=[edges[0], edges[-1], edges[0], edges[-1]])
        ax.set_title(KIND_LABEL.get(label, label))
        ax.set_xlabel("bin (°C)")
    axs[0, 0].set_ylabel("bin (°C)")
    if im is not None:
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label="cosine sim")
    fig.suptitle(f"split={split}", fontsize=7, y=0.99)
    save_fig(fig, f"fig_embed_per_bin_{split}", fig_dir)


def fig_embed_cka(config: dict, runs: list[tuple[str, str]] | None = None,
                  split: str = "val") -> None:
    runs = runs or DEFAULT_EMBED_RUNS
    fig_dir = Path(config["figures_dir"])
    loaded = [(label, _load_embedding(config, run, split=split)) for run, label in runs]
    loaded = [(l, d) for l, d in loaded if d is not None]
    if len(loaded) < 2:
        print(f"[skip] fig_embed_cka_{split}: need >=2 embedding files"); return
    n = len(loaded)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = _linear_cka(loaded[i][1]["features"], loaded[j][1]["features"])
    fig, ax = plt.subplots(figsize=figsize("single"))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ticks = [KIND_LABEL.get(l, l) for l, _ in loaded]
    ax.set_xticks(range(n)); ax.set_xticklabels(ticks, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(ticks)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
    fig.colorbar(im, ax=ax, shrink=0.8, label="linear CKA")
    fig.suptitle(f"split={split}", fontsize=7, y=0.99)
    save_fig(fig, f"fig_embed_cka_{split}", fig_dir)


# ============================================================
# Per-bin scatter colored by DIR many/medium/few
# (folded in from the old standalone `plot_embed_by_bin.py`)
# ============================================================

def fig_embed_by_bin(config: dict, run_name: str, fold: int = 0,
                     split: str = "val", method: str = "umap") -> None:
    """PCA or UMAP scatter colored by many/medium/few bin category.

    Replaces the standalone `plot_embed_by_bin.py`. Subsample for speed when N > 5000.
    """
    fig_dir = Path(config["figures_dir"])
    data = _load_embedding(config, run_name, fold=fold, split=split)
    if data is None:
        print(f"[skip] fig_embed_by_bin: no embedding for {run_name}_fold{fold}_{split}")
        return
    # Bin train labels to get many/medium/few classification.
    import json
    train_df = pd.read_csv(config["labels_path"])
    splits = json.loads(Path(config["splits_path"]).read_text())
    train_y = train_df.iloc[splits[fold]["train"]]["TempM"].to_numpy()
    bin_w = 2.0
    lo = float(min(train_y.min(), data["ys"].min()))
    hi = float(max(train_y.max(), data["ys"].max()))
    edges = np.arange(np.floor(lo / bin_w) * bin_w, np.ceil(hi / bin_w) * bin_w + bin_w, bin_w)
    train_hist, _ = np.histogram(train_y, bins=edges)
    test_idx = np.clip(np.digitize(data["ys"], edges) - 1, 0, len(edges) - 2)
    category = np.where(train_hist[test_idx] >= 100, "many",
               np.where(train_hist[test_idx] >= 20, "medium", "few"))

    xy = _project_2d(data["features"], method)
    if xy is None:
        return
    fig, ax = plt.subplots(figsize=figsize("single"))
    # Draw many first (light blue), then medium (mid), then few (red, on top).
    color = {"many": "#a0c4e0", "medium": "#777777", "few": "#d93030"}
    size  = {"many": 1.5,        "medium": 2.0,        "few": 5.0}
    for cat in ["many", "medium", "few"]:
        sel = category == cat
        if not sel.any():
            continue
        ax.scatter(xy[sel, 0], xy[sel, 1], c=color[cat], s=size[cat],
                   alpha=0.5 if cat != "few" else 0.9, edgecolor="none", label=cat)
    ax.legend(frameon=False, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{run_name} fold{fold} {split} ({method.upper()}) — by bin frequency")
    save_fig(fig, f"fig_embed_by_bin_{run_name}_fold{fold}_{split}_{method}", fig_dir)
