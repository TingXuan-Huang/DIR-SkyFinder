"""plot_embed_by_bin.py — embeddings colored by many / medium / few bin category.

Bin category follows the DIR convention used in `baseline.per_bin_mae`:
    many   : 2 °C bin had >= 100 train samples
    medium : 20 <= train samples < 100
    few    : 0 <  train samples < 20

Reads single-snapshot embeddings from `<embeddings_dir>/<run>_fold{F}_<split>.npz`
(produced by `python analysis.py embeddings`). Computes the per-sample category
from `ys` + the train-set histogram for that fold, then plots a 2×N grid:
    rows: PCA, UMAP
    cols: each run passed via --runs (default: 4 ResNet headline configs)

Many is drawn first (light blue, smallest), then medium (orange), then few
(red, larger, on top) — so rare samples are visually prominent.

Usage:
    python plot_embed_by_bin.py                                                # 4 ResNet, test split
    python plot_embed_by_bin.py --split val
    python plot_embed_by_bin.py --runs baseline_resnet50 baseline_vit lds_fds_resnet50 lds_fds_vit
    python plot_embed_by_bin.py --methods pca                                  # PCA only (single row)
    python plot_embed_by_bin.py --bin-width 1.0                                # finer bins (more "few")
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis._config import load_config
from analysis.style import apply_nature_style, figsize, save_fig

apply_nature_style()


# Match per_bin_mae's defaults: 2 °C bins, thresholds {many >= 100, medium 20-100, few < 20}.
DEFAULT_BIN_W = 2.0
CATEGORY_THRESHOLDS = [("many", 100, np.inf), ("medium", 20, 100), ("few", 0, 20)]
CATEGORY_COLOR = {"many": "#9ecae1", "medium": "#fdae6b", "few": "#e6550d"}
CATEGORY_SIZE  = {"many": 1.5,        "medium": 2.0,        "few": 3.5}    # few biggest, on top
CATEGORY_ZORDER = {"many": 1,         "medium": 2,          "few": 3}


def _train_y(config: dict, fold: int) -> np.ndarray:
    df = pd.read_csv(config["labels_path"])
    splits = json.loads(Path(config["splits_path"]).read_text())
    return df.iloc[splits[fold]["train"]]["TempM"].to_numpy()


def _bin_category(y_true: np.ndarray, train_y: np.ndarray, bin_w: float) -> np.ndarray:
    """Returns per-sample category string ('many'/'medium'/'few'). Mirrors per_bin_mae."""
    lo = float(min(train_y.min(), y_true.min()))
    hi = float(max(train_y.max(), y_true.max()))
    edges = np.arange(np.floor(lo / bin_w) * bin_w,
                      np.ceil(hi / bin_w) * bin_w + bin_w, bin_w)
    train_hist, _ = np.histogram(train_y, bins=edges)
    idx = np.clip(np.digitize(y_true, edges) - 1, 0, len(edges) - 2)
    cats = np.full(len(y_true), "few", dtype=object)
    for name, n_lo, n_hi in CATEGORY_THRESHOLDS:
        bin_mask = (train_hist >= n_lo) & (train_hist < n_hi)
        cats[np.isin(idx, np.where(bin_mask)[0])] = name
    return cats


def _project(X: np.ndarray, method: str) -> np.ndarray | None:
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=0).fit_transform(X)
    if method == "umap":
        try:
            import umap   # type: ignore
        except ImportError:
            print("[skip] UMAP requested but `umap-learn` not installed")
            return None
        return umap.UMAP(n_components=2, n_neighbors=15, random_state=0).fit_transform(X)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("analysis_config.yaml"))
    ap.add_argument("--runs",  nargs="+",
                    default=["baseline_resnet50", "lds_resnet50",
                             "fds_resnet50",      "lds_fds_resnet50"])
    ap.add_argument("--fold",       type=int, default=0)
    ap.add_argument("--split",      default="test", choices=["train", "val", "test"])
    ap.add_argument("--methods",    nargs="+", default=["pca", "umap"], choices=["pca", "umap"])
    ap.add_argument("--bin-width",  type=float, default=DEFAULT_BIN_W,
                    help="bin width (°C) for the many/medium/few classification (default 2.0)")
    a = ap.parse_args()
    config = load_config(a.config)

    embed_dir = Path(config["embeddings_dir"])
    train_y = _train_y(config, a.fold)

    n_rows = len(a.methods)
    n_cols = len(a.runs)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize("double", 0.55), squeeze=False)

    for c, run in enumerate(a.runs):
        npz_path = embed_dir / f"{run}_fold{a.fold}_{a.split}.npz"
        if not npz_path.exists():
            for r in range(n_rows):
                axs[r, c].set_axis_off()
            axs[0, c].set_title(f"{run}\n(no npz)")
            continue
        data = dict(np.load(npz_path, allow_pickle=True))
        cats = _bin_category(data["ys"], train_y, bin_w=a.bin_width)

        # Print counts so it's easy to see how the split lands.
        unique, counts = np.unique(cats, return_counts=True)
        cdict = dict(zip(unique, counts))
        print(f"[{run:30s}] " + "  ".join(f"{k}={cdict.get(k, 0)}"
                                          for k in ("many", "medium", "few")))

        for r, method in enumerate(a.methods):
            ax = axs[r, c]
            xy = _project(data["features"], method)
            if xy is None:
                ax.set_axis_off(); continue
            # Order: many -> medium -> few so few is drawn LAST (on top).
            for cat_name, _, _ in CATEGORY_THRESHOLDS:
                m = (cats == cat_name)
                if not m.any():
                    continue
                ax.scatter(xy[m, 0], xy[m, 1],
                           c=CATEGORY_COLOR[cat_name],
                           s=CATEGORY_SIZE[cat_name],
                           zorder=CATEGORY_ZORDER[cat_name],
                           alpha=0.6, edgecolor="none")
            if r == 0:
                ax.set_title(run)
            if c == 0:
                ax.set_ylabel(method.upper())
            ax.set_xticks([]); ax.set_yticks([])

    # Shared legend at the top.
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=CATEGORY_COLOR[c],
                          markersize=5, label=c) for c, _, _ in CATEGORY_THRESHOLDS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"split={a.split}  fold={a.fold}  bin_w={a.bin_width}", fontsize=7, y=0.965)
    save_fig(fig, f"fig_embed_by_bin_{a.split}_fold{a.fold}",
             Path(config["figures_dir"]))


if __name__ == "__main__":
    main()
