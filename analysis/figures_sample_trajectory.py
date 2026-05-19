"""Per-sample embedding trajectory plot.

Tracks a small number of anchor samples (1 from each bin) across training
snapshot epochs, in the 2D PCA projection of that run's final-epoch features.
Each anchor sample is drawn as a connected polyline (with arrowheads); the
right-hand column shows a thumbnail + TempM label so you can see *what* image
is moving.

Bin schemes supported:
  - "freq": many / medium / few from `dir_skyfinder.baseline.per_bin_mae`
            (train-frequency bins with bin_w=2.0).
  - "temp": 4 fixed temperature ranges:
            <0 C / 0–15 C / 15–30 C / >30 C.

Prereq:
  - Snapshot `<run>_fold{f}_ep{N}.pt` files must exist in `config["results_dir"]`
    (training was done with `snapshot_every > 0`).
  - Trajectory `.npz` files must exist in `config["trajectory_dir"]`. Generate
    them with: `python analysis.py trajectory` (which calls trajectory.run_trajectory).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from PIL import Image

from analysis.style import PALETTE, apply_nature_style, save_fig
from analysis.trajectory import find_snapshot_epochs

apply_nature_style()


DEFAULT_RUNS = (
    "baseline_resnet50", "lds_resnet50", "fds_resnet50", "lds_fds_resnet50",
)
DEFAULT_SPLITS = ("val", "test")
KIND_LABEL = {
    "baseline_resnet50": "baseline",
    "lds_resnet50":      "LDS",
    "fds_resnet50":      "FDS",
    "lds_fds_resnet50":  "LDS+FDS",
    "baseline_vit":      "baseline (ViT)",
    "lds_vit":           "LDS (ViT)",
    "fds_vit":           "FDS (ViT)",
    "lds_fds_vit":       "LDS+FDS (ViT)",
}


# ============================================================
# Bin assignment
# ============================================================

def _bin_freq(y: np.ndarray, train_y: np.ndarray, bin_w: float = 2.0) -> np.ndarray:
    """Returns array of strings 'many'/'medium'/'few' matching per_bin_mae."""
    lo = min(train_y.min(), y.min()); hi = max(train_y.max(), y.max())
    edges = np.arange(np.floor(lo / bin_w) * bin_w,
                      np.ceil(hi / bin_w) * bin_w + bin_w, bin_w)
    train_hist, _ = np.histogram(train_y, bins=edges)
    idx = np.clip(np.digitize(y, edges) - 1, 0, len(edges) - 2)
    labels = np.empty(len(y), dtype=object)
    for name, n_lo, n_hi in [("many", 100, np.inf), ("medium", 20, 100), ("few", 0, 20)]:
        which = np.isin(idx, np.where((train_hist >= n_lo) & (train_hist < n_hi))[0])
        labels[which] = name
    return labels


def _bin_temp(y: np.ndarray) -> np.ndarray:
    """4 fixed temperature ranges as strings."""
    out = np.empty(len(y), dtype=object)
    out[y < 0]               = "<0"
    out[(y >= 0)  & (y < 15)] = "0-15"
    out[(y >= 15) & (y < 30)] = "15-30"
    out[y >= 30]             = ">30"
    return out


BIN_ORDERS = {
    "freq": ["many", "medium", "few"],
    "temp": ["<0", "0-15", "15-30", ">30"],
}
BIN_COLORS = {
    "many":   "#4C72B0",  "medium": "#DD8452",  "few":    "#C44E52",
    "<0":     "#3B6FB6",  "0-15":   "#55A868",  "15-30":  "#DD8452",  ">30":   "#C44E52",
}


# ============================================================
# Anchor selection
# ============================================================

def _pick_anchors(ys: np.ndarray, bin_labels: np.ndarray, bin_order: list[str],
                  seed: int = 0) -> dict[str, int]:
    """One anchor index per bin; seeded random pick from each bin."""
    rng = np.random.default_rng(seed)
    anchors: dict[str, int] = {}
    for b in bin_order:
        candidates = np.where(bin_labels == b)[0]
        if len(candidates) == 0:
            print(f"  [warn] no samples in bin {b!r}")
            continue
        anchors[b] = int(rng.choice(candidates))
    return anchors


# ============================================================
# Embedding loading + projection
# ============================================================

def _load_traj_npz(traj_dir: Path, run: str, fold: int, ep: int, split: str) -> dict | None:
    p = traj_dir / f"{run}_fold{fold}_ep{ep}_{split}.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def _fit_pca_on_final(features_final: np.ndarray) -> "PCA":
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=0).fit(features_final)
    return pca


def _load_train_y(config: dict, fold: int) -> np.ndarray:
    df = pd.read_csv(config["labels_path"])
    splits = json.loads(Path(config["splits_path"]).read_text())
    return df.iloc[splits[fold]["train"]]["TempM"].to_numpy()


def _thumbnail(img_path: Path, size: int = 224) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize((size, size))
    return np.asarray(img)


# ============================================================
# Plotting
# ============================================================

def _plot_one(config: dict, run: str, fold: int, split: str,
              bin_scheme: str, seed: int, fig_dir: Path) -> None:
    """One figure per (run, split, bin_scheme)."""
    traj_dir = Path(config["trajectory_dir"])
    eps = find_snapshot_epochs(config, run, fold)
    if len(eps) < 2:
        print(f"[skip] {run}: need >=2 snapshot epochs, found {eps}")
        return

    # Load each epoch's npz for this split.
    by_ep: dict[int, dict] = {}
    for ep in eps:
        d = _load_traj_npz(traj_dir, run, fold, ep, split)
        if d is None:
            print(f"  [skip] missing trajectory npz: {run}_fold{fold}_ep{ep}_{split}.npz "
                  f"-- run `python analysis.py trajectory` first")
            return
        by_ep[ep] = d

    ys = by_ep[eps[-1]]["ys"]
    train_y = _load_train_y(config, fold)
    bin_labels = _bin_freq(ys, train_y) if bin_scheme == "freq" else _bin_temp(ys)
    bin_order = BIN_ORDERS[bin_scheme]
    anchors = _pick_anchors(ys, bin_labels, bin_order, seed=seed)
    if not anchors:
        print(f"[skip] {run} {split}: no anchors selected"); return

    pca = _fit_pca_on_final(by_ep[eps[-1]]["features"])
    xy_by_ep = {ep: pca.transform(d["features"]) for ep, d in by_ep.items()}

    # Image lookup (anchor index in npz == iloc index in `splits[fold][split]`,
    # because extract_one used `df.iloc[fold_info[split]].reset_index(drop=True)`
    # without any shuffling or subsampling for val/test).
    labels_df = pd.read_csv(config["labels_path"])
    splits = json.loads(Path(config["splits_path"]).read_text())
    split_df = labels_df.iloc[splits[fold][split]].reset_index(drop=True)
    img_dir = Path(config["img_dir"])

    # ---- Figure layout: scatter on the left, anchor cards on the right ----
    n_anch = len(anchors)
    fig = plt.figure(figsize=(8.5, max(3.2, 1.6 * n_anch)))
    gs = fig.add_gridspec(n_anch, 4, width_ratios=[3.2, 0.05, 1.0, 1.6], wspace=0.15)
    ax_scatter = fig.add_subplot(gs[:, 0])

    # Background scatter 1: INITIAL-epoch distribution (light green ghost,
    # projected through the same final-epoch PCA so the shift is visible).
    init_xy = xy_by_ep[eps[0]]
    ax_scatter.scatter(init_xy[:, 0], init_xy[:, 1],
                       s=4, alpha=0.18, color="#90EE90", edgecolor="none",
                       label=f"ep {eps[0]} (initial dist.)", zorder=1)

    # Background scatter 2: FINAL-epoch points colored by bin (current state).
    final_xy = xy_by_ep[eps[-1]]
    for b in bin_order:
        mask = bin_labels == b
        if not mask.any(): continue
        ax_scatter.scatter(final_xy[mask, 0], final_xy[mask, 1],
                           s=4, alpha=0.30, color=BIN_COLORS[b], edgecolor="none",
                           label=f"{b} ep {eps[-1]} (n={int(mask.sum())})", zorder=2)

    # Anchor polylines + arrowheads
    for b, idx in anchors.items():
        traj = np.stack([xy_by_ep[ep][idx] for ep in eps])  # (T, 2)
        color = BIN_COLORS[b]
        ax_scatter.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=1.4, alpha=0.95)
        ax_scatter.scatter(traj[0, 0], traj[0, 1], marker="o", s=22,
                           facecolor="white", edgecolor=color, lw=1.0, zorder=5)
        ax_scatter.scatter(traj[-1, 0], traj[-1, 1], marker="*", s=80,
                           color=color, edgecolor="black", lw=0.4, zorder=6)
        # arrowhead from second-to-last to last
        ax_scatter.add_patch(FancyArrowPatch(
            tuple(traj[-2]), tuple(traj[-1]),
            arrowstyle="-|>", mutation_scale=8, color=color, lw=0))

    ax_scatter.set_xlabel(f"PC1 (fit on {run} ep{eps[-1]} {split} features)")
    ax_scatter.set_ylabel("PC2")
    ax_scatter.set_title(f"{KIND_LABEL.get(run, run)} -- {split} -- "
                         f"trajectory across ep {eps[0]}→{eps[-1]} ({len(eps)} snapshots)")
    ax_scatter.legend(loc="best", frameon=False, fontsize=6)

    # ---- Right column: image + TempM card per anchor ----
    for row_i, (b, idx) in enumerate(anchors.items()):
        ax_img = fig.add_subplot(gs[row_i, 2])
        row = split_df.iloc[idx]
        try:
            thumb = _thumbnail(img_dir / str(row["CamId"]) / row["Filename"], size=128)
            ax_img.imshow(thumb)
        except Exception as e:
            ax_img.text(0.5, 0.5, f"<image missing>\n{e}",
                        ha="center", va="center", fontsize=5, transform=ax_img.transAxes)
        ax_img.set_xticks([]); ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_edgecolor(BIN_COLORS[b]); spine.set_linewidth(1.5)

        ax_lbl = fig.add_subplot(gs[row_i, 3])
        ax_lbl.set_axis_off()
        y_val  = float(ys[idx])
        cam    = row["CamId"]
        fname  = row["Filename"]
        month  = row.get("Month", "?")
        hour   = row.get("Hour", "?")
        ax_lbl.text(0.0, 0.7,
                    f"bin: {b}\n"
                    f"TempM: {y_val:+.1f} °C\n"
                    f"cam: {cam}\n"
                    f"month {month}, hour {hour}\n"
                    f"file: {fname[:24]}{'…' if len(fname) > 24 else ''}",
                    fontsize=6, va="top", transform=ax_lbl.transAxes)

    save_fig(fig, f"fig_sample_traj_{run}_fold{fold}_{split}_{bin_scheme}", fig_dir)


def make_sample_trajectory(config: dict, runs: tuple[str, ...] = DEFAULT_RUNS,
                           fold: int = 0, splits: tuple[str, ...] = DEFAULT_SPLITS,
                           bin_schemes: tuple[str, ...] = ("freq", "temp"),
                           seed: int = 0) -> None:
    """Emit one PDF per (run, split, bin_scheme). Skips quietly if data missing."""
    fig_dir = Path(config["figures_dir"])
    for run in runs:
        for split in splits:
            for scheme in bin_schemes:
                print(f"[traj-samples] {run} fold={fold} split={split} bin_scheme={scheme}")
                _plot_one(config, run, fold, split, scheme, seed, fig_dir)


# ============================================================
# Bin-distribution filmstrip across epochs
# ============================================================

def _plot_bin_dist(config: dict, run: str, fold: int, split: str,
                   bin_scheme: str, every: int, subsample: int | None,
                   seed: int, fig_dir: Path) -> None:
    """One PDF: panels = sampled snapshot epochs, all colored by bin, shared PCA basis."""
    traj_dir = Path(config["trajectory_dir"])
    eps_all = find_snapshot_epochs(config, run, fold)
    if len(eps_all) < 2:
        print(f"[skip] {run}: need >=2 snapshot epochs, found {eps_all}")
        return
    # Filter: keep ep == 0 (or smallest), every Nth, and always the final.
    eps = [ep for ep in eps_all if ep % every == 0 or ep == eps_all[0] or ep == eps_all[-1]]
    eps = sorted(set(eps))
    if len(eps) < 2:
        print(f"[skip] {run}: every={every} left <2 epochs after filter; have {eps_all}")
        return

    by_ep: dict[int, dict] = {}
    for ep in eps:
        d = _load_traj_npz(traj_dir, run, fold, ep, split)
        if d is None:
            print(f"  [skip] missing trajectory npz: {run}_fold{fold}_ep{ep}_{split}.npz "
                  f"-- run `python analysis.py trajectory` first")
            return
        by_ep[ep] = d

    ys = by_ep[eps[-1]]["ys"]
    train_y = _load_train_y(config, fold)
    bin_labels = _bin_freq(ys, train_y) if bin_scheme == "freq" else _bin_temp(ys)
    bin_order  = BIN_ORDERS[bin_scheme]

    pca = _fit_pca_on_final(by_ep[eps[-1]]["features"])
    xy_by_ep = {ep: pca.transform(d["features"]) for ep, d in by_ep.items()}

    # Optional subsample (consistent across epochs so same points are shown each panel).
    n_total = len(ys)
    if subsample is not None and n_total > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=subsample, replace=False)
        idx.sort()
        bin_labels = bin_labels[idx]
        xy_by_ep = {ep: xy[idx] for ep, xy in xy_by_ep.items()}

    # Shared axis limits so motion is visible (use union over all epochs).
    all_xy = np.concatenate(list(xy_by_ep.values()), axis=0)
    pad = 0.05 * (all_xy.max(axis=0) - all_xy.min(axis=0))
    xlim = (all_xy[:, 0].min() - pad[0], all_xy[:, 0].max() + pad[0])
    ylim = (all_xy[:, 1].min() - pad[1], all_xy[:, 1].max() + pad[1])

    # Layout: up to 6 columns.
    n_panels = len(eps)
    n_cols = min(6, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows + 0.6),
                            squeeze=False, sharex=True, sharey=True)
    fig.suptitle(f"{KIND_LABEL.get(run, run)} — {split} — bin distribution every {every} ep "
                 f"(PCA fit on ep {eps[-1]} features, bin scheme={bin_scheme})",
                 fontsize=8, y=0.995)

    handles_for_legend = {}
    for i, ep in enumerate(eps):
        ax = axs[i // n_cols, i % n_cols]
        xy = xy_by_ep[ep]
        for b in bin_order:
            mask = bin_labels == b
            if not mask.any(): continue
            sc = ax.scatter(xy[mask, 0], xy[mask, 1], s=3, alpha=0.45,
                            color=BIN_COLORS[b], edgecolor="none")
            handles_for_legend[b] = sc
        ax.set_title(f"ep {ep}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    # blank unused panels
    for j in range(n_panels, n_rows * n_cols):
        axs[j // n_cols, j % n_cols].set_axis_off()

    # One legend for the whole figure.
    if handles_for_legend:
        fig.legend(
            [handles_for_legend[b] for b in bin_order if b in handles_for_legend],
            [b for b in bin_order if b in handles_for_legend],
            loc="lower center", ncol=len(bin_order), frameon=False, fontsize=7,
            bbox_to_anchor=(0.5, -0.005),
        )
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.03, right=0.99, hspace=0.18, wspace=0.05)
    save_fig(fig, f"fig_bin_traj_{run}_fold{fold}_{split}_{bin_scheme}_every{every}", fig_dir)


def make_bin_distribution_traj(config: dict, runs: tuple[str, ...] = DEFAULT_RUNS,
                               fold: int = 0, splits: tuple[str, ...] = DEFAULT_SPLITS,
                               bin_schemes: tuple[str, ...] = ("freq", "temp"),
                               every: int = 5, subsample: int | None = None,
                               seed: int = 0) -> None:
    """Emit one PDF per (run, split, bin_scheme): the full embedding distribution at
    every `every`-th snapshot epoch, panels colored by bin, shared PCA basis.

    `subsample=None` (default) plots every point. Pass an int to randomly subsample
    each panel to that many points; same indices are used across panels so motion
    stays consistent visually.
    """
    fig_dir = Path(config["figures_dir"])
    for run in runs:
        for split in splits:
            for scheme in bin_schemes:
                print(f"[bin-traj] {run} fold={fold} split={split} "
                      f"bin_scheme={scheme} every={every}")
                _plot_bin_dist(config, run, fold, split, scheme, every,
                               subsample, seed, fig_dir)
