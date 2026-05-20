"""Headline-sweep figures: main bar chart, scatter, dist + per-bin errbar."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from skyfinder.analysis.figures._helpers import (BIN_ORDER, DEFAULT_HEADLINE_RUNS,
                                                  KIND_LABEL, KIND_ORDER,
                                                  _empty, _get_test_preds,
                                                  _ref_lines, _resnet, _train_y)
from skyfinder.analysis.style import PALETTE, REF_COLORS, figsize, save_fig
from skyfinder.training.engine import per_bin_mae_by_edges


def fig_main_sweep(config: dict, df, metric: str = "test") -> None:
    """Per-bin MAE bars over 4 configs.

    `metric` ∈ {"test", "val"} — column prefix in `df`.
    Output filename includes the metric suffix.
    """
    fig_dir = Path(config["figures_dir"])
    sub = _resnet(df[df["group"] == "main"])
    if _empty(sub, f"fig_main_sweep_{metric}"):
        return
    fig, ax = plt.subplots(figsize=figsize("double", 0.45))
    width = 0.18
    x = np.arange(len(BIN_ORDER))
    for i, kind in enumerate(KIND_ORDER):
        rows = sub[sub["config_kind"] == kind]
        if rows.empty:
            continue
        means = [rows[f"{metric}_{b}"].mean()  for b in BIN_ORDER]
        stds  = [rows[f"{metric}_{b}"].std(ddof=0) for b in BIN_ORDER]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=2,
               color=PALETTE[kind], label=KIND_LABEL[kind], edgecolor="none")
    refs = _ref_lines(config)
    for j, (tag, val) in enumerate(refs.items()):
        ax.axhline(val, color=REF_COLORS[list(REF_COLORS)[j % 3]],
                   linestyle="--", linewidth=0.5, label=tag, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(BIN_ORDER)
    ax.set_ylabel(f"{metric} MAE (°C)")
    ax.set_xlabel("bin (DIR many/medium/few)")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    save_fig(fig, f"fig_main_sweep_{metric}", fig_dir)


def fig_pred_vs_true_scatter(
    config: dict,
    pairs: list[tuple[str, str]] | None = None,
    fold: int = 0,
) -> None:
    """Pred-vs-true scatter for one or more runs. Defaults to baseline vs LDS+FDS."""
    if pairs is None:
        pairs = [("baseline_resnet50", "baseline"), ("lds_fds_resnet50", "LDS+FDS")]
    fig_dir = Path(config["figures_dir"])
    runs = [(label, _get_test_preds(config, f"{name}_fold{fold}")) for name, label in pairs]
    if not any(arr is not None for _, arr in runs):
        print("[skip] fig_pred_vs_true_scatter: no test_inference data (run skyfinder inference)")
        return
    fig, axs = plt.subplots(1, len(pairs), figsize=figsize("double", 0.45),
                            sharex=True, sharey=True, squeeze=False)
    for ax, (label, arrs) in zip(axs[0], runs):
        if arrs is None:
            ax.set_title(f"{label} (missing)"); continue
        y_true, y_pred = arrs
        color_key = "lds_fds" if label == "LDS+FDS" else label.lower()
        ax.scatter(y_true, y_pred, s=2, alpha=0.3,
                   color=PALETTE.get(color_key, "#888888"), edgecolor="none")
        lo = float(min(y_true.min(), y_pred.min())) - 1
        hi = float(max(y_true.max(), y_pred.max())) + 1
        ax.plot([lo, hi], [lo, hi], "k-", linewidth=0.5)
        ax.set_title(label)
        ax.set_xlabel("true TempM (°C)")
    axs[0, 0].set_ylabel("predicted TempM (°C)")
    save_fig(fig, f"fig_pred_vs_true_scatter_fold{fold}", fig_dir)


def fig_dist_and_errbar(
    config: dict,
    pairs: list[tuple[str, str]] | None = None,
    fold: int = 0,
) -> None:
    """For each (name, kind) pair: train-dist histogram + per-bin test MAE."""
    if pairs is None:
        pairs = list(DEFAULT_HEADLINE_RUNS)
    fig_dir = Path(config["figures_dir"])
    for name, kind in pairs:
        arrs = _get_test_preds(config, f"{name}_fold{fold}")
        if arrs is None:
            print(f"[skip] fig_dist_and_errbar: no test preds for {name}_fold{fold}")
            continue
        y_true, y_pred = arrs
        train_y = _train_y(config, fold)
        bin_w = 1.0
        lo = float(min(train_y.min(), y_true.min(), y_pred.min()))
        hi = float(max(train_y.max(), y_true.max(), y_pred.max()))
        edges = np.arange(np.floor(lo / bin_w) * bin_w, np.ceil(hi / bin_w) * bin_w + bin_w, bin_w)
        centers = (edges[:-1] + edges[1:]) / 2

        # Shared per-bin MAE utility (was duplicated logic; now lives in engine.py)
        per_bin = per_bin_mae_by_edges(y_true, y_pred, edges)

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=figsize("single", 0.95),
            gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08}, sharex=True,
        )
        ax_top.hist(train_y, bins=edges, color="#888888", alpha=0.5, label="train true")
        ax_top.hist(y_pred,  bins=edges, color=PALETTE[kind],
                    alpha=0.65, label="test pred", histtype="step", linewidth=1.0)
        ax_top.set_ylabel("count")
        ax_top.legend(loc="upper right", frameon=False)
        ax_top.set_title(KIND_LABEL.get(kind, kind))

        ax_bot.bar(centers, per_bin, width=bin_w * 0.9,
                   color=PALETTE[kind], edgecolor="none")
        ax_bot.set_ylabel("test MAE (°C)")
        ax_bot.set_xlabel("TempM bin (°C)")
        save_fig(fig, f"fig_dist_and_errbar_{kind}_fold{fold}", fig_dir)
