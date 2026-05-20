"""Training-curve figures (loss curves over epochs).

THE REUSABLE LOSS-CURVE FUNCTION (`fig_training_curves`) — accepts any list of
runs and any grid shape. Subsumes the old standalone `plot_loss_curves.py`.

Usage:
    # Default: 4 ResNet runs in a 2x2 grid (backwards-compatible).
    fig_training_curves(config)

    # All 8 runs in a 2x4 grid:
    runs = [(f"{k}_resnet50", k) for k in KIND_ORDER] + [(f"{k}_vit", k) for k in KIND_ORDER]
    fig_training_curves(config, runs=runs, grid=(2, 4), output_name="fig_loss_curves_all_fold0")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from skyfinder.analysis.figures._helpers import (DEFAULT_HEADLINE_RUNS, KIND_LABEL,
                                                  _load_run_json)
from skyfinder.analysis.style import PALETTE, figsize, save_fig


def fig_training_curves(
    config: dict,
    runs: list[tuple[str, str]] | None = None,
    fold: int = 0,
    grid: tuple[int, int] | None = None,
    output_name: str | None = None,
    sharex: bool = True,
    sharey: bool = True,
) -> None:
    """Render train + val MAE curves for an arbitrary list of runs in an arbitrary grid.

    Parameters
    ----------
    config : analysis_config dict (must contain `figures_dir`, `results_dir`).
    runs : list of `(run_name_prefix, label)` tuples. The prefix is looked up at
        `<results_dir>/<prefix>/<prefix>_fold{fold}.json`. The label is the
        title of the subplot (mapped through KIND_LABEL if it's a known key).
        Defaults to the 4-ResNet headline sweep (backwards-compatible).
    fold : which CV fold's JSON to read.
    grid : (rows, cols). If None, auto-fit: ncols = min(len(runs), 4),
        nrows = ceil(len(runs) / ncols).
    output_name : optional override for the saved filename stem.
    sharex, sharey : passed to matplotlib subplots.
    """
    if runs is None:
        runs = list(DEFAULT_HEADLINE_RUNS)
    if grid is None:
        ncols = min(len(runs), 4)
        nrows = (len(runs) + ncols - 1) // ncols
    else:
        nrows, ncols = grid
    output_name = output_name or f"fig_training_curves_fold{fold}"

    fig_dir = Path(config["figures_dir"])
    loaded = [(label, _load_run_json(config, name, fold)) for name, label in runs]
    if not any(d for _, d in loaded):
        print(f"[skip] {output_name}: no result JSONs found for any of {[r[0] for r in runs]}")
        return

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize("double", 0.55 * nrows / 2),
                            sharex=sharex, sharey=sharey, squeeze=False)
    axs_flat = axs.ravel()

    for ax, (kind, data) in zip(axs_flat, loaded):
        title = KIND_LABEL.get(kind, kind)
        if data is None:
            ax.set_axis_off(); ax.set_title(f"{title} (missing)"); continue
        history = data.get("history", [])
        if not history:
            ax.set_axis_off(); ax.set_title(f"{title} (no history)"); continue
        epochs = [h["epoch"] for h in history]
        train  = [h["train_mae"] for h in history]
        val    = [h["val_mae"]   for h in history]
        c = PALETTE.get(kind, "#666666")
        ax.plot(epochs, train, color=c, linestyle="--", alpha=0.55, label="train")
        ax.plot(epochs, val,   color=c, linestyle="-",                label="val")
        best = data.get("best_epoch")
        if best is not None and 0 <= best < len(val):
            ax.scatter([best], [val[best]], color=c, marker="*", s=25, zorder=5,
                       edgecolor="white", linewidth=0.4)
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper right")

    # Hide unused axes (when runs < grid cells).
    for ax in axs_flat[len(runs):]:
        ax.set_axis_off()

    for ax in axs[-1, :]:
        ax.set_xlabel("epoch")
    for ax in axs[:, 0]:
        ax.set_ylabel("MAE (°C)")

    save_fig(fig, output_name, fig_dir)
