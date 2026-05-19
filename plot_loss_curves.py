"""plot_loss_curves.py — train + val MAE curves for all 8 main-sweep runs.

Renders a 2×4 grid:
  rows: ResNet-50, ViT-B/16
  cols: baseline, LDS, FDS, LDS+FDS

Each panel: train MAE (dashed, light) + val MAE (solid) over epochs, with a
star at best_epoch. Reads each run's results JSON from
`results/<sub>/<run>_fold{F}.json` (nested or flat -- both supported via
`baseline._resolve_load_path`).

Usage:
    python plot_loss_curves.py                       # fold 0
    python plot_loss_curves.py --fold 2
    python plot_loss_curves.py --config /path/to/analysis_config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from analysis._config import load_config
from analysis.style import PALETTE, apply_nature_style, figsize, save_fig
from dir_skyfinder.baseline import _resolve_load_path

apply_nature_style()


ARCHS = [("resnet50", "ResNet-50"), ("vit", "ViT-B/16")]
KINDS = [("baseline", "baseline"), ("lds", "LDS"), ("fds", "FDS"), ("lds_fds", "LDS+FDS")]


def _load(config: dict, run_name: str, fold: int) -> dict | None:
    p = _resolve_load_path(f"{run_name}_fold{fold}", ".json", Path(config["results_dir"]))
    return json.loads(p.read_text()) if p is not None else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("analysis_config.yaml"))
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    config = load_config(a.config)

    fig, axs = plt.subplots(len(ARCHS), len(KINDS),
                            figsize=figsize("double", 0.55),
                            sharex=True, sharey=True)
    n_any = 0
    for r, (arch_tag, arch_label) in enumerate(ARCHS):
        for c, (kind, kind_label) in enumerate(KINDS):
            ax = axs[r, c]
            run_name = f"{kind}_{arch_tag}"
            data = _load(config, run_name, a.fold)
            if data is None:
                ax.set_axis_off()
                if r == 0:
                    ax.set_title(f"{kind_label}\n(missing)")
                continue
            history = data.get("history", [])
            if not history:
                ax.set_axis_off(); continue
            n_any += 1
            eps = [h["epoch"]     for h in history]
            train = [h["train_mae"] for h in history]
            val   = [h["val_mae"]   for h in history]
            color = PALETTE[kind]
            ax.plot(eps, train, color=color, linestyle="--", alpha=0.55, label="train")
            ax.plot(eps, val,   color=color, linestyle="-",                label="val")
            best = data.get("best_epoch")
            if best is not None and 0 <= best < len(val):
                ax.scatter([best], [val[best]], color=color, marker="*", s=25,
                           zorder=5, edgecolor="white", linewidth=0.4)
            if r == 0:
                ax.set_title(kind_label)
            if c == 0:
                ax.set_ylabel(f"{arch_label}\nMAE (°C)")
            if r == len(ARCHS) - 1:
                ax.set_xlabel("epoch")
            ax.legend(frameon=False, loc="upper right", fontsize=5)

    if n_any == 0:
        print("[skip] no run JSONs found — nothing to plot")
        return
    save_fig(fig, f"fig_loss_curves_all_fold{a.fold}", Path(config["figures_dir"]))


if __name__ == "__main__":
    main()
