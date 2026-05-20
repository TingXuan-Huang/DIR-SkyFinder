"""Ablation-family figures: A1-A5 (DIR hyperparams), D1 (sky-mask), D4 (linear-probe),
E1 (seed variance), F1-F5 (label corruption)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from skyfinder.analysis.figures._helpers import (KIND_LABEL, KIND_ORDER,
                                                  _empty, _resnet)
from skyfinder.analysis.style import MARKERS, PALETTE, figsize, save_fig


# ============================================================
# A-family helpers (DIR hyperparams)
# ============================================================

def _render_ablation_line(config: dict, df, group: str, x_col: str,
                          x_label: str, name: str) -> None:
    fig_dir = Path(config["figures_dir"])
    sub = _resnet(df[df["group"] == group]).sort_values(x_col)
    if _empty(sub, name):
        return
    fig, ax = plt.subplots(figsize=figsize("single"))
    ax.plot(sub[x_col], sub["test_overall"], marker="o", color=PALETTE["lds_fds"], label="overall")
    ax.plot(sub[x_col], sub["test_few"],     marker="^", color=PALETTE["fds"],      label="few")
    ax.set_xlabel(x_label); ax.set_ylabel("test MAE (°C)")
    ax.legend(frameon=False)
    save_fig(fig, name, fig_dir)


def fig_a1_sigma(config, df):        _render_ablation_line(config, df, "A1", "lds_sigma",        "lds_sigma (°C)", "fig_a1_sigma")
def fig_a3_bucket(config, df):       _render_ablation_line(config, df, "A3", "bin_width",        "bin_width (°C)", "fig_a3_bucket")
def fig_a4_momentum(config, df):     _render_ablation_line(config, df, "A4", "fds_momentum",     "fds_momentum",   "fig_a4_momentum")
def fig_a5_start_smooth(config, df): _render_ablation_line(config, df, "A5", "fds_start_smooth", "fds_start_smooth (epoch)", "fig_a5_start_smooth")


def fig_a2_reweight(config: dict, df) -> None:
    fig_dir = Path(config["figures_dir"])
    sub = _resnet(df[df["group"] == "A2"])
    if _empty(sub, "fig_a2_reweight"):
        return
    fig, ax = plt.subplots(figsize=figsize("single"))
    schemes = sub["lds_reweight"].fillna("none").tolist()
    x = np.arange(len(schemes))
    width = 0.4
    ax.bar(x - width/2, sub["test_overall"], width, color=PALETTE["lds_fds"], label="overall")
    ax.bar(x + width/2, sub["test_few"],     width, color=PALETTE["fds"],     label="few")
    ax.set_xticks(x); ax.set_xticklabels(schemes)
    ax.set_xlabel("lds_reweight"); ax.set_ylabel("test MAE (°C)")
    ax.legend(frameon=False)
    save_fig(fig, "fig_a2_reweight", fig_dir)


# ============================================================
# D1 sky-mask raw vs masked
# ============================================================

def fig_d1_skymask(config: dict) -> None:
    fig_dir = Path(config["figures_dir"])
    p = Path(config["skymask_path"])
    if not p.exists():
        print("[skip] fig_d1_skymask: d1 results not found")
        return
    d = json.loads(p.read_text())
    rows = [r for r in d["per_fold"] if "raw" in r]
    if not rows:
        print("[skip] fig_d1_skymask: empty per_fold"); return
    fig, ax = plt.subplots(figsize=figsize("single"))
    width = 0.35
    x = np.arange(len(rows))
    raws  = [r["raw"]["overall"]    for r in rows]
    masks = [r["masked"]["overall"] for r in rows]
    ax.bar(x - width/2, raws,  width, label="raw",    color=PALETTE["baseline"])
    ax.bar(x + width/2, masks, width, label="masked", color=PALETTE["lds_fds"])
    ax.set_xticks(x)
    ax.set_xticklabels([KIND_LABEL.get(r["config_kind"], r["config_kind"]) for r in rows])
    ax.set_ylabel("test MAE (°C)"); ax.legend(frameon=False)
    save_fig(fig, "fig_d1_skymask", fig_dir)


# ============================================================
# D4 linear-probe vs full fine-tune
# ============================================================

def fig_d4_linprobe(config: dict, df) -> None:
    fig_dir = Path(config["figures_dir"])
    main = _resnet(df[(df["group"] == "main") & (df["fold"] == 0)
                      & df["config_kind"].isin(KIND_ORDER)])
    d4 = _resnet(df[df["group"] == "D4"])
    if _empty(d4, "fig_d4_linprobe"):
        return
    fig, ax = plt.subplots(figsize=figsize("single"))
    x = np.arange(len(KIND_ORDER)); width = 0.35
    full = [main[main["config_kind"] == k]["test_overall"].mean() for k in KIND_ORDER]
    lin  = [d4[d4["config_kind"] == k]["test_overall"].mean()     for k in KIND_ORDER]
    ax.bar(x - width/2, full, width, color=PALETTE["baseline"], label="full fine-tune")
    ax.bar(x + width/2, lin,  width, color=PALETTE["lds_fds"],  label="linear probe")
    ax.set_xticks(x); ax.set_xticklabels([KIND_LABEL[k] for k in KIND_ORDER], rotation=20, ha="right")
    ax.set_ylabel("test MAE (°C)"); ax.legend(frameon=False)
    save_fig(fig, "fig_d4_linprobe", fig_dir)


# ============================================================
# E1 seed variance
# ============================================================

def fig_e1_seeds(config: dict, df) -> None:
    fig_dir = Path(config["figures_dir"])
    sub = _resnet(df[df["group"] == "E1"])
    if _empty(sub, "fig_e1_seeds"):
        return
    fig, ax = plt.subplots(figsize=figsize("single"))
    metrics = ["overall", "few"]
    width = 0.35; x = np.arange(len(metrics))
    means = [sub[f"test_{m}"].mean() for m in metrics]
    stds  = [sub[f"test_{m}"].std(ddof=0) for m in metrics]
    ax.bar(x, means, width, yerr=stds, capsize=3, color=PALETTE["lds_fds"], edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("test MAE (°C)"); ax.set_title(f"seed variance ({len(sub)} seeds)")
    save_fig(fig, "fig_e1_seeds", fig_dir)


# ============================================================
# F-family (label corruption)
# ============================================================

def _render_corruption_line(config: dict, df, group: str, x_label: str, y_col: str,
                             name: str, ref_lookup: tuple[str, str] | None = None) -> None:
    fig_dir = Path(config["figures_dir"])
    sub = _resnet(df[df["group"] == group])
    if _empty(sub, name):
        return
    fig, ax = plt.subplots(figsize=figsize("single"))
    for kind in KIND_ORDER:
        rows = sub[sub["config_kind"] == kind].sort_values("corruption_param")
        if rows.empty:
            continue
        ax.plot(rows["corruption_param"], rows[y_col],
                marker=MARKERS[kind], color=PALETTE[kind], label=KIND_LABEL[kind])
    if ref_lookup is not None:
        main = _resnet(df[(df["group"] == "main") & (df["fold"] == 0)])
        for kind in ["baseline", "lds_fds"]:
            v = main[main["config_kind"] == kind][y_col]
            if not v.empty:
                ax.scatter([0], [v.mean()], color=PALETTE[kind], marker="*", s=30, zorder=5)
    ax.set_xlabel(x_label); ax.set_ylabel(f"test {y_col.replace('test_', '')} MAE (°C)")
    ax.legend(frameon=False)
    save_fig(fig, name, fig_dir)


def fig_f1_rate(config, df):  _render_corruption_line(config, df, "F1", "corruption rate",           "test_overall", "fig_f1_rate", ref_lookup=("main", "fold0"))
def fig_f3_drop(config, df):  _render_corruption_line(config, df, "F3", "rare-bin drop prob",        "test_few",     "fig_f3_drop")
def fig_f5_noise(config, df): _render_corruption_line(config, df, "F5", "Gaussian noise sigma (°C)", "test_overall", "fig_f5_noise")


def fig_f2_range(config: dict, df) -> None:
    fig_dir = Path(config["figures_dir"])
    sub_i = _resnet(df[df["group"] == "F2_impute"])
    sub_d = _resnet(df[df["group"] == "F2_drop"])
    if _empty(sub_i, "fig_f2_range") and _empty(sub_d, "fig_f2_range"):
        return
    fig, (ax_i, ax_d) = plt.subplots(1, 2, figsize=figsize("double", 0.45), sharey=True)
    for ax, sub, title in [(ax_i, sub_i, "impute"), (ax_d, sub_d, "drop")]:
        if sub.empty:
            ax.set_title(f"{title} (missing)"); continue
        ranges = sorted(sub["corruption_param"].unique())
        kinds_present = [k for k in KIND_ORDER if (sub["config_kind"] == k).any()]
        width = 0.8 / max(len(kinds_present), 1)
        x = np.arange(len(ranges))
        for i, kind in enumerate(kinds_present):
            ys = [sub[(sub["corruption_param"] == r) & (sub["config_kind"] == kind)]["test_overall"].mean()
                  for r in ranges]
            ax.bar(x + (i - (len(kinds_present)-1)/2) * width, ys, width,
                   color=PALETTE[kind], label=KIND_LABEL[kind], edgecolor="none")
        ax.set_xticks(x); ax.set_xticklabels([f"{r:.0f}±2.5" for r in ranges])
        ax.set_title(title); ax.set_xlabel("dropped range center (°C)")
    ax_i.set_ylabel("test MAE (°C)"); ax_i.legend(frameon=False, loc="best")
    save_fig(fig, "fig_f2_range", fig_dir)
