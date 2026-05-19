"""All Nature-style figures for the DIR-SkyFinder runs and ablations.

Every public function takes `config: dict` (loaded from `analysis_config.yaml`) as
its first argument; paths/directories are read from `config`, not from module
globals. Code reuse from `dir_skyfinder` is limited to `analysis.style` and
nothing else here.

If the relevant rows/files don't exist yet (e.g., sweep hasn't run), the function
prints `[skip]` and returns -- so the pipeline degrades gracefully on partial
data instead of crashing.

ViT is filtered out at figure time per the project decision.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.style import (MARKERS, PALETTE, REF_COLORS, apply_nature_style,
                            figsize, save_fig)

apply_nature_style()


# Order used everywhere we group bars/lines by config_kind.
KIND_ORDER = ["baseline", "lds", "fds", "lds_fds"]
KIND_LABEL = {"baseline": "baseline", "lds": "LDS", "fds": "FDS", "lds_fds": "LDS+FDS"}
BIN_ORDER = ["overall", "many", "medium", "few"]

EMBED_RUNS = [
    ("baseline_resnet50", "baseline"),
    ("lds_resnet50",      "lds"),
    ("fds_resnet50",      "fds"),
    ("lds_fds_resnet50",  "lds_fds"),
]

_TRAJ_EP_RE = re.compile(r"_ep(\d+)_")


# ============================================================
# helpers (all take config; no module-level paths)
# ============================================================

def _resnet(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["model"] == "resnet50"]


def _empty(df: pd.DataFrame, name: str) -> bool:
    if len(df) == 0:
        print(f"[skip] {name}: no matching rows")
        return True
    return False


def _load_run_json(config: dict, name: str, fold: int = 0) -> dict | None:
    p = Path(config["results_dir"]) / f"{name}_fold{fold}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_test_runs(config: dict) -> dict:
    """`test_inference_path` -> {run_name: row_dict}. Empty if not generated yet."""
    p = Path(config["test_inference_path"])
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("runs", {}) or {}


def _get_test_preds(config: dict, run_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (test_ys, test_preds) for `run_name` (full, with `_foldN`), or None."""
    r = _load_test_runs(config).get(run_name)
    if not r or "test_preds" not in r or "test_ys" not in r:
        return None
    return (np.asarray(r["test_ys"],    dtype=np.float32),
            np.asarray(r["test_preds"], dtype=np.float32))


def _train_y(config: dict, fold: int = 0) -> np.ndarray:
    df = pd.read_csv(config["labels_path"])
    splits = json.loads(Path(config["splits_path"]).read_text())
    return df.iloc[splits[fold]["train"]]["TempM"].to_numpy()


def _ref_lines(config: dict, c1_kind: str = "per_cam_month_mean") -> dict[str, float]:
    """Return {label: mean test MAE} for C1 (best predictor), C2, D1 (baseline raw)."""
    out: dict[str, float] = {}
    for tag, path_key, key in [
        ("C1 (per-cam-month)", "c1_results_path", c1_kind),
        ("C2 (metadata GBM)",  "c2_results_path", None),
        ("D1 (sky mask)",      "d1_results_path", None),
    ]:
        path = Path(config[path_key])
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        if "per_fold" in d:
            rows = d["per_fold"]
            if key:
                rows = [r for r in rows if r.get("predictor") == key]
            xs = [r.get("test", r.get("masked", {})).get("overall") for r in rows]
            xs = [x for x in xs if x is not None]
            if xs:
                out[tag] = float(np.mean(xs))
    return out


# ============================================================
# 1. Main sweep -- per-bin MAE bars over 4 configs, errors over folds
# ============================================================

def fig_main_sweep(config: dict, df: pd.DataFrame) -> None:
    fig_dir = Path(config["figures_dir"])
    sub = _resnet(df[df["group"] == "main"])
    if _empty(sub, "fig_main_sweep"):
        return
    fig, ax = plt.subplots(figsize=figsize("double", 0.45))
    width = 0.18
    x = np.arange(len(BIN_ORDER))
    for i, kind in enumerate(KIND_ORDER):
        rows = sub[sub["config_kind"] == kind]
        if rows.empty:
            continue
        means = [rows[f"test_{b}"].mean()  for b in BIN_ORDER]
        stds  = [rows[f"test_{b}"].std(ddof=0) for b in BIN_ORDER]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=2,
               color=PALETTE[kind], label=KIND_LABEL[kind], edgecolor="none")
    refs = _ref_lines(config)
    for j, (tag, val) in enumerate(refs.items()):
        ax.axhline(val, color=REF_COLORS[list(REF_COLORS)[j % 3]],
                   linestyle="--", linewidth=0.5, label=tag, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(BIN_ORDER)
    ax.set_ylabel("test MAE (°C)")
    ax.set_xlabel("bin (DIR many/medium/few)")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    save_fig(fig, "fig_main_sweep", fig_dir)


# ============================================================
# 1b. Training curves -- train + val MAE per epoch, 4 headline configs
# ============================================================

def fig_training_curves(config: dict, fold: int = 0,
                        runs: tuple[tuple[str, str], ...] = (
                            ("baseline_resnet50", "baseline"),
                            ("lds_resnet50",      "lds"),
                            ("fds_resnet50",      "fds"),
                            ("lds_fds_resnet50",  "lds_fds"),
                        )) -> None:
    fig_dir = Path(config["figures_dir"])
    loaded = [(label, _load_run_json(config, name, fold)) for name, label in runs]
    if not any(d for _, d in loaded):
        print("[skip] fig_training_curves: no headline JSONs found")
        return
    fig, axs = plt.subplots(2, 2, figsize=figsize("double", 0.55), sharey=True, sharex=True)
    for ax, (kind, data) in zip(axs.ravel(), loaded):
        if data is None:
            ax.set_axis_off(); ax.set_title(f"{KIND_LABEL.get(kind, kind)} (missing)"); continue
        history = data.get("history", [])
        if not history:
            ax.set_axis_off(); ax.set_title(f"{KIND_LABEL.get(kind, kind)} (no history)"); continue
        epochs = [h["epoch"] for h in history]
        train  = [h["train_mae"] for h in history]
        val    = [h["val_mae"]   for h in history]
        c = PALETTE[kind]
        ax.plot(epochs, train, color=c, linestyle="--", alpha=0.55, label="train")
        ax.plot(epochs, val,   color=c, linestyle="-",                label="val")
        best = data.get("best_epoch")
        if best is not None and 0 <= best < len(val):
            ax.scatter([best], [val[best]], color=c, marker="*", s=25, zorder=5,
                       edgecolor="white", linewidth=0.4)
        ax.set_title(KIND_LABEL.get(kind, kind))
        ax.legend(frameon=False, loc="upper right")
    for ax in axs[-1, :]:
        ax.set_xlabel("epoch")
    for ax in axs[:, 0]:
        ax.set_ylabel("MAE (°C)")
    save_fig(fig, f"fig_training_curves_fold{fold}", fig_dir)


# ============================================================
# 2. Pred-vs-true scatter (baseline vs LDS+FDS)
# ============================================================

def fig_pred_vs_true_scatter(config: dict, fold: int = 0) -> None:
    fig_dir = Path(config["figures_dir"])
    pairs = [("baseline_resnet50", "baseline"), ("lds_fds_resnet50", "LDS+FDS")]
    runs = [(label, _get_test_preds(config, f"{name}_fold{fold}")) for name, label in pairs]
    if not any(arr is not None for _, arr in runs):
        print("[skip] fig_pred_vs_true_scatter: no test_inference data (run inference.py)")
        return
    fig, axs = plt.subplots(1, 2, figsize=figsize("double", 0.45), sharex=True, sharey=True)
    for ax, (label, arrs) in zip(axs, runs):
        if arrs is None:
            ax.set_title(f"{label} (missing)"); continue
        y_true, y_pred = arrs
        ax.scatter(y_true, y_pred, s=2, alpha=0.3,
                   color=PALETTE["lds_fds" if label == "LDS+FDS" else "baseline"], edgecolor="none")
        lo = float(min(y_true.min(), y_pred.min())) - 1
        hi = float(max(y_true.max(), y_pred.max())) + 1
        ax.plot([lo, hi], [lo, hi], "k-", linewidth=0.5)
        ax.set_title(label)
        ax.set_xlabel("true TempM (°C)")
    axs[0].set_ylabel("predicted TempM (°C)")
    save_fig(fig, f"fig_pred_vs_true_scatter_fold{fold}", fig_dir)


# ============================================================
# 3. Dist (true vs pred) + per-bin error -- canonical DIR figure
# ============================================================

def fig_dist_and_errbar(config: dict, fold: int = 0) -> None:
    fig_dir = Path(config["figures_dir"])
    pairs = [
        ("baseline_resnet50", "baseline"),
        ("lds_resnet50", "lds"),
        ("fds_resnet50", "fds"),
        ("lds_fds_resnet50", "lds_fds"),
    ]
    for name, kind in pairs:
        arrs = _get_test_preds(config, f"{name}_fold{fold}")
        if arrs is None:
            print(f"[skip] fig_dist_and_errbar: no test preds for {name}_fold{fold}")
            continue
        y_true, y_pred = arrs
        train_y = _train_y(config, fold)

        # Val preds come from the per-run JSON (best-checkpoint val_preds/val_ys).
        run_json = _load_run_json(config, name, fold)
        val_y    = np.asarray((run_json or {}).get("val_ys", []),    dtype=np.float32)
        val_pred = np.asarray((run_json or {}).get("val_preds", []), dtype=np.float32)
        has_val  = val_y.size > 0 and val_pred.size == val_y.size

        bin_w = 1.0
        lo_vals = [train_y.min(), y_true.min(), y_pred.min()]
        hi_vals = [train_y.max(), y_true.max(), y_pred.max()]
        if has_val:
            lo_vals += [val_y.min(), val_pred.min()]
            hi_vals += [val_y.max(), val_pred.max()]
        lo = float(min(lo_vals)); hi = float(max(hi_vals))
        edges = np.arange(np.floor(lo / bin_w) * bin_w, np.ceil(hi / bin_w) * bin_w + bin_w, bin_w)
        centers = (edges[:-1] + edges[1:]) / 2

        def _per_bin(y_t, y_p):
            i = np.clip(np.digitize(y_t, edges) - 1, 0, len(edges) - 2)
            e = np.abs(y_t - y_p)
            return np.array([e[i == k].mean() if (i == k).any() else np.nan
                             for k in range(len(centers))])
        test_mae = _per_bin(y_true, y_pred)
        val_mae  = _per_bin(val_y, val_pred) if has_val else None

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=figsize("single", 0.95),
            gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08}, sharex=True,
        )
        ax_top.hist(train_y, bins=edges, color="#888888", alpha=0.5, label="train true")
        if has_val:
            ax_top.hist(val_pred, bins=edges, color="#666666",
                        alpha=0.85, label="val pred", histtype="step", linewidth=1.0, linestyle="--")
        ax_top.hist(y_pred,  bins=edges, color=PALETTE[kind],
                    alpha=0.85, label="test pred", histtype="step", linewidth=1.0)
        ax_top.set_ylabel("count")
        ax_top.legend(loc="upper right", frameon=False)
        ax_top.set_title(KIND_LABEL[kind])

        # Test MAE as filled bars (current behavior); val MAE overlaid as a
        # line+markers so paired comparison is readable across ~80 bins.
        ax_bot.bar(centers, test_mae, width=bin_w * 0.9,
                   color=PALETTE[kind], edgecolor="none", alpha=0.7, label="test MAE")
        if val_mae is not None:
            ax_bot.plot(centers, val_mae, color="black", linewidth=0.7,
                        marker="o", markersize=2.5, markerfacecolor="white",
                        markeredgewidth=0.5, label="val MAE")
            ax_bot.legend(loc="upper right", frameon=False)
        ax_bot.set_ylabel("MAE (°C)")
        ax_bot.set_xlabel("TempM bin (°C)")
        save_fig(fig, f"fig_dist_and_errbar_{kind}_fold{fold}", fig_dir)


# ============================================================
# 4-8. A-family
# ============================================================

def _a_line(config: dict, df: pd.DataFrame, group: str, x_col: str,
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


def fig_a1_sigma(config, df):        _a_line(config, df, "A1", "lds_sigma",        "lds_sigma (°C)", "fig_a1_sigma")
def fig_a3_bucket(config, df):       _a_line(config, df, "A3", "bin_width",        "bin_width (°C)", "fig_a3_bucket")
def fig_a4_momentum(config, df):     _a_line(config, df, "A4", "fds_momentum",     "fds_momentum",   "fig_a4_momentum")
def fig_a5_start_smooth(config, df): _a_line(config, df, "A5", "fds_start_smooth", "fds_start_smooth (epoch)", "fig_a5_start_smooth")


def fig_a2_reweight(config: dict, df: pd.DataFrame) -> None:
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
# 9. D1 sky-mask raw vs masked
# ============================================================

def fig_d1_skymask(config: dict) -> None:
    fig_dir = Path(config["figures_dir"])
    p = Path(config["d1_results_path"])
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
    ax.set_xticks(x); ax.set_xticklabels([KIND_LABEL.get(r["config_kind"], r["config_kind"]) for r in rows])
    ax.set_ylabel("test MAE (°C)"); ax.legend(frameon=False)
    save_fig(fig, "fig_d1_skymask", fig_dir)


# ============================================================
# 10. D4 linear-probe vs full fine-tune
# ============================================================

def fig_d4_linprobe(config: dict, df: pd.DataFrame) -> None:
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
# 11. E1 seed variance
# ============================================================

def fig_e1_seeds(config: dict, df: pd.DataFrame) -> None:
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
# 12-15. F-family
# ============================================================

def _f_line(config: dict, df: pd.DataFrame, group: str, x_label: str, y_col: str,
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


def fig_f1_rate(config, df):  _f_line(config, df, "F1", "corruption rate",           "test_overall", "fig_f1_rate", ref_lookup=("main", "fold0"))
def fig_f3_drop(config, df):  _f_line(config, df, "F3", "rare-bin drop prob",        "test_few",     "fig_f3_drop")
def fig_f5_noise(config, df): _f_line(config, df, "F5", "Gaussian noise sigma (°C)", "test_overall", "fig_f5_noise")


def fig_f2_range(config: dict, df: pd.DataFrame) -> None:
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


# ============================================================
# 16-20. Embedding figures
# ============================================================

def _load_embedding(config: dict, run_name: str, fold: int = 0, split: str = "val") -> dict | None:
    p = Path(config["embeddings_dir"]) / f"{run_name}_fold{fold}_{split}.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def _project_2d(features: np.ndarray, method: str) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=0).fit_transform(features)
    if method == "umap":
        try:
            import umap   # type: ignore
        except ImportError:
            print("[skip] UMAP requested but `umap-learn` not installed")
            return None  # type: ignore[return-value]
        return umap.UMAP(n_components=2, n_neighbors=15, random_state=0).fit_transform(features)
    raise ValueError(method)


def _fig_embed_scatter(config: dict, color_by: str, name: str, split: str) -> None:
    fig_dir = Path(config["figures_dir"])
    loaded = [(label, _load_embedding(config, run, split=split)) for run, label in EMBED_RUNS]
    if not any(d is not None for _, d in loaded):
        print(f"[skip] {name}_{split}: no embedding npz files for split={split}")
        return
    fig, axs = plt.subplots(2, 4, figsize=figsize("double", 0.55))
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
                ax.set_title(KIND_LABEL[label])
            if col == 0:
                ax.set_ylabel(method.upper())
            ax.set_xticks([]); ax.set_yticks([])
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.6, pad=0.02)
        cbar.set_label("true TempM (°C)" if color_by == "temp" else "camera id")
    fig.suptitle(f"split={split}", fontsize=7, y=0.99)
    save_fig(fig, f"{name}_{split}", fig_dir)


def fig_embed_temp(config: dict, split: str = "val"): _fig_embed_scatter(config, "temp", "fig_embed_temp", split)
def fig_embed_cam(config: dict, split: str = "val"):  _fig_embed_scatter(config, "cam",  "fig_embed_cam",  split)


def fig_embed_knn(config: dict, split: str = "val") -> None:
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsRegressor
    fig_dir = Path(config["figures_dir"])
    rows = []
    for run, label in EMBED_RUNS:
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
           color=[PALETTE[r[0]] for r in rows], edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels([KIND_LABEL[r[0]] for r in rows])
    ax.set_ylabel(f"5-fold k-NN MAE on features (°C, split={split})")
    save_fig(fig, f"fig_embed_knn_{split}", fig_dir)


def fig_embed_per_bin(config: dict, split: str = "val") -> None:
    fig_dir = Path(config["figures_dir"])
    bin_w = 2.0
    panels = [(label, _load_embedding(config, run, split=split)) for run, label in EMBED_RUNS]
    panels = [(l, d) for l, d in panels if d is not None]
    if not panels:
        print(f"[skip] fig_embed_per_bin_{split}: no embeddings"); return
    all_ys = np.concatenate([d["ys"] for _, d in panels])
    lo, hi = float(all_ys.min()), float(all_ys.max())
    edges = np.arange(np.floor(lo / bin_w) * bin_w, np.ceil(hi / bin_w) * bin_w + bin_w, bin_w)

    fig, axs = plt.subplots(1, len(panels), figsize=figsize("double", 0.30), sharey=True)
    im = None
    for ax, (label, data) in zip(np.atleast_1d(axs), panels):
        idx = np.clip(np.digitize(data["ys"], edges) - 1, 0, len(edges) - 2)
        mean_feat = np.stack([
            data["features"][idx == k].mean(axis=0) if (idx == k).any()
            else np.zeros(data["features"].shape[1])
            for k in range(len(edges) - 1)
        ])
        norm = np.linalg.norm(mean_feat, axis=1, keepdims=True) + 1e-9
        sim = (mean_feat / norm) @ (mean_feat / norm).T
        im = ax.imshow(sim, origin="lower", cmap="viridis", vmin=-1, vmax=1,
                       extent=[edges[0], edges[-1], edges[0], edges[-1]])
        ax.set_title(KIND_LABEL[label])
        ax.set_xlabel("bin (°C)")
    np.atleast_1d(axs)[0].set_ylabel("bin (°C)")
    if im is not None:
        fig.colorbar(im, ax=np.atleast_1d(axs).tolist(), shrink=0.6, label="cosine sim")
    fig.suptitle(f"split={split}", fontsize=7, y=0.99)
    save_fig(fig, f"fig_embed_per_bin_{split}", fig_dir)


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0); Y = Y - Y.mean(0)
    xy = float(np.linalg.norm(X.T @ Y, ord="fro") ** 2)
    xx = float(np.linalg.norm(X.T @ X, ord="fro"))
    yy = float(np.linalg.norm(Y.T @ Y, ord="fro"))
    return xy / (xx * yy + 1e-12)


def fig_embed_cka(config: dict, split: str = "val") -> None:
    fig_dir = Path(config["figures_dir"])
    loaded = [(label, _load_embedding(config, run, split=split)) for run, label in EMBED_RUNS]
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
    ticks = [KIND_LABEL[l] for l, _ in loaded]
    ax.set_xticks(range(n)); ax.set_xticklabels(ticks, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(ticks)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
    fig.colorbar(im, ax=ax, shrink=0.8, label="linear CKA")
    fig.suptitle(f"split={split}", fontsize=7, y=0.99)
    save_fig(fig, f"fig_embed_cka_{split}", fig_dir)


# ============================================================
# Trajectory figures (require analysis/trajectory.py to have run)
# ============================================================

def _list_traj_epochs(config: dict, run_name: str, fold: int, split: str) -> list[int]:
    traj_dir = Path(config["trajectory_dir"])
    if not traj_dir.exists():
        return []
    eps = []
    for p in traj_dir.glob(f"{run_name}_fold{fold}_ep*_{split}.npz"):
        m = _TRAJ_EP_RE.search(p.name)
        if m:
            eps.append(int(m.group(1)))
    return sorted(set(eps))


def _load_traj_npz(config: dict, run_name: str, fold: int, ep: int, split: str) -> dict | None:
    p = Path(config["trajectory_dir"]) / f"{run_name}_fold{fold}_ep{ep}_{split}.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def fig_traj_pca(config: dict, run_name: str, fold: int = 0, split: str = "val") -> None:
    from sklearn.decomposition import PCA
    fig_dir = Path(config["figures_dir"])
    eps = _list_traj_epochs(config, run_name, fold, split)
    if not eps:
        print(f"[skip] fig_traj_pca: no snapshots for {run_name}/{split}")
        return
    n = len(eps)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize("double", 0.30 * max(nrows, 1)))
    axs_flat = np.atleast_1d(axs).ravel()
    sc = None
    for ax, ep in zip(axs_flat, eps):
        data = _load_traj_npz(config, run_name, fold, ep, split)
        if data is None:
            ax.set_axis_off(); continue
        xy = PCA(n_components=2, random_state=0).fit_transform(data["features"])
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=data["ys"], cmap="viridis",
                        s=1, alpha=0.5, edgecolor="none")
        ax.set_title(f"ep {ep}")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axs_flat[n:]:
        ax.set_axis_off()
    if sc is not None:
        fig.colorbar(sc, ax=axs_flat.tolist(), shrink=0.6, label="true TempM (°C)")
    fig.suptitle(f"{run_name} fold{fold} {split} — PCA trajectory", fontsize=8, y=0.99)
    save_fig(fig, f"fig_traj_pca_{run_name}_fold{fold}_{split}", fig_dir)


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


def make_trajectory(config: dict, runs: tuple[str, ...] | None = None, fold: int = 0) -> None:
    from analysis.trajectory import DEFAULT_RUNS
    runs = runs if runs is not None else DEFAULT_RUNS
    for r in runs:
        for s in ("val", "test"):
            fig_traj_pca(config, r, fold=fold, split=s)
            fig_traj_cka(config, r, fold=fold, split=s)
        fig_traj_knn_mae(config, r, fold=fold)


# ============================================================
# Driver
# ============================================================

def make_all(config: dict, df: pd.DataFrame) -> None:
    fig_main_sweep(config, df)
    fig_training_curves(config)
    fig_pred_vs_true_scatter(config)
    fig_dist_and_errbar(config)
    fig_a1_sigma(config, df)
    fig_a2_reweight(config, df)
    fig_a3_bucket(config, df)
    fig_a4_momentum(config, df)
    fig_a5_start_smooth(config, df)
    fig_d1_skymask(config)
    fig_d4_linprobe(config, df)
    fig_e1_seeds(config, df)
    fig_f1_rate(config, df)
    fig_f2_range(config, df)
    fig_f3_drop(config, df)
    fig_f5_noise(config, df)
    for _split in ("val", "test"):
        fig_embed_temp(config, _split)
        fig_embed_cam(config, _split)
        fig_embed_knn(config, _split)
        fig_embed_per_bin(config, _split)
        fig_embed_cka(config, _split)


def main(config: dict | None = None):
    if config is None:
        from analysis._config import load_config
        config = load_config()
    from analysis.aggregate import build_dataframe
    df = build_dataframe(config)
    make_all(config, df)


if __name__ == "__main__":
    main()
