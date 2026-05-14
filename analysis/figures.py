"""All Nature-style figures for the DIR-SkyFinder runs and ablations.

Each `fig_*` function:
  - takes `df` (aggregate DataFrame from `analysis/aggregate.py`),
  - filters to its rows,
  - returns nothing -- saves to `figures/<name>.{pdf,png}` via `style.save_fig`.

If the relevant rows don't exist yet (e.g., sweep hasn't run), the function
prints `[skip]` and returns -- so the pipeline degrades gracefully on partial
data instead of crashing.

ViT is filtered out at figure time per the project decision; aggregate.py still
ingests ViT JSONs so the CSV stays complete.
"""
from __future__ import annotations

import json
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

RESULTS_DIR = Path("results")
ABL_RESULTS = Path("ablations/results")
FIG_DIR = Path("figures")
EMBED_DIR = ABL_RESULTS / "embeddings"


# ============================================================
# helpers
# ============================================================

def _resnet(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["model"] == "resnet50"]


def _load_run_json(name: str, fold: int = 0) -> dict | None:
    p = RESULTS_DIR / f"{name}_fold{fold}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _train_y(fold: int = 0) -> np.ndarray:
    from dir_skyfinder.baseline import LABELS, SPLITS
    df = pd.read_csv(LABELS)
    splits = json.loads(SPLITS.read_text())
    return df.iloc[splits[fold]["train"]]["TempM"].to_numpy()


def _empty(df: pd.DataFrame, name: str) -> bool:
    if len(df) == 0:
        print(f"[skip] {name}: no matching rows")
        return True
    return False


def _ref_lines(c1_kind: str = "per_cam_month_mean") -> dict[str, float]:
    """Return {label: mean test MAE} for C1 (best predictor), C2, D1 (baseline raw)."""
    out: dict[str, float] = {}
    for tag, path, key in [
        ("C1 (per-cam-month)", ABL_RESULTS / "c1_constants.json", c1_kind),
        ("C2 (metadata GBM)",  ABL_RESULTS / "c2_metadata_only.json", None),
        ("D1 (sky mask)",      ABL_RESULTS / "d1_skymask.json",     None),
    ]:
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

def fig_main_sweep(df: pd.DataFrame) -> None:
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
        means = [rows[f"val_{b}"].mean()  for b in BIN_ORDER]
        stds  = [rows[f"val_{b}"].std(ddof=0) for b in BIN_ORDER]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=2,
               color=PALETTE[kind], label=KIND_LABEL[kind], edgecolor="none")
    # Reference lines (overall column only -- they don't have per-bin breakdowns)
    refs = _ref_lines()
    for j, (tag, val) in enumerate(refs.items()):
        ax.axhline(val, color=REF_COLORS[list(REF_COLORS)[j % 3]],
                   linestyle="--", linewidth=0.5, label=tag, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(BIN_ORDER)
    ax.set_ylabel("val MAE (°C)")
    ax.set_xlabel("bin (DIR many/medium/few)")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    save_fig(fig, "fig_main_sweep", FIG_DIR)


# ============================================================
# 1b. Training curves -- train + val MAE per epoch, 4 headline configs, fold 0
# ============================================================

def fig_training_curves(fold: int = 0,
                        runs: tuple[tuple[str, str], ...] = (
                            ("baseline_resnet50", "baseline"),
                            ("lds_resnet50",      "lds"),
                            ("fds_resnet50",      "fds"),
                            ("lds_fds_resnet50",  "lds_fds"),
                        )) -> None:
    loaded = [(label, _load_run_json(name, fold)) for name, label in runs]
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
    save_fig(fig, f"fig_training_curves_fold{fold}", FIG_DIR)


# ============================================================
# 2. Pred-vs-true scatter (baseline vs LDS+FDS, fold 0)
# ============================================================

def fig_pred_vs_true_scatter() -> None:
    pairs = [("baseline_resnet50", "baseline"), ("lds_fds_resnet50", "LDS+FDS")]
    runs = [(_load_run_json(name), label) for name, label in pairs]
    if not any(r for r, _ in runs):
        print("[skip] fig_pred_vs_true_scatter: no headline JSONs found")
        return
    fig, axs = plt.subplots(1, 2, figsize=figsize("double", 0.45), sharex=True, sharey=True)
    for ax, (data, label) in zip(axs, runs):
        if data is None:
            ax.set_title(f"{label} (missing)"); continue
        y_true = np.array(data["val_ys"])
        y_pred = np.array(data["val_preds"])
        ax.scatter(y_true, y_pred, s=2, alpha=0.3, color=PALETTE["lds_fds" if label == "LDS+FDS" else "baseline"], edgecolor="none")
        lo = float(min(y_true.min(), y_pred.min())) - 1
        hi = float(max(y_true.max(), y_pred.max())) + 1
        ax.plot([lo, hi], [lo, hi], "k-", linewidth=0.5)
        ax.set_title(label)
        ax.set_xlabel("true TempM (°C)")
    axs[0].set_ylabel("predicted TempM (°C)")
    save_fig(fig, "fig_pred_vs_true_scatter", FIG_DIR)


# ============================================================
# 3. Dist (true vs pred) + per-bin error  -- the canonical DIR figure
# ============================================================

def fig_dist_and_errbar() -> None:
    pairs = [("baseline_resnet50", "baseline"), ("lds_fds_resnet50", "LDS+FDS")]
    for name, label in pairs:
        data = _load_run_json(name)
        if data is None:
            print(f"[skip] fig_dist_and_errbar: {name} not found")
            continue
        train_y = _train_y(data["config"].get("fold", 0))
        y_true = np.array(data["val_ys"])
        y_pred = np.array(data["val_preds"])
        bin_w = 1.0
        lo = float(min(train_y.min(), y_true.min(), y_pred.min()))
        hi = float(max(train_y.max(), y_true.max(), y_pred.max()))
        edges = np.arange(np.floor(lo / bin_w) * bin_w, np.ceil(hi / bin_w) * bin_w + bin_w, bin_w)
        centers = (edges[:-1] + edges[1:]) / 2

        # Per-bin MAE on val (true bin classification)
        idx = np.clip(np.digitize(y_true, edges) - 1, 0, len(edges) - 2)
        err = np.abs(y_true - y_pred)
        per_bin_mae = np.array([err[idx == k].mean() if (idx == k).any() else np.nan
                                for k in range(len(centers))])

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=figsize("single", 0.95),
            gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08}, sharex=True,
        )
        # top: histograms (train true vs val pred)
        ax_top.hist(train_y, bins=edges, color="#888888", alpha=0.5, label="train true")
        ax_top.hist(y_pred,  bins=edges, color=PALETTE["lds_fds" if label == "LDS+FDS" else "baseline"],
                    alpha=0.65, label="val pred", histtype="step", linewidth=1.0)
        ax_top.set_ylabel("count")
        ax_top.legend(loc="upper right", frameon=False)
        ax_top.set_title(label)

        # bottom: per-bin MAE on val
        ax_bot.bar(centers, per_bin_mae, width=bin_w * 0.9,
                   color=PALETTE["lds_fds" if label == "LDS+FDS" else "baseline"], edgecolor="none")
        ax_bot.set_ylabel("val MAE (°C)")
        ax_bot.set_xlabel("TempM bin (°C)")
        save_fig(fig, f"fig_dist_and_errbar_{label.replace('+', '_').replace(' ', '_').lower()}", FIG_DIR)


# ============================================================
# 4-8. A-family (lds_sigma / reweight / bin_width / fds_momentum / start_smooth)
# ============================================================

def _a_line(df: pd.DataFrame, group: str, x_col: str, x_label: str, name: str) -> None:
    sub = _resnet(df[df["group"] == group]).sort_values(x_col)
    if _empty(sub, name):
        return
    fig, ax = plt.subplots(figsize=figsize("single"))
    ax.plot(sub[x_col], sub["val_overall"], marker="o", color=PALETTE["lds_fds"], label="overall")
    ax.plot(sub[x_col], sub["val_few"],     marker="^", color=PALETTE["fds"],      label="few")
    ax.set_xlabel(x_label); ax.set_ylabel("val MAE (°C)")
    ax.legend(frameon=False)
    save_fig(fig, name, FIG_DIR)


def fig_a1_sigma(df):       _a_line(df, "A1", "lds_sigma",        "lds_sigma (°C)", "fig_a1_sigma")
def fig_a3_bucket(df):      _a_line(df, "A3", "bin_width",        "bin_width (°C)", "fig_a3_bucket")
def fig_a4_momentum(df):    _a_line(df, "A4", "fds_momentum",     "fds_momentum",   "fig_a4_momentum")
def fig_a5_start_smooth(df):_a_line(df, "A5", "fds_start_smooth", "fds_start_smooth (epoch)", "fig_a5_start_smooth")


def fig_a2_reweight(df: pd.DataFrame) -> None:
    sub = _resnet(df[df["group"] == "A2"])
    if _empty(sub, "fig_a2_reweight"):
        return
    fig, ax = plt.subplots(figsize=figsize("single"))
    schemes = sub["lds_reweight"].fillna("none").tolist()
    x = np.arange(len(schemes))
    width = 0.4
    ax.bar(x - width/2, sub["val_overall"], width, color=PALETTE["lds_fds"], label="overall")
    ax.bar(x + width/2, sub["val_few"],     width, color=PALETTE["fds"],     label="few")
    ax.set_xticks(x); ax.set_xticklabels(schemes)
    ax.set_xlabel("lds_reweight"); ax.set_ylabel("val MAE (°C)")
    ax.legend(frameon=False)
    save_fig(fig, "fig_a2_reweight", FIG_DIR)


# ============================================================
# 9. D1 sky-mask raw vs masked, 4 configs
# ============================================================

def fig_d1_skymask() -> None:
    p = ABL_RESULTS / "d1_skymask.json"
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
    ax.set_ylabel("val MAE (°C)"); ax.legend(frameon=False)
    save_fig(fig, "fig_d1_skymask", FIG_DIR)


# ============================================================
# 10. D4 linear-probe vs full fine-tune
# ============================================================

def fig_d4_linprobe(df: pd.DataFrame) -> None:
    main = _resnet(df[(df["group"] == "main") & (df["fold"] == 0)
                      & df["config_kind"].isin(["baseline", "lds_fds"])])
    d4 = _resnet(df[df["group"] == "D4"])
    if _empty(d4, "fig_d4_linprobe"):
        return
    fig, ax = plt.subplots(figsize=figsize("single"))
    x = np.arange(2); width = 0.35
    full = [main[main["config_kind"] == k]["val_overall"].mean() for k in ["baseline", "lds_fds"]]
    lin  = [d4[d4["config_kind"] == k]["val_overall"].mean()     for k in ["baseline", "lds_fds"]]
    ax.bar(x - width/2, full, width, color=PALETTE["baseline"], label="full fine-tune")
    ax.bar(x + width/2, lin,  width, color=PALETTE["lds_fds"],  label="linear probe")
    ax.set_xticks(x); ax.set_xticklabels(["baseline", "LDS+FDS"])
    ax.set_ylabel("val MAE (°C)"); ax.legend(frameon=False)
    save_fig(fig, "fig_d4_linprobe", FIG_DIR)


# ============================================================
# 11. E1 seed variance
# ============================================================

def fig_e1_seeds(df: pd.DataFrame) -> None:
    sub = _resnet(df[df["group"] == "E1"])
    if _empty(sub, "fig_e1_seeds"):
        return
    fig, ax = plt.subplots(figsize=figsize("single"))
    metrics = ["overall", "few"]
    width = 0.35; x = np.arange(len(metrics))
    means = [sub[f"val_{m}"].mean() for m in metrics]
    stds  = [sub[f"val_{m}"].std(ddof=0) for m in metrics]
    ax.bar(x, means, width, yerr=stds, capsize=3, color=PALETTE["lds_fds"], edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("val MAE (°C)"); ax.set_title(f"seed variance ({len(sub)} seeds)")
    save_fig(fig, "fig_e1_seeds", FIG_DIR)


# ============================================================
# 12-15. F-family
# ============================================================

def _f_line(df: pd.DataFrame, group: str, x_label: str, y_col: str,
            name: str, ref_lookup: tuple[str, str] | None = None) -> None:
    """Generic F-family line: x = corruption_param, y = y_col, lines per config_kind."""
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
    # Reference at corruption_param=0 from main sweep.
    if ref_lookup is not None:
        main = _resnet(df[(df["group"] == "main") & (df["fold"] == 0)])
        for kind in ["baseline", "lds_fds"]:
            v = main[main["config_kind"] == kind][y_col]
            if not v.empty:
                ax.scatter([0], [v.mean()], color=PALETTE[kind], marker="*", s=30, zorder=5)
    ax.set_xlabel(x_label); ax.set_ylabel(f"val {y_col.replace('val_', '')} MAE (°C)")
    ax.legend(frameon=False)
    save_fig(fig, name, FIG_DIR)


def fig_f1_rate(df):  _f_line(df, "F1", "corruption rate",       "val_overall", "fig_f1_rate", ref_lookup=("main", "fold0"))
def fig_f3_drop(df):  _f_line(df, "F3", "rare-bin drop prob",    "val_few",     "fig_f3_drop")
def fig_f5_noise(df): _f_line(df, "F5", "Gaussian noise sigma (°C)", "val_overall", "fig_f5_noise")


def fig_f2_range(df: pd.DataFrame) -> None:
    """Grouped bar: 3 ranges x {baseline, LDS+FDS, (LDS, FDS once ab2 lands)} x {impute, drop}."""
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
            ys = [sub[(sub["corruption_param"] == r) & (sub["config_kind"] == kind)]["val_overall"].mean()
                  for r in ranges]
            ax.bar(x + (i - (len(kinds_present)-1)/2) * width, ys, width,
                   color=PALETTE[kind], label=KIND_LABEL[kind], edgecolor="none")
        ax.set_xticks(x); ax.set_xticklabels([f"{r:.0f}±2.5" for r in ranges])
        ax.set_title(title); ax.set_xlabel("dropped range center (°C)")
    ax_i.set_ylabel("val MAE (°C)"); ax_i.legend(frameon=False, loc="best")
    save_fig(fig, "fig_f2_range", FIG_DIR)


# ============================================================
# 16-20. Embedding figures
# ============================================================

def _load_embedding(run_name: str, fold: int = 0) -> dict | None:
    p = EMBED_DIR / f"{run_name}_fold{fold}.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


EMBED_RUNS = [
    ("baseline_resnet50", "baseline"),
    ("lds_resnet50",      "lds"),
    ("fds_resnet50",      "fds"),
    ("lds_fds_resnet50",  "lds_fds"),
]


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


def _fig_embed_scatter(color_by: str, name: str) -> None:
    """2x4 grid: rows = {PCA, UMAP}, cols = 4 configs. Points colored by `color_by`."""
    loaded = [(label, _load_embedding(run)) for run, label in EMBED_RUNS]
    if not any(d is not None for _, d in loaded):
        print(f"[skip] {name}: no embedding npz files")
        return
    fig, axs = plt.subplots(2, 4, figsize=figsize("double", 0.55))
    for col, (label, data) in enumerate(loaded):
        for row, method in enumerate(["pca", "umap"]):
            ax = axs[row, col]
            if data is None:
                ax.set_axis_off(); ax.set_title(f"{label} (missing)"); continue
            xy = _project_2d(data["features"], method)
            if xy is None:
                ax.set_axis_off(); continue
            color_vals = data["ys"] if color_by == "temp" else pd.Categorical(data["cam_ids"]).codes
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=color_vals, cmap="viridis" if color_by == "temp" else "tab20",
                            s=1, alpha=0.5, edgecolor="none")
            if row == 0:
                ax.set_title(KIND_LABEL[label])
            if col == 0:
                ax.set_ylabel(method.upper())
            ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("true TempM (°C)" if color_by == "temp" else "camera id")
    save_fig(fig, name, FIG_DIR)


def fig_embed_temp(): _fig_embed_scatter("temp", "fig_embed_temp")
def fig_embed_cam():  _fig_embed_scatter("cam",  "fig_embed_cam")


def fig_embed_knn() -> None:
    """k-NN regression MAE on features (5-fold CV over val), per config."""
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsRegressor
    rows = []
    for run, label in EMBED_RUNS:
        data = _load_embedding(run)
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
        print("[skip] fig_embed_knn: no embeddings"); return
    fig, ax = plt.subplots(figsize=figsize("single"))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], yerr=[r[2] for r in rows], capsize=3,
           color=[PALETTE[r[0]] for r in rows], edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels([KIND_LABEL[r[0]] for r in rows])
    ax.set_ylabel("5-fold k-NN MAE on features (°C)")
    save_fig(fig, "fig_embed_knn", FIG_DIR)


def fig_embed_per_bin() -> None:
    """Per-bin mean-feature cosine similarity matrices, one panel per config."""
    bin_w = 2.0
    panels = [(label, _load_embedding(run)) for run, label in EMBED_RUNS]
    panels = [(l, d) for l, d in panels if d is not None]
    if not panels:
        print("[skip] fig_embed_per_bin: no embeddings"); return
    # Common bin edges across all configs.
    all_ys = np.concatenate([d["ys"] for _, d in panels])
    lo, hi = float(all_ys.min()), float(all_ys.max())
    edges = np.arange(np.floor(lo / bin_w) * bin_w, np.ceil(hi / bin_w) * bin_w + bin_w, bin_w)

    fig, axs = plt.subplots(1, len(panels), figsize=figsize("double", 0.30), sharey=True)
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
    fig.colorbar(im, ax=np.atleast_1d(axs).tolist(), shrink=0.6, label="cosine sim")
    save_fig(fig, "fig_embed_per_bin", FIG_DIR)


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear centered-kernel-alignment, Kornblith et al. 2019."""
    X = X - X.mean(0); Y = Y - Y.mean(0)
    xy = float(np.linalg.norm(X.T @ Y, ord="fro") ** 2)
    xx = float(np.linalg.norm(X.T @ X, ord="fro"))
    yy = float(np.linalg.norm(Y.T @ Y, ord="fro"))
    return xy / (xx * yy + 1e-12)


def fig_embed_cka() -> None:
    """Pairwise CKA between all 4 configs -- one 4x4 heatmap."""
    loaded = [(label, _load_embedding(run)) for run, label in EMBED_RUNS]
    loaded = [(l, d) for l, d in loaded if d is not None]
    if len(loaded) < 2:
        print("[skip] fig_embed_cka: need >=2 embedding files"); return
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
    save_fig(fig, "fig_embed_cka", FIG_DIR)


# ============================================================
# Driver
# ============================================================

def make_all(df: pd.DataFrame) -> None:
    fig_main_sweep(df)
    fig_training_curves()
    fig_pred_vs_true_scatter()
    fig_dist_and_errbar()
    fig_a1_sigma(df)
    fig_a2_reweight(df)
    fig_a3_bucket(df)
    fig_a4_momentum(df)
    fig_a5_start_smooth(df)
    fig_d1_skymask()
    fig_d4_linprobe(df)
    fig_e1_seeds(df)
    fig_f1_rate(df)
    fig_f2_range(df)
    fig_f3_drop(df)
    fig_f5_noise(df)
    fig_embed_temp()
    fig_embed_cam()
    fig_embed_knn()
    fig_embed_per_bin()
    fig_embed_cka()


def main():
    from analysis.aggregate import build_dataframe
    df = build_dataframe()
    make_all(df)


if __name__ == "__main__":
    main()
