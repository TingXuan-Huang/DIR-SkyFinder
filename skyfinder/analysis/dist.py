"""Train/val/test distribution-shift analysis for DIR-SkyFinder.

This script is intentionally standalone: run it after splits are available, and
optionally after embeddings/trajectory extraction if you want feature-shift
diagnostics.

Default output:
    figures/dist/

Examples:
    python dist.py
    python dist.py --fold 0 --feature-run baseline_resnet50
    python dist.py --skip-features
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from skyfinder.analysis.config_loader import load_config


SPLITS = ("train", "val", "test")
SPLIT_COLORS = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}
BIN_LABELS = ("many", "medium", "few")
_EP_RE = re.compile(r"_ep(\d+)_")


def _apply_style():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    try:
        from skyfinder.analysis.style import apply_nature_style
        apply_nature_style()
    except Exception:
        pass
    return plt


def _skip_plot(name: str) -> None:
    print(f"[skip] {name}: matplotlib is not installed")


def _save(fig, name: str, out_dir: Path) -> None:
    try:
        from skyfinder.analysis.style import save_fig
        save_fig(fig, name, out_dir)
    except Exception:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"[saved] {out_dir / (name + '.pdf')}")


def _load_split_dfs(config: dict, fold: int) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(config["labels_path"])
    split_info = json.loads(Path(config["splits_path"]).read_text())[fold]
    out = {}
    for split in SPLITS:
        out[split] = df.iloc[split_info[split]].reset_index(drop=True).copy()
        out[split]["split"] = split
    return out


def _bin_edges(values: np.ndarray, bin_w: float) -> np.ndarray:
    lo = np.floor(float(np.nanmin(values)) / bin_w) * bin_w
    hi = np.ceil(float(np.nanmax(values)) / bin_w) * bin_w
    return np.arange(lo, hi + bin_w, bin_w)


def _split_temp_summary(split_dfs: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    rows = []
    for split, df in split_dfs.items():
        y = df["TempM"].to_numpy()
        rows.append({
            "split": split,
            "n_rows": int(len(df)),
            "n_cameras": int(df["CamId"].nunique()),
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
            "min": float(np.min(y)),
            "p05": float(np.percentile(y, 5)),
            "median": float(np.median(y)),
            "p95": float(np.percentile(y, 95)),
            "max": float(np.max(y)),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "dist_tempm_summary.csv", index=False)
    print(f"[saved] {out_dir / 'dist_tempm_summary.csv'}")
    return summary


def _pairwise_temp_shift(split_dfs: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    from scipy.stats import ks_2samp, wasserstein_distance

    train = split_dfs["train"]["TempM"].to_numpy()
    rows = []
    for split in ("val", "test"):
        y = split_dfs[split]["TempM"].to_numpy()
        ks = ks_2samp(train, y)
        rows.append({
            "reference": "train",
            "target": split,
            "wasserstein": float(wasserstein_distance(train, y)),
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "mean_delta_target_minus_train": float(np.mean(y) - np.mean(train)),
            "std_ratio_target_over_train": float(np.std(y) / max(np.std(train), 1e-9)),
        })
    shift = pd.DataFrame(rows)
    shift.to_csv(out_dir / "dist_tempm_pairwise_shift.csv", index=False)
    print(f"[saved] {out_dir / 'dist_tempm_pairwise_shift.csv'}")
    return shift


def plot_tempm(split_dfs: dict[str, pd.DataFrame], out_dir: Path, bin_w: float) -> None:
    plt = _apply_style()
    if plt is None:
        _skip_plot("dist_tempm_hist/dist_tempm_ecdf")
        return
    all_y = np.concatenate([d["TempM"].to_numpy() for d in split_dfs.values()])
    edges = _bin_edges(all_y, bin_w)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for split in SPLITS:
        ax.hist(split_dfs[split]["TempM"], bins=edges, density=True, histtype="step",
                linewidth=1.5, color=SPLIT_COLORS[split], label=split)
    ax.set_xlabel("TempM (C)")
    ax.set_ylabel("density")
    ax.set_title("TempM distribution by split")
    ax.legend(frameon=False)
    _save(fig, "dist_tempm_hist", out_dir)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for split in SPLITS:
        y = np.sort(split_dfs[split]["TempM"].to_numpy())
        ax.plot(y, np.arange(1, len(y) + 1) / len(y),
                color=SPLIT_COLORS[split], label=split)
    ax.set_xlabel("TempM (C)")
    ax.set_ylabel("empirical CDF")
    ax.set_title("TempM empirical CDF by split")
    ax.legend(frameon=False)
    _save(fig, "dist_tempm_ecdf", out_dir)


def plot_train_frequency_bins(split_dfs: dict[str, pd.DataFrame],
                              out_dir: Path, bin_w: float) -> pd.DataFrame:
    train_y = split_dfs["train"]["TempM"].to_numpy()
    all_y = np.concatenate([d["TempM"].to_numpy() for d in split_dfs.values()])
    edges = _bin_edges(all_y, bin_w)
    train_hist, _ = np.histogram(train_y, bins=edges)

    bin_kind = np.full(len(train_hist), "few", dtype=object)
    bin_kind[(train_hist >= 20) & (train_hist < 100)] = "medium"
    bin_kind[train_hist >= 100] = "many"

    rows = []
    for split, df in split_dfs.items():
        y = df["TempM"].to_numpy()
        idx = np.clip(np.digitize(y, edges) - 1, 0, len(train_hist) - 1)
        for kind in BIN_LABELS:
            n = int(np.sum(bin_kind[idx] == kind))
            rows.append({
                "split": split,
                "bin_kind": kind,
                "n": n,
                "fraction": n / max(len(y), 1),
            })
    coverage = pd.DataFrame(rows)
    coverage.to_csv(out_dir / "dist_train_frequency_bin_coverage.csv", index=False)
    print(f"[saved] {out_dir / 'dist_train_frequency_bin_coverage.csv'}")

    plt = _apply_style()
    if plt is None:
        _skip_plot("dist_train_frequency_bin_coverage")
        return coverage
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bottoms = np.zeros(len(SPLITS))
    x = np.arange(len(SPLITS))
    colors = {"many": "#4C72B0", "medium": "#DD8452", "few": "#C44E52"}
    for kind in BIN_LABELS:
        vals = [coverage[(coverage["split"] == s) & (coverage["bin_kind"] == kind)]["fraction"].iloc[0]
                for s in SPLITS]
        ax.bar(x, vals, bottom=bottoms, color=colors[kind], label=kind, edgecolor="none")
        bottoms += np.asarray(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(SPLITS)
    ax.set_ylabel("fraction of rows")
    ax.set_title("Rows by train-defined temperature-bin frequency")
    ax.legend(frameon=False)
    _save(fig, "dist_train_frequency_bin_coverage", out_dir)
    return coverage


def _camera_role(split_dfs: dict[str, pd.DataFrame]) -> dict:
    train_val = set(split_dfs["train"]["CamId"]).union(set(split_dfs["val"]["CamId"]))
    test = set(split_dfs["test"]["CamId"])
    return {cam: ("test" if cam in test else "train_val") for cam in train_val.union(test)}


def camera_stats(split_dfs: dict[str, pd.DataFrame], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_df = pd.concat(split_dfs.values(), ignore_index=True)
    roles = _camera_role(split_dfs)

    rows = []
    for (split, cam), g in all_df.groupby(["split", "CamId"]):
        rows.append({
            "split": split,
            "camera_role": roles.get(cam, split),
            "CamId": cam,
            "n": int(len(g)),
            "mean_temp": float(g["TempM"].mean()),
            "std_temp": float(g["TempM"].std(ddof=0)),
            "min_temp": float(g["TempM"].min()),
            "max_temp": float(g["TempM"].max()),
            "Latitude": float(g["Latitude"].iloc[0]) if "Latitude" in g else np.nan,
            "Longitude": float(g["Longitude"].iloc[0]) if "Longitude" in g else np.nan,
            "n_months": int(g["Month"].nunique()) if "Month" in g else 0,
            "n_hours": int(g["Hour"].nunique()) if "Hour" in g else 0,
        })
    per_split_cam = pd.DataFrame(rows)
    per_split_cam.to_csv(out_dir / "dist_per_split_camera_stats.csv", index=False)
    print(f"[saved] {out_dir / 'dist_per_split_camera_stats.csv'}")

    by_cam = all_df.groupby("CamId").agg(
        n=("TempM", "size"),
        mean_temp=("TempM", "mean"),
        std_temp=("TempM", "std"),
        min_temp=("TempM", "min"),
        max_temp=("TempM", "max"),
        Latitude=("Latitude", "first"),
        Longitude=("Longitude", "first"),
    ).reset_index()
    by_cam["camera_role"] = by_cam["CamId"].map(roles)
    by_cam.to_csv(out_dir / "dist_per_camera_stats.csv", index=False)
    print(f"[saved] {out_dir / 'dist_per_camera_stats.csv'}")
    return per_split_cam, by_cam


def plot_camera_stats(by_cam: pd.DataFrame, out_dir: Path) -> None:
    plt = _apply_style()
    if plt is None:
        _skip_plot("dist_per_camera_mean_std/dist_camera_geo_mean_temp")
        return
    ordered = by_cam.sort_values("mean_temp").reset_index(drop=True)
    colors = ordered["camera_role"].map({"train_val": "#4C72B0", "test": "#C44E52"}).fillna("#999999")

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.errorbar(np.arange(len(ordered)), ordered["mean_temp"], yerr=ordered["std_temp"],
                fmt="none", ecolor="#BBBBBB", linewidth=0.6, alpha=0.8, zorder=0)
    ax.scatter(np.arange(len(ordered)), ordered["mean_temp"], c=colors, s=16, edgecolor="none")
    ax.set_xlabel("camera sorted by mean TempM")
    ax.set_ylabel("mean TempM +/- std (C)")
    ax.set_title("Per-camera temperature statistics")
    _save(fig, "dist_per_camera_mean_std", out_dir)

    if {"Latitude", "Longitude"}.issubset(by_cam.columns):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for role, marker, label in [("train_val", "o", "train/val cameras"), ("test", "^", "test cameras")]:
            sub = by_cam[by_cam["camera_role"] == role]
            if sub.empty:
                continue
            sc = ax.scatter(sub["Longitude"], sub["Latitude"], c=sub["mean_temp"],
                            cmap="viridis", s=30, marker=marker, edgecolor="black",
                            linewidth=0.2, label=label)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title("Camera geography colored by mean TempM")
        ax.legend(frameon=False)
        fig.colorbar(sc, ax=ax, label="mean TempM (C)")
        _save(fig, "dist_camera_geo_mean_temp", out_dir)


def plot_month_hour(split_dfs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    plt = _apply_style()
    if plt is None:
        _skip_plot("dist_month_hour")
        return
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for split in SPLITS:
        df = split_dfs[split]
        if "Month" in df:
            m = df["Month"].value_counts(normalize=True).sort_index()
            axs[0].plot(m.index, m.values, marker="o", color=SPLIT_COLORS[split], label=split)
        if "Hour" in df:
            h = df["Hour"].value_counts(normalize=True).sort_index()
            axs[1].plot(h.index, h.values, marker="o", color=SPLIT_COLORS[split], label=split)
    axs[0].set_xlabel("month")
    axs[0].set_ylabel("fraction")
    axs[0].set_title("Month distribution")
    axs[1].set_xlabel("hour")
    axs[1].set_ylabel("fraction")
    axs[1].set_title("Hour distribution")
    axs[1].legend(frameon=False)
    _save(fig, "dist_month_hour", out_dir)


def _latest_traj_npz(traj_dir: Path, run_name: str, fold: int, split: str) -> Path | None:
    matches = []
    for p in traj_dir.glob(f"{run_name}_fold{fold}_ep*_{split}.npz"):
        m = _EP_RE.search(p.name)
        if m:
            matches.append((int(m.group(1)), p))
    if not matches:
        return None
    return sorted(matches)[-1][1]


def _feature_path(config: dict, run_name: str, fold: int, split: str) -> Path | None:
    embed = Path(config["embeddings_dir"]) / f"{run_name}_fold{fold}_{split}.npz"
    if embed.exists():
        return embed
    return _latest_traj_npz(Path(config["trajectory_dir"]), run_name, fold, split)


def _sample_features(data: dict, split: str, max_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n = len(data["ys"])
    idx = np.arange(n)
    if n > max_rows:
        idx = rng.choice(idx, size=max_rows, replace=False)
    df = pd.DataFrame({
        "split": split,
        "TempM": data["ys"][idx].astype(float),
        "CamId": data["cam_ids"][idx].astype(str),
    })
    return df, data["features"][idx].astype(np.float32)


def _sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X2 = np.sum(X * X, axis=1, keepdims=True)
    Y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    return np.maximum(X2 + Y2 - 2.0 * X @ Y.T, 0.0)


def _rbf_mmd2(X: np.ndarray, Y: np.ndarray) -> float:
    Z = np.vstack([X, Y])
    sample = Z[np.random.RandomState(0).choice(len(Z), size=min(len(Z), 800), replace=False)]
    d = _sq_dists(sample, sample)
    med = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    gamma = 1.0 / max(med, 1e-9)
    Kxx = np.exp(-gamma * _sq_dists(X, X))
    Kyy = np.exp(-gamma * _sq_dists(Y, Y))
    Kxy = np.exp(-gamma * _sq_dists(X, Y))
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def feature_shift(config: dict, out_dir: Path, run_name: str, fold: int,
                  max_rows: int, max_mmd_rows: int, seed: int) -> dict:
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        print(f"[skip] feature shift: {exc}")
        return {"skipped": True, "reason": str(exc), "feature_run": run_name}

    paths = {split: _feature_path(config, run_name, fold, split) for split in SPLITS}
    missing = [s for s, p in paths.items() if p is None]
    if missing:
        print(f"[skip] feature shift: missing {missing} npz files for {run_name}_fold{fold}")
        return {"skipped": True, "missing_splits": missing, "feature_run": run_name}

    frames = []
    feats = {}
    sources = {}
    for split, path in paths.items():
        data = dict(np.load(path, allow_pickle=True))
        frame, X = _sample_features(data, split, max_rows=max_rows, seed=seed)
        frames.append(frame)
        feats[split] = X
        sources[split] = str(path)
    meta = pd.concat(frames, ignore_index=True)
    X_all = np.vstack([feats[s] for s in SPLITS])

    scaler = StandardScaler().fit(feats["train"])
    Z_all = scaler.transform(X_all)
    Z = {s: scaler.transform(feats[s]) for s in SPLITS}

    pca = PCA(n_components=2, random_state=seed).fit_transform(Z_all)
    meta["pc1"] = pca[:, 0]
    meta["pc2"] = pca[:, 1]
    meta.to_csv(out_dir / "dist_feature_pca_points.csv", index=False)
    print(f"[saved] {out_dir / 'dist_feature_pca_points.csv'}")

    plt = _apply_style()
    if plt is None:
        _skip_plot("dist_feature_pca_split/dist_feature_pca_tempm")
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for split in SPLITS:
            sub = meta[meta["split"] == split]
            ax.scatter(sub["pc1"], sub["pc2"], s=3, alpha=0.45,
                       color=SPLIT_COLORS[split], label=split, edgecolor="none")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Feature PCA by split ({run_name}, fold {fold})")
        ax.legend(frameon=False)
        _save(fig, "dist_feature_pca_split", out_dir)

        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        sc = ax.scatter(meta["pc1"], meta["pc2"], c=meta["TempM"], cmap="viridis",
                        s=3, alpha=0.5, edgecolor="none")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Feature PCA colored by TempM ({run_name}, fold {fold})")
        fig.colorbar(sc, ax=ax, label="TempM (C)")
        _save(fig, "dist_feature_pca_tempm", out_dir)

    nn = NearestNeighbors(n_neighbors=1).fit(Z["train"])
    pair_rows = []
    nn_rows = []
    for split in ("val", "test"):
        dist = nn.kneighbors(Z[split], return_distance=True)[0].ravel()
        nn_rows.extend({"split": split, "nearest_train_distance": float(x)} for x in dist)
        pair_rows.append({
            "target": split,
            "nearest_train_distance_mean": float(dist.mean()),
            "nearest_train_distance_median": float(np.median(dist)),
            "nearest_train_distance_p95": float(np.percentile(dist, 95)),
        })

    nn_df = pd.DataFrame(nn_rows)
    nn_df.to_csv(out_dir / "dist_feature_nearest_train_distances.csv", index=False)
    print(f"[saved] {out_dir / 'dist_feature_nearest_train_distances.csv'}")

    if plt is None:
        _skip_plot("dist_feature_nearest_train_distance")
    else:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        vals = [nn_df[nn_df["split"] == s]["nearest_train_distance"] for s in ("val", "test")]
        ax.boxplot(vals, labels=["val", "test"], showfliers=False)
        ax.set_ylabel("nearest train-feature distance")
        ax.set_title("Feature distance to train distribution")
        _save(fig, "dist_feature_nearest_train_distance", out_dir)

    rng = np.random.RandomState(seed)
    for split in ("val", "test"):
        n = min(len(Z["train"]), len(Z[split]), max_rows)
        tr_idx = rng.choice(len(Z["train"]), size=n, replace=False)
        tg_idx = rng.choice(len(Z[split]), size=n, replace=False)
        X_domain = np.vstack([Z["train"][tr_idx], Z[split][tg_idx]])
        y_domain = np.array([0] * n + [1] * n)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_domain, y_domain, test_size=0.35, stratify=y_domain, random_state=seed,
        )
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        pred = prob >= 0.5

        m = min(n, max_mmd_rows)
        a = Z["train"][rng.choice(len(Z["train"]), size=m, replace=False)]
        b = Z[split][rng.choice(len(Z[split]), size=m, replace=False)]
        pair_rows[-1 if split == "test" else 0].update({
            "domain_balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
            "domain_auc": float(roc_auc_score(y_te, prob)),
            "mmd2_rbf": _rbf_mmd2(a, b),
            "mean_feature_l2_distance": float(np.linalg.norm(Z["train"].mean(0) - Z[split].mean(0))),
        })

    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(out_dir / "dist_feature_pairwise_shift.csv", index=False)
    print(f"[saved] {out_dir / 'dist_feature_pairwise_shift.csv'}")

    return {
        "skipped": False,
        "feature_run": run_name,
        "fold": fold,
        "sources": sources,
        "n_sampled": {s: int(len(feats[s])) for s in SPLITS},
        "pairwise_to_train": pairwise.to_dict(orient="records"),
    }


def run(args) -> None:
    config = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_dfs = _load_split_dfs(config, args.fold)

    label_summary = _split_temp_summary(split_dfs, out_dir)
    temp_shift = _pairwise_temp_shift(split_dfs, out_dir)
    plot_tempm(split_dfs, out_dir, args.bin_width)
    bin_coverage = plot_train_frequency_bins(split_dfs, out_dir, args.bin_width)
    _per_split_cam, by_cam = camera_stats(split_dfs, out_dir)
    plot_camera_stats(by_cam, out_dir)
    plot_month_hour(split_dfs, out_dir)

    feature_result = {"skipped": True, "reason": "disabled"}
    if not args.skip_features:
        feature_result = feature_shift(
            config=config,
            out_dir=out_dir,
            run_name=args.feature_run,
            fold=args.fold,
            max_rows=args.max_feature_rows,
            max_mmd_rows=args.max_mmd_rows,
            seed=args.seed,
        )

    report = {
        "fold": args.fold,
        "out_dir": str(out_dir),
        "bin_width": args.bin_width,
        "label_summary": label_summary.to_dict(orient="records"),
        "tempm_pairwise_shift": temp_shift.to_dict(orient="records"),
        "train_frequency_bin_coverage": bin_coverage.to_dict(orient="records"),
        "feature_shift": feature_result,
    }
    (out_dir / "dist_summary.json").write_text(json.dumps(report, indent=2))
    print(f"[saved] {out_dir / 'dist_summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("analysis_config.yaml"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("figures/dist"))
    ap.add_argument("--bin-width", type=float, default=2.0)
    ap.add_argument("--feature-run", type=str, default="baseline_resnet50",
                    help="run name used for feature-shift npz files")
    ap.add_argument("--max-feature-rows", type=int, default=5000,
                    help="max rows sampled per split for PCA/domain/NN analysis")
    ap.add_argument("--max-mmd-rows", type=int, default=1000,
                    help="max rows per split for RBF MMD")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-features", action="store_true",
                    help="only run label/metadata/camera distribution analysis")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
