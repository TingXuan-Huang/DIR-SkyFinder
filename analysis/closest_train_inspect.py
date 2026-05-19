"""High-loss test-image inspector.

For each test camera in fold 0, pick the *median-residual* test image, then
show it alongside two training-set neighbours:

  (a) feature-NN: train image with highest cosine similarity in this model's
      trained embedding space.
  (b) geo-NN:    one image from the train camera with the closest (lat, lon)
      to the test camera (seeded random choice within that train cam).

One figure per model (4 figures total at fold 0). Rows = test cameras;
columns = [test image | feat-NN train | geo-NN train].

Inputs:
  - results/test_inference.json (and/or test_inference_vit.json) for residuals
  - <embeddings_dir>/<run>_fold{f}_test.npz  -- test features (per model)
  - <embeddings_dir>/<run>_fold{f}_train.npz -- train features (per model);
    auto-extracted via embeddings.extract_one(split="train") if missing.
  - data/labels_with_images.csv -- columns Filename, CamId, TempM, Month, Hour,
                                   Latitude, Longitude
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from analysis.style import apply_nature_style, save_fig

apply_nature_style()


DEFAULT_RUNS = (
    "baseline_resnet50", "lds_resnet50", "fds_resnet50", "lds_fds_resnet50",
)
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
# Helpers
# ============================================================

def _load_test_residuals(config: dict, run: str) -> dict | None:
    """Return {"preds": np.array, "ys": np.array} for one run from test_inference.json(s)."""
    candidates = [
        Path(config["test_inference_path"]),                              # default ResNet
        Path(config["test_inference_path"]).with_name("test_inference_vit.json"),
    ]
    for p in candidates:
        if not p.exists(): continue
        runs_dict = json.loads(p.read_text()).get("runs", {})
        for key, row in runs_dict.items():
            # Match either "<run>_fold0" or exact "<run>".
            if key == f"{run}_fold0" or key == run:
                if "test_preds" not in row or "test_ys" not in row:
                    print(f"  [skip] {p.name}: {key} has no test_preds/test_ys "
                          f"(rerun `python inference.py` without --no-preds)")
                    return None
                return {"preds": np.asarray(row["test_preds"], dtype=np.float32),
                        "ys":    np.asarray(row["test_ys"],    dtype=np.float32)}
    print(f"  [skip] no test_inference entry found for {run}")
    return None


def _ensure_embedding(config: dict, run: str, fold: int, split: str) -> dict | None:
    """Load <run>_fold{f}_{split}.npz; auto-extract if missing."""
    emb_dir = Path(config["embeddings_dir"])
    p = emb_dir / f"{run}_fold{fold}_{split}.npz"
    if not p.exists():
        print(f"  [extract] missing {p.name}; running embeddings.extract_one(split={split!r})")
        from analysis.embeddings import extract_one
        out = extract_one(config, run_name=run, fold=fold, out_dir=emb_dir,
                          split=split, subsample=None)  # full split; no subsample
        if out is None:
            return None
    if not p.exists():
        print(f"  [skip] extraction did not produce {p.name}"); return None
    return dict(np.load(p, allow_pickle=True))


def _cam_latlon(labels_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """One (lat, lon) per CamId (cameras are static, so any row gives the location)."""
    if "Latitude" not in labels_df.columns or "Longitude" not in labels_df.columns:
        raise RuntimeError("labels CSV missing Latitude/Longitude columns")
    g = labels_df.groupby("CamId")[["Latitude", "Longitude"]].first()
    return {str(c): (float(g.loc[c, "Latitude"]), float(g.loc[c, "Longitude"])) for c in g.index}


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """km between two (lat, lon) points. Plenty accurate at this scale."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def _median_residual_pick(test_df: pd.DataFrame, preds: np.ndarray, ys: np.ndarray,
                          cam_id: str) -> int | None:
    """Index (into test_df) of the median-|residual| image for this camera."""
    mask = (test_df["CamId"].astype(str) == str(cam_id)).to_numpy()
    if not mask.any(): return None
    idxs = np.where(mask)[0]
    residuals = np.abs(preds[idxs] - ys[idxs])
    order = np.argsort(residuals)
    return int(idxs[order[len(order) // 2]])  # median by rank


def _read_thumb(img_dir: Path, cam: str, filename: str, size: int = 192) -> np.ndarray | None:
    try:
        return np.asarray(Image.open(img_dir / cam / filename).convert("RGB").resize((size, size)))
    except Exception as e:
        print(f"  [warn] failed to read image {cam}/{filename}: {e}")
        return None


# ============================================================
# Main per-run pipeline
# ============================================================

def _process_one_run(config: dict, run: str, fold: int, seed: int, fig_dir: Path) -> None:
    """Emit one figure for one model."""
    res = _load_test_residuals(config, run)
    if res is None:
        return

    test_emb = _ensure_embedding(config, run, fold, "test")
    if test_emb is None: return
    train_emb = _ensure_embedding(config, run, fold, "train")
    if train_emb is None: return

    labels_df = pd.read_csv(config["labels_path"])
    splits = json.loads(Path(config["splits_path"]).read_text())
    test_df  = labels_df.iloc[splits[fold]["test"]].reset_index(drop=True)
    train_df = labels_df.iloc[splits[fold]["train"]].reset_index(drop=True)

    # Sanity: test inference and embeddings must align in length with the df.
    n_test = len(test_df)
    if len(res["preds"]) != n_test or len(test_emb["features"]) != n_test:
        print(f"[skip] {run}: length mismatch — "
              f"test_df={n_test} preds={len(res['preds'])} emb={len(test_emb['features'])}")
        return

    test_cams_in_fold = list(splits[fold].get("test_cams", []))
    if not test_cams_in_fold:
        test_cams_in_fold = sorted(test_df["CamId"].astype(str).unique())

    cam_to_latlon = _cam_latlon(labels_df)

    # L2-normalize features for cosine similarity.
    def _norm(x):  # (N, D) -> (N, D)
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(n, 1e-12, None)
    train_feat = _norm(train_emb["features"].astype(np.float32))
    test_feat  = _norm(test_emb["features"].astype(np.float32))

    rng = np.random.default_rng(seed)
    img_dir = Path(config["img_dir"])

    rows = []  # one tuple per test cam
    for cam in test_cams_in_fold:
        cam = str(cam)  # splits JSON stores ints; CSV CamId can be either — normalize
        pick = _median_residual_pick(test_df, res["preds"], res["ys"], cam)
        if pick is None:
            print(f"  [skip] cam {cam}: no test rows"); continue
        test_row = test_df.iloc[pick]
        test_y, test_pred = float(res["ys"][pick]), float(res["preds"][pick])
        resid = test_pred - test_y

        # Feature-NN (cosine similarity over train, no same-cam mask needed -- test cams are LOCO)
        sims = train_feat @ test_feat[pick]
        nn_idx = int(np.argmax(sims))
        nn_row = train_df.iloc[nn_idx]

        # Geo-NN: closest train *camera* in haversine, then random image from it
        if cam not in cam_to_latlon:
            print(f"  [warn] no lat/lon for cam {cam}; geo-NN row skipped"); geo_row = None
        else:
            tlat, tlon = cam_to_latlon[cam]
            train_cams = sorted(train_df["CamId"].astype(str).unique())
            geo_dists = [(c, _haversine(tlat, tlon, *cam_to_latlon[c])) for c in train_cams
                         if c in cam_to_latlon]
            if not geo_dists:
                geo_row = None
            else:
                geo_cam, geo_km = min(geo_dists, key=lambda t: t[1])
                cand = train_df[train_df["CamId"].astype(str) == geo_cam]
                geo_row = cand.iloc[int(rng.integers(0, len(cand)))]
                geo_row = geo_row.copy(); geo_row["_geo_km"] = geo_km

        rows.append({
            "test_cam": cam, "test_row": test_row, "test_y": test_y,
            "test_pred": test_pred, "resid": resid,
            "nn_row": nn_row, "nn_sim": float(sims[nn_idx]),
            "geo_row": geo_row,
        })

    if not rows:
        print(f"[skip] {run}: no usable test cams"); return

    # ---- Plot ----
    n = len(rows)
    fig, axs = plt.subplots(n, 3, figsize=(7.2, 2.3 * n), squeeze=False)
    fig.suptitle(f"Closest-train inspector — {KIND_LABEL.get(run, run)} (fold {fold}, "
                 f"median-residual per test cam)", fontsize=8, y=0.995)
    col_titles = ["test image (held-out cam)", "feature-NN in train", "geo-NN in train (lat/lon)"]
    for c, t in enumerate(col_titles):
        axs[0, c].set_title(t, fontsize=7)

    for r, row in enumerate(rows):
        # col 0: test
        ax = axs[r, 0]
        thumb = _read_thumb(img_dir, str(row["test_row"]["CamId"]), row["test_row"]["Filename"])
        if thumb is not None: ax.imshow(thumb)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"cam {row['test_cam']}", fontsize=6)
        ax.set_xlabel(f"TempM={row['test_y']:+.1f}  pred={row['test_pred']:+.1f}  "
                      f"resid={row['resid']:+.1f} °C", fontsize=6)

        # col 1: feat-NN
        ax = axs[r, 1]
        nn = row["nn_row"]
        thumb = _read_thumb(img_dir, str(nn["CamId"]), nn["Filename"])
        if thumb is not None: ax.imshow(thumb)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"cam {nn['CamId']}  TempM={float(nn['TempM']):+.1f}  "
                      f"cos={row['nn_sim']:.2f}", fontsize=6)

        # col 2: geo-NN
        ax = axs[r, 2]
        if row["geo_row"] is None:
            ax.set_axis_off(); ax.text(0.5, 0.5, "(no geo-NN)",
                                       ha="center", va="center", transform=ax.transAxes,
                                       fontsize=6)
        else:
            geo = row["geo_row"]
            thumb = _read_thumb(img_dir, str(geo["CamId"]), geo["Filename"])
            if thumb is not None: ax.imshow(thumb)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"cam {geo['CamId']}  TempM={float(geo['TempM']):+.1f}  "
                          f"{float(geo['_geo_km']):.0f} km", fontsize=6)

    plt.subplots_adjust(top=0.96, bottom=0.02, left=0.06, right=0.99, hspace=0.55, wspace=0.05)
    save_fig(fig, f"fig_closest_train_{run}_fold{fold}", fig_dir)


def run_closest_train_inspect(config: dict, runs: tuple[str, ...] = DEFAULT_RUNS,
                              fold: int = 0, seed: int = 0) -> None:
    """One figure per run; safely skips runs with missing artifacts."""
    fig_dir = Path(config["figures_dir"])
    for run in runs:
        print(f"[closest-train] {run} fold={fold}")
        _process_one_run(config, run, fold, seed, fig_dir)
