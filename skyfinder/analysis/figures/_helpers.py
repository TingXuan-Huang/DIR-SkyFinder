"""Shared constants and small helper functions for all figure subpackages.

This module holds:
  - Display-convention constants (KIND_ORDER, KIND_LABEL, BIN_ORDER)
  - Default run lists (DEFAULT_HEADLINE_RUNS, DEFAULT_EMBED_RUNS) — overrideable per call
  - JSON / npz / preds loading helpers
  - per_bin_mae_by_edges for figure-internal per-bin error
  - _linear_cka for CKA heatmaps

Public `fig_*` functions live in sibling files (main_sweep, curves, ablations,
embeddings, trajectory).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from skyfinder.analysis.style import apply_nature_style

apply_nature_style()


# ============================================================
# Display constants
# ============================================================

KIND_ORDER = ["baseline", "lds", "fds", "lds_fds"]
KIND_LABEL = {"baseline": "baseline", "lds": "LDS", "fds": "FDS", "lds_fds": "LDS+FDS"}
BIN_ORDER = ["overall", "many", "medium", "few"]

# Default run lists. Override at call time by passing `runs=...`.
# These tuples are (run_name_prefix, label_in_KIND_LABEL).
DEFAULT_HEADLINE_RUNS = [
    ("baseline_resnet50", "baseline"),
    ("lds_resnet50",      "lds"),
    ("fds_resnet50",      "fds"),
    ("lds_fds_resnet50",  "lds_fds"),
]
DEFAULT_EMBED_RUNS = list(DEFAULT_HEADLINE_RUNS)   # same 4-run default

TRAJ_EP_RE = re.compile(r"_ep(\d+)_")


# ============================================================
# DataFrame filters
# ============================================================

def _resnet(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["model"] == "resnet50"]


def _empty(df: pd.DataFrame, name: str) -> bool:
    if len(df) == 0:
        print(f"[skip] {name}: no matching rows")
        return True
    return False


# ============================================================
# JSON / npz loaders
# ============================================================

def _load_run_json(config: dict, name: str, fold: int = 0) -> dict | None:
    from skyfinder.training.checkpoint import find_artifact
    p = find_artifact(f"{name}_fold{fold}", ".json", Path(config["results_dir"]))
    if p is None:
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


def _ref_lines(config: dict, constant_kind: str = "per_cam_month_mean") -> dict[str, float]:
    """Return {label: mean test MAE} for the non-DL baselines (used as reference
    lines on the headline figures).

    Labels keep the "C1 / C2 / D1" prefixes because `docs/figure_report.md`
    references them; the keys read from `config` follow the new naming.
    """
    out: dict[str, float] = {}
    for tag, path_key, key in [
        ("C1 (per-cam-month)", "baselines_constant_path", constant_kind),
        ("C2 (metadata GBM)",  "baselines_metadata_path", None),
        ("D1 (sky mask)",      "skymask_path",            None),
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


def _load_npz(dir_path: Path, stem: str) -> dict | None:
    """Generic .npz loader. Used by embeddings and trajectory figures."""
    p = Path(dir_path) / f"{stem}.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def _load_embedding(config: dict, run_name: str, fold: int = 0, split: str = "val") -> dict | None:
    return _load_npz(Path(config["embeddings_dir"]), f"{run_name}_fold{fold}_{split}")


def _load_traj_npz(config: dict, run_name: str, fold: int, ep: int, split: str) -> dict | None:
    return _load_npz(Path(config["trajectory_dir"]), f"{run_name}_fold{fold}_ep{ep}_{split}")


def _list_traj_epochs(config: dict, run_name: str, fold: int, split: str) -> list[int]:
    traj_dir = Path(config["trajectory_dir"])
    if not traj_dir.exists():
        return []
    eps = []
    for p in traj_dir.glob(f"{run_name}_fold{fold}_ep*_{split}.npz"):
        m = TRAJ_EP_RE.search(p.name)
        if m:
            eps.append(int(m.group(1)))
    return sorted(set(eps))


# ============================================================
# Math helpers
# ============================================================

def _project_2d(features: np.ndarray, method: str):
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=0).fit_transform(features)
    if method == "umap":
        try:
            import umap   # type: ignore
        except ImportError:
            print("[skip] UMAP requested but `umap-learn` not installed")
            return None
        return umap.UMAP(n_components=2, n_neighbors=15, random_state=0).fit_transform(features)
    raise ValueError(method)


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0); Y = Y - Y.mean(0)
    xy = float(np.linalg.norm(X.T @ Y, ord="fro") ** 2)
    xx = float(np.linalg.norm(X.T @ X, ord="fro"))
    yy = float(np.linalg.norm(Y.T @ Y, ord="fro"))
    return xy / (xx * yy + 1e-12)


def per_bin_feature_cosine_sim(features: np.ndarray, ys: np.ndarray,
                                 edges: np.ndarray) -> np.ndarray:
    """Return (n_bins, n_bins) cosine-similarity matrix over per-bin mean features."""
    idx = np.clip(np.digitize(ys, edges) - 1, 0, len(edges) - 2)
    mean_feat = np.stack([
        features[idx == k].mean(axis=0) if (idx == k).any()
        else np.zeros(features.shape[1])
        for k in range(len(edges) - 1)
    ])
    norm = np.linalg.norm(mean_feat, axis=1, keepdims=True) + 1e-9
    unit = mean_feat / norm
    return unit @ unit.T
