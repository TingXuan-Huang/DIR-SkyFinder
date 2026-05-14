"""D1: sky-mask inference. NO training -- just zero out non-sky pixels at eval time.

For each of the 4 checkpoint configs (baseline / lds / fds / lds_fds) on fold 0,
evaluate the val loader twice: (a) raw images, (b) images x sky-mask. Reports
per-bin MAE for both, plus the delta.

Mask convention:
  - One PNG per camera at `data/masks/<CamId>.png` (configurable via --masks).
  - Pixels > 127 are "sky" (kept); pixels <= 127 are "ground" (zeroed).
  - Mask is resized to 224x224 with nearest-neighbour to match model input.
  - Applied AFTER ImageNet normalization (non-sky pixels become exactly 0, which
    is roughly the normalized-image mean -- safer than pre-norm masking which
    would push those pixels to ~-2 sigma, a regime the model has never seen).

Run:
    python analysis.py d1
        [--masks data/masks]
        [--fold 0]
        [--out ablations/results/d1_skymask.json]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dir_skyfinder.baseline import (
    EVAL_TF, IMG_DIR, LABELS, SPLITS, RESULTS_DIR,
    FDSModel, get_device, make_model, per_bin_mae,
)
from dir_skyfinder.fds import FDS
from dir_skyfinder.lds import MIN_TEMP


# 4 configs to evaluate. Names match the headline runs in `results/`.
DEFAULT_RUNS = (
    "baseline_resnet50",
    "lds_resnet50",
    "fds_resnet50",
    "lds_fds_resnet50",
)


# ============================================================
# Mask loading + dataset
# ============================================================

def _load_mask(mask_dir: Path, cam_id, hw: int = 224) -> torch.Tensor:
    """Load mask for one camera. Returns float32 tensor (hw, hw) with 0/1 values."""
    path = mask_dir / f"{cam_id}.png"
    if not path.exists():
        raise FileNotFoundError(
            f"sky mask not found: {path}\n"
            f"hint: D1 expects one PNG per CamId under {mask_dir}/. "
            f"Download from SkyFinder (cs.valdosta.edu/~rpmihail/skyfinder/) "
            f"or point --masks at the directory where they live."
        )
    m = Image.open(path).convert("L").resize((hw, hw), Image.NEAREST)
    arr = (np.array(m) > 127).astype(np.float32)
    return torch.from_numpy(arr)


class MaskedSkyDataset(Dataset):
    """Yields (image, temp) with optional per-camera mask applied AFTER normalization."""

    def __init__(self, df: pd.DataFrame, transform, mask_dir: Path | None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mask_dir = mask_dir
        self._mask_cache: dict = {}

    def __len__(self):
        return len(self.df)

    def _get_mask(self, cam_id) -> torch.Tensor:
        if cam_id not in self._mask_cache:
            self._mask_cache[cam_id] = _load_mask(self.mask_dir, cam_id)
        return self._mask_cache[cam_id]

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(IMG_DIR / str(row["CamId"]) / row["Filename"]).convert("RGB")
        x = self.transform(img)
        if self.mask_dir is not None:
            x = x * self._get_mask(row["CamId"]).unsqueeze(0)   # broadcast over channels
        y = torch.tensor(row["TempM"], dtype=torch.float32)
        return x, y


# ============================================================
# Model loading
# ============================================================

def _build_model_from_results(results_path: Path) -> tuple[torch.nn.Module, dict]:
    """Reconstruct the trained model from its results JSON. Returns (model, config_dict)."""
    d = json.loads(results_path.read_text())
    cfg = d["config"]
    vanilla = make_model(cfg["model"], freeze_backbone=False)
    if cfg.get("use_fds", False):
        # Mirror baseline.run_baseline's FDS construction (see baseline.py:425-439).
        feature_dim = 2048 if cfg["model"] == "resnet50" else 768
        bin_width = float(cfg.get("bin_width", 1.0))
        bucket_num = int(np.ceil((55.0 - MIN_TEMP) / bin_width))
        fds = FDS(
            feature_dim=feature_dim,
            bucket_num=bucket_num,
            kernel=cfg.get("fds_kernel", "gaussian"),
            ks=cfg.get("fds_ks", 5),
            sigma=cfg.get("fds_sigma", 2.0),
            momentum=cfg.get("fds_momentum", 0.9),
            start_smooth=cfg.get("fds_start_smooth", 1),
        )
        model = FDSModel(vanilla, fds, bin_width=bin_width, min_temp=MIN_TEMP)
    else:
        model = vanilla

    # Load weights from the saved best-state path. baseline.save_checkpoint writes
    # <run_name>.pt next to the JSON; results['checkpoint'] is the absolute path.
    ckpt_path = d.get("checkpoint") or (results_path.parent / f"{d['run_name']}.pt")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, cfg


# ============================================================
# Eval
# ============================================================

def _evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    preds, ys = [], []
    model.to(device)
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            out = out.squeeze(-1) if out.ndim > 1 else out
            preds.append(out.cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def _eval_one_config(run_name: str, fold: int, mask_dir: Path,
                     batch_size: int, num_workers: int, device: str) -> dict:
    results_path = RESULTS_DIR / f"{run_name}_fold{fold}.json"
    if not results_path.exists():
        return {"run_name": run_name, "skipped": f"no results json at {results_path}"}

    model, cfg = _build_model_from_results(results_path)
    df = pd.read_csv(LABELS)
    splits = json.loads(SPLITS.read_text())
    val_df = df.iloc[splits[fold]["val"]].reset_index(drop=True)
    train_y = df.iloc[splits[fold]["train"]]["TempM"].to_numpy()

    raw_loader = DataLoader(
        MaskedSkyDataset(val_df, EVAL_TF, mask_dir=None),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    msk_loader = DataLoader(
        MaskedSkyDataset(val_df, EVAL_TF, mask_dir=mask_dir),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    raw_pred, ys = _evaluate(model, raw_loader, device)
    msk_pred, _  = _evaluate(model, msk_loader, device)

    raw_mae = per_bin_mae(ys, raw_pred, train_y)
    msk_mae = per_bin_mae(ys, msk_pred, train_y)
    delta = {k: (msk_mae[k] - raw_mae[k]) for k in raw_mae}

    return {
        "run_name": run_name,
        "config_kind": ("lds_fds" if cfg.get("use_lds") and cfg.get("use_fds")
                        else "lds" if cfg.get("use_lds")
                        else "fds" if cfg.get("use_fds")
                        else "baseline"),
        "fold": fold,
        "model": cfg.get("model"),
        "n_val": int(len(val_df)),
        "raw":    raw_mae,
        "masked": msk_mae,
        "delta":  delta,
        "raw_preds":    raw_pred.tolist(),
        "masked_preds": msk_pred.tolist(),
        "ys":           ys.tolist(),
    }


def run_d1(out_path: Path, mask_dir: Path, fold: int = 0,
           runs: tuple[str, ...] = DEFAULT_RUNS,
           batch_size: int = 32, num_workers: int = 2) -> dict:
    device = get_device()
    print(f"[device] {device}")

    results = {"per_fold": [], "mask_dir": str(mask_dir), "fold": fold}
    for name in runs:
        print(f"[d1] evaluating {name} fold={fold}")
        row = _eval_one_config(name, fold, mask_dir, batch_size, num_workers, device)
        results["per_fold"].append(row)
        if "skipped" in row:
            print(f"  -> SKIPPED ({row['skipped']})")
            continue
        print(f"  raw   : overall={row['raw']['overall']:.3f}  few={row['raw']['few']:.3f}")
        print(f"  masked: overall={row['masked']['overall']:.3f}  few={row['masked']['few']:.3f}")
        print(f"  delta : overall={row['delta']['overall']:+.3f}  few={row['delta']['few']:+.3f}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[saved] {out_path}")
    return results
