"""Extract ResNet-50 penultimate features (2048-d) for the 4 headline checkpoints.

One forward pass per checkpoint over the fold-0 val set. Features come from the
output of `model.avgpool` (post-flatten, pre-`fc`) -- captured via forward hook
so the same code handles both vanilla ResNet and FDS-wrapped variants (FDS lives
between avgpool and the head, so avgpool's output is the "raw" feature either way).

For each run we save a `.npz` with:
  features  (N, 2048) float32
  preds     (N,)       float32   -- model output
  ys        (N,)       float32   -- true TempM
  cam_ids   (N,)       object    -- CamId strings

Output dir: `ablations/results/embeddings/<run_name>_fold<k>.npz`.

Run:
    python analysis.py embeddings [--fold 0]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dir_skyfinder.baseline import EVAL_TF, IMG_DIR, LABELS, SPLITS, get_device

# Re-use the same model-loading helper used by D1 so we don't duplicate the
# FDS construction recipe.
from analysis.d1 import _build_model_from_results


DEFAULT_RUNS = (
    "baseline_resnet50",
    "lds_resnet50",
    "fds_resnet50",
    "lds_fds_resnet50",
)


class _EmbedDataset(Dataset):
    """Like SkyFinderDataset but yields (image, temp, cam_id)."""

    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(IMG_DIR / str(row["CamId"]) / row["Filename"]).convert("RGB")
        return self.transform(img), float(row["TempM"]), str(row["CamId"])


def _find_penultimate_module(model: torch.nn.Module) -> torch.nn.Module:
    """Return the module whose output is the penultimate feature.

    For vanilla ResNet-50: `model.avgpool` (output shape (B, 2048, 1, 1) which we
    flatten in the hook).
    For FDSModel: `model.backbone.avgpool` -- FDS sits between this and the head.
    """
    if hasattr(model, "avgpool"):
        return model.avgpool
    if hasattr(model, "backbone") and hasattr(model.backbone, "avgpool"):
        return model.backbone.avgpool
    raise ValueError(f"don't know how to find penultimate module in {type(model).__name__}")


def extract_one(run_name: str, fold: int, out_dir: Path,
                batch_size: int = 32, num_workers: int = 2) -> Path | None:
    from dir_skyfinder.baseline import RESULTS_DIR
    results_path = RESULTS_DIR / f"{run_name}_fold{fold}.json"
    if not results_path.exists():
        print(f"[skip] no results json at {results_path}")
        return None

    model, _cfg = _build_model_from_results(results_path)
    device = get_device()
    model.to(device).eval()

    # Hook on avgpool to capture penultimate features.
    target = _find_penultimate_module(model)
    bag: list[torch.Tensor] = []
    def hook(_module, _inp, out):
        bag.append(out.detach().flatten(1).cpu())
    handle = target.register_forward_hook(hook)

    df = pd.read_csv(LABELS)
    splits = json.loads(SPLITS.read_text())
    val_df = df.iloc[splits[fold]["val"]].reset_index(drop=True)
    loader = DataLoader(_EmbedDataset(val_df, EVAL_TF),
                        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    preds_all, ys_all, cams_all = [], [], []
    with torch.no_grad():
        for x, y, cam in loader:
            out = model(x.to(device))
            out = out.squeeze(-1) if out.ndim > 1 else out
            preds_all.append(out.cpu().numpy())
            ys_all.append(np.asarray(y, dtype=np.float32))
            cams_all.extend(cam)
    handle.remove()

    features = torch.cat(bag, dim=0).numpy().astype(np.float32)
    preds    = np.concatenate(preds_all).astype(np.float32)
    ys       = np.concatenate(ys_all).astype(np.float32)
    cam_ids  = np.array(cams_all, dtype=object)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}_fold{fold}.npz"
    np.savez_compressed(out_path,
                        features=features, preds=preds, ys=ys, cam_ids=cam_ids)
    print(f"[saved] {out_path}  features={features.shape}")
    return out_path


def run_embeddings(out_dir: Path, fold: int = 0,
                   runs: tuple[str, ...] = DEFAULT_RUNS,
                   batch_size: int = 32, num_workers: int = 2) -> list[Path]:
    paths = []
    for name in runs:
        p = extract_one(name, fold, out_dir, batch_size, num_workers)
        if p is not None:
            paths.append(p)
    return paths
