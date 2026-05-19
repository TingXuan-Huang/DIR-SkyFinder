"""Extract ResNet-50 penultimate features (2048-d) for trained checkpoints.

Two forward passes per checkpoint by default: one over the fold-0 val set
(within-distribution: same cameras as train, held-out frames) and one over the
test set (LOCO: held-out cameras). Features come from the output of
`model.avgpool` (post-flatten, pre-`fc`) -- captured via forward hook so the
same code handles both vanilla ResNet and FDS-wrapped variants.

For each (run, split) we save a `.npz` with:
  features  (N, 2048) float32
  preds     (N,)       float32   -- model output
  ys        (N,)       float32   -- true TempM
  cam_ids   (N,)       object    -- CamId strings

Output dir: `<embeddings_dir>/<run_name>_fold<k>_<split>.npz`.

All paths come from `config` (loaded from `analysis_config.yaml`). Only
`EVAL_TF` and `get_device` are imported from `dir_skyfinder.baseline` (code).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dir_skyfinder.baseline import EVAL_TF, _resolve_load_path, get_device

# Re-use the FDS-aware model reconstruction we already have for D1.
from analysis.d1 import _build_model_from_results


DEFAULT_RUNS = (
    "baseline_resnet50",
    "lds_resnet50",
    "fds_resnet50",
    "lds_fds_resnet50",
)


class _EmbedDataset(Dataset):
    """Like SkyFinderDataset but yields (image, temp, cam_id) and takes img_dir explicitly."""

    def __init__(self, df: pd.DataFrame, transform, img_dir: Path):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_dir = Path(img_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.img_dir / str(row["CamId"]) / row["Filename"]).convert("RGB")
        return self.transform(img), float(row["TempM"]), str(row["CamId"])


def _find_penultimate_module(model: torch.nn.Module) -> tuple[torch.nn.Module, str]:
    """Return (module, hook_kind) where hook_kind is 'post' (capture output) or
    'pre' (capture input).

    Supported model shapes:
      FDSModel(any arch)    -> hook backbone OUTPUT (FDSModel routes its input through
                               backbone(x) which already strips the head).
      Vanilla ResNet-50     -> hook avgpool OUTPUT (B, 2048, 1, 1); flatten in the hook.
      Vanilla ViT-B/16      -> hook heads INPUT (768-d CLS token before the final
                               Linear). torchvision's ViT has no avgpool; the
                               penultimate feature is the input to `model.heads`,
                               so we use `register_forward_pre_hook`.
    """
    if hasattr(model, "backbone"):
        return model.backbone, "post"
    if hasattr(model, "avgpool"):
        return model.avgpool, "post"
    if hasattr(model, "heads"):
        return model.heads, "pre"
    raise ValueError(f"don't know how to find penultimate module in {type(model).__name__}")


def extract_one(config: dict, run_name: str, fold: int, out_dir: Path | str,
                split: str = "val",
                ckpt_override: Path | None = None,
                epoch_tag: str | int | None = None,
                subsample: int | None = None,
                batch_size: int = 32, num_workers: int = 2) -> Path | None:
    """Extract features for one checkpoint on one split ('train', 'val', or 'test').

    Paths come from `config`:
      - results_dir   : where to find `<run>_fold{fold}.json`
      - labels_path   : labels CSV
      - splits_path   : LOCO splits JSON
      - img_dir       : root for `<CamId>/<Filename>` lookups

    `ckpt_override`: load weights from this `.pt` instead of the JSON's `checkpoint` field.
    `epoch_tag`: filename suffix (used by trajectory.py for `_ep{N}`).
    `subsample`: if int, random-subsample that many rows (seed=0). Useful for train.
    """
    results_dir = Path(config["results_dir"])
    labels_path = Path(config["labels_path"])
    splits_path = Path(config["splits_path"])
    img_dir     = Path(config["img_dir"])

    results_path = _resolve_load_path(f"{run_name}_fold{fold}", ".json", results_dir)
    if results_path is None:
        print(f"[skip] no results json for {run_name}_fold{fold} under {results_dir}")
        return None

    model, _cfg = _build_model_from_results(config, results_path, ckpt_override=ckpt_override)
    device = get_device()
    model.to(device).eval()

    target, hook_kind = _find_penultimate_module(model)
    bag: list[torch.Tensor] = []
    if hook_kind == "post":
        # Capture OUTPUT of `target` (e.g. avgpool, FDS backbone).
        def hook(_module, _inp, out):
            bag.append(out.detach().flatten(1).cpu())
        handle = target.register_forward_hook(hook)
    else:  # "pre" — capture INPUT to `target` (e.g. ViT.heads input = CLS token)
        def hook(_module, args):
            bag.append(args[0].detach().flatten(1).cpu())
        handle = target.register_forward_pre_hook(hook)

    df = pd.read_csv(labels_path)
    fold_info = json.loads(splits_path.read_text())[fold]
    if split not in fold_info or not fold_info[split]:
        handle.remove()
        print(f"[skip] fold {fold} has no '{split}' split")
        return None
    split_df = df.iloc[fold_info[split]].reset_index(drop=True)
    if subsample is not None and len(split_df) > subsample:
        split_df = split_df.sample(n=subsample, random_state=0).reset_index(drop=True)
    loader = DataLoader(_EmbedDataset(split_df, EVAL_TF, img_dir=img_dir),
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
    stem = f"{run_name}_fold{fold}"
    if epoch_tag is not None:
        stem = f"{stem}_ep{epoch_tag}"
    out_path = out_dir / f"{stem}_{split}.npz"
    np.savez_compressed(out_path,
                        features=features, preds=preds, ys=ys, cam_ids=cam_ids)
    print(f"[saved] {out_path}  features={features.shape}")
    return out_path


def run_embeddings(config: dict, out_dir: Path | str | None = None, fold: int = 0,
                   runs: tuple[str, ...] = DEFAULT_RUNS,
                   which_splits: tuple[str, ...] = ("val", "test"),
                   batch_size: int = 32, num_workers: int = 2) -> list[Path]:
    """Extract features for each (run, split). Default: both val and test."""
    out_dir = Path(out_dir) if out_dir is not None else Path(config["embeddings_dir"])
    paths: list[Path] = []
    for split in which_splits:
        for name in runs:
            p = extract_one(config, name, fold, out_dir, split=split,
                            batch_size=batch_size, num_workers=num_workers)
            if p is not None:
                paths.append(p)
    return paths
