"""Dataset, DataLoader builders, and image transforms for SkyFinder training.

Public API:
    NORM, TRAIN_TF, EVAL_TF  — torchvision transform compositions
    SkyFinderDataset         — (image, temp, weight) triples
    build_loaders            — train+val DataLoaders for cfg.fold, optionally with LDS weights
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import Config, IMG_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate the handful of partial JPEGs

# --- transforms ---
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(), NORM,
])
EVAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), NORM,
])


class SkyFinderDataset(Dataset):
    """Returns (image, temp, weight). Weight is 1.0 by default; supply `weights` for LDS."""

    def __init__(self, df: pd.DataFrame, transform, img_dir: Path | None = None,
                 weights: np.ndarray | None = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_dir = img_dir if img_dir is not None else IMG_DIR
        self.weights = weights  # numpy array aligned with df rows, or None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = self.img_dir / str(row["CamId"]) / row["Filename"]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(row["TempM"], dtype=torch.float32)
        w = torch.tensor(self.weights[i] if self.weights is not None else 1.0,
                         dtype=torch.float32)
        return x, y, w


def build_loaders(cfg: Config, *, train_weights=None):
    """Build train+val DataLoaders for cfg.fold.

    Returns (train_loader, val_loader, train_df, val_df).

    Every run parameter (fold, batch_size, subsets, seed, corruption, paths) comes from cfg.

    `train_weights`: optional numpy array aligned with the *unsampled* train rows
        (df.iloc[f["train"]]). If cfg.train_subset is set, weights are subset to match.
        (The trainer currently attaches LDS weights post-build via
        `train_loader.dataset.weights = ...` instead of passing them here.)
    """
    df = pd.read_csv(cfg.labels_path)
    splits = json.loads(cfg.splits_path.read_text())
    f = splits[cfg.fold]
    train_df = df.iloc[f["train"]].reset_index(drop=True)
    val_df = df.iloc[f["val"]].reset_index(drop=True)

    # F-family corruption (applied BEFORE subsetting so LDS bins reflect the
    # corrupted distribution).
    if cfg.corruption is not None:
        from skyfinder.analysis.corrupt_labels import corrupt_train_labels
        train_df = corrupt_train_labels(train_df, cfg.corruption, seed=cfg.seed)

    if cfg.train_subset:
        sampled = train_df.sample(cfg.train_subset, random_state=cfg.seed)
        if train_weights is not None:
            train_weights = train_weights[sampled.index.to_numpy()]
        train_df = sampled.reset_index(drop=True)
    if cfg.val_subset:
        val_df = val_df.sample(cfg.val_subset, random_state=cfg.seed).reset_index(drop=True)

    train_loader = DataLoader(
        SkyFinderDataset(train_df, TRAIN_TF, img_dir=cfg.img_dir, weights=train_weights),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        SkyFinderDataset(val_df, EVAL_TF, img_dir=cfg.img_dir),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )
    return train_loader, val_loader, train_df, val_df
