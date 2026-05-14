"""Baseline image -> TempM regressor (ResNet-50 / ViT-B/16).

Designed for experimental, callable use:
  Script:    python dir_skyfinder/baseline.py --model resnet50 --epochs 20
  Notebook:  from dir_skyfinder.baseline import run_baseline
             results = run_baseline(model="resnet50", epochs=20)

Each building block is a plain function — override or replace as needed:
  build_loaders, make_model, train_one_epoch, evaluate, per_bin_mae, save_results

Paths (DATA, LABELS, SPLITS, IMG_DIR, RESULTS_DIR) are module-level constants.
On Colab, reassign before calling run_baseline:
  import dir_skyfinder.baseline as b
  b.DATA = Path("/content/drive/MyDrive/DIR_Code/data")
  b.LABELS, b.SPLITS, b.IMG_DIR = b.DATA/"labels_with_images.csv", b.DATA/"splits/loco_5fold.json", b.DATA/"images"
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (ResNet50_Weights, ViT_B_16_Weights, resnet50,
                                vit_b_16)
from tqdm import tqdm

from dir_skyfinder.fds import FDS
from dir_skyfinder.lds import MIN_TEMP, compute_lds_weights, weighted_l1_loss

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate the handful of partial JPEGs

# --- paths (reassign in a notebook if your data lives elsewhere) ---
PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"
LABELS = DATA / "labels_with_images.csv"
SPLITS = DATA / "splits" / "loco_5fold.json"
IMG_DIR = DATA / "images"
RESULTS_DIR = PROJ / "results"

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


@dataclass
class Config:
    """Baseline training config. Pass to run_baseline(cfg=...) or via kwargs.

    All LDS/FDS fields default off so existing baseline calls are unchanged.
    Toggle one or both via `use_lds=True`, `use_fds=True`.
    """
    model: str = "resnet50"          # "resnet50" or "vit_b_16"
    fold: int = 0                    # 0 through 4 for the current 5-fold split
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    num_workers: int = 2
    train_subset: int | None = None  # cap train rows for smoke tests
    val_subset: int | None = None
    seed: int = 0
    run_name: str | None = None      # auto-generated from model/fold/epochs/time if None

    # LDS (loss-side reweighting; no architecture change)
    use_lds: bool = False
    lds_kernel: str = "gaussian"     # "gaussian" | "triang" | "laplace"
    lds_ks: int = 5                  # odd
    lds_sigma: float = 2.0
    lds_reweight: str = "sqrt_inv"   # "none" | "sqrt_inv" | "inverse"

    # FDS (architecture-side feature calibration)
    use_fds: bool = False
    fds_kernel: str = "gaussian"
    fds_ks: int = 5
    fds_sigma: float = 2.0
    fds_momentum: float = 0.9
    fds_start_smooth: int = 1        # epoch to begin applying calibration

    # Bucket scheme — applies to both LDS and FDS when active
    bin_width: float = 1.0           # C per bucket; default 1 C matches DIR paper convention

    # Ablation hooks (see ablation.py)
    freeze_backbone: bool = False    # D4: linear probe — train only the regression head
    corruption: dict | None = None   # F-family: train-label corruption (see corrupt_train_labels)


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


class FDSModel(nn.Module):
    """Wraps a vanilla ResNet-50 or ViT-B/16 with an FDS calibration module
    between its backbone and the regression head.

    Forward signature:
        net(x)              # eval mode; returns predictions (B,)
        net(x, labels=y)    # train mode; returns (preds, raw_features) for FDS update
    """

    def __init__(self, vanilla_model: nn.Module, fds: FDS,
                 bin_width: float = 1.0, min_temp: float = MIN_TEMP):
        super().__init__()
        self.fds = fds
        self.bin_width = bin_width
        self.min_temp = min_temp
        self.current_epoch = 0

        # Split off the final FC for both architectures.
        if hasattr(vanilla_model, "fc") and isinstance(vanilla_model.fc, nn.Linear):
            self.head = vanilla_model.fc
            vanilla_model.fc = nn.Identity()
        elif hasattr(vanilla_model, "heads"):
            self.head = vanilla_model.heads.head
            vanilla_model.heads.head = nn.Identity()
        else:
            raise ValueError(f"Don't know how to FDS-wrap model {type(vanilla_model)}")
        self.backbone = vanilla_model

    def _bucketize(self, temps: torch.Tensor) -> torch.Tensor:
        return ((temps - self.min_temp) / self.bin_width).long().clamp(0, self.fds.bucket_num - 1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = feats.flatten(1)
        smoothed = feats
        if self.training and labels is not None and self.current_epoch >= self.fds.start_smooth:
            smoothed = self.fds.smooth(feats, self._bucketize(labels), self.current_epoch)
        pred = self.head(smoothed).squeeze(-1)
        if self.training:
            return pred, feats   # raw feats for the FDS running-stats update
        return pred


# --- building blocks ---
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_model(name: str, freeze_backbone: bool = False) -> nn.Module:
    """Build a fresh model with our 1-output regression head.

    `freeze_backbone` (D4 linear probe): freezes all pretrained params, then
    swaps in a new head. The new `nn.Linear` is constructed AFTER the freeze,
    so it stays trainable by default — no name-matching needed downstream.
    """
    if name == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for p in m.parameters():
                p.requires_grad_(False)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m
    if name == "vit_b_16":
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in m.parameters():
                p.requires_grad_(False)
        m.heads.head = nn.Linear(m.heads.head.in_features, 1)
        return m
    raise ValueError(f"unknown model: {name}")


def build_loaders(fold=0, batch_size=32, num_workers=2,
                  train_subset=None, val_subset=None, seed=0,
                  train_weights=None, corruption_cfg: dict | None = None):
    """Returns (train_loader, val_loader, train_df, val_df).

    `train_weights`: optional numpy array aligned with the *unsampled* train_df rows
    (i.e. with df.iloc[f["train"]]). If `train_subset` is used, we subset weights
    consistently. NOTE: when `corruption_cfg` is also set with `drop_mode="drop"`,
    row counts change so passing `train_weights` here is unsupported — attach
    weights to `train_loader.dataset.weights` after construction instead.

    `corruption_cfg`: optional F-family corruption applied to train labels
    (val is never touched). See `ablation.corrupt_train_labels` for the schema.
    """
    df = pd.read_csv(LABELS)
    splits = json.loads(SPLITS.read_text())
    f = splits[fold]
    train_df = df.iloc[f["train"]].reset_index(drop=True)
    val_df = df.iloc[f["val"]].reset_index(drop=True)

    # F-family corruption (applied BEFORE subsetting so LDS bins, computed by
    # the caller from the returned train_df, reflect the corrupted distribution).
    if corruption_cfg is not None:
        from ablation import corrupt_train_labels
        train_df = corrupt_train_labels(train_df, corruption_cfg, seed=seed)

    if train_subset:
        sampled = train_df.sample(train_subset, random_state=seed)
        if train_weights is not None:
            train_weights = train_weights[sampled.index.to_numpy()]
        train_df = sampled.reset_index(drop=True)
    if val_subset:
        val_df = val_df.sample(val_subset, random_state=seed).reset_index(drop=True)

    train_loader = DataLoader(
        SkyFinderDataset(train_df, TRAIN_TF, weights=train_weights),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        SkyFinderDataset(val_df, EVAL_TF),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader, train_df, val_df


def train_one_epoch(model, loader, optimizer, device, fds_state: dict | None = None):
    """Returns mean training loss across the epoch.

    If `fds_state` is a dict with empty lists `"feats"` and `"labels"`, treats
    the model as an FDSModel: passes labels into forward and accumulates the
    raw features for the FDS running-stats update at end of epoch.
    """
    model.train()
    tot_err = tot_n = 0
    is_fds = fds_state is not None
    for x, y, w in tqdm(loader, desc="train", leave=False):
        x, y, w = x.to(device), y.to(device), w.to(device)
        if is_fds:
            pred, feats = model(x, labels=y)
            fds_state["feats"].append(feats.detach().cpu())
            fds_state["labels"].append(y.detach().cpu())
        else:
            pred = model(x).squeeze(-1)
        loss = weighted_l1_loss(pred, y, w)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        tot_err += loss.item() * len(y)
        tot_n += len(y)
    return tot_err / tot_n


def evaluate(model, loader, device):
    """Returns (preds, ys) as 1-D numpy arrays. Works with any loader that yields (x, y, *)."""
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            out = model(x.to(device))
            out = out.squeeze(-1) if out.ndim > 1 else out
            preds.append(out.cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def per_bin_mae(y_true, y_pred, train_y, bin_w=2.0):
    """MAE in `bin_w`-C bins, classified by training-set frequency (DIR many/medium/few)."""
    lo = min(train_y.min(), y_true.min())
    hi = max(train_y.max(), y_true.max())
    edges = np.arange(np.floor(lo / bin_w) * bin_w,
                      np.ceil(hi / bin_w) * bin_w + bin_w, bin_w)
    train_hist, _ = np.histogram(train_y, bins=edges)
    idx = np.clip(np.digitize(y_true, edges) - 1, 0, len(edges) - 2)
    err = np.abs(y_true - y_pred)
    out = {"overall": float(err.mean())}
    for name, n_lo, n_hi in [("many", 100, np.inf), ("medium", 20, 100), ("few", 0, 20)]:
        sel = np.isin(idx, np.where((train_hist >= n_lo) & (train_hist < n_hi))[0])
        out[name] = float(err[sel].mean()) if sel.any() else float("nan")
    return out


def save_results(results: dict, results_dir: Path | None = None) -> Path:
    """Write results dict to <results_dir>/<run_name>.json. Returns the path."""
    out_dir = results_dir if results_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{results['run_name']}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"[saved] {path}")
    return path


def load_results(path) -> dict:
    return json.loads(Path(path).read_text())


def save_checkpoint(state_dict, run_name: str, results_dir: Path | None = None) -> Path:
    """Save a model state_dict to <results_dir>/<run_name>.pt."""
    out_dir = results_dir if results_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name}.pt"
    torch.save(state_dict, path)
    print(f"[saved] {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_checkpoint(run_name_or_path, results_dir: Path | None = None, map_location="cpu"):
    """Load a saved state_dict. Pass a run_name (looks in results_dir) or a full path.

    Example:
        sd = load_checkpoint("baseline_resnet50_fold0")
        net = make_model("resnet50")
        net.load_state_dict(sd)
        net.to(get_device()).eval()
    """
    p = Path(run_name_or_path)
    if not p.exists():
        out_dir = results_dir if results_dir is not None else RESULTS_DIR
        p = out_dir / (run_name_or_path if str(run_name_or_path).endswith(".pt")
                       else f"{run_name_or_path}.pt")
    return torch.load(p, map_location=map_location, weights_only=True)


def save_full_checkpoint(state: dict, run_name: str, results_dir: Path | None = None) -> Path:
    """Save full training state to <results_dir>/<run_name>_last.pt, atomically.

    `state` is a dict produced by run_baseline that holds everything needed to
    resume training: model state_dict, optimizer state, scheduler state, current
    epoch, history, best-val state, and RNG state.

    Atomic via .part-then-rename, so a preempted job won't leave a corrupt file.
    """
    out_dir = results_dir if results_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{run_name}_last.pt"
    tmp = final.with_name(final.name + ".part")
    torch.save(state, tmp)
    tmp.rename(final)
    return final


def load_full_checkpoint(run_name: str, results_dir: Path | None = None) -> dict | None:
    """Load full training state from <results_dir>/<run_name>_last.pt, or return
    None if no resume file exists. `weights_only=False` because the dict
    contains pickled Python objects (optimizer / scheduler state).
    """
    out_dir = results_dir if results_dir is not None else RESULTS_DIR
    path = out_dir / f"{run_name}_last.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


# --- main entrypoint ---
def run_baseline(cfg: Config | None = None, save: bool = True, **kwargs) -> dict:
    """Train + evaluate + (optionally) save. Returns results dict.

    Three calling patterns, all equivalent for plain runs:
        run_baseline(model="vit_b_16", epochs=2)        # kwargs -> Config
        run_baseline(cfg=Config(model="vit_b_16"))      # pre-built Config
        run_baseline(cfg=my_cfg, epochs=5)              # Config + per-call override
    """
    if cfg is None:
        cfg = Config(**kwargs)
    elif kwargs:
        cfg = replace(cfg, **kwargs)

    if cfg.run_name is None:
        cfg = replace(cfg, run_name=f"{cfg.model}_fold{cfg.fold}_ep{cfg.epochs}_"
                                    f"{datetime.now():%Y%m%d_%H%M%S}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = get_device()
    print(f"[env] device={device}  run={cfg.run_name}")
    print(f"[cfg] {asdict(cfg)}")

    # --- Build loaders first (this applies F-family corruption to train labels
    # if cfg.corruption is set), then compute LDS weights from the resulting
    # train_df so bin frequencies reflect the post-corruption distribution. ---
    train_loader, val_loader, train_df, val_df = build_loaders(
        fold=cfg.fold, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        train_subset=cfg.train_subset, val_subset=cfg.val_subset, seed=cfg.seed,
        corruption_cfg=cfg.corruption,
    )
    print(f"[data] train={len(train_df):,}  val={len(val_df):,}")

    if cfg.use_lds:
        train_weights = compute_lds_weights(
            train_df["TempM"].to_numpy(),
            bin_width=cfg.bin_width,
            reweight=cfg.lds_reweight,
            lds_kernel=cfg.lds_kernel,
            lds_ks=cfg.lds_ks,
            lds_sigma=cfg.lds_sigma,
        )
        train_loader.dataset.weights = train_weights  # attach post-construction
        print(f"[lds] weights computed: min={train_weights.min():.3f}  "
              f"max={train_weights.max():.3f}  mean={train_weights.mean():.3f}")

    # --- Model: vanilla or FDS-wrapped ---
    # When cfg.freeze_backbone is true, make_model freezes the pretrained
    # weights and then attaches a fresh head (which stays trainable).
    vanilla = make_model(cfg.model, freeze_backbone=cfg.freeze_backbone)
    if cfg.use_fds:
        feature_dim = 2048 if cfg.model == "resnet50" else 768
        bucket_num = int(np.ceil((55.0 - MIN_TEMP) / cfg.bin_width))  # MIN_TEMP from lds
        fds_module = FDS(
            feature_dim=feature_dim,
            bucket_num=bucket_num,
            kernel=cfg.fds_kernel, ks=cfg.fds_ks, sigma=cfg.fds_sigma,
            momentum=cfg.fds_momentum, start_smooth=cfg.fds_start_smooth,
        )
        net = FDSModel(vanilla, fds_module,
                       bin_width=cfg.bin_width, min_temp=MIN_TEMP).to(device)
        print(f"[fds] wrapped {cfg.model} with FDS (feature_dim={feature_dim}, buckets={bucket_num})")
    else:
        net = vanilla.to(device)

    trainable = [p for p in net.parameters() if p.requires_grad]
    if cfg.freeze_backbone:
        n_train = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in net.parameters())
        print(f"[freeze] backbone frozen: {n_train:,}/{n_total:,} "
              f"trainable ({100*n_train/max(n_total,1):.2f}%)")
    opt = torch.optim.Adam(trainable, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    history = []
    train_y = train_df["TempM"].to_numpy()
    best_val_mae = float("inf")
    best_state = best_preds = best_ys = None
    best_epoch = -1
    start_epoch = 0

    # --- Resume from <run_name>_last.pt if it exists (Hyak preempt-safe) ---
    resume = load_full_checkpoint(cfg.run_name) if save else None
    if resume is not None:
        print(f"[resume] found {cfg.run_name}_last.pt — continuing from epoch {resume['epoch'] + 1}")
        net.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        torch.set_rng_state(resume["torch_rng"])
        np.random.set_state(resume["np_rng"])
        history = resume["history"]
        best_val_mae = resume["best_val_mae"]
        best_state = resume["best_state"]
        best_preds = resume["best_preds"]
        best_ys = resume["best_ys"]
        best_epoch = resume["best_epoch"]
        start_epoch = resume["epoch"] + 1

    for ep in range(start_epoch, cfg.epochs):
        t0 = time.time()
        # Tell the FDS wrapper which epoch this is (for start_smooth gating)
        if cfg.use_fds:
            net.current_epoch = ep
            fds_state = {"feats": [], "labels": []}
        else:
            fds_state = None

        train_mae = train_one_epoch(net, train_loader, opt, device, fds_state=fds_state)
        sched.step()

        # FDS post-epoch update: feed accumulated raw features + labels to running stats
        if cfg.use_fds:
            all_feats = torch.cat(fds_state["feats"]).to(device)
            all_labels = torch.cat(fds_state["labels"]).to(device)
            bucket_labels = net._bucketize(all_labels)
            net.fds.update_running_stats(all_feats, bucket_labels, ep)
            net.fds.update_last_epoch_stats(ep + 1)

        val_preds, val_ys = evaluate(net, val_loader, device)
        val_mae = float(np.mean(np.abs(val_preds - val_ys)))
        dt = time.time() - t0

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_preds, best_ys = val_preds, val_ys
            best_epoch = ep
            # Write best weights immediately so an interrupted run still has a usable best.
            if save:
                save_checkpoint(best_state, cfg.run_name)

        print(f"ep {ep:2d}  train_mae={train_mae:.3f}  val_mae={val_mae:.3f}"
              f"{' [best]' if is_best else ''}  ({dt:.0f}s)")
        history.append({"epoch": ep, "train_mae": train_mae, "val_mae": val_mae,
                        "sec": dt, "is_best": is_best})

        # Save full resume state every epoch so preemption costs at most one epoch.
        if save:
            save_full_checkpoint({
                "epoch": ep,
                "model": net.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "history": history,
                "best_val_mae": best_val_mae,
                "best_state": best_state,
                "best_preds": best_preds,
                "best_ys": best_ys,
                "best_epoch": best_epoch,
                "torch_rng": torch.get_rng_state(),
                "np_rng": np.random.get_state(),
            }, cfg.run_name)

    # Report and persist using the best-val checkpoint, not the final epoch
    val_preds = best_preds if best_state is not None else val_preds
    val_ys = best_ys if best_state is not None else val_ys
    binned = per_bin_mae(val_ys, val_preds, train_y) if val_preds is not None else {}
    print(f"[best val MAE @ epoch {best_epoch}] overall={binned.get('overall', float('nan')):.3f}  "
          f"many={binned.get('many', float('nan')):.3f}  "
          f"medium={binned.get('medium', float('nan')):.3f}  "
          f"few={binned.get('few', float('nan')):.3f}")

    checkpoint_path = None
    if save and best_state is not None:
        checkpoint_path = save_checkpoint(best_state, cfg.run_name)

    results = {
        "run_name": cfg.run_name,
        "config": asdict(cfg),
        "device": device,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "final_val": binned,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "val_preds": val_preds.tolist() if val_preds is not None else [],
        "val_ys": val_ys.tolist() if val_ys is not None else [],
    }
    if save:
        save_results(results)
        # Run completed cleanly — the resume checkpoint is no longer needed.
        last = RESULTS_DIR / f"{cfg.run_name}_last.pt"
        if last.exists():
            last.unlink()
    return results


# --- CLI ---
def _parse_args():
    d = Config()  # use dataclass defaults as the argparse defaults
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet50", "vit_b_16"], default=d.model)
    ap.add_argument("--fold", type=int, default=d.fold)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--num-workers", type=int, default=d.num_workers)
    ap.add_argument("--train-subset", type=int, default=d.train_subset)
    ap.add_argument("--val-subset", type=int, default=d.val_subset)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--run-name", type=str, default=d.run_name)
    ap.add_argument("--no-save", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    cfg = Config(
        model=a.model, fold=a.fold, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, num_workers=a.num_workers, train_subset=a.train_subset,
        val_subset=a.val_subset, seed=a.seed, run_name=a.run_name,
    )
    run_baseline(cfg=cfg, save=not a.no_save)
