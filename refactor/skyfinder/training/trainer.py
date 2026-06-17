"""Training-loop orchestration: `run_baseline(cfg)`.

The model factories, Dataset, engine helpers, and checkpoint I/O live in
their own modules (`model.py`, `dataloader.py`, `engine.py`, `checkpoint.py`).
This file is pure orchestration: build → resume? → epoch loop → save.

Usage:
    from skyfinder.training.trainer import run_baseline
    results = run_baseline(model="resnet50", epochs=20)

For YAML-driven experiment runs, use `skyfinder train --family X` (see `skyfinder.cli`).
"""
from __future__ import annotations

import time
from dataclasses import asdict, replace
from datetime import datetime

import numpy as np
import torch

from . import config as cfg_module
from .checkpoint import (load_training_state, save_model_weights,
                                            save_results, save_training_state,
                                            subdir_for)
from .config import Config
from .dataloader import build_loaders
from .engine import (get_device, per_bin_mae, predict_split,
                                        train_one_epoch)
from .fds import FDS
from .lds import MIN_TEMP, compute_lds_weights
from .model import FDSModel, build_model


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

    # --- Build loaders first (applies F-family corruption to train labels if
    # cfg.corruption is set), then compute LDS weights from the resulting train_df
    # so bin frequencies reflect the post-corruption distribution. ---
    train_loader, val_loader, train_df, val_df = build_loaders(cfg)
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
        train_loader.dataset.weights = train_weights
        print(f"[lds] weights computed: min={train_weights.min():.3f}  "
              f"max={train_weights.max():.3f}  mean={train_weights.mean():.3f}")

    # --- Model: vanilla or FDS-wrapped ---
    vanilla = build_model(cfg.model, freeze_backbone=cfg.freeze_backbone)
    if cfg.use_fds:
        feature_dim = 2048 if cfg.model == "resnet50" else 768
        bucket_num = int(np.ceil((55.0 - MIN_TEMP) / cfg.bin_width))
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
              f"trainable ({100 * n_train / max(n_total, 1):.2f}%)")
    opt = torch.optim.Adam(trainable, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    history: list = []
    train_y = train_df["TempM"].to_numpy()
    best_val_mae = float("inf")
    best_state = best_preds = best_ys = None
    best_epoch = -1
    start_epoch = 0
    val_preds: np.ndarray | None = None
    val_ys: np.ndarray | None = None

    # --- Resume from <run_name>_last.pt if it exists (Hyak preempt-safe) ---
    resume = load_training_state(cfg.run_name) if save else None
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

    # Snapshot trajectory: dump initial state (pre-FT) on a fresh run.
    if save and cfg.snapshot_every > 0 and start_epoch == 0:
        save_model_weights(net.state_dict(), f"{cfg.run_name}_ep0")
        print(f"[snapshot] saved initial state -> {cfg.run_name}_ep0.pt")

    for ep in range(start_epoch, cfg.epochs):
        t0 = time.time()
        if cfg.use_fds:
            net.current_epoch = ep
            fds_state: dict | None = {"feats": [], "labels": []}
        else:
            fds_state = None

        train_mae = train_one_epoch(net, train_loader, opt, device, fds_state=fds_state)
        sched.step()

        if cfg.use_fds and fds_state is not None:
            all_feats = torch.cat(fds_state["feats"]).to(device)
            all_labels = torch.cat(fds_state["labels"]).to(device)
            bucket_labels = net._bucketize(all_labels)
            net.fds.update_running_stats(all_feats, bucket_labels, ep)
            net.fds.update_last_epoch_stats(ep + 1)

        val_preds, val_ys = predict_split(net, val_loader, device)
        val_mae = float(np.mean(np.abs(val_preds - val_ys)))
        dt = time.time() - t0

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_preds, best_ys = val_preds, val_ys
            best_epoch = ep
            if save:
                save_model_weights(best_state, cfg.run_name)

        print(f"ep {ep:2d}  train_mae={train_mae:.3f}  val_mae={val_mae:.3f}"
              f"{' [best]' if is_best else ''}  ({dt:.0f}s)")
        history.append({"epoch": ep, "train_mae": train_mae, "val_mae": val_mae,
                        "sec": dt, "is_best": is_best})

        # Trajectory snapshot every Nth completed epoch.
        if save and cfg.snapshot_every > 0 and (ep + 1) % cfg.snapshot_every == 0:
            save_model_weights(net.state_dict(), f"{cfg.run_name}_ep{ep + 1}")
            print(f"[snapshot] saved {cfg.run_name}_ep{ep + 1}.pt")

        if save:
            save_training_state({
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

    # Report and persist using the best-val checkpoint, not the final epoch.
    val_preds = best_preds if best_state is not None else val_preds
    val_ys = best_ys if best_state is not None else val_ys
    binned = per_bin_mae(val_ys, val_preds, train_y) if val_preds is not None else {}
    print(f"[best val MAE @ epoch {best_epoch}] overall={binned.get('overall', float('nan')):.3f}  "
          f"many={binned.get('many', float('nan')):.3f}  "
          f"medium={binned.get('medium', float('nan')):.3f}  "
          f"few={binned.get('few', float('nan')):.3f}")

    checkpoint_path = None
    if save and best_state is not None:
        checkpoint_path = save_model_weights(best_state, cfg.run_name)

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
        last = cfg_module.RESULTS_DIR / subdir_for(cfg.run_name) / f"{cfg.run_name}_last.pt"
        if last.exists():
            last.unlink()
    return results
