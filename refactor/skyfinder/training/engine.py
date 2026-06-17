"""Per-epoch training and evaluation. Device detection. Per-bin MAE metric.

This is the standard PyTorch "engine" file: one function per training-loop primitive.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .lds import weighted_l1_loss


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_one_epoch(model, loader, optimizer, device, fds_state: dict | None = None):
    """One pass over `loader`. Returns mean training loss.

    If `fds_state` is a dict with `"feats"` and `"labels"` empty lists, treats
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


def predict_split(model, loader, device):
    """Return raw (preds, ys) as 1-D numpy arrays. Works with any loader yielding (x, y, *)."""
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


# Backwards-compatible alias for older callers. New code should use `predict_split`.
evaluate = predict_split


def per_bin_mae(y_true, y_pred, train_y, bin_w=1.0):
    """MAE in `bin_w`-°C bins, classified by training-set frequency (DIR many/medium/few).

    Default 1.0 °C: paper-faithful (matches LDS/FDS bin_width) and populates the few-bin
    better than 2.0 (see experiments/restart-2026-05-24/measure_bins.py — the U3 finding).
    """
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


def per_bin_mae_by_edges(y_true, y_pred, edges):
    """Per-bin MAE on caller-provided bin edges. Returns array of length len(edges)-1.

    NaN for empty bins. Used by figures (e.g. `fig_dist_and_errbar`) that need
    raw per-bin error on arbitrary edges, not the DIR many/medium/few classification.
    """
    idx = np.clip(np.digitize(y_true, edges) - 1, 0, len(edges) - 2)
    err = np.abs(y_true - y_pred)
    return np.array([err[idx == k].mean() if (idx == k).any() else np.nan
                     for k in range(len(edges) - 1)])
