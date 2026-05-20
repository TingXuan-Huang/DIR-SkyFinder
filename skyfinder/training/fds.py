"""Feature Distribution Smoothing (FDS) — architecture-side intervention.

FDS sits between a backbone and a regression head. It tracks running per-bucket
mean and variance of the features going into the head, smooths them across
neighboring buckets, and calibrates features so rare-bucket samples look more
like their common-bucket neighbors.

This file is adapted from upstream DIR (https://github.com/YyzHarry/imbalanced-regression):
  - imdb-wiki-dir/fds.py     (FDS class)
  - imdb-wiki-dir/utils.py   (calibrate_mean_var, get_lds_kernel_window)

Changes vs upstream:
  - Made device-agnostic (removed hardcoded .cuda(); kernel_window registered as buffer
    so .to(device) moves it).
  - Inlined calibrate_mean_var.
  - Removed `print = logging.info` shim.
  - Smoothing kernel is sum-normalized (matches upstream's FDS convolution conv1d).

See questions.md #4 for the theory.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang


def _kernel_window(kernel: str, ks: int, sigma: float) -> torch.Tensor:
    """1-D smoothing kernel (Gaussian / triang / Laplace), sum-normalized to 1.0."""
    half_ks = (ks - 1) // 2
    if kernel == "gaussian":
        base = np.zeros(ks, dtype=np.float32)
        base[half_ks] = 1.0
        kw = gaussian_filter1d(base, sigma=sigma)
    elif kernel == "triang":
        kw = triang(ks).astype(np.float32)
    elif kernel == "laplace":
        grid = np.arange(-half_ks, half_ks + 1)
        kw = np.exp(-np.abs(grid) / sigma) / (2.0 * sigma)
    else:
        raise ValueError(f"unknown kernel: {kernel}")
    kw = kw / kw.sum()
    return torch.tensor(kw, dtype=torch.float32)


def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10.0):
    """Whiten `matrix` using (m1, v1), then re-color with (m2, v2).

    Used to make features from rare-label buckets look statistically similar
    to features from common-label buckets.
    """
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.0).any():
        valid = (v1 != 0.0)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix
    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


class FDS(nn.Module):
    """Feature Distribution Smoother (placeholder between backbone and head).

    Parameters
    ----------
    feature_dim : int
        Dimensionality of features being smoothed. 2048 for ResNet-50, 768 for ViT-B/16.
    bucket_num : int
        Number of label buckets. For TempM in [-30, 55] with 1 C bins, 85 buckets.
    bucket_start : int
        First bucket index to track (default 0).
    start_update : int
        Epoch at which to start updating running stats.
    start_smooth : int
        Epoch at which to begin applying feature calibration.
    kernel : {"gaussian", "triang", "laplace"}
        Smoothing kernel.
    ks : int
        Kernel size (odd).
    sigma : float
        Kernel sigma (Gaussian/Laplace).
    momentum : float
        EMA momentum for running stats.
    """

    def __init__(self, feature_dim: int, bucket_num: int = 85, bucket_start: int = 0,
                 start_update: int = 0, start_smooth: int = 1,
                 kernel: str = "gaussian", ks: int = 5, sigma: float = 2.0,
                 momentum: float = 0.9):
        super().__init__()
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.half_ks = (ks - 1) // 2
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth

        # All registered as buffers so .to(device) moves them.
        self.register_buffer("kernel_window", _kernel_window(kernel, ks, sigma))
        self.register_buffer("epoch", torch.zeros(1).fill_(start_update))
        n = bucket_num - bucket_start
        self.register_buffer("running_mean", torch.zeros(n, feature_dim))
        self.register_buffer("running_var", torch.ones(n, feature_dim))
        self.register_buffer("running_mean_last_epoch", torch.zeros(n, feature_dim))
        self.register_buffer("running_var_last_epoch", torch.ones(n, feature_dim))
        self.register_buffer("smoothed_mean_last_epoch", torch.zeros(n, feature_dim))
        self.register_buffer("smoothed_var_last_epoch", torch.ones(n, feature_dim))
        self.register_buffer("num_samples_tracked", torch.zeros(n))

    def _update_smoothed_stats(self):
        """Smooth running mean/var across label buckets via 1-D conv with kernel_window."""
        self.running_mean_last_epoch = self.running_mean
        self.running_var_last_epoch = self.running_var

        kw = self.kernel_window.view(1, 1, -1)
        self.smoothed_mean_last_epoch = F.conv1d(
            input=F.pad(self.running_mean_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode="reflect"),
            weight=kw, padding=0
        ).permute(2, 1, 0).squeeze(1)
        self.smoothed_var_last_epoch = F.conv1d(
            input=F.pad(self.running_var_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode="reflect"),
            weight=kw, padding=0
        ).permute(2, 1, 0).squeeze(1)

    def update_last_epoch_stats(self, epoch: int):
        """Call at end of train epoch — snapshots running stats and computes their smoothed form."""
        if epoch == self.epoch + 1:
            self.epoch += 1
            self._update_smoothed_stats()

    def update_running_stats(self, features: torch.Tensor, labels: torch.Tensor, epoch: int):
        """EMA update of per-bucket running mean/var. `labels` must be int bucket indices."""
        if epoch < self.epoch:
            return
        assert self.feature_dim == features.size(1)
        assert features.size(0) == labels.size(0)

        for label in torch.unique(labels):
            lbl = int(label.item())
            if lbl >= self.bucket_num or lbl < self.bucket_start:
                continue
            if lbl == self.bucket_start:
                curr_feats = features[labels <= label]
            elif lbl == self.bucket_num - 1:
                curr_feats = features[labels >= label]
            else:
                curr_feats = features[labels == label]
            n_curr = curr_feats.size(0)
            curr_mean = curr_feats.mean(0)
            curr_var = curr_feats.var(0, unbiased=(n_curr != 1))

            idx = lbl - self.bucket_start
            self.num_samples_tracked[idx] += n_curr
            factor = self.momentum if self.momentum is not None else (
                1.0 - n_curr / float(self.num_samples_tracked[idx])
            )
            factor = 0.0 if epoch == self.start_update else factor
            self.running_mean[idx] = (1 - factor) * curr_mean + factor * self.running_mean[idx]
            self.running_var[idx] = (1 - factor) * curr_var + factor * self.running_var[idx]

    def smooth(self, features: torch.Tensor, labels: torch.Tensor, epoch: int) -> torch.Tensor:
        """Apply feature calibration. `labels` must be int bucket indices, shape (B,) or (B, 1).
        Returns a new tensor (does NOT modify `features` in place — we clone inside)."""
        if epoch < self.start_smooth:
            return features
        if labels.dim() == 2:
            labels = labels.squeeze(1)

        out = features.clone()
        for label in torch.unique(labels):
            lbl = int(label.item())
            if lbl >= self.bucket_num or lbl < self.bucket_start:
                continue
            if lbl == self.bucket_start:
                mask = labels <= label
            elif lbl == self.bucket_num - 1:
                mask = labels >= label
            else:
                mask = labels == label
            idx = lbl - self.bucket_start
            out[mask] = calibrate_mean_var(
                out[mask],
                self.running_mean_last_epoch[idx], self.running_var_last_epoch[idx],
                self.smoothed_mean_last_epoch[idx], self.smoothed_var_last_epoch[idx],
            )
        return out

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1.0)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1.0)
        self.num_samples_tracked.zero_()
