"""Label Distribution Smoothing (LDS) — per-sample weights for the loss.

LDS is a *loss-side* intervention. It doesn't change the model architecture;
it just multiplies the L1 loss by a per-sample weight that's larger for
rare-label samples and smaller for common-label samples.

Recipe:
  1. Bucketize labels into integer bin indices (default 1 C bins).
  2. Count samples per bucket -> histogram.
  3. Convolve histogram with a smoothing kernel (Gaussian sigma=2 default).
  4. Per-sample weight = 1 / sqrt(smoothed_count[my_bucket]),
     normalized so mean weight = 1.0.

Adapted from upstream DIR (https://github.com/YyzHarry/imbalanced-regression):
  - imdb-wiki-dir/datasets.py:_prepare_weights
  - imdb-wiki-dir/utils.py:get_lds_kernel_window
  - imdb-wiki-dir/loss.py:weighted_l1_loss

See questions.md #3 for the theory.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang

# SkyFinder TempM range is roughly -27.2 to +50.0 C; pad to -30..+55 for safety.
MIN_TEMP = -30.0
MAX_TEMP = 55.0


def bucketize(temps, bin_width: float = 1.0, min_temp: float = MIN_TEMP) -> np.ndarray:
    """Map continuous temperatures (C) to integer bucket indices."""
    return ((np.asarray(temps) - min_temp) / bin_width).astype(int)


def get_lds_kernel_window(kernel: str, ks: int, sigma: float) -> np.ndarray:
    """1-D smoothing kernel for the label histogram (Gaussian / triang / Laplace).
    Normalized so the peak value is 1.0 (matches upstream convention)."""
    assert kernel in {"gaussian", "triang", "laplace"}
    half_ks = (ks - 1) // 2
    if kernel == "gaussian":
        base = np.zeros(ks, dtype=np.float64)
        base[half_ks] = 1.0
        kw = gaussian_filter1d(base, sigma=sigma)
        return kw / kw.max()
    if kernel == "triang":
        return triang(ks)
    grid = np.arange(-half_ks, half_ks + 1)
    kw = np.exp(-np.abs(grid) / sigma) / (2.0 * sigma)
    return kw / kw.max()


def compute_lds_weights(
    temps,
    bin_width: float = 1.0,
    min_temp: float = MIN_TEMP,
    max_temp: float = MAX_TEMP,
    reweight: str = "sqrt_inv",
    lds_kernel: str = "gaussian",
    lds_ks: int = 5,
    lds_sigma: float = 2.0,
) -> np.ndarray:
    """Per-sample LDS weights for training. Returns float32 array, mean=1.0.

    Parameters
    ----------
    temps : array-like of float
        Temperature labels (C) for the training set.
    bin_width : float
        Bucket width in C. Default 1.0 (matches DIR paper's age-in-years convention).
    reweight : {"none", "sqrt_inv", "inverse"}
        "sqrt_inv": weight = 1/sqrt(smoothed_density). Paper default, conservative.
        "inverse":  weight = 1/smoothed_density. More aggressive.
        "none":     all weights = 1.0.
    lds_kernel, lds_ks, lds_sigma :
        Smoothing kernel parameters.
    """
    assert reweight in {"none", "sqrt_inv", "inverse"}
    temps = np.asarray(temps)

    if reweight == "none":
        return np.ones(len(temps), dtype=np.float32)

    n_buckets = int(np.ceil((max_temp - min_temp) / bin_width))
    buckets = np.clip(bucketize(temps, bin_width, min_temp), 0, n_buckets - 1)
    counts = np.bincount(buckets, minlength=n_buckets).astype(np.float64)

    kernel = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
    smoothed = convolve1d(counts, weights=kernel, mode="constant")
    smoothed = np.maximum(smoothed, 1.0)  # avoid div-by-zero in empty bins

    per_bucket = 1.0 / (np.sqrt(smoothed) if reweight == "sqrt_inv" else smoothed)
    weights = per_bucket[buckets]
    weights *= len(weights) / weights.sum()  # normalize mean to 1.0
    return weights.astype(np.float32)


def weighted_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                     weight: torch.Tensor | None = None) -> torch.Tensor:
    """L1 loss with optional per-sample weight. Returns scalar mean loss."""
    loss = F.l1_loss(pred, target, reduction="none")
    if weight is not None:
        loss = loss * weight
    return loss.mean()
