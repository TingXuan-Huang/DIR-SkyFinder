"""Measure DIR shot-bin populations at 1 C and 2 C — resolves U3 + the metric bin-width decision.

For each LOCO fold and each eval split (val, test), classify temperature bins into DIR
many/medium/few by TRAIN-set frequency (>=100 / 20-99 / <20 samples per bin), then count
how many eval samples land in each shot. Mirrors engine.per_bin_mae's binning exactly.

Run from repo root:  python experiments/restart-2026-05-24/measure_bins.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
df = pd.read_csv(REPO / "data" / "labels_with_images.csv")
splits = json.loads((REPO / "data" / "splits" / "loco_5fold.json").read_text())
temp = df["TempM"].to_numpy()


def shot_counts(train_y, eval_y, bin_w):
    """Replicate per_bin_mae binning; return (samples_per_shot, bins_per_shot)."""
    lo = min(train_y.min(), eval_y.min())
    hi = max(train_y.max(), eval_y.max())
    edges = np.arange(np.floor(lo / bin_w) * bin_w,
                      np.ceil(hi / bin_w) * bin_w + bin_w, bin_w)
    train_hist, _ = np.histogram(train_y, bins=edges)
    idx = np.clip(np.digitize(eval_y, edges) - 1, 0, len(edges) - 2)
    samples = {"many": 0, "medium": 0, "few": 0}
    nbins = {"many": 0, "medium": 0, "few": 0}
    for b in range(len(edges) - 1):
        c = train_hist[b]
        shot = "many" if c >= 100 else ("medium" if c >= 20 else "few")
        nbins[shot] += 1
        samples[shot] += int((idx == b).sum())
    return samples, nbins


for bin_w in (1.0, 2.0):
    print(f"\n===== bin_w = {bin_w} C =====")
    for split_name in ("val", "test"):
        tot = {"many": 0, "medium": 0, "few": 0}
        few_per_fold = []
        nbins_last = None
        for f in splits:
            s, nb = shot_counts(temp[f["train"]], temp[f[split_name]], bin_w)
            for k in tot:
                tot[k] += s[k]
            few_per_fold.append(s["few"])
            nbins_last = nb
        total = sum(tot.values())
        print(f"  {split_name}: many={tot['many']:,}  medium={tot['medium']:,}  few={tot['few']:,}"
              f"   (few = {100 * tot['few'] / total:.2f}% of {total:,} eval samples)")
        print(f"        few samples per fold: {few_per_fold}")
        print(f"        bins per shot (last fold): {nbins_last}")
