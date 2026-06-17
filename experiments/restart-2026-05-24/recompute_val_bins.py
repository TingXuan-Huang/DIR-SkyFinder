"""Phase 4: recompute VAL per-bin MAE for the completed ResNet-50 DIR sweep.

The stored final_val in each archived run JSON was computed at bin_w=2.0 (old default).
Here we recompute from raw val_preds/val_ys at BOTH 2.0 (to verify our pipeline matches
the stored numbers) and 1.0 (the U3-corrected granularity), aggregating mean +/- std across
the available folds per condition.

Raw runs: data/old_outputs/server_results/results/<cond>_fold<k>.json
Run from repo root:  python experiments/restart-2026-05-24/recompute_val_bins.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "refactor")
from skyfinder.training.engine import per_bin_mae

REPO = Path(".").resolve()
df = pd.read_csv(REPO / "data" / "labels_with_images.csv")
splits = json.loads((REPO / "data" / "splits" / "loco_5fold.json").read_text())
temp = df["TempM"].to_numpy()
SR = REPO / "data" / "old_outputs" / "server_results" / "results"

CONDS = ["baseline_resnet50", "lds_resnet50", "fds_resnet50", "lds_fds_resnet50"]
BINS = ["overall", "many", "medium", "few"]


def agg(cond, bin_w):
    rows, stored = [], []
    for jp in sorted(glob.glob(str(SR / f"{cond}_fold*.json"))):
        d = json.load(open(jp))
        fold = int(jp.split("_fold")[1].split(".")[0])
        train_y = temp[splits[fold]["train"]]
        m = per_bin_mae(np.array(d["val_ys"]), np.array(d["val_preds"]), train_y, bin_w=bin_w)
        rows.append([m.get(b, np.nan) for b in BINS])
        stored.append(d["final_val"].get("overall", np.nan))
    return np.array(rows), np.array(stored)


def show(bin_w):
    print(f"\n===== VAL per-bin MAE  (bin_w = {bin_w} C) =====")
    print(f"{'condition':20s} | " + " | ".join(f"{b:>12s}" for b in BINS) + " | folds")
    for cond in CONDS:
        a, _ = agg(cond, bin_w)
        cells = []
        for j in range(len(BINS)):
            col = a[:, j][~np.isnan(a[:, j])]
            cells.append(f"{col.mean():5.2f}±{col.std():4.2f}" if len(col) else "    n/a    ")
        print(f"{cond:20s} | " + " | ".join(f"{c:>12s}" for c in cells) + f" | {a.shape[0]}")


# Verification: our 2.0 recompute should match each run's stored final_val.overall
print("=== verify recompute matches stored final_val.overall (bin_w=2.0) ===")
for cond in CONDS:
    a, stored = agg(cond, 2.0)
    diff = np.abs(a[:, 0] - stored)
    print(f"{cond:20s}: max |recompute - stored| = {diff.max():.4f}  (n={len(stored)})")

show(2.0)
show(1.0)
